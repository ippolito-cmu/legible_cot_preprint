import json
import os
import pickle
import re
import time
import warnings
import shutil
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from dotenv import load_dotenv
import asyncio
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
import re
os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['GRPC_TRACE'] = ''
os.environ['GOOGLE_CLOUD_DISABLE_GRPC_FOR_REST'] = 'true'
from google import genai
from google.genai import types
import logging
logging.getLogger('google').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
import sys
from src.utils.prompts import system_instruction_backtracking
_BATCH_PRICE_INPUT_PER_1M_TOKENS_USD = 0.30
_BATCH_PRICE_OUTPUT_PER_1M_TOKENS_USD = 2.50
def _pricing_rates_usd_per_1m_tokens(*, pricing_tier: str) -> Tuple[float, float]:
    """Return (input_rate, output_rate) in USD per 1M tokens.

    Args:
        pricing_tier (str): 'batch' (default in this repo) or 'standard'.

    Returns:
        Tuple[float, float]: (input_usd_per_1m, output_usd_per_1m)
    """
    tier = (pricing_tier or 'batch').strip().lower()
    if tier == 'batch':
        return _BATCH_PRICE_INPUT_PER_1M_TOKENS_USD, _BATCH_PRICE_OUTPUT_PER_1M_TOKENS_USD
    if tier == 'standard':
        return 2.0 * _BATCH_PRICE_INPUT_PER_1M_TOKENS_USD, 2.0 * _BATCH_PRICE_OUTPUT_PER_1M_TOKENS_USD
    raise ValueError(f"Unknown pricing_tier={pricing_tier!r} (expected 'batch' or 'standard')")
def _token_counter_model_name() -> str:
    """Return the model name used for token counting.

    Defaults to Gemini 2.5 Flash, per user request.
    """
    return os.getenv('BACKTRACKING_TOKEN_COUNTER_MODEL', 'gemini-2.5-flash')
@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
def _count_tokens_for_text(client: Any, text: str) -> int:
    """Count tokens for a text using Gemini's official token counter."""
    model_name = _token_counter_model_name()
    resp = client.models.count_tokens(model=model_name, contents=text)
    total = getattr(resp, 'total_tokens', None)
    if total is None:
        try:
            total = resp.get('total_tokens')
        except Exception:
            total = None
    if total is None:
        raise ValueError('count_tokens response missing total_tokens')
    return int(total)
def _sample_indices(n: int, k: int) -> List[int]:
    """Deterministic sample: first k indices in [0, n)."""
    return list(range(min(n, max(0, k))))
def _estimate_output_tokens_from_existing(
    *,
    client: Any,
    existing_results: List['BacktrackingResult'],
    sample_k: int,
) -> Optional[float]:
    """Estimate average output tokens from already-parsed results.

    We serialize the canonical JSON object we asked for and count tokens on that.
    Returns None if there are no usable existing results.
    """
    if not existing_results:
        print("no results")
        return None
    with_usage = [
        r
        for r in existing_results
        if isinstance(getattr(r, 'output_tokens', None), int) and r.output_tokens is not None
    ]
    if with_usage:
        sample = with_usage[: min(sample_k, len(with_usage))]
        return float(np.mean([int(r.total_tokens - r.prompt_tokens) for r in sample]))
    good = [r for r in existing_results if not r.error]
    if not good:
        print("not good")
        return None
    sample = good[: min(sample_k, len(good))]
    token_counts: List[int] = []
    for r in sample:
        payload = {
            'backtracking_detected': bool(r.backtracking_detected),
            'final_answer': r.final_answer or '',
            'backtracking_steps': r.backtracking_steps or [],
            'confidence': float(r.confidence) if r.confidence is not None else 0.0,
            'overall_reasoning': r.overall_reasoning or '',
        }
        token_counts.append(_count_tokens_for_text(client, json.dumps(payload, ensure_ascii=False)))
    return float(np.mean(token_counts)) if token_counts else None
def _estimate_cost_for_pair(
    *,
    dataset: str,
    model: str,
    output_dir: Path,
    smoke_n: int,
    fail_rate: float,
    sample_k_prompts: int,
    sample_k_outputs: int,
    pricing_tier: str,
) -> Optional[Dict[str, Any]]:
    """Estimate token cost for remaining traces for a single (dataset, model).

    Uses Gemini token counter for sampled prompts and sampled outputs, then extrapolates.
    """
    pickle_path = Path('outputs/traces') / dataset / f'traces_{model}.pkl'
    if not pickle_path.exists():
        return None
    data = load_trace_pickle(pickle_path)
    traces, _ = get_traces_and_correctness(data, dataset)
    if smoke_n and smoke_n > 0:
        traces = traces[: min(smoke_n, len(traces))]
    results_path = _results_path(output_dir, dataset, model)
    existing_results = _load_existing_results(results_path)
    processed = _processed_indices(existing_results)
    failed_pool = _load_failed_pool(output_dir, dataset, model)
    inflight = _inflight_indices(_batch_dir(output_dir), dataset, model)
    print(len(inflight), "inflight")
    print(len(processed), "processed")
    print(len(failed_pool), "failed pool")
    print(len(traces), "total traces")
    print(results_path)
    available_indices = set(range(len(traces)))
    target = (available_indices - processed) | failed_pool
    target = {i for i in target if i not in inflight and 0 <= i < len(traces)}
    print(len(target), "remaining targets")
    remaining = len(target)
    if remaining == 0:
        return {
            'dataset': dataset,
            'model': model,
            'pricing_tier': (pricing_tier or 'batch').strip().lower(),
            'remaining_traces': 0,
            'assumed_fail_rate': fail_rate,
            'estimated_reprocess_count': 0,
            'avg_prompt_tokens_est': 0.0,
            'avg_output_tokens_est': 0.0,
            'estimated_total_input_tokens': 0,
            'estimated_total_output_tokens': 0,
            'estimated_total_cost_usd': 0.0,
            'estimated_input_cost_usd': 0.0,
            'estimated_output_cost_usd': 0.0,
            'token_counter_model': _token_counter_model_name(),
        }
    client = _ensure_gemini_client()
    system_instruction = system_instruction_backtracking[dataset]
    remaining_indices = sorted(target)
    sample_indices = [remaining_indices[i] for i in _sample_indices(len(remaining_indices), sample_k_prompts)]
    prompt_tokens: List[int] = []
    for idx in sample_indices:
        user_prompt = build_backtracking_prompt(traces[idx])
        prompt_tokens.append(_count_tokens_for_text(client, system_instruction + "\n\n" + user_prompt))
    avg_prompt_tokens = float(np.mean(prompt_tokens)) if prompt_tokens else 0.0
    est_total_input_tokens_initial = int(round(avg_prompt_tokens * remaining))
    avg_output_tokens = _estimate_output_tokens_from_existing(
        client=client,
        existing_results=existing_results,
        sample_k=sample_k_outputs,
    )
    if avg_output_tokens is None:
        try:
            max_output_tokens = int(os.getenv('BACKTRACKING_MAX_OUTPUT_TOKENS', '2048'))
        except Exception:
            max_output_tokens = 2048 * 2
        avg_output_tokens = float(2048)
    est_total_output_tokens_initial = int(round(avg_output_tokens * remaining))
    est_total_input_tokens = (fail_rate + 1) * (est_total_input_tokens_initial)
    est_total_output_tokens = (fail_rate + 1) * (est_total_output_tokens_initial)
    input_rate, output_rate = _pricing_rates_usd_per_1m_tokens(pricing_tier=pricing_tier)
    input_cost = (est_total_input_tokens / 1_000_000.0) * input_rate
    output_cost = (est_total_output_tokens / 1_000_000.0) * output_rate
    return {
        'dataset': dataset,
        'model': model,
        'pricing_tier': (pricing_tier or 'batch').strip().lower(),
        'remaining_traces': remaining,
        'assumed_fail_rate': fail_rate,
        'estimated_reprocess_count': int(np.ceil(remaining * max(0.0, min(1.0, fail_rate)))),
        'avg_prompt_tokens_est': avg_prompt_tokens,
        'avg_output_tokens_est': float(avg_output_tokens),
        'estimated_total_input_tokens': est_total_input_tokens,
        'estimated_total_output_tokens': est_total_output_tokens,
        'estimated_input_cost_usd': float(input_cost),
        'estimated_output_cost_usd': float(output_cost),
        'estimated_total_cost_usd': float(input_cost + output_cost),
        'token_counter_model': _token_counter_model_name(),
    }
@dataclass
class SubmittedBatch:
    job_name: str
    batch_file_path: Path
    metadata_file: Path
    dataset: str
    model: str
    batch_type: str
    attempt: int
    indices: List[int]
MODEL_COLORS = {
    'Qwen/Qwen3-0.6B': 'hsl(15, 85%, 45%)',
    'Qwen/Qwen3-4B': 'hsl(22, 85%, 40%)',
    'Qwen/Qwen3-8B': 'hsl(30, 85%, 35%)',
    'Qwen/QwQ-32B': 'hsl(38, 85%, 38%)',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B': 'hsl(230, 85%, 40%)',
    'deepseek-ai/deepseek-r1-0528': 'hsl(245, 85%, 45%)',
    'google/gemma-3-12b-it': 'hsl(172, 85%, 35%)',
    'google/gemma-3-27b-it': 'hsl(184, 85%, 32%)',
    'gpt-5': 'hsl(285, 85%, 45%)',
    'openai/gpt-oss-20b': 'hsl(300, 85%, 42%)',
    'openai/gpt-oss-120b': 'hsl(315, 85%, 40%)',
    'meta-llama/Meta-Llama-3.1-8B-Instruct': 'hsl(203, 85%, 42%)',
    'meta-llama/Meta-Llama-3.1-70B-Instruct': 'hsl(215, 85%, 38%)',
    'mistralai/Magistral-Small-2509': 'hsl(48, 80%, 40%)',
    'nvidia/Llama-3.1-Nemotron-Nano-8B-v1': 'hsl(138, 85%, 35%)',
    'nvidia/OpenReasoning-Nemotron-32B': 'hsl(150, 85%, 32%)',
    'math_ground_truth': 'hsl(145, 85%, 42%)',
}
def get_model_color(model_name):
    """Get consistent color for a model"""
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    normalized_name = model_name.replace('/', '_')
    if normalized_name in MODEL_COLORS:
        return MODEL_COLORS[normalized_name]
    alt_name = model_name.replace('_', '/')
    if alt_name in MODEL_COLORS:
        return MODEL_COLORS[alt_name]
    hash_val = hash(model_name)
    hue = (hash_val % 360)
    lightness = 35 + (abs(hash_val) % 15)
    return f'hsl({hue}, 80%, {lightness}%)'
def hsl_to_rgb(hsl_str):
    """Convert HSL string to RGB tuple for matplotlib."""
    import colorsys
    if not hsl_str.startswith('hsl('):
        return hsl_str
    parts = hsl_str[4:-1].split(',')
    h = int(parts[0]) / 360.0
    s = int(parts[1].strip('%')) / 100.0
    l = int(parts[2].strip('%')) / 100.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)
load_dotenv(f".{os.getenv('USER')}.env" if os.getenv("USER") else ".env", override=False)
class ParseErrorFilter(logging.Filter):
    def filter(self, record):
        return "Failed to parse result for idx" not in record.getMessage() and "countTokens" not in record.getMessage()
log_level = os.getenv('BACKTRACKING_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.addFilter(ParseErrorFilter())
logger.info("Loaded environment variables from .env")
_api_configured = False
_client = None
def _ensure_gemini_client():
    global _api_configured, _client
    if not _api_configured:
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        _client = genai.Client(api_key=api_key)
        _api_configured = True
        logger.info("Gemini client configured")
    return _client
@dataclass
class BacktrackingResult:
    index: int
    backtracking_detected: bool
    final_answer: str
    backtracking_steps: List[Dict]
    confidence: float
    overall_reasoning: str
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
def load_trace_pickle(pickle_path: Path) -> Dict:
    """Load a trace pickle file."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)
def get_traces_and_correctness(data: Dict, dataset: str) -> Tuple[List[str], List[int]]:
    """Extract traces and correctness scores from loaded data."""
    traces = data['data']['traces']
    scores = data['data']['scores']
    return traces, scores
def _results_path(output_dir: Path, dataset: str, model: str) -> Path:
    return output_dir / f'backtracking_{dataset}_{model}.pkl'
def _batch_dir(output_dir: Path) -> Path:
    d = output_dir / 'batch_jobs'
    d.mkdir(exist_ok=True, parents=True)
    return d
def _failed_pool_path(output_dir: Path, dataset: str, model: str) -> Path:
    return _batch_dir(output_dir) / f'failed_to_reprocess_{dataset}_{model}.json'
def _load_failed_pool(output_dir: Path, dataset: str, model: str) -> Set[int]:
    path = _failed_pool_path(output_dir, dataset, model)
    if not path.exists():
        return set()
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return {int(x) for x in data}
        return set()
    except Exception:
        return set()
def _save_failed_pool(output_dir: Path, dataset: str, model: str, indices: Set[int]) -> None:
    path = _failed_pool_path(output_dir, dataset, model)
    with open(path, 'w') as f:
        json.dump(sorted(indices), f, indent=2)
def _load_existing_results(results_path: Path) -> List['BacktrackingResult']:
    if not results_path.exists():
        return []
    try:
        with open(results_path, 'rb') as f:
            existing_data = pickle.load(f)
        if isinstance(existing_data, dict):
            raw_results = existing_data.get('results', [])
        elif isinstance(existing_data, list):
            raw_results = existing_data
        else:
            return []
        parsed: List[BacktrackingResult] = []
        for item in raw_results:
            if isinstance(item, BacktrackingResult):
                parsed.append(item)
                continue
            if isinstance(item, dict):
                normalized = _normalize_backtracking_dict(item)
                payload = {
                    'index': int(item.get('index', -1)),
                    'backtracking_detected': bool(normalized.get('backtracking_detected', False)),
                    'final_answer': str(normalized.get('final_answer', '') or ''),
                    'backtracking_steps': normalized.get('backtracking_steps', []) or [],
                    'confidence': float(normalized.get('confidence', 0.0) or 0.0),
                    'overall_reasoning': str(normalized.get('overall_reasoning', '') or ''),
                    'error': item.get('error', None),
                    'prompt_tokens': item.get('prompt_tokens', None),
                    'output_tokens': item.get('output_tokens', None),
                    'total_tokens': item.get('total_tokens', None),
                    'finish_reason': item.get('finish_reason', None),
                }
                parsed.append(BacktrackingResult(**payload))
        return parsed
    except Exception as e:
        print(e)
        return []
def _processed_indices(existing_results: List['BacktrackingResult']) -> Set[int]:
    return {r.index for r in existing_results}
def _inflight_indices(batch_dir: Path, dataset: str, model: str) -> Set[int]:
    inflight: Set[int] = set()
    for meta_path in batch_dir.glob('*_metadata.json'):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get('dataset') != dataset or meta.get('model') != model:
                continue
            if meta.get('status') in {'completed', 'failed', 'cancelled', 'expired'}:
                continue
            if isinstance(meta.get('indices'), list):
                for idx in meta['indices']:
                    inflight.add(int(idx))
            else:
                start = meta.get('start_index')
                end = meta.get('end_index')
                if isinstance(start, int) and isinstance(end, int) and end > start:
                    inflight |= set(range(start, end))
        except Exception:
            continue
    return inflight
def _correctness_for_results(results: List['BacktrackingResult'], correctness: List[int]) -> List[int]:
    out: List[int] = []
    for r in results:
        if 0 <= r.index < len(correctness):
            out.append(int(correctness[r.index]))
        else:
            out.append(0)
    return out
def str2bool(v):
    """Convert common string representations to boolean for argparse.

    Accepts values like 'true', 'false', '1', '0', 'yes', 'no'.
    If called with no value (None) it returns True (matches present flag semantics).
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    if isinstance(v, str):
        v_lower = v.lower()
        if v_lower in ('true', '1', 't', 'yes', 'y'):
            return True
        if v_lower in ('false', '0', 'f', 'no', 'n'):
            return False
    raise argparse.ArgumentTypeError('Boolean value expected.')
def smoke_arg(v):
    """Parse --smoke-test which can be: absent, present, true/false, or an integer N.

    - If flag is present with no value: returns 75
    - If value is truthy: returns 75
    - If value is falsy: returns 0
    - If value is an int (or int-like string): returns that int
    """
    if v is None:
        return 75
    if isinstance(v, int):
        return v
    if isinstance(v, bool):
        return 75 if v else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s.isdigit():
            return int(s)
        if s in ('true', '1', 't', 'yes', 'y'):
            return 75
        if s in ('false', '0', 'f', 'no', 'n'):
            return 0
    raise argparse.ArgumentTypeError('smoke-test expects true/false or an integer N')
@retry(wait=wait_random_exponential(multiplier=1, min=4, max=120), stop=stop_after_attempt(5))
def call_gemini_api(prompt: str, system_instruction: str, model_name: str = "gemini-2.5-flash") -> str:
    """Calls Gemini API with retry logic."""
    client = _ensure_gemini_client()
    logger.debug("Calling Gemini API with model %s", model_name)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            max_output_tokens=8192,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini response missing text")
    logger.debug("Gemini API call successful, response length: %d", len(text))
    return text.strip()
def _schema_text() -> str:
    return (
        "Return ONLY valid JSON (no markdown, no extra text).\n"
        "Schema (all keys required):\n"
        "{\n"
        "  \"backtracking_detected\": boolean,\n"
        "  \"final_answer\": string,\n"
        "  \"backtracking_steps\": [\n"
        "    {\n"
        "      \"step_number\": integer,\n"
        "      \"reason\": string\n"
        "    }\n"
        "  ],\n"
        "  \"confidence\": number,\n"
        "  \"overall_reasoning\": string\n"
        "}\n"
        "Constraints:\n"
        "- backtracking_steps must be [] if no backtracking.\n"
        "- confidence must be in [0, 1].\n"
    )
def build_backtracking_prompt(trace: str) -> str:
    return (
        "Reasoning trace:\n"
        "--- TRACE START ---\n"
        f"{trace}\n"
        "--- TRACE END ---\n"
    )
def build_reprocess_prompt(bad_output: str) -> str:
    return (
        "You are a JSON repair tool. "
        "You will be given an invalid or truncated JSON-like output. "
        "Your job is to output ONLY valid JSON that matches the schema.\n\n"
        f"{_schema_text()}\n"
        "Here is the invalid output to repair (may be incomplete):\n"
        "--- BAD OUTPUT START ---\n"
        f"{bad_output}\n"
        "--- BAD OUTPUT END ---\n\n"
        "Output ONLY the repaired JSON."
    )
def _extract_json_text(text: str) -> str:
    if not text:
        return text
    content = str(text).strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif content.startswith("```"):
        parts = content.split("```", 2)
        if len(parts) >= 2:
            content = parts[1].strip()
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1 and end > start:
        return content[start:end + 1]
    return content
def _parse_backtracking_json(text: str) -> Dict[str, Any]:
    return json.loads(_extract_json_text(text))
def _normalize_backtracking_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        'backtracking_detected': bool(d.get('backtracking_detected', False)),
        'final_answer': str(d.get('final_answer', '') or ''),
        'backtracking_steps': d.get('backtracking_steps', []) or [],
        'confidence': float(d.get('confidence', 0.0) or 0.0),
        'overall_reasoning': str(d.get('overall_reasoning', '') or ''),
    }
    if out['confidence'] < 0:
        out['confidence'] = 0.0
    if out['confidence'] > 1:
        out['confidence'] = 1.0
    if not isinstance(out['backtracking_steps'], list):
        out['backtracking_steps'] = []
    normalized_steps: List[Dict[str, Any]] = []
    for s in out['backtracking_steps']:
        if not isinstance(s, dict):
            continue
        step_number = s.get('step_number', None)
        reason = s.get('reason', None)
        if reason is None:
            reason = s.get('reasoning', None)
        if reason is None:
            parts: List[str] = []
            for k in ('original_claim', 'contradiction_found', 'correction_made'):
                v = s.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(f"{k}: {v.strip()}")
            reason = '; '.join(parts)
        try:
            step_number_int = int(step_number) if step_number is not None else 0
        except Exception:
            step_number_int = 0
        normalized_steps.append({'step_number': step_number_int, 'reason': str(reason or '')})
    out['backtracking_steps'] = normalized_steps
    return out
def create_batch_jsonl_file(request_lines: List[Dict[str, Any]], batch_file_path: Path) -> None:
    """Create a JSONL file for batch processing.

    Each line must be: {"key": str, "request": <GenerateContentRequest dict>}.
    """
    logger.info("Creating batch JSONL file with %d requests", len(request_lines))
    with open(batch_file_path, 'w', encoding='utf-8') as f:
        for line in request_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    logger.info("Created batch file: %s", batch_file_path)
def submit_batch_job(jsonl_file_path: Path, display_name: str) -> str:
    """Submit a batch job and return the job name."""
    client = _ensure_gemini_client()
    logger.info(f"Uploading batch file: {jsonl_file_path}")
    uploaded_file = client.files.upload(
        file=str(jsonl_file_path),
        config=types.UploadFileConfig(
            display_name=f"{display_name}_input",
            mime_type='jsonl'
        )
    )
    logger.info(f"Uploaded file: {uploaded_file.name}")
    logger.info("Creating batch job...")
    batch_job = client.batches.create(
        model="models/gemini-2.5-flash",
        src=types.BatchJobSource(file_name=uploaded_file.name),
        config=types.CreateBatchJobConfig(display_name=display_name),
    )
    job_name = batch_job.name
    logger.info(f"Created batch job: {job_name}")
    return job_name
def submit_traces_batch(
    *,
    dataset: str,
    model: str,
    batch_type: str,
    attempt: int,
    indices: List[int],
    prompts: List[str],
    system_instruction: Optional[str] = None,
    output_dir: Path,
    max_output_tokens_override: Optional[int] = None,
    max_output_tokens_multiplier: float = 1.0,
) -> SubmittedBatch:
    """Submit a single batch job and exit (no waiting)."""
    batch_dir = _batch_dir(output_dir)
    try:
        max_output_tokens = int(os.getenv('BACKTRACKING_MAX_OUTPUT_TOKENS', '2048'))
    except Exception:
        max_output_tokens = 2048
    if max_output_tokens <= 0:
        max_output_tokens = 2048
    if isinstance(max_output_tokens_override, int) and max_output_tokens_override > 0:
        max_output_tokens = max_output_tokens_override
    else:
        try:
            max_output_tokens = int(round(float(max_output_tokens) * float(max_output_tokens_multiplier)))
        except Exception:
            max_output_tokens = max_output_tokens
    if max_output_tokens <= 0:
        max_output_tokens = 2048
    timestamp = int(time.time())
    batch_file_path = batch_dir / f"batch_{batch_type}_{dataset}_{model}_{timestamp}.jsonl"
    request_lines: List[Dict[str, Any]] = []
    for idx, prompt in zip(indices, prompts):
        request: Dict[str, Any] = {
            'contents': [
                {
                    'parts': [{'text': prompt}],
                    'role': 'user',
                }
            ],
            'generationConfig': {
                'temperature': 0.1,
                'maxOutputTokens': max_output_tokens,
                'responseMimeType': 'application/json',
            },
        }
        if system_instruction:
            request['systemInstruction'] = {'parts': [{'text': system_instruction}]}
        request_lines.append(
            {
                'key': f'idx_{idx}',
                'request': request,
            }
        )
    create_batch_jsonl_file(request_lines, batch_file_path)
    display_name = f"backtracking_{batch_type}_{dataset}_{model}_{timestamp}"
    job_name = submit_batch_job(batch_file_path, display_name)
    metadata = {
        'job_name': job_name,
        'batch_file': str(batch_file_path),
        'dataset': dataset,
        'model': model,
        'batch_type': batch_type,
        'attempt': attempt,
        'indices': indices,
        'num_traces': len(indices),
        'submitted_at': timestamp,
        'status': 'submitted',
    }
    metadata_file = batch_dir / f"{job_name.replace('/', '_')}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("Submitted %s batch job %s with %d items", batch_type, job_name, len(indices))
    return SubmittedBatch(
        job_name=job_name,
        batch_file_path=batch_file_path,
        metadata_file=metadata_file,
        dataset=dataset,
        model=model,
        batch_type=batch_type,
        attempt=attempt,
        indices=indices,
    )
def monitor_batch_job(job_name: str, check_interval: int = 30) -> dict:
    """Monitor a batch job until completion and return results."""
    client = _ensure_gemini_client()
    logger.info(f"Monitoring batch job: {job_name}")
    completed_states = {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_PARTIALLY_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    }
    while True:
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state.name if hasattr(batch_job.state, "name") else str(batch_job.state)
        logger.info(f"Job state: {state}")
        if state in completed_states:
            break
        time.sleep(check_interval)
    if state in {'JOB_STATE_SUCCEEDED', 'JOB_STATE_PARTIALLY_SUCCEEDED'}:
        logger.info("Batch job succeeded, retrieving results...")
        return retrieve_batch_results(client, batch_job)
    else:
        error_msg = f"Batch job failed with state: {state}"
        if hasattr(batch_job, 'error') and batch_job.error:
            error_msg += f" - Error: {batch_job.error}"
        logger.error(error_msg)
        raise Exception(error_msg)
def retrieve_batch_results(client, batch_job) -> dict:
    """Retrieve results from a completed batch job."""
    results = {}
    if getattr(batch_job, "dest", None) and getattr(batch_job.dest, "file_name", None):
        result_file_name = batch_job.dest.file_name
        logger.info(f"Downloading result file: {result_file_name}")
        file_content = client.files.download(file=result_file_name)
        content_str = file_content.decode('utf-8')
        for line in content_str.strip().split('\n'):
            if line.strip():
                result_obj = json.loads(line)
                key = result_obj['key']
                results[key] = result_obj
    elif getattr(batch_job, "dest", None) and getattr(batch_job.dest, "inlined_responses", None):
        for i, inline_response in enumerate(batch_job.dest.inlined_responses):
            key = f"trace_{i}"
            results[key] = {
                'response': inline_response.response,
                'error': inline_response.error
            }
    logger.info(f"Retrieved {len(results)} results from batch job")
    return results
def process_batch_results(batch_results: dict) -> Tuple[List[BacktrackingResult], Dict[int, str]]:
    """Parse batch JSONL results into BacktrackingResult objects.

    Returns: (results, bad_outputs) where bad_outputs maps index->raw text for parse failures.
    """
    results: List[BacktrackingResult] = []
    bad_outputs: Dict[int, str] = {}
    for key, result_obj in batch_results.items():
        if not isinstance(key, str) or not key.startswith('idx_'):
            continue
        try:
            idx = int(key.split('_', 1)[1])
        except Exception:
            continue
        if isinstance(result_obj, dict) and result_obj.get('error'):
            results.append(
                BacktrackingResult(
                    index=idx,
                    backtracking_detected=False,
                    final_answer='',
                    backtracking_steps=[],
                    confidence=0.0,
                    overall_reasoning='',
                    error=f"API error: {result_obj.get('error')}",
                )
            )
            continue
        response_obj: Optional[Dict[str, Any]] = None
        if isinstance(result_obj, dict):
            candidate = result_obj.get('response')
            if isinstance(candidate, dict):
                response_obj = candidate
        text: Optional[str] = None
        finish_reason: Optional[str] = None
        prompt_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        try:
            if not isinstance(response_obj, dict):
                raise ValueError('Missing response object')
            usage = response_obj.get('usageMetadata') or response_obj.get('usage_metadata') or {}
            if isinstance(usage, dict):
                pt = usage.get('promptTokenCount') or usage.get('prompt_token_count') or usage.get('prompt_tokens')
                ct = (
                    usage.get('candidatesTokenCount')
                    or usage.get('candidates_token_count')
                    or usage.get('output_tokens')
                )
                tt = usage.get('totalTokenCount') or usage.get('total_token_count') or usage.get('total_tokens')
                if isinstance(pt, int):
                    prompt_tokens = pt
                if isinstance(ct, int):
                    output_tokens = ct
                if isinstance(tt, int):
                    total_tokens = tt
            candidates = response_obj.get('candidates')
            if isinstance(candidates, list) and candidates:
                cand0 = candidates[0]
                if isinstance(cand0, dict):
                    fr = cand0.get('finishReason') or cand0.get('finish_reason')
                    if isinstance(fr, str):
                        finish_reason = fr
                    content = cand0.get('content') or {}
                    parts = content.get('parts') or []
                    text_chunks: List[str] = []
                    if isinstance(parts, list):
                        for part in parts:
                            if isinstance(part, dict) and isinstance(part.get('text'), str):
                                text_chunks.append(part['text'])
                    if text_chunks:
                        text = ''.join(text_chunks)
            if not text:
                raise ValueError('Missing response text')
            parsed = _normalize_backtracking_dict(_parse_backtracking_json(str(text)))
            results.append(
                BacktrackingResult(
                    index=idx,
                    backtracking_detected=parsed['backtracking_detected'],
                    final_answer=parsed['final_answer'],
                    backtracking_steps=parsed['backtracking_steps'],
                    confidence=parsed['confidence'],
                    overall_reasoning=parsed['overall_reasoning'],
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    finish_reason=finish_reason,
                )
            )
        except Exception as e:
            if text is not None:
                logger.warning(
                    "Failed to parse result for idx %s: %s (finish_reason=%s, text_len=%s)",
                    idx,
                    e,
                    finish_reason,
                    len(text),
                )
            else:
                logger.warning(
                    "Failed to parse result for idx %s: %s (finish_reason=%s)",
                    idx,
                    e,
                    finish_reason,
                )
            bad_outputs[idx] = str(text or '')
            results.append(
                BacktrackingResult(
                    index=idx,
                    backtracking_detected=False,
                    final_answer='',
                    backtracking_steps=[],
                    confidence=0.0,
                    overall_reasoning='',
                    error=f"Parse error: {str(e)}",
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    finish_reason=finish_reason,
                )
            )
    results.sort(key=lambda r: r.index)
    return results, bad_outputs
def submit_missing_traces(*, dataset: str, model: str, output_dir: Path, smoke_n: int) -> None:
    """Submit ONE batch job containing all missing traces for (dataset, model) and exit."""
    pickle_path = Path('outputs/traces') / dataset / f'traces_{model}.pkl'
    logger.info("Loading %s", pickle_path)
    data = load_trace_pickle(pickle_path)
    traces, _ = get_traces_and_correctness(data, dataset)
    if smoke_n and smoke_n > 0:
        traces = traces[: min(smoke_n, len(traces))]
    results_path = _results_path(output_dir, dataset, model)
    existing_results = _load_existing_results(results_path)
    processed = _processed_indices(existing_results)
    failed_pool = _load_failed_pool(output_dir, dataset, model)
    inflight = _inflight_indices(_batch_dir(output_dir), dataset, model)
    available_indices = set(range(len(traces)))
    target = (available_indices - processed) | failed_pool
    target = {i for i in target if i not in inflight and 0 <= i < len(traces)}
    if not target:
        print(f"No missing traces to submit for {dataset}/{model}.")
        return
    indices = sorted(target)
    system_instruction = system_instruction_backtracking[dataset]
    prompts = [build_backtracking_prompt(traces[i]) for i in indices]
    submit_traces_batch(
        dataset=dataset,
        model=model,
        batch_type='initial',
        attempt=0,
        indices=indices,
        prompts=prompts,
        system_instruction=system_instruction,
        output_dir=output_dir,
    )
    print(f"Submitted 1 batch with {len(indices)} traces for {dataset}/{model}.")
def check_batches_and_collect(*, dataset: str, model: str, output_dir: Path, dry_run: bool = False) -> None:
    """Check batch jobs, collect completed results, and resubmit parse failures with original prompts."""
    batch_dir = _batch_dir(output_dir)
    results_path = _results_path(output_dir, dataset, model)
    pickle_path = Path('outputs/traces') / dataset / f'traces_{model}.pkl'
    data = load_trace_pickle(pickle_path)
    traces, correctness = get_traces_and_correctness(data, dataset)
    existing_results = _load_existing_results(results_path)
    merged_by_idx: Dict[int, BacktrackingResult] = {r.index: r for r in existing_results}
    failed_pool = _load_failed_pool(output_dir, dataset, model)
    meta_files = sorted(batch_dir.glob('*_metadata.json'))
    if not meta_files:
        print("No batch metadata files found.")
        return
    client = _ensure_gemini_client()
    any_collected = False
    all_collected_ids, all_failed_ids = 0, 0
    for meta_path in meta_files:
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception:
            continue
        if meta.get('dataset') != dataset or meta.get('model') != model:
            continue
        job_name = meta.get('job_name')
        if not job_name:
            continue
        if meta.get('status') in {'completed', 'failed', 'cancelled', 'expired'}:
            continue
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
        meta['last_state'] = state
        meta['last_checked_at'] = int(time.time())
        if state not in {'JOB_STATE_SUCCEEDED', 'JOB_STATE_PARTIALLY_SUCCEEDED'}:
            meta['status'] = 'submitted'
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"{job_name}: {state}")
            continue
        batch_results = retrieve_batch_results(client, batch_job)
        parsed_results, bad_outputs = process_batch_results(batch_results)
        parse_failed: List[int] = []
        good_indices: Set[int] = set()
        for r in parsed_results:
            if r.error:
                if r.error.startswith('Parse error'):
                    parse_failed.append(r.index)
                existing = merged_by_idx.get(r.index)
                if existing is None or existing.error:
                    merged_by_idx[r.index] = r
                continue
            merged_by_idx[r.index] = r
            good_indices.add(r.index)
        if good_indices & failed_pool:
            failed_pool -= good_indices
        if bad_outputs:
            bad_path = batch_dir / f"{job_name.replace('/', '_')}_bad_outputs.json"
            with open(bad_path, 'w') as f:
                json.dump({str(k): v for k, v in bad_outputs.items()}, f)
        if parse_failed:
            parse_failed = sorted(set(parse_failed))
            failed_pool |= set(parse_failed)
            if dry_run:
                logger.info(
                    "Dry-run enabled: would resubmit %s/%s with %d items (job=%s)",
                    dataset,
                    model,
                    len(parse_failed),
                    job_name,
                )
            else:
                system_instruction = system_instruction_backtracking[dataset]
                resubmit_prompts: List[str] = [build_backtracking_prompt(traces[i]) for i in parse_failed]
                submit_traces_batch(
                    dataset=dataset,
                    model=model,
                    batch_type='resubmit',
                    attempt=int(meta.get('attempt', 0)) + 1,
                    indices=parse_failed,
                    prompts=resubmit_prompts,
                    system_instruction=system_instruction,
                    output_dir=output_dir,
                    max_output_tokens_multiplier=16.0,
                )
        _save_failed_pool(output_dir, dataset, model, failed_pool)
        merged_results = [merged_by_idx[i] for i in sorted(merged_by_idx.keys())]
        analysis = analyze_backtracking(merged_results, _correctness_for_results(merged_results, correctness))
        save_results(merged_results, analysis, results_path)
        any_collected = True
        if not dry_run:
            meta['status'] = 'completed'
            meta['completed_at'] = int(time.time())
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
        print(f"{job_name}: collected {len(good_indices)} ok, {len(parse_failed)} parse-failed")
        all_failed_ids += len(parse_failed)
        all_collected_ids += len(good_indices)
    if not any_collected:
        print("No completed batches to collect on this check.")
    return all_collected_ids, all_failed_ids
def check_all_batches_and_collect(*, output_dir: Path, dry_run: bool = False) -> None:
    """Dataset/model agnostic batch collector.

    Discovers all (dataset, model) pairs from batch metadata files under output_dir/batch_jobs
    and runs collection for each.
    """
    batch_dir = _batch_dir(output_dir)
    meta_files = sorted(batch_dir.glob('*_metadata.json'))
    if not meta_files:
        print("No batch metadata files found.")
        return
    pairs: Set[tuple[str, str]] = set()
    for meta_path in meta_files:
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception:
            continue
        dataset = meta.get('dataset')
        model = meta.get('model')
        if isinstance(dataset, str) and isinstance(model, str) and dataset and model:
            pairs.add((dataset, model))
    if not pairs:
        print("No (dataset, model) pairs found in batch metadata.")
        return
    all_c, all_f = 0, 0
    for dataset, model in sorted(pairs):
        print(f"\nCollecting batches for {dataset}/{model}")
        try:
            c, f = check_batches_and_collect(dataset=dataset, model=model, output_dir=output_dir, dry_run=dry_run)
            all_c += c
            all_f += f
        except FileNotFoundError as e:
            print(f"Skipping {dataset}/{model}: {e}")
        except Exception as e:
            print(f"Error collecting {dataset}/{model}: {e}")
    print(f"Total: {all_c} collected, {all_f} failed")
def analyze_backtracking(results: List[BacktrackingResult], correctness: List[int]) -> Dict:
    """Analyze backtracking results and compute metrics."""
    backtracks = [len(r.backtracking_steps) for r in results if r.backtracking_detected]
    if any(c not in (0, 1) for c in correctness):
        correctness = [1 if c == 4 else 0 for c in correctness]
    correct_backtracks = [len(r.backtracking_steps) for r, corr in zip(results, correctness) if corr == 1 and r.backtracking_detected]
    incorrect_backtracks = [len(r.backtracking_steps) for r, corr in zip(results, correctness) if corr == 0 and r.backtracking_detected]
    return {
        'total_traces': len(results),
        'backtracking_detected': sum(1 for r in results if r.backtracking_detected),
        'avg_backtracking_steps': np.mean(backtracks) if backtracks else 0,
        'correct_backtracks': len(correct_backtracks),
        'incorrect_backtracks': len(incorrect_backtracks),
        'correlation_with_correctness': stats.pearsonr(
            [len(r.backtracking_steps) if r.backtracking_detected else 0 for r in results],
            correctness
        )[0] if results else 0
    }
def plot_results(results: List[BacktrackingResult], correctness: List[int], output_dir: Path):
    """Generate plots for backtracking analysis."""
    output_dir.mkdir(exist_ok=True)
    steps = [len(r.backtracking_steps) for r in results if r.backtracking_detected]
    if steps:
        plt.figure(figsize=(10, 6))
        sns.histplot(steps, bins=20)
        plt.title('Distribution of Backtracking Steps')
        plt.xlabel('Number of Backtracking Steps')
        plt.ylabel('Frequency')
        plt.savefig(output_dir / 'backtracking_steps_dist.png')
        plt.close()
    backtrack_counts = [len(r.backtracking_steps) if r.backtracking_detected else 0 for r in results]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=correctness, y=backtrack_counts)
    plt.title('Backtracking Steps by Correctness')
    plt.xlabel('Correct (1) vs Incorrect (0)')
    plt.ylabel('Backtracking Steps')
    plt.savefig(output_dir / 'backtracking_vs_correctness.png')
    plt.close()
def save_results(results: List[BacktrackingResult], analysis: Dict, output_path: Path):
    """Save results in migratable format (pickle for now, can be adapted for DB)."""
    if output_path.exists():
        backup_dir = output_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True, parents=True)
        backup_path = backup_dir / (output_path.name + f".bak_{int(time.time())}")
        try:
            shutil.move(str(output_path), str(backup_path))
            logger.warning("Backed up existing results to %s", backup_path)
        except Exception as e:
            logger.warning("Failed to back up existing results %s: %s", output_path, e)
    data = {
        'results': [r.__dict__ for r in results],
        'analysis': analysis,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
def check_coverage(output_dir: str = 'outputs/backtracking_analysis'):
    """Check coverage of backtracking analysis files vs total traces needed."""
    output_dir_path = Path(output_dir)
    traces_dir = Path('outputs/traces')
    datasets = ['math', 'gpqa', 'connections']
    coverage = {}
    for dataset in datasets:
        dataset_traces_dir = traces_dir / dataset
        if not dataset_traces_dir.exists():
            continue
        coverage[dataset] = {}
        total_traces_needed = 0
        trace_files = list(dataset_traces_dir.glob('traces_*.pkl'))
        for trace_file in trace_files:
            model = trace_file.stem.replace('traces_', '')
            backtracking_file = output_dir_path / f'backtracking_{dataset}_{model}.pkl'
            try:
                with open(trace_file, 'rb') as f:
                    trace_data = pickle.load(f)
                num_traces = len(trace_data['data']['traces'])
                total_traces_needed += num_traces
            except Exception as e:
                print(f"Error loading {trace_file}: {e}")
                continue
            if backtracking_file.exists():
                try:
                    with open(backtracking_file, 'rb') as f:
                        back_data = pickle.load(f)
                    num_backtracked = 0
                    for result in back_data['results']:
                        if result['error'] is None:
                            num_backtracked += 1
                    coverage[dataset][model] = {
                        'total_traces': num_traces,
                        'backtracked_traces': num_backtracked,
                        'coverage': num_backtracked / num_traces if num_traces > 0 else 0
                    }
                except Exception as e:
                    print(f"Error loading {backtracking_file}: {e}")
                    coverage[dataset][model] = {
                        'total_traces': num_traces,
                        'backtracked_traces': 0,
                        'coverage': 0,
                        'error': str(e)
                    }
            else:
                coverage[dataset][model] = {
                    'total_traces': num_traces,
                    'backtracked_traces': 0,
                    'coverage': 0
                }
        coverage[dataset]['total_traces_needed'] = total_traces_needed
    print("Backtracking Coverage Report:")
    for dataset, data in coverage.items():
        print(f"\n{dataset.upper()}:")
        total_needed = data.get('total_traces_needed', 0)
        total_backtracked = sum(m['backtracked_traces'] for m in data.values() if isinstance(m, dict) and 'backtracked_traces' in m)
        print(f"  Total traces needed: {total_needed}")
        print(f"  Total backtracked: {total_backtracked}")
        print(".1%")
        for model, stats in data.items():
            if model == 'total_traces_needed':
                continue
            print(f"    {model}: {stats['backtracked_traces']}/{stats['total_traces']} ({stats['coverage']:.1%})")
    return coverage
def process_need_to_complete(output_dir: str = 'outputs/backtracking_analysis', dataset: str = None, model: str = None):
    raise NotImplementedError(
        "Deprecated. Use --check-batches (it downloads completed batch results and schedules resubmit batches)."
    )
def plot_backtracking_analysis(output_dir: str = 'outputs/backtracking_analysis'):
    output_dir_path = Path(output_dir)
    plots_dir = output_dir_path / 'plots'
    plots_dir.mkdir(exist_ok=True, parents=True)
    datasets = ['gpqa', 'connections', 'math']
    all_data = {}
    for dataset in datasets:
        dataset_data = {}
        traces_dir = output_dir_path / f'backtracking_{dataset}_*.pkl'
        for pkl_file in output_dir_path.glob(f'backtracking_{dataset}_*.pkl'):
            model = pkl_file.stem.replace(f'backtracking_{dataset}_', '')
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            dataset_data[model] = data
        all_data[dataset] = dataset_data
    violin_data = []
    box_data = []
    for dataset in datasets:
        for model, data in all_data[dataset].items():
            results = data['results']
            analysis = data['analysis']
            for result in results:
                if result['backtracking_detected']:
                    steps = len(result['backtracking_steps'])
                else:
                    steps = 0
                violin_data.append({
                    'dataset': dataset,
                    'model': model,
                    'backtracking_steps': steps,
                    'color': get_model_color(model)
                })
            pickle_path = Path('outputs/traces') / dataset / f'traces_{model}.pkl'
            if pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    orig_data = pickle.load(f)
                correctness = orig_data['data']['scores']
            else:
                correctness = [0] * len(results)
            for i, result in enumerate(results):
                corr = correctness[i] if i < len(correctness) else 0
                if result['backtracking_detected']:
                    steps = len(result['backtracking_steps'])
                else:
                    steps = 0
                box_data.append({
                    'dataset': dataset,
                    'model': model,
                    'backtracking_steps': steps,
                    'correctness': 'Correct' if corr == 1 else 'Incorrect',
                    'color': 'green' if corr == 1 else 'red'
                })
    for dataset in datasets + ['combined']:
        if dataset == 'combined':
            plot_data = violin_data
            title = 'Backtracking Steps Across All Datasets'
            filename = 'violin_combined.pdf'
        else:
            plot_data = [d for d in violin_data if d['dataset'] == dataset]
            title = f'Backtracking Steps for {dataset.upper()} Dataset'
            filename = f'violin_{dataset}.png'
        if not plot_data:
            continue
        df = pd.DataFrame(plot_data)
        models_sorted = sorted(df['model'].unique())
        palette = {row['model']: hsl_to_rgb(row['color']) for row in plot_data}
        plt.figure(figsize=(12, 8))
        ax = sns.violinplot(data=df, x='model', y='backtracking_steps', palette=palette, hue='model', legend=False, order=models_sorted)
        plt.title(title)
        plt.xlabel('Model')
        plt.ylabel('Backtracking Steps')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / filename)
        plt.close()
    for dataset in datasets:
        plot_data = [d for d in box_data if d['dataset'] == dataset]
        if not plot_data:
            continue
        df = pd.DataFrame(plot_data)
        models_sorted = sorted(df['model'].unique())
        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(data=df, x='model', y='backtracking_steps', hue='correctness', palette={'Correct': 'green', 'Incorrect': 'red'}, order=models_sorted)
        plt.title(f'Backtracking Steps by Correctness for {dataset.upper()} Dataset')
        plt.xlabel('Model')
        plt.ylabel('Backtracking Steps')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Ground Truth')
        plt.tight_layout()
        plt.savefig(plots_dir / f'boxplot_{dataset}.png')
        plt.close()
    table_data = {}
    for dataset in datasets:
        dataset_violin = [d for d in violin_data if d['dataset'] == dataset]
        dataset_box = [d for d in box_data if d['dataset'] == dataset]
        if not dataset_violin:
            continue
        df_violin = pd.DataFrame(dataset_violin)
        df_box = pd.DataFrame(dataset_box)
        models = sorted(df_violin['model'].unique())
        table_data[dataset] = {}
        for model in models:
            model_violin = df_violin[df_violin['model'] == model]
            model_box = df_box[df_box['model'] == model]
            avg_steps = model_violin['backtracking_steps'].mean()
            std_steps = model_violin['backtracking_steps'].std()
            correct_data = model_box[model_box['correctness'] == 'Correct']['backtracking_steps']
            incorrect_data = model_box[model_box['correctness'] == 'Incorrect']['backtracking_steps']
            avg_correct = correct_data.mean() if not correct_data.empty else 0
            avg_incorrect = incorrect_data.mean() if not incorrect_data.empty else 0
            table_data[dataset][model] = {
                'avg_steps': avg_steps,
                'std_steps': std_steps,
                'avg_steps_correct': avg_correct,
                'avg_steps_incorrect': avg_incorrect
            }
    import json
    with open(plots_dir / 'backtracking_table.json', 'w') as f:
        json.dump(table_data, f, indent=2)
    all_models = set()
    for dataset in datasets:
        all_models.update(table_data[dataset].keys())
    all_models = sorted(all_models)
    max_min = {}
    for dataset in datasets:
        max_min[dataset] = {}
        for metric in ['avg_steps', 'std_steps', 'avg_steps_correct', 'avg_steps_incorrect']:
            values = [table_data[dataset][model][metric] for model in table_data[dataset] if not np.isnan(table_data[dataset][model][metric])]
            if values:
                max_min[dataset][metric] = {'max': max(values), 'min': min(values)}
            else:
                max_min[dataset][metric] = {'max': None, 'min': None}
    latex_content = r"""
\begin{table}[h]
\centering
\caption{Backtracking Analysis Results}
\label{tab:backtracking}
\begin{tabular}{l cccc cccc}
\hline
\multirow{2}{*}{Model} & \multicolumn{4}{c}{GPQA} & \multicolumn{4}{c}{Connections} \\
\cline{2-9}
 & Avg Steps & Std Steps & Avg Correct & Avg Incorrect & Avg Steps & Std Steps & Avg Correct & Avg Incorrect \\
\hline
"""
    for model in all_models:
        row = [model]
        for dataset in datasets:
            for metric in ['avg_steps', 'std_steps', 'avg_steps_correct', 'avg_steps_incorrect']:
                if model in table_data[dataset]:
                    val = table_data[dataset][model][metric]
                    if np.isnan(val):
                        cell = "-"
                    else:
                        formatted = f"{val:.2f}"
                        if val == max_min[dataset][metric]['max']:
                            cell = f"\\textbf{{{formatted}}}"
                        elif val == max_min[dataset][metric]['min']:
                            cell = f"\\underline{{{formatted}}}"
                        else:
                            cell = formatted
                else:
                    cell = "-"
                row.append(cell)
        latex_content += " & ".join(row) + " \\\\\n"
    latex_content += r"""
\hline
\end{tabular}
\end{table}
"""
    with open(plots_dir / 'backtracking_table.tex', 'w') as f:
        f.write(latex_content)
    with open(plots_dir / 'backtracking_table.tex', 'w') as f:
        f.write(latex_content)
    print(f"Plots and table saved to {plots_dir}")
def main():
    parser = argparse.ArgumentParser(description='Analyze backtracking in reasoning traces')
    parser.add_argument('--dataset', choices=['math', 'gpqa', 'connections', 'all'], help='Dataset to analyze or "all" for smoke test')
    parser.add_argument('--model', help='Model name (e.g., Qwen_QwQ-32B); required unless dataset=all')
    parser.add_argument(
        '--smoke-test', nargs='?', const=75, default=0, type=smoke_arg,
        help='Submit only the first N traces for quick testing (default N=75). Accepts true/false or N.'
    )
    parser.add_argument('--max-concurrent', type=int, default=5, help='(ignored) Max concurrent API calls')
    parser.add_argument('--delay', type=float, default=5.0, help='(ignored) Delay between API calls (seconds)')
    parser.add_argument('--batch-size', type=int, default=200, help='(ignored) Number of traces per batch job')
    parser.add_argument('--output-dir', default='outputs/backtracking_analysis', help='Output directory')
    parser.add_argument('--plot-only', action='store_true', help='Only generate plots from existing analysis files')
    parser.add_argument('--check-coverage', action='store_true', help='Check coverage of backtracking analysis vs total traces')
    parser.add_argument('--check-batches', action='store_true', help='Check existing batch jobs and collect results (may submit resubmit batches)')
    parser.add_argument(
        '--check-batches-dry-run',
        action='store_true',
        help='Like --check-batches, but never submits new reprocess batches (download/parse/merge only)'
    )
    parser.add_argument(
        '--estimate-cost',
        action='store_true',
        help='Estimate token cost for remaining traces using Gemini token counter (assumes 10% reprocess fail rate)'
    )
    parser.add_argument(
        '--pricing-tier',
        choices=['batch', 'standard'],
        default='batch',
        help='Pricing tier for --estimate-cost. Batch is ~50% off Standard for token costs.'
    )
    parser.add_argument('--submit-batch', action='store_true', help='Submit ONE batch job with all missing traces and exit (default)')
    args = parser.parse_args()
    print("Starting backtracking analysis with arguments:", args)
    if args.check_coverage:
        check_coverage(args.output_dir)
        return
    if args.plot_only:
        plot_backtracking_analysis(args.output_dir)
        return
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    if args.estimate_cost:
        fail_rate = 0.10
        sample_single_prompts = int(os.getenv('BACKTRACKING_ESTIMATE_SAMPLE_PROMPTS', '200'))
        sample_single_outputs = int(os.getenv('BACKTRACKING_ESTIMATE_SAMPLE_OUTPUTS', '100'))
        sample_multi_prompts = int(os.getenv('BACKTRACKING_ESTIMATE_SAMPLE_PROMPTS_MULTI', '50'))
        sample_multi_outputs = int(os.getenv('BACKTRACKING_ESTIMATE_SAMPLE_OUTPUTS_MULTI', '50'))
        def _print_estimate(est: Dict[str, Any]) -> None:
            print(
                f"{est['dataset']}/{est['model']}: remaining={est['remaining_traces']}, "
                f"reprocess~{est['estimated_reprocess_count']} (fail_rate={est['assumed_fail_rate']:.2f}), "
                f"input_tokens~{est['estimated_total_input_tokens']:,}, output_tokens~{est['estimated_total_output_tokens']:,}, "
                f"cost=${est['estimated_total_cost_usd']:.4f} (in=${est['estimated_input_cost_usd']:.4f}, out=${est['estimated_output_cost_usd']:.4f}) "
                f"tier={est.get('pricing_tier', args.pricing_tier)}"
            )
        input_rate, output_rate = _pricing_rates_usd_per_1m_tokens(pricing_tier=args.pricing_tier)
        print(
            f"Pricing assumptions ({args.pricing_tier}): "
            f"input=${input_rate}/1M tokens, output=${output_rate}/1M tokens. "
            "(Gemini 2.5 Flash token rates; output includes thinking tokens.)"
        )
        if args.dataset not in (None, 'all') and args.model:
            est = _estimate_cost_for_pair(
                dataset=args.dataset,
                model=args.model,
                output_dir=output_dir_path,
                smoke_n=int(args.smoke_test),
                fail_rate=fail_rate,
                sample_k_prompts=sample_single_prompts,
                sample_k_outputs=sample_single_outputs,
                pricing_tier=args.pricing_tier,
            )
            if not est:
                raise FileNotFoundError(f"Missing traces file for {args.dataset}/{args.model}")
            _print_estimate(est)
            return
        datasets = ['math', 'gpqa', 'connections'] if args.dataset in (None, 'all') else [args.dataset]
        traces_root = Path('outputs/traces')
        estimates: List[Dict[str, Any]] = []
        for ds in datasets:
            ds_dir = traces_root / ds
            if not ds_dir.exists():
                continue
            for trace_file in sorted(ds_dir.glob('traces_*.pkl')):
                m = trace_file.stem.replace('traces_', '')
                est = _estimate_cost_for_pair(
                    dataset=ds,
                    model=m,
                    output_dir=output_dir_path,
                    smoke_n=int(args.smoke_test),
                    fail_rate=fail_rate,
                    sample_k_prompts=sample_multi_prompts,
                    sample_k_outputs=sample_multi_outputs,
                    pricing_tier=args.pricing_tier,
                )
                if est:
                    estimates.append(est)
        if not estimates:
            print("No trace pickles found to estimate.")
            return
        total_in = sum(e['estimated_total_input_tokens'] for e in estimates)
        total_out = sum(e['estimated_total_output_tokens'] for e in estimates)
        total_in_cost = sum(e['estimated_input_cost_usd'] for e in estimates)
        total_out_cost = sum(e['estimated_output_cost_usd'] for e in estimates)
        total_cost = total_in_cost + total_out_cost
        for e in estimates:
            _print_estimate(e)
        print(
            f"\nTOTAL: input_tokens~{total_in:,}, output_tokens~{total_out:,}, "
            f"cost=${total_cost:.4f} (in=${total_in_cost:.4f}, out=${total_out_cost:.4f}) "
            f"using token_counter_model={_token_counter_model_name()}"
        )
        return
    if args.check_batches or args.check_batches_dry_run:
        dry_run = bool(args.check_batches_dry_run)
        if args.dataset in (None, 'all') or not args.model:
            check_all_batches_and_collect(output_dir=output_dir_path, dry_run=dry_run)
        else:
            check_batches_and_collect(dataset=args.dataset, model=args.model, output_dir=output_dir_path, dry_run=dry_run)
        return
    if args.dataset == 'all':
        if not args.smoke_test:
            raise ValueError("dataset='all' only allowed with --smoke-test")
        datasets = ['math', 'gpqa', 'connections']
        model = args.model or 'Qwen_QwQ-32B'
    else:
        datasets = [args.dataset]
        model = args.model
        if not model:
            raise ValueError("model required when dataset != 'all'")
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        submit_missing_traces(dataset=dataset, model=model, output_dir=output_dir_path, smoke_n=int(args.smoke_test))
if __name__ == '__main__':
    main()
