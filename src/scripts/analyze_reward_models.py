"""Reward Model Analysis Script (GPU)

Loads one or more Hugging Face reward models, scores each trace output from a
`outputs/traces/<dataset>/traces_<teacher_model>.pkl` file, and assigns a scalar
reward per datapoint.

This is scaffolding intended to mirror the sharding + merging conventions used
by other analysis scripts in this repo (e.g. `analyze_efficiency.py`).

Typical usage (single shard):
  python3 -m src.scripts.analyze_reward_models \
    --teacher_model "Qwen/QwQ-32B" \
    --dataset_name math \
    --reward_models OpenAssistant/reward-model-deberta-v3-large-v2

Sharded usage:
  python3 -m src.scripts.analyze_reward_models ... --num_shards 4 --shard_id 0

Notes
- Reward model I/O formats differ across checkpoints. This script assumes
  `AutoModelForSequenceClassification` with logits that can be converted into a
  single scalar score via `--logit_strategy auto`.
- The input text format defaults to a simple conversation:
    "System: ...\nHuman: <question>\nAssistant: <response>"
  Customize via `--input_mode` and `--include_system_prompt`.
"""
from __future__ import annotations
import argparse
import hashlib
import os
import os.path as osp
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
try:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _ML_DEPS_AVAILABLE = True
except Exception:
    torch = None
    tqdm = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    _ML_DEPS_AVAILABLE = False
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward Model Analysis Script")
    parser.add_argument("--teacher_model", type=str, required=True, help="Teacher model name (e.g. 'Qwen/QwQ-32B')")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (math|gpqa|connections|...)")
    parser.add_argument("--base_dir", type=str, default="outputs", help="Base directory containing outputs/traces")
    parser.add_argument("--trace_file", type=str, default=None, help="Optional explicit path to a traces_*.pkl")
    parser.add_argument(
        "--reward_model",
        type=str,
        default=None,
        help=(
            "Single HF model ID to load. This script is designed to run one reward model per output. "
            "If omitted, uses a conservative default."
        ),
    )
    parser.add_argument(
        "--reward_models",
        type=str,
        nargs="+",
        default=None,
        help=(
            "DEPRECATED: Use --reward_model instead. If provided with exactly one value, it is treated as --reward_model. "
            "If multiple values are provided, this script will error to avoid producing non-RM-specific outputs."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loading (use only if you trust the repo).",
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        default="question_trace",
        choices=[
            "trace_only",
            "completion_only",
            "question_trace",
            "question_completion",
        ],
        help="How to build the text fed to the reward model.",
    )
    parser.add_argument(
        "--include_system_prompt",
        action="store_true",
        help="Include trace file's system_prompt (if present) in the scored text.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for reward model scoring")
    parser.add_argument("--max_length", type=int, default=2048, help="Tokenizer max_length (truncates long inputs)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype on GPU",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: 'auto' (cuda if available) or 'cpu' or 'cuda:0'",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Optional HF device_map (e.g. 'auto'). Useful for huge models.",
    )
    parser.add_argument(
        "--logit_strategy",
        type=str,
        default="auto",
        choices=["auto", "single", "class1", "diff_last_first"],
        help="How to convert logits -> scalar score.",
    )
    parser.add_argument(
        "--reward_transform",
        type=str,
        default="identity",
        choices=["identity", "sigmoid", "tanh"],
        help="Optional post-transform on scalar score.",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "min", "max", "first"],
        help="(Deprecated) Aggregation method. With a single reward model this has no effect.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of datapoints")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard id (0-indexed)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/reward_model_analysis",
        help="Base output directory for reward model analysis",
    )
    return parser.parse_args()
def _safe_model_dirname(model_id: str) -> str:
    return model_id.replace("/", "_")
def get_trace_path(base_dir: str, dataset_name: str, teacher_model: str) -> str:
    teacher_us = teacher_model.replace("/", "_")
    trace_path = osp.join(base_dir, "traces", dataset_name, f"traces_{teacher_us}.pkl")
    if not osp.exists(trace_path):
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    return trace_path
def load_pickle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)
def chunked_indices(indices: Sequence[int], batch_size: int) -> Iterable[Sequence[int]]:
    for i in range(0, len(indices), batch_size):
        yield indices[i : i + batch_size]
def _resolve_device(device_arg: str) -> torch.device:
    if not _ML_DEPS_AVAILABLE:
        raise RuntimeError("Missing deps: install torch + transformers (and run inside the project venv).")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)
def _resolve_dtype(dtype_arg: str, device: torch.device) -> Optional[torch.dtype]:
    if not _ML_DEPS_AVAILABLE:
        raise RuntimeError("Missing deps: install torch + transformers (and run inside the project venv).")
    if dtype_arg == "auto":
        if device.type == "cuda":
            return torch.float16
        return None
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_arg]
def build_scoring_text(
    *,
    input_mode: str,
    include_system_prompt: bool,
    system_prompt: str,
    question: str,
    trace: str,
    completion: str,
) -> str:
    sys = (system_prompt or "").strip()
    q = (question or "").strip()
    if input_mode == "trace_only":
        body = (trace or "").strip()
        parts = []
        if include_system_prompt and sys:
            parts.append(f"System: {sys}")
        parts.append(body)
        return "\n".join([p for p in parts if p])
    if input_mode == "completion_only":
        body = (completion or "").strip()
        parts = []
        if include_system_prompt and sys:
            parts.append(f"System: {sys}")
        parts.append(body)
        return "\n".join([p for p in parts if p])
    if input_mode == "question_trace":
        a = (trace or "").strip()
        parts = []
        if include_system_prompt and sys:
            parts.append(f"System: {sys}")
        parts.append(f"Human: {q}")
        parts.append(f"Assistant: {a}")
        return "\n".join(parts)
    if input_mode == "question_completion":
        a = (completion or "").strip()
        parts = []
        if include_system_prompt and sys:
            parts.append(f"System: {sys}")
        parts.append(f"Human: {q}")
        parts.append(f"Assistant: {a}")
        return "\n".join(parts)
    raise ValueError(f"Unknown input_mode: {input_mode}")
def logits_to_score(logits: torch.Tensor, strategy: str) -> torch.Tensor:
    """Convert logits to a single scalar score per example."""
    if logits.ndim == 1:
        return logits
    if logits.ndim != 2:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if strategy == "single":
        if logits.shape[1] != 1:
            raise ValueError("logit_strategy=single expects logits with shape [B,1]")
        return logits[:, 0]
    if strategy == "class1":
        if logits.shape[1] < 2:
            raise ValueError("logit_strategy=class1 expects at least 2 classes")
        return logits[:, 1]
    if strategy == "diff_last_first":
        if logits.shape[1] < 2:
            raise ValueError("logit_strategy=diff_last_first expects at least 2 classes")
        return logits[:, -1] - logits[:, 0]
    if logits.shape[1] == 1:
        return logits[:, 0]
    if logits.shape[1] == 2:
        return logits[:, 1] - logits[:, 0]
    return logits[:, -1]
def transform_score(score: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "identity":
        return score
    if transform == "sigmoid":
        return torch.sigmoid(score)
    if transform == "tanh":
        return torch.tanh(score)
    raise ValueError(f"Unknown reward_transform: {transform}")
def aggregate_rewards(rewards_by_model: Dict[str, float], mode: str) -> float:
    vals = list(rewards_by_model.values())
    if not vals:
        return float("nan")
    if mode == "first":
        return float(vals[0])
    if mode == "mean":
        return float(np.mean(vals))
    if mode == "min":
        return float(np.min(vals))
    if mode == "max":
        return float(np.max(vals))
    raise ValueError(f"Unknown aggregate mode: {mode}")
DEFAULT_REWARD_MODELS: List[str] = [
    "OpenAssistant/reward-model-deberta-v3-large-v2",
]
@dataclass
class LoadedRewardModel:
    model_id: str
    tokenizer: Any
    model: Any
def load_reward_model(
    model_id: str,
    *,
    device: torch.device,
    dtype: Optional[torch.dtype],
    device_map: Optional[str],
    trust_remote_code: bool,
) -> LoadedRewardModel:
    if not _ML_DEPS_AVAILABLE:
        raise RuntimeError("Missing deps: install torch + transformers (and run inside the project venv).")
    kwargs_tok = {"use_fast": True, "trust_remote_code": trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs_tok)
    added_pad_token = False
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            added_pad_token = True
    if getattr(tokenizer, "padding_side", None) != "right":
        tokenizer.padding_side = "right"
    kwargs_model: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        kwargs_model["torch_dtype"] = dtype
    if device_map is not None:
        kwargs_model["device_map"] = device_map
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs_model)
    if added_pad_token:
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception as e:
            raise RuntimeError(f"Failed to resize token embeddings after adding pad token for {model_id}: {e}")
    if getattr(model, "config", None) is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if device_map is None:
        model.to(device)
    model.eval()
    print(f"Loaded reward model: {model_id} on device {device} with dtype {dtype}")
    return LoadedRewardModel(model_id=model_id, tokenizer=tokenizer, model=model)
def analyze_reward_models(args: argparse.Namespace) -> Dict[str, Any]:
    if not _ML_DEPS_AVAILABLE:
        raise RuntimeError(
            "Missing required deps. Activate the repo environment (see run_*.sh) "
            "and ensure torch + transformers are installed."
        )
    trace_path = args.trace_file or get_trace_path(args.base_dir, args.dataset_name, args.teacher_model)
    traces_pkl = load_pickle(trace_path)
    data = traces_pkl.get("data", {})
    metadata = traces_pkl.get("metadata", {})
    system_prompt = metadata.get("system_prompt", "")
    questions: List[str] = data.get("questions", [])
    completions: List[str] = data.get("completions", [])
    traces: List[str] = data.get("traces", [])
    if not questions:
        raise ValueError("Trace pickle missing data['questions']")
    total = len(questions)
    if args.limit is not None:
        total = min(total, args.limit)
    if args.shard_id >= args.num_shards:
        raise ValueError(f"shard_id ({args.shard_id}) must be < num_shards ({args.num_shards})")
    shard_indices = list(range(args.shard_id, total, args.num_shards))
    reward_model_id: Optional[str] = None
    if args.reward_model:
        reward_model_id = args.reward_model
    elif args.reward_models is not None:
        if len(args.reward_models) != 1:
            raise ValueError(
                "--reward_models is deprecated and must contain exactly one model id. "
                "Use --reward_model <id> and run one RM per invocation."
            )
        reward_model_id = args.reward_models[0]
    else:
        reward_model_id = DEFAULT_REWARD_MODELS[0]
    reward_model_safe = _safe_model_dirname(reward_model_id)
    reward_model_id_hash = hashlib.sha1(reward_model_id.encode("utf-8")).hexdigest()[:10]
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    rm = load_reward_model(
        reward_model_id,
        device=device,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    results: List[Dict[str, Any]] = []
    with torch.no_grad():
        print("Starting reward model scoring...")
        for batch_ids in tqdm(chunked_indices(shard_indices, args.batch_size), total=(len(shard_indices) + args.batch_size - 1) // args.batch_size, desc="Scoring Batches"):
            texts: List[str] = []
            batch_meta: List[Tuple[int, str]] = []
            for idx in batch_ids:
                q = questions[idx]
                t = traces[idx] if idx < len(traces) else ""
                c = completions[idx] if idx < len(completions) else ""
                text = build_scoring_text(
                    input_mode=args.input_mode,
                    include_system_prompt=args.include_system_prompt,
                    system_prompt=system_prompt,
                    question=q,
                    trace=t,
                    completion=c,
                )
                texts.append(text)
                batch_meta.append((idx, q))
            tok = rm.tokenizer
            model = rm.model
            enc = tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            if args.device_map is None:
                enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits
            scores = logits_to_score(logits, args.logit_strategy)
            scores = transform_score(scores, args.reward_transform)
            batch_scores = [float(x) for x in scores.detach().cpu().tolist()]
            for j, (idx, q) in enumerate(batch_meta):
                rm_reward = batch_scores[j]
                rewards_by_model = {reward_model_id: rm_reward}
                results.append(
                    {
                        "index": idx,
                        "question": q,
                        "rm_reward": rm_reward,
                        "reward": rm_reward,
                        "reward_model": reward_model_id,
                        "rewards_by_model": rewards_by_model,
                    }
                )
    out = {
        "metadata": {
            "teacher_model": args.teacher_model,
            "dataset_name": args.dataset_name,
            "trace_file": trace_path,
            "trace_metadata": metadata,
            "reward_model": reward_model_id,
            "reward_model_safe": reward_model_safe,
            "reward_model_id_hash": reward_model_id_hash,
            "input_mode": args.input_mode,
            "include_system_prompt": bool(args.include_system_prompt),
            "logit_strategy": args.logit_strategy,
            "reward_transform": args.reward_transform,
            "aggregate": args.aggregate,
            "collected_on": datetime.now().isoformat(),
            "shard_info": {
                "num_shards": args.num_shards,
                "shard_id": args.shard_id,
                "num_datapoints": len(shard_indices),
                "interleaved": True,
            },
        },
        "data": results,
    }
    return out
def save_output(result: Dict[str, Any], args: argparse.Namespace) -> str:
    dataset = args.dataset_name
    teacher_us = args.teacher_model.replace("/", "_")
    shard_dir = osp.join(args.output_dir, dataset, teacher_us, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    rm_safe = result["metadata"]["reward_model_safe"]
    out_path = osp.join(
        shard_dir,
        f"reward_model_{rm_safe}_shard{args.shard_id}of{args.num_shards}.pkl",
    )
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    return out_path
def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and "HF_HOME" not in os.environ:
        pass
    result = analyze_reward_models(args)
    out_path = save_output(result, args)
    rewards = [r.get("rm_reward") for r in result["data"] if isinstance(r.get("rm_reward"), (int, float))]
    if rewards:
        print(f"Saved shard to: {out_path}")
        print(f"Shard reward mean: {float(np.mean(rewards)):.4f} (n={len(rewards)})")
    else:
        print(f"Saved shard (empty) to: {out_path}")
if __name__ == "__main__":
    main()
