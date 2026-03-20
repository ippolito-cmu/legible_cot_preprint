from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time
import time as clock_time
import random
from dotenv import load_dotenv
import os
from fastapi import params
import torch
import gc
import re
from tqdm import tqdm
from vllm import SamplingParams, vllm
import pickle
import argparse
import yaml
import os.path as osp
import multiprocessing as mp
from typing import List,Dict,Tuple
from src.utils import load_dataset, get_question_answer
from src.utils.prompts import *
from src.utils.graders import grade_answer
from src.utils.extractors import extract_answer, extract_trace
from src.utils.logging import get_logger
try:
    from sympy import N, simplify
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr
    from latex2sympy2_extended import latex2sympy
except ImportError:
    pass
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
def parse_args():
    parser = argparse.ArgumentParser(description="Pedagogical Utility Experiment Script")
    parser.add_argument("--student_model", type=str, required=True, help="Name of the student model")
    parser.add_argument("--teacher_model", type=str, required=True, help="Name of the teacher model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--base_dir", type=str, default="outputs", help="Base directory for traces")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log file")
    parser.add_argument("--output_dir", type=str, default="outputs/pu_results", help="Output directory for results")
    parser.add_argument("--config_dir", type=str, default="src/scripts/pu_config", help="Config directory for generation params")
    parser.add_argument("--logprobs", action='store_true', help="Whether to log logprobs during generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of datapoints to process")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards to split data into")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID to process (0-indexed)")
    args = parser.parse_args()
    return args
def get_trace_path(base_dir: str, dataset_name: str, model_name: str):
    model_name = model_name.replace("/", "_")
    trace_path = osp.join(base_dir, "traces", dataset_name, f"traces_{model_name}.pkl")
    assert f"traces_{model_name}.pkl" in os.listdir(osp.join(base_dir, "traces", dataset_name)), f"Trace file {trace_path} not found."
    return trace_path
def load_trace(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)
def get_generation_params(config_dir: str, dataset_name: str, model_name: str, logger = None) -> Tuple[Dict, Dict]:
    '''
    Load generation parameters from config file based on model name and dataset name.
    If no specific config for model, fall back to default based on model size (if that fails, go small)

    args:
    - config_dir (str): base directory containing config yaml files
    - dataset_name (str): name of dataset to check for overrides
    - model_name (str): name of model to load config for
    - logger: optional logger for routing prints

    returns: (sampling_params (dict), vllm_kwargs (dict))

    '''
    model_name = model_name.lower().replace("/","--")
    printf = lambda msg: logger.info(msg) if logger else print(msg)
    config_path = osp.join(config_dir, f"{model_name}.yaml")
    if f"{model_name}.yaml" not in os.listdir(config_dir):
        size = re.findall(r'(\d+\.?\d*)b', model_name)
        printf(f"Debug: Model name is {model_name}, size match is {size}")
        if size:
            size_str = float(size[-1].lower().replace("b", ""))
            if size_str > 50:
                printf(f"Model size detected as {size_str}B, using large default config.")
                config_path = osp.join(config_dir, "default_large.yaml")
            else:
                printf(f"Model size detected as {size_str}B, using small default config.")
                config_path = osp.join(config_dir, "default_small.yaml")
        else:
            printf(f"Model size could not be detected, using small default config.")
            config_path = osp.join(config_dir, "default_small.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    sampling_params = config.get('sampling_params', {})
    vllm_kwargs = config.get('vllm_kwargs', {})
    if 'dataset_overrides' in config and dataset_name in config['dataset_overrides']:
        dataset_overrides = config['dataset_overrides'][dataset_name]
        sampling_params.update(dataset_overrides.get('sampling_params', {}))
        vllm_kwargs.update(dataset_overrides.get('vllm_kwargs', {}))
    if sampling_params == {}:
        printf("Warning: No sampling parameters found, using empty dict.")
    if vllm_kwargs == {}:
        printf("Warning: No vLLM kwargs found, using empty dict.")
    return sampling_params, vllm_kwargs
def process_datapoint(idx: int, tok, system_prompt: str, datapoint: dict, student: str, teacher: str, step_interval: int = 10) -> List[Tuple[dict, str]]: 
    '''
    split a trace into multiple steps, log the data.
    Now supports step_interval to sample every N steps instead of all steps.

    args:
    - idx (int): index of original trace (and, generally, the dataset as well?)
    - tok (AutoTokenizer): tokenizer object for the student model, usually an AutoTokenizer
    - system_prompt (str): dataset system prompt
    - datapoint (dict): row-major point from generate_traces.py (which is saved as a column usually)
    - student (str): student model
    - teacher (str): teacher model
    - step_interval (int): sample every N steps (default: 10). Use 1 for all steps.

    returns: [(meta (dict), prompt (str)), ... ], where meta contains index, teacher, student, num_steps, total_steps
    '''
    sentences = sent_tokenize(datapoint['traces'])
    n = len(sentences)
    results = []
    sample_points = list(range(0, n, step_interval))
    if 0 not in sample_points:
        sample_points.insert(0, 0)
    if (n-1) not in sample_points and n > 0:
        sample_points.append(n-1)
    for j in sample_points:
        meta = {
            'index': idx,
            'num_steps': j,
            'total_steps': n,
        }
        partial_reasoning = "\n".join(sentences[:j])
        prompt = templatize(tok, system_prompt, datapoint['questions'], partial_reasoning)
        results.append((meta, prompt))
    return results
def templatize(tok, system_prompt: str, problem: str, reasoning: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": reasoning},
    ]
    return tok.apply_chat_template(messages, tokenize=True, continue_final_message = True)
def clip_eot(tok, text):
    eos = tok.eos_token
    if not text.endswith(eos):
        return text
    i = 0
    while i < len(text) and eos in text[i:]:
     i += text[i:].index(eos) + len(eos)
    return text[:i - len(eos)]
def force_response(tok, text: str, force_prompt: str):
    prompt_clipped = clip_eot(tok, text)
    forced_prompt = prompt_clipped + "\n" + force_prompt
    return tok(forced_prompt, add_special_tokens=False).input_ids
_worker_initialized = False
def _grade_worker(dataset_name, answer, reference):
    """
    Simple worker function that runs in separate process.
    Must be defined at module level for pickle compatibility.
    Warms up on first call to avoid timeout from lazy imports.
    """
    global _worker_initialized
    if not _worker_initialized:
        from src.utils.graders import warmup_grader
        warmup_grader()
        _worker_initialized = True
    from src.utils.graders import grade_answer
    return grade_answer(dataset_name, answer, reference)
def init_worker():
    """Initialize worker process by warming up the grader"""
    from src.utils.graders import warmup_grader
    try:
        warmup_grader()
    except Exception as e:
        import sys
        print(f"Warning: Worker warmup failed: {e}", file=sys.stderr)
def _make_hashable(obj):
    """Convert an object to a hashable type for set membership checks."""
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return tuple(_make_hashable(x) for x in obj)
    elif isinstance(obj, set):
        return frozenset(_make_hashable(x) for x in obj)
    return obj
def check_correctness(answer, reference, dataset_name, idx, pool_container, evil_evals, logger=None):
    """
    Worker function with hard timeout using multiprocessing
    Returns (idx, result, error_msg, pool_needs_restart)
    pool_container is a dict with 'pool' key so we can update it
    evil_evals is a set that accumulates problematic (answer, reference) pairs
    """
    start_time = clock_time.time()
    fprint = lambda msg: logger.warning(msg) if logger else print(msg)
    hashable_key = (_make_hashable(answer), _make_hashable(reference))
    if hashable_key in evil_evals:
        fprint(f"[idx {idx}] Evil eval detected - skipping computation")
        return (idx, False, "Evil eval detected", False)
    if answer == 'THIS WAS AN NA RESULT':
        fprint(f"[idx {idx}] NA result - elapsed: {clock_time.time()-start_time:.2f}s")
        return (idx, False, None, False)
    pool = pool_container['pool']
    try:
        result = pool.apply_async(_grade_worker, (dataset_name, answer, reference))
        correct = result.get(timeout=10)
        elapsed = clock_time.time() - start_time
        if elapsed > 2.0:
            fprint(f"[idx {idx}] Slow grading ({elapsed:.2f}s) for {(answer, reference)}")
        return (idx, correct, None, False)
    except mp.TimeoutError:
        fprint(f"[idx {idx}] Timeout after 10s (treating as False) {(answer, reference)} - elapsed: {clock_time.time()-start_time:.2f}s")
        return (idx, False, "Timeout after 10 seconds", True)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)[:100]}"
        fprint(f"[idx {idx}] Exception (treating as False): {error_msg} - elapsed: {clock_time.time()-start_time:.2f}s")
        return (idx, False, error_msg, False)
def pedagogical_utility_experiment(args, logger = None):    
    student_model = args.student_model
    teacher_model = args.teacher_model
    dataset_name = args.dataset_name
    base_dir = args.base_dir
    config_dir = args.config_dir
    logger = logger or get_logger("./outputs/pedagogical_utility")
    limit = args.limit
    num_shards = args.num_shards
    shard_id = args.shard_id
    if shard_id >= num_shards:
        raise ValueError(f"shard_id ({shard_id}) must be less than num_shards ({num_shards})")
    logger.info(f"Processing shard {shard_id + 1}/{num_shards}")
    trace_path = get_trace_path(base_dir, dataset_name, teacher_model)
    traces = load_trace(trace_path)
    trace_metadata = traces['metadata']
    system_prompt = trace_metadata['system_prompt']
    datapoints = traces['data']
    logger.info(f"Loaded {len(datapoints['traces'])} traces from {trace_path} for dataset {dataset_name} from {teacher_model}.")
    total_datapoints = min(len(datapoints['questions']), limit) if limit else len(datapoints['questions'])
    shard_indices = list(range(shard_id, total_datapoints, num_shards))
    logger.info(f"Total datapoints: {total_datapoints}")
    logger.info(f"Shard {shard_id}: processing {len(shard_indices)} datapoints (interleaved sharding)")
    sampling_params, vllm_kwargs = get_generation_params(config_dir, dataset_name, student_model, logger)
    if sampling_params.get('max_tokens') is None:
        sampling_params['max_tokens'] = 8192
    logger.info(f"Using sampling params: {sampling_params}")
    pu_metadata = {
        'student_model': student_model,
        'teacher_model': teacher_model,
        'dataset_name': dataset_name,
        'generation_params': {
            'sampling_params': sampling_params,
            'vllm_kwargs': vllm_kwargs,
        },
        'trace_file': trace_path,
        'trace_metadata': trace_metadata,
        'collected_on': datetime.now().isoformat(),
        'shard_info': {
            'num_shards': num_shards,
            'shard_id': shard_id,
            'num_datapoints': len(shard_indices),
            'interleaved': True,
        }
    }
    student = vllm.LLM(student_model,
                       hf_token=os.getenv("HF_TOKEN"),
                       tokenizer=student_model,
                       **vllm_kwargs)
    logger.info(f"Loaded student model {student_model} with vLLM kwargs: {vllm_kwargs}")
    tokenizer = student.get_tokenizer()
    metadata = []
    inputs = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for idx in shard_indices:
            if len(datapoints['traces']) == 0:
                logger.warning(f"No stored traces found, extracting trace at index {idx}.")
                trace = extract_trace(datapoints['completions'][idx], teacher_model)
            elif datapoints['traces'][idx] != extract_trace(datapoints['completions'][idx], teacher_model):
                logger.warning(f"Trace mismatch at index {idx} between stored and extracted trace.")
                trace = extract_trace(datapoints['completions'][idx], teacher_model)
            else:
                trace = datapoints['traces'][idx]
            datapoint = {
                'questions': datapoints['questions'][idx],
                'traces': trace,
            }
            futures.append(executor.submit(process_datapoint, idx, tokenizer, system_prompt, datapoint, student_model, teacher_model, 3))
        for future in tqdm(futures, desc=f"Preparing datapoints (shard {shard_id})"):
            results = future.result()
            for meta, prompt in results:
                metadata.append(meta)
                inputs.append({"prompt_token_ids": prompt})
    logger.info(f"Prepared {len(inputs)} inputs for student model generation.")
    logger.info("Sorting inputs to maximize prefix cache efficiency...")
    indices_sorted = sorted(range(len(inputs)), key=lambda i: (metadata[i]['index'], metadata[i]['num_steps']))
    inputs_sorted = [inputs[i] for i in indices_sorted]
    metadata_sorted = [metadata[i] for i in indices_sorted]
    reverse_mapping = {sorted_idx: orig_idx for orig_idx, sorted_idx in enumerate(indices_sorted)}
    logger.info(f"Starting generation with {len(inputs_sorted)} prompts (sorted for prefix caching)...")
    first_pass_params = SamplingParams(**sampling_params)
    first_outputs_sorted = student.generate(inputs_sorted, sampling_params=first_pass_params)
    first_outputs = [None] * len(first_outputs_sorted)
    for sorted_idx, output in enumerate(first_outputs_sorted):
        orig_idx = reverse_mapping[sorted_idx]
        first_outputs[orig_idx] = output
    metadata = [metadata_sorted[indices_sorted.index(i)] for i in range(len(metadata))]
    inputs = [inputs_sorted[indices_sorted.index(i)] for i in range(len(inputs))]
    logger.info(f"Completed student model generation.")
    ctx = mp.get_context('spawn')
    logger.info("Creating worker pool (first call will be slow due to lazy imports)...")
    pool_container = {'pool': ctx.Pool(processes=1)}
    logger.info("Worker pool created.")
    evil_evals = set()
    second_pass_needed = {}
    final_outputs = {}
    for i, resp in enumerate(first_outputs):
        text = resp.outputs[0].text
        initial_prompt = tokenizer.decode(inputs[i]['prompt_token_ids'])
        answer = extract_answer(initial_prompt + text, dataset_name, student_model)
        ground_truth = datapoints['ground_truth_answers'][metadata[i]['index']]
        if answer is None or answer == '':
            second_pass_needed[i] = inputs[i]['prompt_token_ids'] + force_response(tokenizer, text, FOLLOW_UP_PROMPT[dataset_name])
        else:
            idx, correct, error_msg, needs_restart = check_correctness(answer, ground_truth, dataset_name, i, pool_container, evil_evals, logger)
            if needs_restart:
                logger.warning(f"Restarting grading pool due to timeout.")
                try:
                    pool_container['pool'].terminate()
                    pool_container['pool'].join()
                except Exception:
                    pass
                pool_container['pool'] = ctx.Pool(processes=1)
                logger.info(f"Grading pool restarted.")
                idx, correct, error_msg, needs_restart = check_correctness(answer, ground_truth, dataset_name, i, pool_container, evil_evals, logger)
            output = {
                'full_output': initial_prompt + resp.outputs[0].text,
                'completion_starts_at': len(initial_prompt),
                'extracted_answer': answer,
                'score': correct,
                'grading_error': error_msg,
                'logprobs': None,
                'ran_second_pass': False,
            }
            final_outputs[i] = output
    try:
        pool_container['pool'].close()
        pool_container['pool'].join()
        logger.info("Worker pool closed after first pass")
    except Exception as e:
        logger.warning(f"Error closing pool: {e}")
    if len(second_pass_needed) > 0:
        logger.info(f"Starting second pass for {len(second_pass_needed)} incomplete/malformed answers.")
        second_params = sampling_params
        second_params['max_tokens'] = 100
        second_pass_params = SamplingParams(**second_params)
        second_inputs = [{"prompt_token_ids": v} for v in second_pass_needed.values()]
        second_outputs = student.generate(second_inputs, sampling_params=second_pass_params)
        ctx = mp.get_context('spawn') 
        pool_container = {'pool': ctx.Pool(processes=1)}
        logger.info("Second pass worker pool created.")
        for idx, (i, resp) in enumerate(zip(second_pass_needed.keys(), second_outputs)):
            text = resp.outputs[0].text
            initial_prompt = tokenizer.decode(second_inputs[idx]['prompt_token_ids'])
            answer = extract_answer(initial_prompt + text, dataset_name, student_model)
            ground_truth = datapoints['ground_truth_answers'][metadata[i]['index']]
            idx, correct, error_msg, needs_restart = check_correctness(answer, ground_truth, dataset_name, i, pool_container, evil_evals, logger)
            if needs_restart:
                logger.warning(f"Restarting grading pool due to timeout (second pass).")
                try:
                    pool_container['pool'].terminate()
                    pool_container['pool'].join()
                except Exception:
                    pass
                pool_container['pool'] = ctx.Pool(processes=1)
                logger.info(f"Grading pool restarted (second pass).")
                idx, correct, error_msg, needs_restart = check_correctness(answer, ground_truth, dataset_name, i, pool_container, evil_evals, logger)
            output = {
                'full_output': initial_prompt + resp.outputs[0].text,
                'completion_starts_at': len(initial_prompt),
                'extracted_answer': answer,
                'score': correct,
                'grading_error': error_msg,
                'logprobs': None,
                'ran_second_pass': True,
            }
            final_outputs[i] = output
        try:
            pool_container['pool'].close()
            pool_container['pool'].join()
            logger.info("Worker pool closed after second pass")
        except Exception as e:
            logger.warning(f"Error closing pool: {e}")
    for i in range(len(metadata)):
        metadata[i].update(final_outputs[i])
    total_graded = len(metadata)
    timeout_errors = sum(1 for m in metadata if m.get('grading_error') == 'Timeout after 10 seconds')
    exception_errors = sum(1 for m in metadata if m.get('grading_error') and m.get('grading_error') != 'Timeout after 10 seconds')
    successful_grades = sum(1 for m in metadata if m.get('grading_error') is None)
    correct_answers = sum(1 for m in metadata if m.get('score') == 1)
    logger.info(f"Grading summary for shard {shard_id}/{num_shards}:")
    logger.info(f"  Total datapoints: {total_graded}")
    logger.info(f"  Successful grades: {successful_grades} ({100*successful_grades/total_graded:.1f}%)")
    logger.info(f"  Timeout errors: {timeout_errors} ({100*timeout_errors/total_graded:.1f}%)")
    logger.info(f"  Exception errors: {exception_errors} ({100*exception_errors/total_graded:.1f}%)")
    logger.info(f"  Correct answers: {correct_answers} ({100*correct_answers/total_graded:.1f}%)")
    results = {
        'metadata': pu_metadata,
        'data': metadata,
    }
    output_path = osp.join(args.output_dir, dataset_name, teacher_model.replace("/","_"))
    os.makedirs(output_path, exist_ok=True)
    if num_shards > 1:
        shard_dir = osp.join(output_path, "shards")
        os.makedirs(shard_dir, exist_ok=True)
        output_file = osp.join(shard_dir, f"pu_{student_model.replace('/','_')}_shard{shard_id}of{num_shards}.pkl")
    else:
        output_file = osp.join(output_path, f"pu_{student_model.replace('/','_')}.pkl")
    if osp.exists(output_file):
        logger.warning(f"Output file {output_file} already exists, moving to backup.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = osp.join(args.output_dir, "archived", dataset_name, teacher_model.replace("/","_"))
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = osp.join(backup_dir, osp.basename(output_file) + f".{timestamp}.bak")
        os.rename(output_file, backup_file)
        logger.info(f"Moved existing file to {backup_file}.")
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved pedagogical utility results (shard {shard_id}/{num_shards}) to {output_file}.")
if __name__ == "__main__":
    load_dotenv(f".{os.getenv('USER')}.env" if os.getenv("USER") else ".env")
    args = parse_args()
    logger = get_logger(args.log_path or "./logs/pedagogical_utility", args)
    pedagogical_utility_experiment(args, logger)
