"""
Reasoning Efficiency Analysis Script (Basic Metrics - CPU Friendly)

Analyzes reasoning traces for:
1. Token length (using tiktoken)
2. Number of reasoning steps (sentence count)
3. Relationship to correctness

Similar structure to pedagogical_utility.py
For semantic redundancy analysis, see analyze_redundancy.py
"""
import os
import pickle
import argparse
import yaml
import os.path as osp
from typing import Dict, List
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize
from scipy.stats import spearmanr, pearsonr
from dotenv import load_dotenv
from src.utils.logging import get_logger
from src.utils.extractors import extract_trace
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
def parse_args():
    parser = argparse.ArgumentParser(description="Reasoning Efficiency Analysis Script (Basic Metrics)")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to analyze")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--base_dir", type=str, default="outputs", help="Base directory for traces")
    parser.add_argument("--output_dir", type=str, default="outputs/efficiency_analysis", help="Output directory")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of datapoints")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards to split data into")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID to process (0-indexed)")
    return parser.parse_args()
def get_trace_path(base_dir: str, dataset_name: str, model_name: str):
    """Get path to trace file."""
    model_name = model_name.replace("/", "_")
    trace_path = osp.join(base_dir, "traces", dataset_name, f"traces_{model_name}.pkl")
    if not osp.exists(trace_path):
        raise FileNotFoundError(f"Trace file {trace_path} not found.")
    return trace_path
def load_trace(filename: str):
    """Load pickle file."""
    with open(filename, "rb") as f:
        return pickle.load(f)
def count_sentences(text: str) -> int:
    """Count number of sentences using NLTK."""
    if not isinstance(text, str) or text.strip() == "":
        return 0
    return len(sent_tokenize(text))
def count_tokens(text: str, encoder) -> int:
    """Count tokens using tiktoken."""
    if not isinstance(text, str) or text.strip() == "":
        return 0
    try:
        return len(encoder.encode(text, disallowed_special=()))
    except Exception:
        return 0
def analyze_efficiency(args, logger=None):
    """Main efficiency analysis function (basic metrics only)."""
    model_name = args.model_name
    dataset_name = args.dataset_name
    base_dir = args.base_dir
    output_dir = args.output_dir
    limit = args.limit
    num_shards = args.num_shards
    shard_id = args.shard_id
    logger = logger or get_logger("./outputs/efficiency_analysis")
    if shard_id >= num_shards:
        raise ValueError(f"shard_id ({shard_id}) must be less than num_shards ({num_shards})")
    logger.info(f"Processing shard {shard_id + 1}/{num_shards}")
    trace_path = get_trace_path(base_dir, dataset_name, model_name)
    traces = load_trace(trace_path)
    trace_metadata = traces.get('metadata', {})
    datapoints = traces['data']
    logger.info(f"Loaded {len(datapoints['traces'])} traces from {trace_path}")
    total_datapoints = min(len(datapoints['traces']), limit) if limit else len(datapoints['traces'])
    shard_size = (total_datapoints + num_shards - 1) // num_shards
    shard_start = shard_id * shard_size
    shard_end = min(shard_start + shard_size, total_datapoints)
    logger.info(f"Total datapoints: {total_datapoints}")
    logger.info(f"Shard {shard_id}: processing indices [{shard_start}, {shard_end}) ({shard_end - shard_start} datapoints)")
    logger.info("Initializing tiktoken encoder...")
    encoder = tiktoken.get_encoding("cl100k_base")
    results = []
    logger.info(f"Starting analysis...")
    for idx in tqdm(range(shard_start, shard_end), desc=f"Analyzing traces (shard {shard_id})"):
        try:
            completion = datapoints['completions'][idx]
            trace = extract_trace(completion, model_name)
            result = {
                'index': idx,
                'question': datapoints['questions'][idx],
                'ground_truth': datapoints['ground_truth_answers'][idx],
                'extracted_answer': datapoints['extracted_answers'][idx],
                'is_correct': datapoints['matches'][idx] if 'matches' in datapoints else None,
            }
            result['token_length'] = count_tokens(trace, encoder)
            result['num_sentences'] = count_sentences(trace)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing index {idx}: {e}")
            continue
    logger.info(f"Completed analysis of {len(results)} traces")
    logger.info("Computing statistics...")
    df = pd.DataFrame(results)
    if len(df) == 0:
        logger.warning("No traces were processed in this shard. Saving empty results.")
        stats = {
            'total_traces': 0,
            'avg_token_length': None,
            'std_token_length': None,
            'avg_num_sentences': None,
            'std_num_sentences': None,
        }
    else:
        stats = {
            'total_traces': len(df),
            'avg_token_length': float(df['token_length'].mean()),
            'std_token_length': float(df['token_length'].std()),
            'avg_num_sentences': float(df['num_sentences'].mean()),
            'std_num_sentences': float(df['num_sentences'].std()),
        }
        if df['is_correct'].notna().any():
            correct_df = df[df['is_correct'] == True]
            incorrect_df = df[df['is_correct'] == False]
            stats['correct_avg_token_length'] = float(correct_df['token_length'].mean()) if len(correct_df) > 0 else None
            stats['incorrect_avg_token_length'] = float(incorrect_df['token_length'].mean()) if len(incorrect_df) > 0 else None
            stats['correct_avg_num_sentences'] = float(correct_df['num_sentences'].mean()) if len(correct_df) > 0 else None
            stats['incorrect_avg_num_sentences'] = float(incorrect_df['num_sentences'].mean()) if len(incorrect_df) > 0 else None
        if len(df) > 1:
            spearman_rho, spearman_p = spearmanr(df['token_length'], df['num_sentences'])
            pearson_r, pearson_p = pearsonr(df['token_length'], df['num_sentences'])
            stats['spearman_rho_length_sentences'] = float(spearman_rho)
            stats['spearman_p_length_sentences'] = float(spearman_p)
            stats['pearson_r_length_sentences'] = float(pearson_r)
            stats['pearson_p_length_sentences'] = float(pearson_p)
    logger.info(f"Statistics: {stats}")
    metadata = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'trace_file': trace_path,
        'trace_metadata': trace_metadata,
        'collected_on': datetime.now().isoformat(),
        'shard_info': {
            'num_shards': num_shards,
            'shard_id': shard_id,
            'shard_start': shard_start,
            'shard_end': shard_end,
        },
        'statistics': stats,
    }
    output_data = {
        'metadata': metadata,
        'data': results,
    }
    output_path = osp.join(output_dir, dataset_name, model_name.replace("/", "_"))
    os.makedirs(output_path, exist_ok=True)
    if num_shards > 1:
        output_file = osp.join(output_path, f"efficiency_analysis_shard{shard_id}of{num_shards}.pkl")
    else:
        output_file = osp.join(output_path, f"efficiency_analysis.pkl")
    if osp.exists(output_file):
        logger.warning(f"Output file {output_file} already exists, moving to backup.")
        backup_dir = osp.join(output_dir, "archived")
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = osp.join(backup_dir, osp.basename(output_file) + ".bak")
        os.rename(output_file, backup_file)
        logger.info(f"Moved existing file to {backup_file}.")
    with open(output_file, "wb") as f:
        pickle.dump(output_data, f)
    logger.info(f"Saved efficiency analysis (shard {shard_id}/{num_shards}) to {output_file}.")
    return output_file
if __name__ == "__main__":
    load_dotenv(f".{os.getenv('USER')}.env" if os.getenv("USER") else ".env")
    args = parse_args()
    logger = get_logger(args.log_path or "./logs/efficiency_analysis", args)
    analyze_efficiency(args, logger)
