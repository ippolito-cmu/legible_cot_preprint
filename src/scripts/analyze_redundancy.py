"""
Semantic Redundancy Analysis Script (GPU-Accelerated)

Analyzes reasoning traces for semantic redundancy using sentence embeddings.
Computes:
1. Step-by-step semantic similarity (using sentence-transformers)
2. Redundancy scores (fraction of steps above similarity threshold)
3. Per-step redundancy and source tracking
4. Correlation with token length and correctness

Similar structure to pedagogical_utility.py
For basic efficiency metrics (token length, sentence count), see analyze_efficiency.py
"""
import os
import pickle
import argparse
import os.path as osp
from typing import Dict, List
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
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
    parser = argparse.ArgumentParser(description="Semantic Redundancy Analysis Script")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to analyze")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--base_dir", type=str, default="outputs", help="Base directory for traces")
    parser.add_argument("--output_dir", type=str, default="outputs/redundancy_analysis", help="Output directory")
    parser.add_argument("--efficiency_dir", type=str, default="outputs/efficiency_analysis", help="Directory with efficiency analysis results")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of datapoints")
    parser.add_argument("--redundancy_threshold", type=float, default=0.85, help="Similarity threshold for redundancy")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards to split data into")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID to process (0-indexed)")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for encoding")
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
def load_efficiency_data(efficiency_dir: str, dataset_name: str, model_name: str):
    """Load efficiency analysis results if available."""
    model_name_safe = model_name.replace("/", "_")
    efficiency_path = osp.join(efficiency_dir, dataset_name, model_name_safe, "efficiency_analysis.pkl")
    if osp.exists(efficiency_path):
        with open(efficiency_path, "rb") as f:
            return pickle.load(f)
    return None
def compute_semantic_redundancy(trace_text: str, model, device, similarity_threshold: float = 0.85):
    """
    Compute semantic redundancy for a reasoning trace.
    
    Returns:
        - mean_score: fraction of steps with similarity > threshold
        - per_step_scores: similarity to most similar previous step for each step
        - max_sim_indices: index of most similar previous step for each step
    """
    steps = sent_tokenize(trace_text)
    n = len(steps)
    if n <= 1:
        return 0.0, [0.0], [-1]
    embeddings = model.encode(
        steps,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=device
    )
    sim_matrix = util.cos_sim(embeddings, embeddings)
    per_step_scores = [0.0]
    max_sim_indices = [-1]
    for i in range(1, n):
        prior_sims = sim_matrix[i, :i]
        max_val = torch.max(prior_sims).item()
        max_idx = torch.argmax(prior_sims).item()
        per_step_scores.append(max_val)
        max_sim_indices.append(max_idx)
    mean_score = float(np.mean(np.array(per_step_scores[1:]) > similarity_threshold))
    return mean_score, per_step_scores, max_sim_indices
def analyze_redundancy(args, logger=None):
    """Main redundancy analysis function."""
    model_name = args.model_name
    dataset_name = args.dataset_name
    base_dir = args.base_dir
    output_dir = args.output_dir
    efficiency_dir = args.efficiency_dir
    limit = args.limit
    num_shards = args.num_shards
    shard_id = args.shard_id
    redundancy_threshold = args.redundancy_threshold
    embedding_model_name = args.embedding_model
    batch_size = args.batch_size
    logger = logger or get_logger("./outputs/redundancy_analysis")
    if shard_id >= num_shards:
        raise ValueError(f"shard_id ({shard_id}) must be less than num_shards ({num_shards})")
    logger.info(f"Processing shard {shard_id + 1}/{num_shards}")
    trace_path = get_trace_path(base_dir, dataset_name, model_name)
    traces = load_trace(trace_path)
    trace_metadata = traces.get('metadata', {})
    datapoints = traces['data']
    logger.info(f"Loaded {len(datapoints.get('traces', []))} stored traces from {trace_path}")
    if len(datapoints.get('traces', [])) == 0:
        logger.warning("No stored traces found, will extract from completions during processing")
        total_raw = len(datapoints.get('completions', datapoints.get('questions', [])))
    else:
        total_raw = len(datapoints['traces'])
    logger.info(f"Total raw datapoints available: {total_raw}")
    efficiency_data = load_efficiency_data(efficiency_dir, dataset_name, model_name)
    if efficiency_data:
        logger.info(f"Loaded efficiency analysis data from {efficiency_dir}")
    total_datapoints = min(total_raw, limit) if limit else total_raw
    shard_size = (total_datapoints + num_shards - 1) // num_shards
    shard_start = shard_id * shard_size
    shard_end = min(shard_start + shard_size, total_datapoints)
    logger.info(f"Total datapoints: {total_datapoints}")
    logger.info(f"Shard {shard_id}: processing indices [{shard_start}, {shard_end}) ({shard_end - shard_start} datapoints)")
    logger.info(f"Initializing sentence embedding model: {embedding_model_name}")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    results = []
    logger.info(f"Starting redundancy analysis...")
    for idx in tqdm(range(shard_start, shard_end), desc=f"Analyzing redundancy (shard {shard_id})"):
        try:
            if len(datapoints['traces']) == 0:
                logger.warning(f"No stored traces found, extracting trace at index {idx}.")
                trace = extract_trace(datapoints['completions'][idx], model_name)
            elif datapoints['traces'][idx] != extract_trace(datapoints['completions'][idx], model_name):
                logger.warning(f"Trace mismatch at index {idx} between stored and extracted trace.")
                trace = extract_trace(datapoints['completions'][idx], model_name)
            else:
                trace = datapoints['traces'][idx]
            result = {
                'index': idx,
                'question': datapoints['questions'][idx],
                'ground_truth': datapoints['ground_truth_answers'][idx],
                'extracted_answer': datapoints['extracted_answers'][idx],
                'is_correct': datapoints['matches'][idx] if 'matches' in datapoints else None,
            }
            if efficiency_data and 'data' in efficiency_data:
                eff_results = efficiency_data['data']
                matching = [r for r in eff_results if r['index'] == idx]
                if matching:
                    eff = matching[0]
                    result['token_length'] = eff.get('token_length')
                    result['num_sentences'] = eff.get('num_sentences')
            redundancy_score, per_step_scores, match_indices = compute_semantic_redundancy(
                trace, embedding_model, device, redundancy_threshold
            )
            result['redundancy_score'] = redundancy_score
            result['redundancy_per_step'] = per_step_scores
            result['redundancy_match_indices'] = match_indices
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing index {idx}: {e}")
            continue
    logger.info(f"Completed redundancy analysis of {len(results)} traces")
    logger.info("Computing statistics...")
    df = pd.DataFrame(results)
    if len(df) == 0:
        logger.warning("No traces were processed in this shard. Saving empty results.")
        stats = {
            'total_traces': 0,
            'avg_redundancy_score': None,
            'std_redundancy_score': None,
        }
    else:
        stats = {
            'total_traces': len(df),
            'avg_redundancy_score': float(df['redundancy_score'].mean()),
            'std_redundancy_score': float(df['redundancy_score'].std()),
        }
        if df['is_correct'].notna().any():
            correct_df = df[df['is_correct'] == True]
            incorrect_df = df[df['is_correct'] == False]
            stats['correct_avg_redundancy'] = float(correct_df['redundancy_score'].mean()) if len(correct_df) > 0 else None
            stats['incorrect_avg_redundancy'] = float(incorrect_df['redundancy_score'].mean()) if len(incorrect_df) > 0 else None
        if 'token_length' in df.columns and df['token_length'].notna().any():
            stats['avg_token_length'] = float(df['token_length'].mean())
            stats['std_token_length'] = float(df['token_length'].std())
            if len(df) > 1:
                spearman_rho, spearman_p = spearmanr(df['token_length'], df['redundancy_score'])
                pearson_r, pearson_p = pearsonr(df['token_length'], df['redundancy_score'])
                stats['spearman_rho_length_redundancy'] = float(spearman_rho)
                stats['spearman_p_length_redundancy'] = float(spearman_p)
                stats['pearson_r_length_redundancy'] = float(pearson_r)
                stats['pearson_p_length_redundancy'] = float(pearson_p)
    logger.info(f"Statistics: {stats}")
    metadata = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'trace_file': trace_path,
        'trace_metadata': trace_metadata,
        'collected_on': datetime.now().isoformat(),
        'redundancy_threshold': redundancy_threshold,
        'embedding_model': embedding_model_name,
        'batch_size': batch_size,
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
        output_file = osp.join(output_path, f"redundancy_analysis_shard{shard_id}of{num_shards}.pkl")
    else:
        output_file = osp.join(output_path, f"redundancy_analysis.pkl")
    if osp.exists(output_file):
        logger.warning(f"Output file {output_file} already exists, moving to backup.")
        backup_dir = osp.join(output_dir, "archived")
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = osp.join(backup_dir, osp.basename(output_file) + ".bak")
        os.rename(output_file, backup_file)
        logger.info(f"Moved existing file to {backup_file}.")
    with open(output_file, "wb") as f:
        pickle.dump(output_data, f)
    logger.info(f"Saved redundancy analysis (shard {shard_id}/{num_shards}) to {output_file}.")
    return output_file
if __name__ == "__main__":
    load_dotenv(f".{os.getenv('USER')}.env" if os.getenv("USER") else ".env")
    args = parse_args()
    logger = get_logger(args.log_path or "./logs/redundancy_analysis", args)
    analyze_redundancy(args, logger)
