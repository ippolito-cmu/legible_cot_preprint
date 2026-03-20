"""
Preprocessing script to convert large pickle files into optimized format for faster API serving.

This script:
1. Loads pickle files from outputs directory
2. Creates lightweight index files (JSON) with metadata only
3. Creates chunked data files for efficient random access
4. Compresses text data with zlib
5. Generates summary statistics upfront

Run this after generating new traces/analysis data.
"""
import os
import pickle
import json
import zlib
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
TRACES_DIR = OUTPUTS_DIR / 'traces'
EFFICIENCY_DIR = OUTPUTS_DIR / 'efficiency_analysis'
REDUNDANCY_DIR = OUTPUTS_DIR / 'redundancy_analysis'
CACHE_DIR = OUTPUTS_DIR / 'viz_cache'
def normalize_score(score, dataset_name, score_range=[0, 1]):
    """Normalize score to 0-1 range"""
    if dataset_name == 'connections':
        return score / 4.0
    return score
def compress_text(text):
    """Compress text with zlib"""
    if isinstance(text, str):
        text = text.encode('utf-8')
    return zlib.compress(text, level=6)
def decompress_text(compressed):
    """Decompress text"""
    return zlib.decompress(compressed).decode('utf-8')
def create_trace_index(dataset, model, trace_data, efficiency_data=None, redundancy_data=None):
    """
    Create optimized index file with:
    - Metadata
    - Summary statistics
    - Score array
    - Compressed text chunks
    """
    logger.info(f"Creating index for {dataset}/{model}")
    metadata = trace_data.get('metadata', {})
    data = trace_data.get('data', {})
    questions = data.get('questions', [])
    traces = data.get('traces', [])
    extracted_answers = data.get('extracted_answers', [])
    scores = data.get('scores', [])
    ground_truth = data.get('ground_truth_answers', [])
    total = len(questions)
    efficiency_lookup = {}
    if efficiency_data:
        for item in efficiency_data.get('data', []):
            idx = item.get('index', -1)
            efficiency_lookup[idx] = {
                'token_count': item.get('token_length', 0),
                'sentence_count': item.get('num_sentences', 0)
            }
    redundancy_lookup = {}
    if redundancy_data:
        for item in redundancy_data.get('data', []):
            idx = item.get('index', -1)
            redundancy_lookup[idx] = {
                'redundancy_score': item.get('redundancy_score', 0),
                'redundancy_per_step': item.get('redundancy_per_step', []),
                'redundancy_match_indices': item.get('redundancy_match_indices', [])
            }
    normalized_scores = [normalize_score(s, dataset) for s in scores]
    summary = {
        'total_traces': total,
        'accuracy': float(np.mean(normalized_scores)) if normalized_scores else 0,
        'has_efficiency': len(efficiency_lookup) > 0,
        'has_redundancy': len(redundancy_lookup) > 0,
    }
    if efficiency_lookup:
        token_counts = [efficiency_lookup[i]['token_count'] for i in range(total) if i in efficiency_lookup]
        if token_counts:
            summary['avg_tokens'] = float(np.mean(token_counts))
            summary['median_tokens'] = float(np.median(token_counts))
    if redundancy_lookup:
        red_scores = [redundancy_lookup[i]['redundancy_score'] for i in range(total) if i in redundancy_lookup]
        if red_scores:
            summary['avg_redundancy'] = float(np.mean(red_scores))
            summary['median_redundancy'] = float(np.median(red_scores))
    index = {
        'metadata': metadata,
        'summary': summary,
        'dataset': dataset,
        'model': model,
        'total': total,
    }
    cache_dir = CACHE_DIR / dataset / model
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / 'index.json'
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    logger.info(f"Saved index to {index_path}")
    scores_data = {
        'scores': normalized_scores,
        'raw_scores': scores,
    }
    scores_path = cache_dir / 'scores.json'
    with open(scores_path, 'w') as f:
        json.dump(scores_data, f)
    logger.info(f"Saved scores to {scores_path}")
    CHUNK_SIZE = 100
    chunks_dir = cache_dir / 'chunks'
    chunks_dir.mkdir(exist_ok=True)
    logger.info(f"Creating {(total + CHUNK_SIZE - 1) // CHUNK_SIZE} chunks...")
    for chunk_idx in tqdm(range(0, total, CHUNK_SIZE)):
        chunk_data = []
        end_idx = min(chunk_idx + CHUNK_SIZE, total)
        for i in range(chunk_idx, end_idx):
            trace_text = traces[i] if i < len(traces) else ''
            item = {
                'index': i,
                'question': questions[i] if i < len(questions) else '',
                'trace_compressed': compress_text(trace_text).hex(),
                'trace_length': len(trace_text),
                'extracted_answer': extracted_answers[i] if i < len(extracted_answers) else '',
                'score': normalized_scores[i] if i < len(normalized_scores) else 0,
                'raw_score': scores[i] if i < len(scores) else 0,
                'ground_truth': ground_truth[i] if i < len(ground_truth) else '',
            }
            if i in efficiency_lookup:
                item['efficiency'] = efficiency_lookup[i]
            if i in redundancy_lookup:
                red_data = redundancy_lookup[i]
                item['redundancy'] = {
                    'redundancy_score': red_data['redundancy_score'],
                    'has_per_step': len(red_data.get('redundancy_per_step', [])) > 0,
                }
                if len(red_data.get('redundancy_per_step', [])) > 10:
                    per_step_path = chunks_dir / f'redundancy_detail_{i}.json'
                    with open(per_step_path, 'w') as f:
                        json.dump({
                            'redundancy_per_step': red_data['redundancy_per_step'],
                            'redundancy_match_indices': red_data.get('redundancy_match_indices', [])
                        }, f)
                else:
                    item['redundancy']['redundancy_per_step'] = red_data.get('redundancy_per_step', [])
                    item['redundancy']['redundancy_match_indices'] = red_data.get('redundancy_match_indices', [])
            chunk_data.append(item)
        chunk_path = chunks_dir / f'chunk_{chunk_idx:06d}.json'
        with open(chunk_path, 'w') as f:
            json.dump(chunk_data, f)
    logger.info(f"Created {len(list(chunks_dir.glob('chunk_*.json')))} chunks")
    original_size = sum(len(str(q)) + len(str(t)) for q, t in zip(questions, traces))
    compressed_size = sum(len(compress_text(t)) for t in traces)
    compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
    logger.info(f"Text compression: {compression_ratio:.1f}% reduction")
    logger.info(f"Original size: {original_size / 1024 / 1024:.1f} MB")
    logger.info(f"Compressed size: {compressed_size / 1024 / 1024:.1f} MB")
    return index_path
def process_dataset(dataset_name, model_name=None):
    """Process all models in a dataset or a specific model"""
    dataset_dir = TRACES_DIR / dataset_name
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return
    trace_files = list(dataset_dir.glob('traces_*.pkl'))
    if model_name:
        trace_files = [f for f in trace_files if f.name == f'traces_{model_name}.pkl']
        if not trace_files:
            logger.error(f"Model not found: {model_name}")
            return
    logger.info(f"Processing {len(trace_files)} models in dataset {dataset_name}")
    for trace_file in trace_files:
        model = trace_file.name[7:-4]
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {dataset_name}/{model}")
        logger.info(f"{'='*60}")
        with open(trace_file, 'rb') as f:
            trace_data = pickle.load(f)
        efficiency_file = EFFICIENCY_DIR / dataset_name / model / 'efficiency_analysis.pkl'
        efficiency_data = None
        if efficiency_file.exists():
            with open(efficiency_file, 'rb') as f:
                efficiency_data = pickle.load(f)
            logger.info(f"Loaded efficiency data")
        redundancy_file = REDUNDANCY_DIR / dataset_name / model / 'redundancy_analysis.pkl'
        redundancy_data = None
        if redundancy_file.exists():
            with open(redundancy_file, 'rb') as f:
                redundancy_data = pickle.load(f)
            logger.info(f"Loaded redundancy data")
        create_trace_index(dataset_name, model, trace_data, efficiency_data, redundancy_data)
def main():
    parser = argparse.ArgumentParser(description='Preprocess data for faster API serving')
    parser.add_argument('--dataset', type=str, help='Dataset name (e.g., math, gpqa, connections)')
    parser.add_argument('--model', type=str, help='Model name (optional, process specific model only)')
    parser.add_argument('--all', action='store_true', help='Process all datasets')
    args = parser.parse_args()
    CACHE_DIR.mkdir(exist_ok=True)
    if args.all:
        datasets = [d.name for d in TRACES_DIR.iterdir() if d.is_dir() and d.name != 'archived']
        logger.info(f"Processing all datasets: {datasets}")
        for dataset in datasets:
            process_dataset(dataset)
    elif args.dataset:
        process_dataset(args.dataset, args.model)
    else:
        parser.print_help()
        logger.error("\nError: Must specify --dataset or --all")
        return 1
    logger.info("\n" + "="*60)
    logger.info("Preprocessing complete!")
    logger.info(f"Cache directory: {CACHE_DIR}")
    logger.info("="*60)
    total_size = sum(f.stat().st_size for f in CACHE_DIR.rglob('*') if f.is_file())
    logger.info(f"Total cache size: {total_size / 1024 / 1024:.1f} MB")
if __name__ == '__main__':
    main()
