"""
Merge redundancy analysis shards into a single file.

Usage:
    python -m src.scripts.merge_redundancy_shards --input_pattern "path/to/redundancy_*_shard*of*.pkl" --output_file "path/to/redundancy_merged.pkl"
"""
import argparse
import pickle
import glob
import os
import os.path as osp
def parse_args():
    parser = argparse.ArgumentParser(description="Merge redundancy analysis shards")
    parser.add_argument("--input_pattern", type=str, required=True, help="Glob pattern for shard files")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--keep_shards", action="store_true", help="Keep shard files after merging")
    return parser.parse_args()
def merge_shards(shard_files):
    """Merge multiple shard files into a single result."""
    if not shard_files:
        raise ValueError("No shard files found")
    def extract_shard_id(filename):
        basename = osp.basename(filename)
        if 'shard' in basename:
            shard_part = basename.split('shard')[1].split('.pkl')[0]
            shard_id = int(shard_part.split('of')[0])
            return shard_id
        return 0
    shard_files = sorted(shard_files, key=extract_shard_id)
    print(f"Merging {len(shard_files)} shard files...")
    for f in shard_files:
        print(f"  - {osp.basename(f)}")
    with open(shard_files[0], 'rb') as f:
        merged = pickle.load(f)
    base_metadata = merged['metadata'].copy()
    merged_data = merged['data']
    for shard_file in shard_files[1:]:
        with open(shard_file, 'rb') as f:
            shard = pickle.load(f)
        if shard['metadata']['model_name'] != base_metadata['model_name']:
            raise ValueError(f"Model mismatch: {shard['metadata']['model_name']} vs {base_metadata['model_name']}")
        if shard['metadata']['dataset_name'] != base_metadata['dataset_name']:
            raise ValueError(f"Dataset mismatch: {shard['metadata']['dataset_name']} vs {base_metadata['dataset_name']}")
        merged_data.extend(shard['data'])
    import pandas as pd
    import numpy as np
    from scipy.stats import spearmanr, pearsonr
    df = pd.DataFrame(merged_data)
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
    merged['metadata'] = base_metadata
    if 'shard_info' in merged['metadata']:
        del merged['metadata']['shard_info']
    merged['metadata']['merged_from_shards'] = len(shard_files)
    merged['metadata']['statistics'] = stats
    merged['data'] = merged_data
    print(f"Merged {len(merged_data)} total datapoints")
    print(f"\nMerged statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return merged
def main():
    args = parse_args()
    shard_files = glob.glob(args.input_pattern)
    if not shard_files:
        print(f"No files found matching pattern: {args.input_pattern}")
        return
    merged = merge_shards(shard_files)
    output_dir = osp.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(merged, f)
    print(f"\nSaved merged results to: {args.output_file}")
    if not args.keep_shards:
        print(f"\nRemoving {len(shard_files)} shard files...")
        for shard_file in shard_files:
            os.remove(shard_file)
            print(f"  Removed: {osp.basename(shard_file)}")
    else:
        print(f"\nKeeping shard files")
if __name__ == "__main__":
    main()
