"""
Merge pedagogical utility shards into a single file.

Usage:
    python -m src.scripts.merge_pu_shards --input_pattern "path/to/pu_*_shard*of*.pkl" --output_file "path/to/pu_merged.pkl"
"""
import argparse
import pickle
import glob
import os.path as osp
from collections import defaultdict
def parse_args():
    parser = argparse.ArgumentParser(description="Merge pedagogical utility shards")
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
        if shard['metadata']['student_model'] != base_metadata['student_model']:
            raise ValueError(f"Student model mismatch: {shard['metadata']['student_model']} vs {base_metadata['student_model']}")
        if shard['metadata']['teacher_model'] != base_metadata['teacher_model']:
            raise ValueError(f"Teacher model mismatch: {shard['metadata']['teacher_model']} vs {base_metadata['teacher_model']}")
        merged_data.extend(shard['data'])
    merged['metadata'] = base_metadata
    if 'shard_info' in merged['metadata']:
        del merged['metadata']['shard_info']
    merged['metadata']['merged_from_shards'] = len(shard_files)
    merged['data'] = merged_data
    print(f"Merged {len(merged_data)} total datapoints")
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
    if osp.exists(args.output_file):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_parts = args.output_file.split(os.sep)
        if len(path_parts) >= 3:
            dataset = path_parts[-3]
            teacher = path_parts[-2]
            backup_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(args.output_file))), 
                                  "archived", dataset, teacher)
        else:
            backup_dir = osp.join(output_dir, "archived")
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = osp.join(backup_dir, osp.basename(args.output_file) + f".{timestamp}.bak")
        print(f"\nBacking up existing file to: {backup_file}")
        os.rename(args.output_file, backup_file)
    with open(args.output_file, 'wb') as f:
        pickle.dump(merged, f)
    print(f"\nSaved merged results to: {args.output_file}")
    if not args.keep_shards:
        print(f"\nRemoving {len(shard_files)} shard files...")
        for shard_file in shard_files:
            os.remove(shard_file)
            print(f"  Removed: {osp.basename(shard_file)}")
    else:
        print(f"\nKeeping shard files (use --keep_shards=False to remove)")
if __name__ == "__main__":
    import os
    main()
