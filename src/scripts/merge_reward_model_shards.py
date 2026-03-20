"""Merge reward model analysis shards into a single file.

Usage:
  python3 -m src.scripts.merge_reward_model_shards \
        --input_pattern "outputs/reward_model_analysis/<dataset>/<teacher>/shards/reward_model_*_shard*of4.pkl" \
        --output_file "outputs/reward_model_analysis/<dataset>/<teacher>/reward_model_<rm_safe>.pkl"
"""
from __future__ import annotations
import argparse
import glob
import os
import os.path as osp
import pickle
from typing import Any, Dict, List
import numpy as np
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge reward model analysis shards")
    parser.add_argument("--input_pattern", type=str, required=True, help="Glob pattern for shard files")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--keep_shards", action="store_true", help="Keep shard files after merging")
    return parser.parse_args()
def _extract_shard_id(path: str) -> int:
    base = osp.basename(path)
    if "shard" in base and "of" in base:
        try:
            tail = base.split("shard", 1)[1]
            shard_part = tail.split("of", 1)[0]
            return int(shard_part)
        except Exception:
            return 0
    return 0
def merge_shards(shard_files: List[str]) -> Dict[str, Any]:
    if not shard_files:
        raise ValueError("No shard files provided")
    shard_files = sorted(shard_files, key=_extract_shard_id)
    with open(shard_files[0], "rb") as f:
        merged = pickle.load(f)
    base_meta = dict(merged.get("metadata", {}))
    merged_data: List[Dict[str, Any]] = list(merged.get("data", []) or [])
    base_reward_model = base_meta.get("reward_model")
    base_reward_model_safe = base_meta.get("reward_model_safe")
    for path in shard_files[1:]:
        with open(path, "rb") as f:
            shard = pickle.load(f)
        shard_meta = shard.get("metadata", {})
        if shard_meta.get("dataset_name") != base_meta.get("dataset_name"):
            raise ValueError("Dataset mismatch while merging shards")
        if shard_meta.get("teacher_model") != base_meta.get("teacher_model"):
            raise ValueError("Teacher model mismatch while merging shards")
        if shard_meta.get("reward_model") != base_reward_model:
            raise ValueError("Reward model mismatch while merging shards")
        if shard_meta.get("reward_model_safe") != base_reward_model_safe:
            raise ValueError("Reward model safe-name mismatch while merging shards")
        data = shard.get("data", []) or []
        merged_data.extend(data)
    rewards = [d.get("rm_reward") for d in merged_data if isinstance(d.get("rm_reward"), (int, float))]
    stats: Dict[str, Any] = {
        "total_datapoints": len(merged_data),
        "reward_mean": float(np.mean(rewards)) if rewards else None,
        "reward_std": float(np.std(rewards)) if rewards else None,
        "reward_min": float(np.min(rewards)) if rewards else None,
        "reward_max": float(np.max(rewards)) if rewards else None,
    }
    per_model: Dict[str, List[float]] = {}
    for d in merged_data:
        rbm = d.get("rewards_by_model")
        if not isinstance(rbm, dict):
            continue
        for mid, val in rbm.items():
            if isinstance(val, (int, float)):
                per_model.setdefault(mid, []).append(float(val))
    if per_model:
        stats["per_model_mean"] = {k: float(np.mean(v)) for k, v in per_model.items() if v}
    merged["metadata"] = base_meta
    if "shard_info" in merged["metadata"]:
        del merged["metadata"]["shard_info"]
    merged["metadata"]["merged_from_shards"] = len(shard_files)
    merged["metadata"]["statistics"] = stats
    merged["data"] = merged_data
    return merged
def main() -> None:
    args = parse_args()
    shard_files = glob.glob(args.input_pattern)
    if not shard_files:
        print(f"No files found matching pattern: {args.input_pattern}")
        return
    merged = merge_shards(shard_files)
    out_dir = osp.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "wb") as f:
        pickle.dump(merged, f)
    print(f"Saved merged results to: {args.output_file}")
    if not args.keep_shards:
        for p in shard_files:
            os.remove(p)
if __name__ == "__main__":
    main()
