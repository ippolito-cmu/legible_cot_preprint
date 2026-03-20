
from __future__ import annotations
import argparse
import glob
import json
import logging
import os
import os.path as osp
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib
os.makedirs("file_logs", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", osp.join("file_logs", "mpl_config"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from scipy import stats
def _setup_logger() -> logging.Logger:
    os.makedirs("file_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = osp.join("file_logs", f"analyze_trace_level_pu_rm_{timestamp}.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
def _read_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None
def canonical_model_name(model: str) -> str:
    m = str(model or "").strip()
    if not m:
        return ""
    m = m.replace("/", "_")
    low = m.lower()
    if "deepseek" in low and "r1-0528" in low:
        return "deepseek-ai_deepseek-r1-0528"
    return m
def _iter_pu_split_tables(conn: sqlite3.Connection) -> List[Tuple[str, str, str]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name LIKE 'pu_%'
          AND name NOT IN ('pu_percentage_bins','pu_absolute_bins','pu_regression_cache')
          AND name NOT LIKE 'pu_%_bins'
        ORDER BY name
        """
    )
    tables: List[Tuple[str, str, str]] = []
    for (name,) in cur.fetchall():
        parts = str(name).split("_", 2)
        if len(parts) < 3:
            continue
        tables.append((str(name), str(parts[1]).lower(), str(parts[2])))
    return tables
def _dataset_max_score(dataset: str) -> float:
    ds = str(dataset or "").lower()
    if ds == "math":
        return 5.0
    if ds == "gpqa":
        return 1.0
    if ds == "connections":
        return 4.0
    return 1.0
def _load_pu_per_trace(db_path: str, *, logger: logging.Logger, pu_student: str = "both") -> pd.DataFrame:
    """Load per-trace PU metrics from split tables.

    Returns columns: dataset, model, trace_index, pu_avg_correctness, pu_linearity_mse, pu_regression_rate.
    If pu_student == 'both', averages per-trace across students.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        tables = _iter_pu_split_tables(conn)
        if not tables:
            raise RuntimeError("No PU split tables found in DB")
        want_students: Optional[set] = None
        if pu_student == "phi":
            want_students = {"microsoft_phi_3_mini_128k"}
        elif pu_student == "both":
            want_students = None
        else:
            raise ValueError(f"Unknown pu_student: {pu_student}")
        out_rows: List[Dict[str, Any]] = []
        def flush_trace(
            *,
            dataset: str,
            student_short: str,
            model: str,
            trace_index: int,
            total_correct: float,
            total_n: int,
            step_sum_norm: Dict[int, float],
            step_sum_correct: Dict[int, float],
            step_n: Dict[int, int],
        ) -> None:
            if total_n <= 0:
                return
            avg_correct = float(total_correct / float(total_n))
            steps_sorted = sorted(step_n.keys())
            xs: List[float] = []
            y_norm: List[float] = []
            y_correct: List[float] = []
            for s in steps_sorted:
                n = int(step_n.get(s, 0))
                if n <= 0:
                    continue
                xs.append(float(s))
                y_norm.append(float(step_sum_norm.get(s, 0.0)) / float(n))
                y_correct.append(float(step_sum_correct.get(s, 0.0)) / float(n))
            linearity_mse = None
            if len(xs) >= 2:
                coeff = np.polyfit(np.asarray(xs, dtype=float), np.asarray(y_norm, dtype=float), deg=1)
                yhat = coeff[0] * np.asarray(xs, dtype=float) + coeff[1]
                linearity_mse = float(np.mean((np.asarray(y_norm, dtype=float) - yhat) ** 2))
            regressions = 0
            transitions = 0
            if len(y_correct) >= 2:
                for a, b in zip(y_correct[:-1], y_correct[1:]):
                    transitions += 1
                    if a >= 0.5 and b < 0.5:
                        regressions += 1
            regression_rate = float(regressions / transitions) if transitions > 0 else None
            out_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "trace_index": int(trace_index),
                    "student": student_short,
                    "pu_avg_correctness": avg_correct,
                    "pu_linearity_mse": linearity_mse,
                    "pu_regression_rate": regression_rate,
                }
            )
        for table, dataset, student_short in tables:
            if want_students is not None and student_short not in want_students:
                continue
            max_score = float(_dataset_max_score(dataset))
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT teacher_model, trace_index, num_steps, score
                FROM {table}
                ORDER BY teacher_model, trace_index, num_steps
                """
            )
            current_key: Optional[Tuple[str, int]] = None
            total_correct = 0.0
            total_n = 0
            step_sum_norm: Dict[int, float] = {}
            step_sum_correct: Dict[int, float] = {}
            step_n: Dict[int, int] = {}
            n_rows = 0
            for r in cur:
                teacher_model = canonical_model_name(r["teacher_model"])
                trace_index = int(r["trace_index"])
                num_steps = int(r["num_steps"])
                score = _as_float(r["score"]) or 0.0
                key = (teacher_model, trace_index)
                if current_key is None:
                    current_key = key
                if key != current_key:
                    flush_trace(
                        dataset=str(dataset).lower(),
                        student_short=student_short,
                        model=str(current_key[0]),
                        trace_index=int(current_key[1]),
                        total_correct=total_correct,
                        total_n=total_n,
                        step_sum_norm=step_sum_norm,
                        step_sum_correct=step_sum_correct,
                        step_n=step_n,
                    )
                    current_key = key
                    total_correct = 0.0
                    total_n = 0
                    step_sum_norm = {}
                    step_sum_correct = {}
                    step_n = {}
                correct = 1.0 if score > 0.0 else 0.0
                norm = (score / max_score) if max_score > 0 else 0.0
                total_correct += correct
                total_n += 1
                step_sum_norm[num_steps] = float(step_sum_norm.get(num_steps, 0.0)) + float(norm)
                step_sum_correct[num_steps] = float(step_sum_correct.get(num_steps, 0.0)) + float(correct)
                step_n[num_steps] = int(step_n.get(num_steps, 0)) + 1
                n_rows += 1
                if n_rows % 500000 == 0:
                    logger.info(f"PU ingest progress: table={table}, rows={n_rows}")
            if current_key is not None:
                flush_trace(
                    dataset=str(dataset).lower(),
                    student_short=student_short,
                    model=str(current_key[0]),
                    trace_index=int(current_key[1]),
                    total_correct=total_correct,
                    total_n=total_n,
                    step_sum_norm=step_sum_norm,
                    step_sum_correct=step_sum_correct,
                    step_n=step_n,
                )
        if not out_rows:
            raise RuntimeError("No PU rows computed from split tables")
        df = pd.DataFrame(out_rows)
        df["model"] = df["model"].map(canonical_model_name)
        if pu_student == "both":
            df = (
                df.groupby(["dataset", "model", "trace_index"], as_index=False)
                .mean(numeric_only=True)
                .drop(columns=[c for c in ["student"] if c in df.columns], errors="ignore")
            )
        logger.info(f"Loaded PU per-trace: rows={len(df)}")
        return df
    finally:
        conn.close()
def _load_trace_correctness(db_path: str, *, logger: logging.Logger, threshold: float = 0.5) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        tables = [r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        traces_table = "traces" if "traces" in tables else None
        if traces_table is None:
            raise RuntimeError("Could not find traces table")
        cols = [c["name"] for c in conn.execute(f"PRAGMA table_info({traces_table})").fetchall()]
        idx_col = "trace_index" if "trace_index" in cols else ("index" if "index" in cols else None)
        score_col = "score" if "score" in cols else None
        if idx_col is None or score_col is None:
            raise RuntimeError(f"Missing required columns in {traces_table}: idx={idx_col}, score={score_col}")
        q = f"""
            SELECT LOWER(dataset) AS dataset,
                   model AS model_raw,
                   {idx_col} AS trace_index,
                   {score_col} AS trace_score
            FROM {traces_table}
        """
        df = pd.read_sql_query(q, conn)
        df["model"] = df["model_raw"].map(canonical_model_name)
        df["trace_correct"] = pd.to_numeric(df["trace_score"], errors="coerce") > float(threshold)
        df = df[["dataset", "model", "trace_index", "trace_correct"]].copy()
        logger.info(f"Loaded trace correctness: rows={len(df)}")
        return df
    finally:
        conn.close()
def _load_rm_per_trace(outputs_dir: str, *, logger: logging.Logger, max_files: Optional[int] = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    paths = sorted(glob.glob(osp.join(outputs_dir, "reward_model_analysis", "*", "*", "reward_model_*.pkl")))
    logger.info(f"Loading reward_model_analysis from {len(paths)} files")
    count = 0
    for p in paths:
        dataset = osp.basename(osp.dirname(osp.dirname(p))).lower()
        model_raw = osp.basename(osp.dirname(p))
        rm_file = osp.basename(p)
        rm_safe = rm_file.removeprefix("reward_model_").removesuffix(".pkl")
        col = f"rm::{rm_safe}"
        obj = _read_pickle(p)
        data = obj.get("data", []) if isinstance(obj, dict) else []
        if not isinstance(data, list):
            continue
        for d in data:
            if not isinstance(d, dict):
                continue
            idx = d.get("index")
            if idx is None:
                continue
            try:
                idx_i = int(idx)
            except Exception:
                continue
            score = _as_float(d.get("reward"))
            if score is None:
                score = _as_float(d.get("rm_reward"))
            if score is None:
                score = _as_float(d.get("score"))
            if score is None and isinstance(d.get("rewards_by_model"), dict):
                for vv in d["rewards_by_model"].values():
                    score = _as_float(vv)
                    if score is not None:
                        break
            rows.append(
                {
                    "dataset": dataset,
                    "model": canonical_model_name(str(model_raw)),
                    "trace_index": idx_i,
                    col: float(score) if score is not None else float("nan"),
                }
            )
        count += 1
        if max_files is not None and count >= max_files:
            break
    if not rows:
        raise RuntimeError("No RM rows loaded from reward_model_analysis pickles")
    df = pd.DataFrame(rows)
    df = df.groupby(["dataset", "model", "trace_index"], as_index=False).mean(numeric_only=True)
    logger.info(f"Loaded RM per-trace: rows={len(df)}, cols={len(df.columns)}")
    return df
def _pearson_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": n}
    rho, p = stats.pearsonr(x, y)
    return {"rho": float(rho), "p": float(p), "n": n}
def _bootstrap_rho_ci(x: np.ndarray, y: np.ndarray, *, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    rhos: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xx = x[idx]
        yy = y[idx]
        try:
            r, _ = stats.pearsonr(xx, yy)
            rhos.append(float(r))
        except Exception:
            rhos.append(float("nan"))
    arr = np.asarray(rhos, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 10:
        return float("nan"), float("nan")
    lo = float(np.quantile(arr, 0.025))
    hi = float(np.quantile(arr, 0.975))
    return lo, hi
def _save_scatter(df: pd.DataFrame, *, x: str, y: str, title: str, out_path: str) -> None:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.scatter(df[x].to_numpy(dtype=float), df[y].to_numpy(dtype=float), s=8, alpha=0.15, edgecolors="none")
    stats0 = _pearson_stats(df[x].to_numpy(dtype=float), df[y].to_numpy(dtype=float))
    lo, hi = _bootstrap_rho_ci(df[x].to_numpy(dtype=float), df[y].to_numpy(dtype=float))
    txt = f"Pearson $\\rho$={stats0['rho']:.3f} (p={stats0['p']:.2g}, n={stats0['n']})\n95% boot CI: [{lo:.3f}, {hi:.3f}]"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.3, alpha=0.9))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
def _plot_rho_forest(
    df_by_ds: pd.DataFrame,
    *,
    out_path: str,
) -> None:
    """Forest plot for rho by dataset/condition/metric with bootstrap CIs."""
    sns.set_style("whitegrid")
    metric_order = ["pu_avg_correctness", "pu_linearity_mse", "pu_regression_rate"]
    cond_order = ["unconditional", "correct_only"]
    ds_order = ["math", "gpqa", "connections"]
    df = df_by_ds.copy()
    df["metric"] = pd.Categorical(df["metric"], categories=metric_order, ordered=True)
    df["condition"] = pd.Categorical(df["condition"], categories=cond_order, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=ds_order, ordered=True)
    df = df.sort_values(["metric", "dataset", "condition"]).reset_index(drop=True)
    df["ypos"] = np.arange(len(df), dtype=float)
    fig_h = max(4.0, 0.33 * len(df))
    fig, ax = plt.subplots(figsize=(6.6, fig_h))
    color_map = {"unconditional": "#666666", "correct_only": "#111111"}
    marker_map = {"unconditional": "o", "correct_only": "s"}
    for _, r in df.iterrows():
        rho = float(r["rho"])
        lo = float(r["rho_ci95_lo"])
        hi = float(r["rho_ci95_hi"])
        cond = str(r["condition"])
        ax.plot([lo, hi], [r["ypos"], r["ypos"]], color=color_map.get(cond, "#333333"), linewidth=2.0)
        ax.scatter([rho], [r["ypos"]], s=45, marker=marker_map.get(cond, "o"), color=color_map.get(cond, "#333333"), zorder=3)
    y_labels = []
    for _, r in df.iterrows():
        ds = str(r["dataset"]).upper()
        metric = str(r["metric"])
        if metric == "pu_avg_correctness":
            m = "TU avg correctness"
        elif metric == "pu_linearity_mse":
            m = "TU linearity MSE"
        else:
            m = "TU regression rate"
        cond = "uncond" if str(r["condition"]) == "unconditional" else "correct-only"
        y_labels.append(f"{ds} | {m} | {cond}")
    ax.set_yticks(df["ypos"].to_list())
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.axvline(0.0, color="#000000", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Pearson $\\rho$ between RM(mean of 2) and PU metric", fontsize=11)
    ax.set_title("Trace-level RM↔PU correlations (with 95% bootstrap CI)", fontsize=12)
    fig.tight_layout()
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trace-level PU↔RM correlations (conditional/unconditional)")
    p.add_argument("--db_path", type=str, default="outputs/traces.db")
    p.add_argument("--outputs_dir", type=str, default="outputs")
    p.add_argument("--out_dir", type=str, default="analysis/experiments/trace_level_pu_rm")
    p.add_argument("--correctness_threshold", type=float, default=0.5)
    p.add_argument("--max_rm_files", type=int, default=None)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--rm_cols",
        type=str,
        nargs="+",
        default=[
            "rm::Skywork_Skywork-Reward-V2-Llama-3.1-8B",
            "rm::allenai_Llama-3.1-8B-Instruct-RM-RB2",
        ],
        help="Which RM columns to average (must match rm::<name> in pickles)",
    )
    return p.parse_args()
def main() -> None:
    logger = _setup_logger()
    args = parse_args()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = osp.join(args.out_dir, run_stamp)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(
        f"Starting {__name__}: timestamp={datetime.now().isoformat()}, user={os.getenv('USER')}"
    )
    logger.info(f"db_path={args.db_path}")
    logger.info(f"outputs_dir={args.outputs_dir}")
    logger.info(f"run_dir={run_dir}")
    df_pu = _load_pu_per_trace(args.db_path, logger=logger, pu_student="both")
    df_correct = _load_trace_correctness(args.db_path, logger=logger, threshold=args.correctness_threshold)
    df_rm = _load_rm_per_trace(args.outputs_dir, logger=logger, max_files=args.max_rm_files)
    missing = [c for c in args.rm_cols if c not in df_rm.columns]
    if missing:
        raise RuntimeError(f"Missing RM columns in RM table: {missing}. Available: {sorted([c for c in df_rm.columns if c.startswith('rm::')])}")
    df_rm["rm_mean_2"] = df_rm[list(args.rm_cols)].mean(axis=1, skipna=True)
    df = df_pu.merge(df_rm[["dataset", "model", "trace_index", "rm_mean_2"]], on=["dataset", "model", "trace_index"], how="inner")
    df = df.merge(df_correct, on=["dataset", "model", "trace_index"], how="left")
    df["trace_correct"] = df["trace_correct"].fillna(False)
    out_csv = osp.join(run_dir, "trace_level_pu_rm_merged.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote merged trace-level table: {out_csv} (rows={len(df)})")
    stats_out: Dict[str, Any] = {"run": {"generated_on": datetime.now().isoformat(), "rm_cols": args.rm_cols}}
    def compute_block(df0: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for metric in ["pu_avg_correctness", "pu_linearity_mse", "pu_regression_rate"]:
            s = _pearson_stats(df0["rm_mean_2"].to_numpy(dtype=float), df0[metric].to_numpy(dtype=float))
            lo, hi = _bootstrap_rho_ci(df0["rm_mean_2"].to_numpy(dtype=float), df0[metric].to_numpy(dtype=float), n_boot=args.n_boot, seed=args.seed)
            out[metric] = {**s, "rho_ci95": [lo, hi]}
        return out
    stats_out["overall_unconditional"] = compute_block(df)
    stats_out["overall_correct_only"] = compute_block(df[df["trace_correct"]].copy())
    by_ds_rows: List[Dict[str, Any]] = []
    for ds, g in df.groupby("dataset"):
        for cond_name, sub in [("unconditional", g), ("correct_only", g[g["trace_correct"]])]:
            for metric in ["pu_avg_correctness", "pu_linearity_mse", "pu_regression_rate"]:
                s = _pearson_stats(sub["rm_mean_2"].to_numpy(dtype=float), sub[metric].to_numpy(dtype=float))
                lo, hi = _bootstrap_rho_ci(sub["rm_mean_2"].to_numpy(dtype=float), sub[metric].to_numpy(dtype=float), n_boot=args.n_boot, seed=args.seed)
                by_ds_rows.append({"dataset": ds, "condition": cond_name, "metric": metric, **s, "rho_ci95_lo": lo, "rho_ci95_hi": hi})
    df_by_ds = pd.DataFrame(by_ds_rows)
    df_by_ds.to_csv(osp.join(run_dir, "trace_level_correlations_by_dataset.csv"), index=False)
    _plot_rho_forest(
        df_by_ds,
        out_path=osp.join(run_dir, "forest_rho_by_dataset_condition_metric.png"),
    )
    with open(osp.join(run_dir, "trace_level_correlations_overall.json"), "w") as f:
        json.dump(stats_out, f, indent=2, sort_keys=True)
    df_corr = df[df["trace_correct"]].copy()
    _save_scatter(
        df_corr.dropna(subset=["rm_mean_2", "pu_avg_correctness"]),
        x="rm_mean_2",
        y="pu_avg_correctness",
        title="Trace-level: RM vs TU avg correctness (correct-only)",
        out_path=osp.join(run_dir, "scatter_trace_level_rm_vs_tu_avg_correctness_correct_only.png"),
    )
    _save_scatter(
        df_corr.dropna(subset=["rm_mean_2", "pu_linearity_mse"]),
        x="rm_mean_2",
        y="pu_linearity_mse",
        title="Trace-level: RM vs TU linearity MSE (correct-only)",
        out_path=osp.join(run_dir, "scatter_trace_level_rm_vs_tu_linearity_mse_correct_only.png"),
    )
    logger.info("Done")
if __name__ == "__main__":
    main()
