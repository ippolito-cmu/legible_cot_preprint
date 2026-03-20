"""
Database backend for efficient trace storage and retrieval

Uses SQLite for simplicity and portability. Can be upgraded to PostgreSQL for production.

Benefits:
- Indexed queries: Find traces by dataset/model/index in milliseconds
- Pagination: Built-in LIMIT/OFFSET support
- Filtering: SQL WHERE clauses for efficient filtering
- No full file loading: Only retrieve what you need
- Compressed storage: BLOB compression for text fields

Migration script: python db_backend.py --migrate
Query example: python db_backend.py --query --dataset math --model Qwen_Qwen3-8B --limit 10
"""
import sqlite3
import pickle
import json
import zlib
import logging
import random
import glob
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import argparse
import importlib.util
from tqdm import tqdm
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available. Token counting will use a fallback.")
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Sentence counting will use regex fallback.")
SENTENCE_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
TRACES_DIR = OUTPUTS_DIR / 'traces'
PU_DIR = OUTPUTS_DIR / 'pu'
REDUNDANCY_DIR = OUTPUTS_DIR / 'redundancy_analysis'
DB_PATH = OUTPUTS_DIR / 'traces.db'
TOKEN_ENCODER = None
def get_token_encoder():
    """Lazy initialization of tiktoken encoder"""
    global TOKEN_ENCODER
    if TOKEN_ENCODER is None:
        if not TIKTOKEN_AVAILABLE:
            return None
        TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    return TOKEN_ENCODER
def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    if not isinstance(text, str) or not text.strip():
        return 0
    try:
        encoder = get_token_encoder()
        if encoder is None:
            return len(text.split())
        return len(encoder.encode(text, disallowed_special=()))
    except Exception:
        return len(text.split())
def count_sentences(text: str) -> int:
    """Count sentences using NLTK"""
    if not isinstance(text, str) or not text.strip():
        return 0
    try:
        return len(sent_tokenize(text))
    except Exception:
        return 0
def compute_semantic_redundancy(trace_text: str, model, device, similarity_threshold: float = 0.85):
    """Compute semantic redundancy for a reasoning trace
    
    Returns:
        - mean_score: fraction of steps with similarity > threshold
        - per_step_scores: similarity to most similar previous step
        - max_sim_indices: index of most similar previous step
    """
    try:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers not available")
        from sentence_transformers import util
        import torch
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
    except Exception as e:
        logger.warning(f"Error computing redundancy: {e}")
        return 0.0, [0.0], [-1]
class TraceDB:
    """Database interface for trace storage and retrieval"""
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self._connect()
    def _connect(self):
        """Connect to database and enable optimizations"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-64000")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA busy_timeout=30000")
    def initialize_schema(self):
        """Create database schema"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                model TEXT NOT NULL,
                trace_index INTEGER NOT NULL,
                question TEXT,
                trace_text BLOB,  -- Compressed with zlib
                extracted_answer TEXT,
                score REAL,
                raw_score REAL,
                ground_truth TEXT,
                metadata TEXT,  -- JSON
                UNIQUE(dataset, model, trace_index)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS efficiency_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                model TEXT NOT NULL,
                trace_index INTEGER NOT NULL,
                token_count INTEGER,
                sentence_count INTEGER,
                UNIQUE(dataset, model, trace_index),
                FOREIGN KEY(dataset, model, trace_index) 
                    REFERENCES traces(dataset, model, trace_index)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS redundancy_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                model TEXT NOT NULL,
                trace_index INTEGER NOT NULL,
                redundancy_score REAL,
                redundancy_per_step BLOB,  -- Compressed JSON array
                redundancy_match_indices BLOB,  -- Compressed JSON array
                UNIQUE(dataset, model, trace_index),
                FOREIGN KEY(dataset, model, trace_index) 
                    REFERENCES traces(dataset, model, trace_index)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                model TEXT NOT NULL,
                total_traces INTEGER,
                accuracy REAL,
                avg_tokens REAL,
                avg_redundancy REAL,
                metadata_json TEXT,  -- Full metadata as JSON
                UNIQUE(dataset, model)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pu_percentage_bins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                teacher_model TEXT NOT NULL,
                teacher_short_name TEXT NOT NULL,
                student_model TEXT NOT NULL,
                bin_center INTEGER NOT NULL,  -- Right-edge bin in {2,4,...,100}
                mean_correct REAL NOT NULL,
                std_correct REAL NOT NULL,
                count INTEGER NOT NULL,
                hazard_prob REAL NOT NULL DEFAULT 0.0,
                hazard_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(dataset, teacher_model, student_model, bin_center)
            )
        """)
        try:
            cursor.execute("PRAGMA table_info(pu_percentage_bins)")
            cols = {str(r[1]) for r in cursor.fetchall()}
            if 'hazard_prob' not in cols:
                cursor.execute("ALTER TABLE pu_percentage_bins ADD COLUMN hazard_prob REAL NOT NULL DEFAULT 0.0")
            if 'hazard_count' not in cols:
                cursor.execute("ALTER TABLE pu_percentage_bins ADD COLUMN hazard_count INTEGER NOT NULL DEFAULT 0")
        except Exception as e:
            logger.warning(f"Failed ensuring hazard columns on pu_percentage_bins: {e}")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pu_trace_metrics (
                dataset TEXT NOT NULL,
                teacher_model TEXT NOT NULL,
                student_model TEXT NOT NULL,
                trace_index INTEGER NOT NULL,
                trace_fotu REAL,
                trace_regression_rate REAL,
                max_bin INTEGER,
                n_samples INTEGER,
                teacher_score REAL,
                PRIMARY KEY(dataset, teacher_model, student_model, trace_index)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pu_regression_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                teacher_model TEXT NOT NULL,
                teacher_short_name TEXT NOT NULL,
                student_model TEXT NOT NULL,
                total_questions INTEGER NOT NULL,
                regressed_questions INTEGER NOT NULL,
                regression_percentage REAL NOT NULL,
                total_regressions INTEGER NOT NULL,
                avg_regressions_per_question REAL NOT NULL,
                UNIQUE(dataset, teacher_model, student_model)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtracking_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                model TEXT NOT NULL,
                trace_index INTEGER NOT NULL,
                backtracking_detected BOOLEAN,
                num_backtracking_steps INTEGER,
                confidence REAL,
                final_answer TEXT,
                overall_reasoning TEXT,
                backtracking_steps_json TEXT,
                error TEXT,
                prompt_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                finish_reason TEXT,
                UNIQUE(dataset, model, trace_index)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wishlist_metric_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                model TEXT NOT NULL,
                n_traces INTEGER,
                accuracy REAL,
                eff_token_length REAL,
                red_frac_0_8 REAL,
                bt_steps_all REAL,
                pu_avg_correctness REAL,
                pu_linearity_mse REAL,
                pu_regression_rate REAL,
                UNIQUE(dataset, model)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_lookup ON traces(dataset, model, trace_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_score ON traces(dataset, model, score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_efficiency_lookup ON efficiency_metrics(dataset, model, trace_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_redundancy_lookup ON redundancy_metrics(dataset, model, trace_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_meta_lookup ON model_metadata(dataset, model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pu_bins_lookup ON pu_percentage_bins(dataset, teacher_model, student_model, bin_center)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pu_trace_metrics_lookup ON pu_trace_metrics(dataset, teacher_model, student_model, trace_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pu_trace_metrics_ds_teacher ON pu_trace_metrics(dataset, teacher_model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pu_regression_lookup ON pu_regression_cache(dataset, teacher_model, student_model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtracking_lookup ON backtracking_results(dataset, model, trace_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_wishlist_cache_lookup ON wishlist_metric_cache(dataset, model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtracking_model_ds ON backtracking_results(dataset, model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtracking_detected ON backtracking_results(dataset, model, backtracking_detected)")
        self.conn.commit()
        logger.info("Database schema initialized")
    def get_cached_pu_regressions(self, dataset: str, student_model: str = None) -> List[Dict[str, Any]]:
        """Fetch cached PU regression stats for a dataset.

        Args:
            dataset: Dataset name.
            student_model: Full or short student name. If None, returns all students.
        """
        cursor = self.conn.cursor()
        if student_model:
            student_short = self.get_student_short_name(student_model)
            cursor.execute(
                """
                SELECT dataset, teacher_model, teacher_short_name, student_model,
                       total_questions, regressed_questions, regression_percentage,
                       total_regressions, avg_regressions_per_question
                FROM pu_regression_cache
                WHERE dataset = ? AND student_model = ?
                ORDER BY regression_percentage ASC, teacher_model ASC
                """,
                (dataset, student_short),
            )
        else:
            cursor.execute(
                """
                SELECT dataset, teacher_model, teacher_short_name, student_model,
                       total_questions, regressed_questions, regression_percentage,
                       total_regressions, avg_regressions_per_question
                FROM pu_regression_cache
                WHERE dataset = ?
                ORDER BY student_model ASC, regression_percentage ASC, teacher_model ASC
                """,
                (dataset,),
            )
        return [dict(r) for r in cursor.fetchall()]
    def insert_backtracking_result(
        self,
        *,
        dataset: str,
        model: str,
        trace_index: int,
        backtracking_detected: Optional[bool],
        num_backtracking_steps: Optional[int],
        confidence: Optional[float],
        final_answer: Optional[str],
        overall_reasoning: Optional[str],
        backtracking_steps_json: Optional[str],
        error: Optional[str],
        prompt_tokens: Optional[int],
        output_tokens: Optional[int],
        total_tokens: Optional[int],
        finish_reason: Optional[str],
    ) -> None:
        def _sanitize_sqlite_text(val: Optional[str]) -> Optional[str]:
            if val is None:
                return None
            if not isinstance(val, str):
                val = str(val)
            return val.encode('utf-8', errors='replace').decode('utf-8', errors='strict')
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO backtracking_results
            (dataset, model, trace_index, backtracking_detected, num_backtracking_steps,
             confidence, final_answer, overall_reasoning, backtracking_steps_json, error,
             prompt_tokens, output_tokens, total_tokens, finish_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset,
                model,
                int(trace_index),
                int(backtracking_detected) if backtracking_detected is not None else None,
                int(num_backtracking_steps) if num_backtracking_steps is not None else None,
                float(confidence) if confidence is not None else None,
                _sanitize_sqlite_text(final_answer),
                _sanitize_sqlite_text(overall_reasoning),
                _sanitize_sqlite_text(backtracking_steps_json),
                _sanitize_sqlite_text(error),
                int(prompt_tokens) if isinstance(prompt_tokens, (int,)) else None,
                int(output_tokens) if isinstance(output_tokens, (int,)) else None,
                int(total_tokens) if isinstance(total_tokens, (int,)) else None,
                _sanitize_sqlite_text(finish_reason),
            ),
        )
    def get_backtracking_datasets(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT dataset FROM backtracking_results ORDER BY dataset")
        return [r['dataset'] for r in cursor.fetchall()]
    def get_backtracking_models(self, dataset: str) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT model
            FROM backtracking_results
            WHERE dataset = ?
            ORDER BY model
            """,
            (dataset,),
        )
        return [r['model'] for r in cursor.fetchall()]
    def get_leaderboard_rows(self) -> List[Dict[str, Any]]:
        """Return aggregated leaderboard metrics per (dataset, model).

        Uses the precomputed `model_metadata` table for core metrics and joins
        with an aggregate over `backtracking_results` for backtracking metrics.
        """
        pu_stats = self.get_pu_leaderboard_stats(mode='percentage')
        pu_regs = self.get_pu_leaderboard_regressions()
        cursor = self.conn.cursor()
        cursor.execute(
            """
            WITH bt AS (
                SELECT
                    b.dataset AS dataset,
                    b.model AS model,
                    COUNT(*) AS bt_total,
                    SUM(CASE WHEN b.backtracking_detected = 1 THEN 1 ELSE 0 END) AS bt_detected,
                    AVG(CASE WHEN b.backtracking_detected = 1 THEN COALESCE(b.num_backtracking_steps, 0) ELSE NULL END) AS bt_avg_steps_detected,
                    AVG(COALESCE(b.confidence, NULL)) AS bt_avg_confidence,
                    SUM(CASE WHEN t.score > 0.5 THEN 1 ELSE 0 END) AS correct_traces,
                    SUM(CASE WHEN t.score <= 0.5 THEN 1 ELSE 0 END) AS incorrect_traces,
                    SUM(CASE WHEN t.score > 0.5 AND b.backtracking_detected = 1 THEN 1 ELSE 0 END) AS detected_correct,
                    SUM(CASE WHEN t.score <= 0.5 AND b.backtracking_detected = 1 THEN 1 ELSE 0 END) AS detected_incorrect
                FROM backtracking_results b
                LEFT JOIN traces t
                  ON t.dataset = b.dataset AND t.model = b.model AND t.trace_index = b.trace_index
                GROUP BY b.dataset, b.model
            )
            SELECT
                m.dataset,
                m.model,
                m.total_traces,
                m.accuracy,
                m.avg_tokens,
                m.avg_redundancy,
                bt.bt_total,
                bt.bt_detected,
                CASE WHEN bt.bt_total > 0 THEN bt.bt_detected * 1.0 / bt.bt_total ELSE NULL END AS backtracking_detected_rate,
                bt.bt_avg_steps_detected,
                bt.bt_avg_confidence,
                CASE WHEN bt.correct_traces > 0 THEN bt.detected_correct * 1.0 / bt.correct_traces ELSE NULL END AS backtracking_detected_rate_correct,
                CASE WHEN bt.incorrect_traces > 0 THEN bt.detected_incorrect * 1.0 / bt.incorrect_traces ELSE NULL END AS backtracking_detected_rate_incorrect
            FROM model_metadata m
            LEFT JOIN bt
              ON bt.dataset = m.dataset AND bt.model = m.model
            ORDER BY m.dataset ASC, m.accuracy DESC
            """
        )
        base_rows: List[Dict[str, Any]] = []
        for r in cursor.fetchall():
            d = dict(r)
            key = (d.get('dataset'), d.get('model'))
            if key in pu_stats:
                d.update(pu_stats[key])
            if key in pu_regs:
                d.update(pu_regs[key])
            base_rows.append(d)
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='wishlist_metric_cache'")
            if cursor.fetchone():
                cursor.execute(
                    """
                    SELECT dataset, model, n_traces, accuracy, eff_token_length, red_frac_0_8,
                           bt_steps_all, pu_avg_correctness, pu_linearity_mse, pu_regression_rate
                    FROM wishlist_metric_cache
                    """
                )
                cache = {}
                for row in cursor.fetchall():
                    cache[(row['dataset'], row['model'])] = dict(row)
                for d in base_rows:
                    key = (d.get('dataset'), d.get('model'))
                    c = cache.get(key)
                    if c:
                        d['wl_n_traces'] = c.get('n_traces')
                        d['wl_accuracy'] = c.get('accuracy')
                        d['wl_eff_token_length'] = c.get('eff_token_length')
                        d['wl_red_frac_0_8'] = c.get('red_frac_0_8')
                        d['wl_bt_steps_all'] = c.get('bt_steps_all')
                        d['wl_pu_avg_correctness'] = c.get('pu_avg_correctness')
                        d['wl_pu_linearity_mse'] = c.get('pu_linearity_mse')
                        d['wl_pu_regression_rate'] = c.get('pu_regression_rate')
        except Exception:
            pass
        return base_rows
    def get_pu_leaderboard_stats(
        self,
        *,
        datasets: Optional[List[str]] = None,
        mode: str = 'percentage',
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Aggregate PU curve metrics per (dataset, teacher_model) for leaderboard.

        This pools across *all students* by weighted-averaging cached bin means
        (weight = bin count), then computes the same simple line metrics used by
        the Flask PU endpoints (slope via endpoints, r2 against that line, and AUC).
        """
        table = 'pu_percentage_bins'
        cursor = self.conn.cursor()
        where = ''
        params: List[Any] = []
        if datasets:
            placeholders = ','.join(['?'] * len(datasets))
            where = f"WHERE dataset IN ({placeholders})"
            params.extend(datasets)
        cursor.execute(
            f"""
            SELECT
                dataset,
                teacher_model,
                teacher_short_name,
                student_model,
                bin_center,
                mean_correct,
                count,
                hazard_prob
            FROM {table}
            {where}
            ORDER BY dataset, teacher_model, bin_center, student_model
            """,
            tuple(params),
        )
        rows = cursor.fetchall()
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        hazard_by_student: Dict[Tuple[str, str, str], Dict[float, float]] = {}
        for row in rows:
            dataset = row['dataset']
            teacher = row['teacher_model']
            key = (dataset, teacher)
            grouped.setdefault(
                key,
                {
                    'pu_teacher_short_name': row['teacher_short_name'],
                    'by_bin': {},
                    'pu_total_count': 0,
                },
            )
            x = float(row['bin_center'])
            y = float(row['mean_correct'])
            grouped[key]['by_bin'].setdefault(x, []).append(y)
            grouped[key]['pu_total_count'] += int(row['count'] or 0)
            student = str(row['student_model'] or '')
            hs_key = (dataset, teacher, student)
            hazard_by_student.setdefault(hs_key, {})[x] = float(row['hazard_prob'] or 0.0)
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for key, data in grouped.items():
            by_bin = data.get('by_bin') or {}
            points = []
            for x, ys in by_bin.items():
                if not ys:
                    continue
                points.append({'x': float(x), 'y': float(sum(ys) / len(ys))})
            points.sort(key=lambda p: p['x'])
            total_count = int(data.get('pu_total_count') or 0)
            mean_accuracy = float(np.mean([p['y'] for p in points])) if points else None
            ds, teacher = key
            student_keys = [k for k in hazard_by_student.keys() if k[0] == ds and k[1] == teacher]
            hazard_entropies: List[float] = []
            for (_ds, _t, _s) in student_keys:
                hmap = hazard_by_student.get((_ds, _t, _s)) or {}
                if not points:
                    continue
                masses = np.array([float(hmap.get(p['x'], 0.0)) for p in points], dtype=float)
                total = float(np.sum(masses))
                if total <= 0:
                    continue
                q = masses / total
                eps = 1e-12
                h = -float(np.sum(q * np.log(q + eps)))
                denom_h = float(np.log(len(q))) if len(q) > 1 else 1.0
                hazard_entropies.append(h / denom_h if denom_h > 0 else 0.0)
            sotu = float(np.mean(hazard_entropies)) if hazard_entropies else None
            if len(points) >= 2:
                x_vals = np.array([p['x'] for p in points], dtype=float)
                y_vals = np.array([p['y'] for p in points], dtype=float)
                x_mean = float(np.mean(x_vals))
                y_mean = float(np.mean(y_vals))
                denom = float(np.sum((x_vals - x_mean) ** 2))
                if denom > 0:
                    slope = float(np.sum((x_vals - x_mean) * (y_vals - y_mean)) / denom)
                    intercept = y_mean - slope * x_mean
                    y_pred = slope * x_vals + intercept
                    mse = float(np.mean((y_vals - y_pred) ** 2))
                    ss_tot = float(np.sum((y_vals - y_mean) ** 2))
                    ss_res = float(np.sum((y_vals - y_pred) ** 2))
                    r2_ols = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
                else:
                    slope = 0.0
                    mse = 0.0
                    r2_ols = 0.0
                auc = float(np.trapz(y_vals, x_vals) / (x_vals[-1] - x_vals[0])) if x_vals[-1] != x_vals[0] else 0.0
            else:
                slope = None
                mse = None
                r2_ols = None
                auc = None
            out[key] = {
                'pu_total_count': total_count or None,
                'pu_mean_accuracy': float(mean_accuracy) if mean_accuracy is not None else None,
                'pu_r2': sotu,
                'pu_r2_ols': r2_ols,
                'pu_auc': auc,
                'pu_slope': slope,
                'pu_mse': mse,
            }
        return out
    def get_pu_leaderboard_regressions(
        self,
        *,
        datasets: Optional[List[str]] = None,
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Aggregate PU regression stats per (dataset, teacher_model) across students."""
        cursor = self.conn.cursor()
        where = ''
        params: List[Any] = []
        if datasets:
            placeholders = ','.join(['?'] * len(datasets))
            where = f"WHERE dataset IN ({placeholders})"
            params.extend(datasets)
        cursor.execute(
            f"""
            SELECT
                dataset,
                teacher_model,
                teacher_short_name,
                student_model,
                total_questions,
                regressed_questions,
                regression_percentage,
                total_regressions,
                avg_regressions_per_question
            FROM pu_regression_cache
            {where}
            ORDER BY dataset, teacher_model, student_model
            """,
            tuple(params),
        )
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in cursor.fetchall():
            key = (row['dataset'], row['teacher_model'])
            grouped.setdefault(
                key,
                {
                    'total_questions': 0,
                    'regressed_questions': 0,
                    'total_regressions': 0,
                    'pct_list': [],
                    'avg_list': [],
                },
            )
            grouped[key]['total_questions'] += int(row['total_questions'] or 0)
            grouped[key]['regressed_questions'] += int(row['regressed_questions'] or 0)
            grouped[key]['total_regressions'] += int(row['total_regressions'] or 0)
            grouped[key]['pct_list'].append(float(row['regression_percentage'] or 0.0))
            grouped[key]['avg_list'].append(float(row['avg_regressions_per_question'] or 0.0))
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for key, d in grouped.items():
            pct = float(np.mean(d['pct_list'])) if d['pct_list'] else None
            avg_per_q = float(np.mean(d['avg_list'])) if d['avg_list'] else None
            out[key] = {
                'pu_total_questions': d['total_questions'] or None,
                'pu_regressed_questions': d['regressed_questions'] or None,
                'pu_regression_percentage': pct,
                'pu_total_regressions': d['total_regressions'] or None,
                'pu_avg_regressions_per_question': avg_per_q,
            }
        return out
    def get_backtracking_model_summary(self, dataset: str) -> List[Dict[str, Any]]:
        """Aggregate backtracking stats per model for a dataset."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT
                b.model as model,
                COUNT(*) as total_traces,
                SUM(CASE WHEN b.backtracking_detected = 1 THEN 1 ELSE 0 END) as detected_traces,
                SUM(CASE WHEN t.score > 0.5 THEN 1 ELSE 0 END) as correct_traces,
                SUM(CASE WHEN t.score <= 0.5 THEN 1 ELSE 0 END) as incorrect_traces,
                SUM(CASE WHEN t.score > 0.5 AND b.backtracking_detected = 1 THEN 1 ELSE 0 END) as detected_correct,
                SUM(CASE WHEN t.score <= 0.5 AND b.backtracking_detected = 1 THEN 1 ELSE 0 END) as detected_incorrect,
                AVG(COALESCE(b.num_backtracking_steps, 0)) as avg_backtracking_steps_all,
                AVG(CASE WHEN b.backtracking_detected = 1 THEN COALESCE(b.num_backtracking_steps, 0) ELSE NULL END) as avg_backtracking_steps_detected,
                AVG(COALESCE(b.confidence, NULL)) as avg_confidence,
                AVG(CASE WHEN t.score > 0.5 THEN 1.0 ELSE 0.0 END) as trace_accuracy
            FROM backtracking_results b
            LEFT JOIN traces t
              ON t.dataset = b.dataset AND t.model = b.model AND t.trace_index = b.trace_index
            WHERE b.dataset = ?
            GROUP BY b.model
            ORDER BY detected_traces * 1.0 / MAX(total_traces, 1) DESC
            """,
            (dataset,),
        )
        out = []
        for r in cursor.fetchall():
            d = dict(r)
            total = d.get('total_traces') or 0
            detected = d.get('detected_traces') or 0
            d['detected_rate'] = float(detected / total) if total else 0.0
            ctot = d.get('correct_traces') or 0
            itot = d.get('incorrect_traces') or 0
            dc = d.get('detected_correct') or 0
            di = d.get('detected_incorrect') or 0
            d['detected_rate_correct'] = float(dc / ctot) if ctot else 0.0
            d['detected_rate_incorrect'] = float(di / itot) if itot else 0.0
            out.append(d)
        return out
    def get_backtracking_trace_detail(self, dataset: str, model: str, trace_index: int) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT b.*, t.question, t.ground_truth, t.extracted_answer as teacher_extracted_answer,
                   t.score as teacher_score, t.trace_text as teacher_trace
            FROM backtracking_results b
            LEFT JOIN traces t
              ON t.dataset = b.dataset AND t.model = b.model AND t.trace_index = b.trace_index
            WHERE b.dataset = ? AND b.model = ? AND b.trace_index = ?
            """,
            (dataset, model, int(trace_index)),
        )
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d['teacher_trace'] = self.decompress_text(d.get('teacher_trace')) if d.get('teacher_trace') else ''
        except Exception:
            d['teacher_trace'] = ''
        return d
    def get_backtracking_traces(
        self,
        *,
        dataset: str,
        model: str,
        limit: int = 200,
        offset: int = 0,
        detected_only: Optional[bool] = None,
        correctness: str = 'all',
        order: str = 'trace_index_asc',
    ) -> Dict[str, Any]:
        """List backtracking traces for drilldown.

        Args:
            dataset: Dataset name.
            model: Model name.
            limit: Max rows to return.
            offset: Pagination offset.
            detected_only: If True, only traces where backtracking_detected=1.
                If False, only traces where backtracking_detected=0.
                If None, no filter.
            correctness: One of {'all','correct','incorrect'} based on teacher score.
            order: One of {'trace_index_asc','steps_desc','confidence_desc'}.

        Returns:
            Dict containing rows + total.
        """
        limit = int(max(1, min(int(limit), 2000)))
        offset = int(max(0, int(offset)))
        correctness = (correctness or 'all').lower().strip()
        order = (order or 'trace_index_asc').lower().strip()
        where = ["b.dataset = ?", "b.model = ?"]
        params: list[Any] = [dataset, model]
        if detected_only is True:
            where.append("b.backtracking_detected = 1")
        elif detected_only is False:
            where.append("(b.backtracking_detected = 0 OR b.backtracking_detected IS NULL)")
        if correctness == 'correct':
            where.append("t.score > 0.5")
        elif correctness == 'incorrect':
            where.append("t.score <= 0.5")
        elif correctness != 'all':
            raise ValueError(f"Unknown correctness filter: {correctness}")
        if order == 'trace_index_asc':
            order_by = "b.trace_index ASC"
        elif order == 'steps_desc':
            order_by = "COALESCE(b.num_backtracking_steps, -1) DESC, b.trace_index ASC"
        elif order == 'confidence_desc':
            order_by = "COALESCE(b.confidence, -1) DESC, b.trace_index ASC"
        else:
            raise ValueError(f"Unknown order: {order}")
        where_sql = " AND ".join(where)
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM backtracking_results b
            LEFT JOIN traces t
              ON t.dataset = b.dataset AND t.model = b.model AND t.trace_index = b.trace_index
            WHERE {where_sql}
            """,
            tuple(params),
        )
        total = int(cursor.fetchone()['n'] or 0)
        cursor.execute(
            f"""
            SELECT
                b.trace_index,
                b.backtracking_detected,
                b.num_backtracking_steps,
                b.confidence,
                b.error,
                b.finish_reason,
                b.total_tokens,
                t.score AS teacher_score
            FROM backtracking_results b
            LEFT JOIN traces t
              ON t.dataset = b.dataset AND t.model = b.model AND t.trace_index = b.trace_index
            WHERE {where_sql}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
            """,
            tuple(params + [limit, offset]),
        )
        rows = [dict(r) for r in cursor.fetchall()]
        for r in rows:
            if r.get('backtracking_detected') is None:
                r['backtracking_detected'] = None
            else:
                r['backtracking_detected'] = bool(int(r['backtracking_detected']))
        return {
            'dataset': dataset,
            'model': model,
            'limit': limit,
            'offset': offset,
            'total': total,
            'rows': rows,
        }
    def get_student_short_name(self, student_model: str) -> str:
        """Convert full student model name to short name used in database"""
        if not student_model:
            return None
        mapping = {
            'meta-llama/Llama-3.2-1B-Instruct': 'meta_llama_llama_3_2_1b',
            'microsoft/Phi-3-mini-128k-instruct': 'microsoft_phi_3_mini_128k'
        }
        if student_model in mapping:
            return mapping[student_model]
        if '/' in student_model:
            return student_model.split('/')[-1].replace('-Instruct', '').replace('-instruct', '').replace('.', '_').replace('-', '_').lower()
        return student_model.replace('-Instruct', '').replace('-instruct', '').replace('.', '_').replace('-', '_').lower()
    def get_pu_table_name(self, dataset: str, student_model: str) -> str:
        """Get table name for dataset/student PU data"""
        student_short = self.get_student_short_name(student_model)
        if not student_short:
            return None
        return f"pu_{dataset}_{student_short}"
    def create_pu_table(self, dataset: str, student_model: str):
        """Create PU table for specific dataset/student combination"""
        table_name = self.get_pu_table_name(dataset, student_model)
        cursor = self.conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                teacher_model TEXT NOT NULL,
                trace_index INTEGER NOT NULL,
                num_steps INTEGER NOT NULL,
                total_steps INTEGER NOT NULL,
                percentage_complete REAL,
                score REAL,
                extracted_answer TEXT,
                grading_error TEXT,
                ran_second_pass BOOLEAN,
                full_output BLOB,
                completion_starts_at INTEGER,
                teacher_score REAL,
                UNIQUE(teacher_model, trace_index, num_steps)
            )
        """)
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_teacher ON {table_name}(teacher_model, trace_index)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_steps ON {table_name}(teacher_model, num_steps, teacher_score)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_pct ON {table_name}(teacher_model, percentage_complete, teacher_score)")
        self.conn.commit()
        logger.info(f"Created PU table: {table_name}")
    def compress_text(self, text: str) -> bytes:
        """Compress text with zlib"""
        if isinstance(text, str):
            text = text.encode('utf-8')
        return zlib.compress(text, level=6)
    def decompress_text(self, compressed: bytes) -> str:
        """Decompress text"""
        return zlib.decompress(compressed).decode('utf-8')
    def insert_trace(self, dataset: str, model: str, trace_index: int, 
                     question: str, trace_text: str, extracted_answer: str,
                     score: float, raw_score: float, ground_truth: str,
                     metadata: Dict = None):
        """Insert a single trace"""
        from src.utils.tu_unification import normalize_connections_score_value
        ds = (dataset or '').lower()
        if ds == 'connections':
            base_val = raw_score if raw_score is not None else score
            score = normalize_connections_score_value(base_val)
            raw_score = base_val
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO traces 
            (dataset, model, trace_index, question, trace_text, extracted_answer, 
             score, raw_score, ground_truth, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset, model, trace_index, question,
            self.compress_text(trace_text), extracted_answer,
            score, raw_score, ground_truth,
            json.dumps(metadata) if metadata else None
        ))
    def insert_efficiency(self, dataset: str, model: str, trace_index: int,
                         token_count: int, sentence_count: int):
        """Insert efficiency metrics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO efficiency_metrics
            (dataset, model, trace_index, token_count, sentence_count)
            VALUES (?, ?, ?, ?, ?)
        """, (dataset, model, trace_index, token_count, sentence_count))
    def insert_redundancy(self, dataset: str, model: str, trace_index: int,
                         redundancy_score: float, redundancy_per_step: List,
                         redundancy_match_indices: List):
        """Insert redundancy metrics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO redundancy_metrics
            (dataset, model, trace_index, redundancy_score, 
             redundancy_per_step, redundancy_match_indices)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            dataset, model, trace_index, redundancy_score,
            self.compress_text(json.dumps(redundancy_per_step)),
            self.compress_text(json.dumps(redundancy_match_indices))
        ))
    def insert_model_metadata(self, dataset: str, model: str, metadata: Dict):
        """Insert model-level metadata and statistics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO model_metadata
            (dataset, model, total_traces, accuracy, avg_tokens, avg_redundancy, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset, model,
            metadata.get('total_traces', 0),
            metadata.get('accuracy', 0),
            metadata.get('avg_tokens'),
            metadata.get('avg_redundancy'),
            json.dumps(metadata)
        ))
    def insert_pedagogical_utility(self, dataset: str, teacher_model: str, student_model: str,
                                   trace_index: int, num_steps: int, total_steps: int,
                                   score: float, extracted_answer: str, grading_error: str,
                                   ran_second_pass: bool, full_output: str, completion_starts_at: int,
                                   teacher_score: float = None):
        """Insert pedagogical utility data point into dataset/student-specific table"""
        from src.utils.tu_unification import normalize_connections_score_value
        cursor = self.conn.cursor()
        table_name = self.get_pu_table_name(dataset, student_model)
        percentage_complete = (num_steps / total_steps * 100) if total_steps > 0 else 0
        ds = (dataset or '').lower()
        if ds == 'connections':
            score = normalize_connections_score_value(score)
        grading_error_str = str(grading_error) if grading_error is not None else ""
        ran_second_pass_int = 1 if ran_second_pass else 0
        extracted_answer_str = str(extracted_answer) if extracted_answer is not None else ""
        full_output_str = str(full_output) if full_output is not None else ""
        cursor.execute(f"""
            INSERT OR REPLACE INTO {table_name}
            (teacher_model, trace_index, num_steps, total_steps, percentage_complete,
             score, extracted_answer, grading_error, ran_second_pass, full_output, completion_starts_at, teacher_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            teacher_model, trace_index, num_steps, total_steps, percentage_complete,
            score, extracted_answer_str, grading_error_str, ran_second_pass_int,
            self.compress_text(full_output_str), completion_starts_at, teacher_score
        ))
    def compute_and_cache_percentage_bins(self, dataset: str, teacher_model: str, teacher_short_name: str, student_model: str = None):
        """
        DEPRECATED.

        Percentage-bin caches are built during migration via the unified TU logic and stored
        in `pu_percentage_bins`. Computing these on-demand is both slow and historically
        error-prone (including double-normalization for Connections).
        
        Args:
            dataset: Dataset name
            teacher_model: Teacher model name
            teacher_short_name: Short name for teacher model
            student_model: Student model name (if None, caches for all students aggregated)
        """
        raise RuntimeError(
            'compute_and_cache_percentage_bins() is deprecated. '
            'Rebuild caches via migrate_database.py (unified TU) and query with get_cached_percentage_bins().'
        )
    def get_cached_percentage_bins(self, dataset: str, student_model: str = None) -> Dict:
        """
        Retrieve pre-computed percentage bins from cache.
        This is MUCH faster than computing on-demand.
        
        Args:
            dataset: Dataset name
            student_model: Filter by student model (if None, returns all students)
        """
        cursor = self.conn.cursor()
        if student_model:
            student_model = self.get_student_short_name(student_model)
            cursor.execute("""
                SELECT teacher_model, teacher_short_name, student_model, bin_center, mean_correct, std_correct, count,
                       hazard_prob, hazard_count
                FROM pu_percentage_bins
                WHERE dataset = ? AND student_model = ?
                ORDER BY teacher_model, bin_center
            """, (dataset, student_model))
        else:
            cursor.execute("""
                SELECT teacher_model, teacher_short_name, student_model, bin_center, mean_correct, std_correct, count,
                       hazard_prob, hazard_count
                FROM pu_percentage_bins
                WHERE dataset = ?
                ORDER BY teacher_model, student_model, bin_center
            """, (dataset,))
        rows = cursor.fetchall()
        teacher_data = {}
        for row in rows:
            teacher = row['teacher_model']
            student = row['student_model']
            if student_model:
                key = teacher
            else:
                key = f"{teacher}_{student}"
            if key not in teacher_data:
                teacher_data[key] = {
                    'teacher_model': teacher,
                    'teacher_short_name': row['teacher_short_name'],
                    'student_model': student,
                    'bins': []
                }
            teacher_data[key]['bins'].append({
                'bin_center': row['bin_center'],
                'mean_correct': row['mean_correct'],
                'std_correct': row['std_correct'],
                'count': row['count'],
                'hazard_prob': row['hazard_prob'],
                'hazard_count': row['hazard_count'],
            })
        return teacher_data
    def get_cached_absolute_bins(self, dataset: str, student_model: str = None) -> Dict:
        raise RuntimeError('Absolute TU bins are removed; use get_cached_percentage_bins()')
    def get_traces(self, dataset: str, model: str, 
                   offset: int = 0, limit: int = 100,
                   include_text: bool = False,
                   min_score: float = None, max_score: float = None) -> List[Dict]:
        """
        Retrieve traces with pagination and filtering
        
        Args:
            dataset: Dataset name
            model: Model name
            offset: Starting index
            limit: Number of traces to return
            include_text: Whether to decompress and include trace text
            min_score: Minimum score filter
            max_score: Maximum score filter
        """
        cursor = self.conn.cursor()
        query = """
            SELECT t.*, 
                   e.token_count, e.sentence_count,
                   r.redundancy_score
            FROM traces t
            LEFT JOIN efficiency_metrics e ON 
                t.dataset = e.dataset AND t.model = e.model AND t.trace_index = e.trace_index
            LEFT JOIN redundancy_metrics r ON
                t.dataset = r.dataset AND t.model = r.model AND t.trace_index = r.trace_index
            WHERE t.dataset = ? AND t.model = ?
        """
        params = [dataset, model]
        if min_score is not None:
            query += " AND t.score >= ?"
            params.append(min_score)
        if max_score is not None:
            query += " AND t.score <= ?"
            params.append(max_score)
        query += " ORDER BY t.trace_index LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor.execute(query, params)
        rows = cursor.fetchall()
        results = []
        for row in rows:
            item = {
                'index': row['trace_index'],
                'question': row['question'],
                'extracted_answer': row['extracted_answer'],
                'score': row['score'],
                'raw_score': row['raw_score'],
                'ground_truth': row['ground_truth'],
            }
            if include_text and row['trace_text']:
                item['trace'] = self.decompress_text(row['trace_text'])
            else:
                item['trace_available'] = row['trace_text'] is not None
            if row['token_count'] is not None:
                item['efficiency'] = {
                    'token_count': row['token_count'],
                    'sentence_count': row['sentence_count']
                }
            if row['redundancy_score'] is not None:
                item['redundancy'] = {
                    'redundancy_score': row['redundancy_score']
                }
            if row['metadata']:
                item['metadata'] = json.loads(row['metadata'])
            results.append(item)
        return results
    def get_trace_by_index(self, dataset: str, model: str, trace_index: int) -> Optional[Dict]:
        """Get a single trace with all details"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT t.*, 
                   e.token_count, e.sentence_count,
                   r.redundancy_score, r.redundancy_per_step, r.redundancy_match_indices
            FROM traces t
            LEFT JOIN efficiency_metrics e ON 
                t.dataset = e.dataset AND t.model = e.model AND t.trace_index = e.trace_index
            LEFT JOIN redundancy_metrics r ON
                t.dataset = r.dataset AND t.model = r.model AND t.trace_index = r.trace_index
            WHERE t.dataset = ? AND t.model = ? AND t.trace_index = ?
        """, (dataset, model, trace_index))
        row = cursor.fetchone()
        if not row:
            return None
        result = {
            'index': row['trace_index'],
            'question': row['question'],
            'trace': self.decompress_text(row['trace_text']) if row['trace_text'] else None,
            'extracted_answer': row['extracted_answer'],
            'score': row['score'],
            'raw_score': row['raw_score'],
            'ground_truth': row['ground_truth'],
        }
        if row['token_count'] is not None:
            result['efficiency'] = {
                'token_count': row['token_count'],
                'sentence_count': row['sentence_count']
            }
        if row['redundancy_score'] is not None:
            result['redundancy'] = {
                'redundancy_score': row['redundancy_score'],
                'redundancy_per_step': json.loads(self.decompress_text(row['redundancy_per_step'])) if row['redundancy_per_step'] else [],
                'redundancy_match_indices': json.loads(self.decompress_text(row['redundancy_match_indices'])) if row['redundancy_match_indices'] else []
            }
        if row['metadata']:
            result['metadata'] = json.loads(row['metadata'])
        return result
    def get_model_summary(self, dataset: str, model: str) -> Optional[Dict]:
        """Get model-level summary statistics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM model_metadata
            WHERE dataset = ? AND model = ?
        """, (dataset, model))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            'model': model,
            'total_traces': row['total_traces'],
            'accuracy': row['accuracy'],
            'avg_tokens': row['avg_tokens'],
            'avg_redundancy': row['avg_redundancy'],
            'metadata': json.loads(row['metadata_json']) if row['metadata_json'] else {}
        }
    def get_all_models(self, dataset: str) -> List[Dict]:
        """Get all models for a dataset with their summaries"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT model, total_traces, accuracy, avg_tokens, avg_redundancy
            FROM model_metadata
            WHERE dataset = ?
            ORDER BY model
        """, (dataset,))
        return [dict(row) for row in cursor.fetchall()]
    def get_datasets(self) -> List[str]:
        """Get list of all datasets"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT dataset FROM model_metadata ORDER BY dataset")
        return [row['dataset'] for row in cursor.fetchall()]
    def get_pu_students(self, dataset: str = None) -> List[str]:
        """Get list of all student models in PU data from split tables
        
        Args:
            dataset: Filter by dataset, or None for all datasets
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name LIKE 'pu_%'
            """
        )
        students = set()
        allowed_datasets = {'math', 'gpqa', 'connections'}
        excluded_tables = {
            'pu_percentage_bins',
            'pu_absolute_bins',
            'pu_trace_metrics',
            'pu_regression_cache',
        }
        for row in cursor.fetchall():
            table_name = row['name']
            if table_name in excluded_tables:
                continue
            if table_name.endswith('_bins'):
                continue
            parts = table_name.split('_', 2)
            if len(parts) >= 3:
                table_dataset = parts[1]
                student_part = parts[2]
                if table_dataset not in allowed_datasets:
                    continue
                if dataset and table_dataset != dataset:
                    continue
                student_mapping = {
                    'meta_llama_llama_3_2_1b': 'meta-llama/Llama-3.2-1B-Instruct',
                    'microsoft_phi_3_mini_128k': 'microsoft/Phi-3-mini-128k-instruct'
                }
                full_student = student_mapping.get(student_part, student_part)
                students.add(full_student)
        return sorted(list(students))
    def get_pu_samples(self, dataset: str, teacher_model: str, student_model: str = None,
                      step_bin: float = None, mode: str = 'percentage', limit: int = 10) -> List[Dict]:
        """Get pedagogical utility samples with filtering
        
        Args:
            mode: Deprecated; TU is percentage-only. Kept for frontend compatibility.
        """
        cursor = self.conn.cursor()
        if student_model:
            student_model = self.get_student_short_name(student_model)
        if student_model and teacher_model.endswith('_' + student_model):
            teacher_model = teacher_model[:-(len(student_model) + 1)]
        if not student_model and '_' in teacher_model:
            KNOWN_STUDENTS = ['meta_llama_llama_3_2_1b', 'microsoft_phi_3_mini_128k']
            for s in KNOWN_STUDENTS:
                if teacher_model.endswith('_' + s):
                    student_model = s
                    teacher_model = teacher_model[:-(len(s)+1)]
                    break
        excluded_tables = {
            'pu_percentage_bins',
            'pu_absolute_bins',
            'pu_trace_metrics',
            'pu_regression_cache',
        }
        table_specs: List[Dict[str, str]] = []
        if student_model:
            table_name = self.get_pu_table_name(dataset, student_model)
            if not table_name:
                logger.error(f"Cannot fetch PU samples: invalid student model {student_model}")
                return []
            table_specs = [{'table': table_name, 'student_short': student_model}]
        else:
            prefix = f"pu_{dataset}_"
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name LIKE ?
                """,
                (prefix + '%',),
            )
            for r in cursor.fetchall():
                name = r['name']
                if name in excluded_tables or name.endswith('_bins'):
                    continue
                if not name.startswith(prefix):
                    continue
                student_short = name[len(prefix):]
                if not student_short:
                    continue
                table_specs.append({'table': name, 'student_short': student_short})
            if not table_specs:
                logger.error(f"Cannot fetch PU samples: no PU split tables found for dataset={dataset}")
                return []
        candidate_limit = max(500, limit * 200)
        candidates: List[Tuple[str, str, int]] = []
        for spec in table_specs:
            table = spec['table']
            student_short = spec['student_short']
            candidate_query = f"""
                SELECT pu.id
                FROM {table} pu
                WHERE pu.teacher_model = ? AND pu.teacher_score > 0.5
            """
            candidate_params: List[Any] = [teacher_model]
            if step_bin is not None:
                x = float(step_bin)
                low = x - 2.0
                high = x
                candidate_query += " AND pu.percentage_complete > ? AND pu.percentage_complete <= ?"
                candidate_params.extend([low, high])
            candidate_query += " LIMIT ?"
            candidate_params.append(candidate_limit)
            cursor.execute(candidate_query, candidate_params)
            for row in cursor.fetchall():
                candidates.append((table, student_short, int(row['id'])))
        if not candidates:
            return []
        if len(candidates) <= limit:
            chosen = candidates
        else:
            chosen = random.sample(candidates, k=limit)
        random.shuffle(chosen)
        chosen_by_table: Dict[str, Dict[str, Any]] = {}
        for table, student_short, rid in chosen:
            chosen_by_table.setdefault(table, {'student_short': student_short, 'ids': []})
            chosen_by_table[table]['ids'].append(rid)
        rows_with_student: List[Tuple[str, str, Any]] = []
        for table, info in chosen_by_table.items():
            ids = info['ids']
            if not ids:
                continue
            placeholders = ",".join(["?"] * len(ids))
            details_query = f"""
                SELECT pu.*, t.question, t.ground_truth, t.trace_text as teacher_trace
                FROM {table} pu
                JOIN traces t ON
                    t.dataset = ? AND
                    t.model = pu.teacher_model AND
                    t.trace_index = pu.trace_index
                WHERE pu.id IN ({placeholders})
            """
            details_params = [dataset] + ids
            cursor.execute(details_query, details_params)
            for row in cursor.fetchall():
                rows_with_student.append((table, info['student_short'], row))
        id_to_pos = {(t, int(rid)): i for i, (t, _sid, rid) in enumerate(chosen)}
        rows_with_student.sort(key=lambda tr: id_to_pos.get((tr[0], int(tr[2]['id'])), 10**9))
        results = []
        for _table, student_short, row in rows_with_student:
            full_output = self.decompress_text(row['full_output']) if row['full_output'] else ''
            completion_starts_at = row['completion_starts_at'] or 0
            student_completion = full_output[completion_starts_at:] if completion_starts_at > 0 else full_output
            teacher_trace = self.decompress_text(row['teacher_trace']) if row['teacher_trace'] else ''
            try:
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(teacher_trace)
            except Exception:
                import re
                sentences = [s.strip() for s in re.split(r'[.!?]+', teacher_trace) if s.strip()]
            num_steps = row['num_steps']
            total_steps = row['total_steps']
            if num_steps > 0 and num_steps < len(sentences):
                partial_teacher_trace = ' '.join(sentences[:num_steps])
            else:
                paragraphs = [p.strip() for p in teacher_trace.split('\n\n') if p.strip()]
                if num_steps < len(paragraphs):
                    partial_teacher_trace = '\n\n'.join(paragraphs[:num_steps])
                else:
                    partial_teacher_trace = teacher_trace
            results.append({
                'index': row['trace_index'],
                'num_steps': row['num_steps'],
                'total_steps': row['total_steps'],
                'score': row['score'],
                'extracted_answer': row['extracted_answer'],
                'ground_truth': row['ground_truth'],
                'question': row['question'],
                'grading_error': row['grading_error'],
                'ran_second_pass': bool(row['ran_second_pass']),
                'partial_teacher_trace': partial_teacher_trace,
                'full_teacher_trace': teacher_trace,
                'student_completion': student_completion,
                'student_model': student_short,
                'teacher_model': row['teacher_model']
            })
        return results
    def count_traces(self, dataset: str, model: str) -> int:
        """Count total traces for a dataset/model"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as count FROM traces
            WHERE dataset = ? AND model = ?
        """, (dataset, model))
        return cursor.fetchone()['count']
    def commit(self):
        """Commit transaction"""
        self.conn.commit()
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
def migrate_pickle_to_db(dataset_name: str = None, model_name: str = None):
    """
    Migrate pickle files to database (efficiency computed inline, redundancy loaded from pickle)
    
    Args:
        dataset_name: If specified, only migrate this dataset
        model_name: If specified, only migrate this model (requires dataset_name)
    """
    db = TraceDB()
    db.initialize_schema()
    if dataset_name:
        datasets = [dataset_name]
    else:
        datasets = [d.name for d in TRACES_DIR.iterdir() 
                   if d.is_dir() and d.name != 'archived']
    logger.info(f"Migrating {len(datasets)} datasets to database...")
    for dataset in datasets:
        dataset_dir = TRACES_DIR / dataset
        trace_files = list(dataset_dir.glob('traces_*.pkl'))
        if model_name:
            trace_files = [f for f in trace_files if f.name == f'traces_{model_name}.pkl']
        logger.info(f"\nDataset: {dataset} ({len(trace_files)} models)")
        for trace_file in tqdm(trace_files, desc=f"Processing {dataset}"):
            model = trace_file.name[7:-4]
            with open(trace_file, 'rb') as f:
                trace_data = pickle.load(f)
            metadata = trace_data.get('metadata', {})
            data = trace_data.get('data', {})
            questions = data.get('questions', [])
            traces = data.get('traces', [])
            extracted_answers = data.get('extracted_answers', [])
            scores = data.get('scores', [])
            ground_truth = data.get('ground_truth_answers', [])
            redundancy_file = REDUNDANCY_DIR / dataset / model / 'redundancy_analysis.pkl'
            redundancy_lookup = {}
            if redundancy_file.exists():
                with open(redundancy_file, 'rb') as f:
                    red_data = pickle.load(f)
                for item in red_data.get('data', []):
                    idx = item.get('index', -1)
                    redundancy_lookup[idx] = item
            else:
                logger.warning(f"Redundancy file not found: {redundancy_file}")
                logger.warning(f"Run 'python analyze_redundancy.py --dataset {dataset} --model {model}' first")
            logger.info(f"Computing efficiency metrics for {len(traces)} traces...")
            for i in tqdm(range(len(questions)), desc=f"Processing traces for {model}", leave=False):
                trace = traces[i] if i < len(traces) else ''
                raw_score = scores[i] if i < len(scores) else 0
                normalized_score = raw_score
                db.insert_trace(
                    dataset, model, i,
                    questions[i] if i < len(questions) else '',
                    trace,
                    extracted_answers[i] if i < len(extracted_answers) else '',
                    normalized_score, raw_score,
                    ground_truth[i] if i < len(ground_truth) else '',
                    None
                )
                token_count = count_tokens(trace)
                sentence_count = count_sentences(trace)
                db.insert_efficiency(dataset, model, i, token_count, sentence_count)
                if i in redundancy_lookup:
                    red = redundancy_lookup[i]
                    db.insert_redundancy(
                        dataset, model, i,
                        red.get('redundancy_score', 0),
                        red.get('redundancy_per_step', []),
                        red.get('redundancy_match_indices', [])
                    )
            import numpy as np
            model_meta = {
                'total_traces': len(questions),
                'accuracy': 0.0,
                'original_metadata': metadata
            }
            cursor = db.conn.cursor()
            result = cursor.execute(
                """
                SELECT AVG(score) FROM traces
                WHERE dataset = ? AND model = ?
                """,
                (dataset, model),
            ).fetchone()
            if result and result[0] is not None:
                model_meta['accuracy'] = float(result[0])
            result = cursor.execute('''
                SELECT AVG(token_count) FROM efficiency_metrics 
                WHERE dataset = ? AND model = ?
            ''', (dataset, model)).fetchone()
            if result and result[0]:
                model_meta['avg_tokens'] = float(result[0])
            if redundancy_lookup:
                result = cursor.execute('''
                    SELECT AVG(redundancy_score) FROM redundancy_metrics 
                    WHERE dataset = ? AND model = ?
                ''', (dataset, model)).fetchone()
                if result and result[0]:
                    model_meta['avg_redundancy'] = float(result[0])
            db.insert_model_metadata(dataset, model, model_meta)
            db.commit()
    logger.info("\nMigration complete!")
    logger.info(f"Database location: {DB_PATH}")
    logger.info(f"Database size: {DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    db.close()
def migrate_pu_data(dataset_name: str = None):
    """
    Migrate pedagogical utility data from pickle files to database
    
    Args:
        dataset_name: Specific dataset to migrate, or None for all datasets
    """
    db = TraceDB()
    db.initialize_schema()
    if not PU_DIR.exists():
        logger.error(f"PU directory not found: {PU_DIR}")
        return
    logger.info(f"Starting PU data migration from {PU_DIR}")
    pu_files = []
    for dataset_dir in PU_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        if dataset_name and dataset_dir.name != dataset_name:
            continue
        for teacher_dir in dataset_dir.iterdir():
            if not teacher_dir.is_dir():
                continue
            for pu_file in teacher_dir.glob('pu_*.pkl'):
                pu_files.append((dataset_dir.name, teacher_dir.name, pu_file))
    if not pu_files:
        logger.warning(f"No PU pickle files found in {PU_DIR}")
        return
    pu_files = sorted(pu_files, key=lambda x: x[2].stat().st_size)
    logger.info(f"Found {len(pu_files)} PU pickle files to migrate (sorted by size)")
    total_inserted = 0
    BATCH_SIZE = 500
    for dataset, teacher_model, pu_file in tqdm(pu_files, desc="Migrating PU files"):
        try:
            student_model = pu_file.stem[3:]
            file_size_mb = pu_file.stat().st_size / 1024 / 1024
            logger.info(f"Processing {dataset}/{teacher_model} -> {student_model} (size: {file_size_mb:.1f}MB)")
            db.create_pu_table(dataset, student_model)
            table_name = db.get_pu_table_name(dataset, student_model)
            cursor = db.conn.cursor()
            cursor.execute(f"""
                SELECT COUNT(*) as count FROM {table_name}
                WHERE teacher_model = ?
            """, (teacher_model,))
            existing_count = cursor.fetchone()['count']
            if existing_count > 0:
                logger.info(f"Skipping {dataset}/{teacher_model}/{student_model} - already has {existing_count} records")
                continue
            logger.info(f"Loading teacher scores for {dataset}/{teacher_model}...")
            cursor.execute("""
                SELECT trace_index, score 
                FROM traces 
                WHERE dataset = ? AND model = ?
            """, (dataset, teacher_model))
            teacher_scores = {row['trace_index']: row['score'] for row in cursor.fetchall()}
            logger.info(f"Loaded {len(teacher_scores)} teacher scores")
            logger.info(f"Loading pickle file (size: {file_size_mb:.1f}MB, this may take a while)...")
            with open(pu_file, 'rb') as f:
                pu_pkl = pickle.load(f)
            if isinstance(pu_pkl, dict) and 'data' in pu_pkl:
                pu_data = pu_pkl['data']
                metadata = pu_pkl.get('metadata', {})
            else:
                logger.warning(f"Unexpected format in {pu_file.name}, skipping")
                continue
            num_records = len(pu_data)
            logger.info(f"Loaded {num_records} records from {pu_file.name}")
            file_inserted = 0
            conn_unique_vals = set()
            conn_saw_gt1 = False
            conn_tol = 1e-6
            chunk_size = 500
            for chunk_start in range(0, num_records, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_records)
                chunk = pu_data[chunk_start:chunk_end]
                for i, record in enumerate(tqdm(chunk, desc=f"Inserting {dataset}/{teacher_model} ({chunk_start}-{chunk_end})", leave=False)):
                    try:
                        trace_index = record.get('index', 0)
                        teacher_score = teacher_scores.get(trace_index, None)
                        num_steps = record.get('num_steps', 0)
                        total_steps = record.get('total_steps', 0)
                        score = record.get('score', 0.0)
                        if dataset == 'connections':
                            try:
                                s = float(score)
                            except Exception:
                                s = 0.0
                            conn_unique_vals.add(s)
                            if s > 1.0 + conn_tol:
                                conn_saw_gt1 = True
                        extracted_answer = record.get('extracted_answer', '')
                        grading_error = record.get('grading_error', '')
                        ran_second_pass = record.get('ran_second_pass', False)
                        full_output = record.get('full_output', '')
                        completion_starts_at = record.get('completion_starts_at', 0)
                        db.insert_pedagogical_utility(
                            dataset=dataset,
                            teacher_model=teacher_model,
                            student_model=student_model,
                            trace_index=trace_index,
                            num_steps=num_steps,
                            total_steps=total_steps,
                            score=score,
                            extracted_answer=extracted_answer,
                            grading_error=grading_error,
                            ran_second_pass=ran_second_pass,
                            full_output=full_output,
                            completion_starts_at=completion_starts_at,
                            teacher_score=teacher_score
                        )
                        file_inserted += 1
                        if file_inserted % BATCH_SIZE == 0:
                            db.conn.commit()
                            logger.debug(f"Committed batch ({file_inserted}/{num_records})")
                    except Exception as e:
                        logger.error(f"Error inserting PU record: {e}")
                        continue
                db.conn.commit()
                del chunk
                logger.info(f"Processed chunk {chunk_start}-{chunk_end} ({file_inserted} total inserted)")
            db.conn.commit()
            total_inserted += file_inserted
            logger.info(f"✓ Inserted {file_inserted} records from {pu_file.name}")
            if dataset == 'connections' and not conn_saw_gt1:
                if conn_unique_vals.issubset({0.0, 1.0}):
                    logger.warning(
                        f"Connections PU normalization ambiguous for student={student_model}: "
                        f"only observed scores {sorted(conn_unique_vals)}; treating as already-normalized"
                    )
            del pu_data
            del pu_pkl
        except Exception as e:
            logger.error(f"Error processing {pu_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    logger.info(f"PU migration complete! Inserted {total_inserted} total records")
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE 'pu_%'
        AND name NOT LIKE 'pu_%_bins'
    """)
    pu_tables = [row['name'] for row in cursor.fetchall()]
    total_count = 0
    logger.info(f"\nPU tables created: {len(pu_tables)}")
    logger.info("\nPU records by table:")
    for table_name in pu_tables:
        cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
        count = cursor.fetchone()['count']
        total_count += count
        logger.info(f"  {table_name}: {count:,} records")
    logger.info(f"\nTotal PU records across all tables: {total_count:,}")
    db.close()
def migrate_backtracking_data(dataset_name: str = None, model_name: str = None):
    """Migrate backtracking_analysis pickles into SQLite.

    Expected pickle layout:
        outputs/backtracking_analysis/backtracking_{dataset}_{model}.pkl

    Each pickle is a dict with keys: results (list[dict]), analysis (dict), timestamp.

    Args:
        dataset_name: Optional dataset filter (math/gpqa/connections).
        model_name: Optional model filter (must match the filename model segment exactly).
    """
    db = TraceDB()
    db.initialize_schema()
    bt_dir = OUTPUTS_DIR / 'backtracking_analysis'
    if not bt_dir.exists():
        logger.error(f"Backtracking directory not found: {bt_dir}")
        return
    paths = sorted(bt_dir.glob('backtracking_*.pkl'))
    if not paths:
        logger.warning(f"No backtracking pickle files found in {bt_dir}")
        return
    def parse_name(path: Path) -> Tuple[Optional[str], Optional[str]]:
        stem = path.stem
        if not stem.startswith('backtracking_'):
            return None, None
        rest = stem[len('backtracking_'):]
        if '_' not in rest:
            return None, None
        ds, model = rest.split('_', 1)
        return ds, model
    filtered = []
    for p in paths:
        ds, model = parse_name(p)
        if not ds or not model:
            continue
        if dataset_name and ds != dataset_name:
            continue
        if model_name and model != model_name:
            continue
        filtered.append((p, ds, model))
    if not filtered:
        logger.warning("No backtracking pickles matched the provided filters")
        return
    logger.info(f"Found {len(filtered)} backtracking pickles to migrate")
    cursor = db.conn.cursor()
    total_inserted = 0
    for path, ds, model in tqdm(filtered, desc='Migrating backtracking'):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            continue
        results = obj.get('results', []) if isinstance(obj, dict) else []
        if not isinstance(results, list):
            logger.warning(f"Unexpected format in {path.name}: results is not a list")
            continue
        cursor.execute(
            """
            SELECT COUNT(*) as n
            FROM backtracking_results
            WHERE dataset = ? AND model = ?
            """,
            (ds, model),
        )
        existing = cursor.fetchone()['n']
        if existing and existing > 0:
            logger.info(f"Skipping {ds}/{model} (already has {existing} rows)")
            continue
        inserted = 0
        for r in results:
            if not isinstance(r, dict):
                continue
            idx = r.get('index')
            if not isinstance(idx, (int, np.integer)):
                continue
            steps = r.get('backtracking_steps')
            if isinstance(steps, list):
                n_steps = len(steps)
                steps_json = json.dumps(steps, ensure_ascii=False)
            else:
                n_steps = None
                steps_json = json.dumps(steps, ensure_ascii=False) if steps is not None else None
            db.insert_backtracking_result(
                dataset=ds,
                model=model,
                trace_index=int(idx),
                backtracking_detected=r.get('backtracking_detected'),
                num_backtracking_steps=n_steps,
                confidence=r.get('confidence'),
                final_answer=str(r.get('final_answer')) if r.get('final_answer') is not None else None,
                overall_reasoning=r.get('overall_reasoning'),
                backtracking_steps_json=steps_json,
                error=r.get('error'),
                prompt_tokens=r.get('prompt_tokens'),
                output_tokens=r.get('output_tokens'),
                total_tokens=r.get('total_tokens'),
                finish_reason=r.get('finish_reason'),
            )
            inserted += 1
            if inserted % 2000 == 0:
                db.conn.commit()
        db.conn.commit()
        total_inserted += inserted
        logger.info(f"Migrated backtracking rows: dataset={ds} model={model} inserted={inserted}")
    logger.info(f"Backtracking migration complete. Inserted total rows={total_inserted}")
    db.close()
def query_example():
    """Example queries to demonstrate usage"""
    db = TraceDB()
    datasets = db.get_datasets()
    print(f"\nAvailable datasets: {datasets}")
    if datasets:
        dataset = datasets[0]
        models = db.get_all_models(dataset)
        print(f"\nModels in {dataset}:")
        for m in models:
            print(f"  {m['model']}: {m['total_traces']} traces, {m['accuracy']:.2%} accuracy")
        if models:
            model = models[0]['model']
            traces = db.get_traces(dataset, model, offset=0, limit=10, include_text=False)
            print(f"\nFirst 10 traces from {dataset}/{model}:")
            for t in traces[:3]:
                print(f"  Index {t['index']}: score={t['score']:.2f}")
            trace = db.get_trace_by_index(dataset, model, 0)
            if trace:
                print(f"\nTrace 0 details:")
                print(f"  Question length: {len(trace.get('question', ''))}")
                print(f"  Trace length: {len(trace.get('trace', ''))}")
                print(f"  Score: {trace['score']:.2f}")
    db.close()
def main():
    parser = argparse.ArgumentParser(description='Database backend for trace storage')
    parser.add_argument('--migrate', action='store_true', help='Migrate pickle files to database')
    parser.add_argument('--migrate-pu', action='store_true', help='Migrate pedagogical utility data to database')
    parser.add_argument('--dataset', type=str, help='Specific dataset to migrate')
    parser.add_argument('--model', type=str, help='Specific model to migrate (requires --dataset)')
    parser.add_argument('--query', action='store_true', help='Run example queries')
    args = parser.parse_args()
    if args.migrate:
        migrate_pickle_to_db(args.dataset, args.model)
    elif args.migrate_pu:
        migrate_pu_data(args.dataset)
    elif args.query:
        query_example()
    else:
        parser.print_help()
if __name__ == '__main__':
    main()
