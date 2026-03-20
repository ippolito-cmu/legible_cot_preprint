"""
Flask backend for reasoning trace visualization - DATABASE VERSION
Combines full app.py functionality with database backend for performance

This version:
- Uses SQLite database instead of pickle files (10-50x faster)
- Retains ALL interactive features from original app.py
- Supports traces, efficiency, redundancy analysis
- Click-to-load trace details
- Pedagogical utility with sample viewing
"""
import os
import json
import logging
import socket
import pickle
import io
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from flask import Flask, jsonify, render_template, send_from_directory, request, send_file
from flask_cors import CORS
import numpy as np
from db_backend import TraceDB
from urllib.parse import unquote_plus
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
@app.after_request
def add_no_cache_headers(response):
    """Avoid stale cached templates/JS causing UI breakages after edits."""
    try:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    except Exception:
        pass
    return response
logger.info("="*80)
logger.info("Reasoning Trace Visualization Dashboard - DATABASE VERSION")
logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
DB_PATH = OUTPUTS_DIR / 'traces.db'
PU_ANALYSIS_DIR = OUTPUTS_DIR / 'pu_analysis'
TRACES_DIR = OUTPUTS_DIR / 'traces'
CORRELATIONS_JSON_PATH = PROJECT_ROOT / 'analysis' / 'experiments' / 'metric_correlations' / 'cross_correlations.json'
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Database path: {DB_PATH}")
logger.info(f"Database exists: {DB_PATH.exists()}")
if not DB_PATH.exists():
    logger.warning("="*80)
    logger.warning("WARNING: Database not found!")
    logger.warning("Please run migration first: python db_backend.py --migrate")
    logger.warning("="*80)
db = TraceDB(DB_PATH)
_CORR_CACHE = {
    'mtime': None,
    'data': None,
}
def _load_local_font(logger_: logging.Logger, font_dir: Path) -> Optional[str]:
    try:
        if not font_dir.exists():
            return None
        candidates = list(font_dir.glob('*.ttf')) + list(font_dir.glob('*.otf'))
        if not candidates:
            return None
        return font_manager.FontProperties(fname=str(candidates[0])).get_name()
    except Exception as e:
        logger_.warning(f"Failed to load font from {font_dir}: {e}")
        return None
def get_correlation_results() -> Optional[Dict[str, Any]]:
    """Load and cache metric correlation results JSON."""
    if not CORRELATIONS_JSON_PATH.exists():
        return None
    mtime = CORRELATIONS_JSON_PATH.stat().st_mtime
    if _CORR_CACHE['data'] is None or _CORR_CACHE['mtime'] != mtime:
        _CORR_CACHE['data'] = json.loads(CORRELATIONS_JSON_PATH.read_text())
        _CORR_CACHE['mtime'] = mtime
    return _CORR_CACHE['data']
def _corr_condition_key(*, dataset: str, model: str, correctness: str) -> tuple[str, str]:
    """Map (dataset, model, correctness) to (conditions_key, group_key) in the JSON."""
    if dataset == 'all' and model == 'all':
        base = 'unconditional'
        group = 'all'
    elif dataset != 'all' and model == 'all':
        base = 'by_dataset'
        group = dataset
    elif dataset == 'all' and model != 'all':
        base = 'by_model'
        group = model
    else:
        base = 'by_dataset_model'
        group = f"{dataset}::{model}"
    if correctness == 'all':
        suffix = ''
    elif correctness == 'correct_only':
        suffix = '__correct_only'
    elif correctness == 'incorrect_only':
        suffix = '__incorrect_only'
    else:
        raise ValueError(f"Unknown correctness: {correctness}")
    return base + suffix, group
def _metric_label(metric: str) -> str:
    if metric.startswith('rm::'):
        return 'RM: ' + metric[len('rm::'):]
    if metric.startswith('pu_'):
        return 'PU: ' + metric[len('pu_'):]
    if metric.startswith('eff_'):
        return 'EFF: ' + metric[len('eff_'):]
    if metric.startswith('bt_'):
        return 'BT: ' + metric[len('bt_'):]
    if metric.startswith('red_'):
        return 'RED: ' + metric[len('red_'):]
    return metric
def _metrics_for_preset(all_metrics: list[str], preset: str) -> list[str]:
    if preset == 'all':
        return list(all_metrics)
    if preset == 'rm_pu':
        return [m for m in all_metrics if m.startswith('rm::') or m.startswith('pu_')]
    if preset == 'rm_only':
        return [m for m in all_metrics if m.startswith('rm::')]
    if preset == 'pu_only':
        return [m for m in all_metrics if m.startswith('pu_')]
    if preset == 'core':
        keep = {
            'trace_score',
            'trace_correct',
            'pu_avg_correctness',
            'pu_linearity_mse',
            'pu_regression_rate',
        }
        keep |= {m for m in all_metrics if m.startswith('rm::')}
        return [m for m in all_metrics if m in keep]
    raise ValueError(f"Unknown metric preset: {preset}")
    pmat = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        mat[i, i] = 1.0
        pmat[i, i] = 0.0
    for row in pairs:
        a = row.get('metric_x')
        b = row.get('metric_y')
        if a not in idx or b not in idx:
            continue
        r = row.get('r')
        p = row.get(p_kind)
        if r is None:
            continue
        i = idx[a]
        j = idx[b]
        mat[i, j] = float(r)
        mat[j, i] = float(r)
        if p is not None:
            pmat[i, j] = float(p)
            pmat[j, i] = float(p)
    return mat, pmat
def _render_correlation_heatmap_png(
    *,
    corr: np.ndarray,
    pvals: np.ndarray,
    metric_labels: list[str],
    title: str,
    alpha: float,
    annotate: bool,
    rim_significant: bool,
) -> bytes:
    n = len(metric_labels)
    side = max(7.0, min(22.0, 0.7 * n + 3.0))
    fig, ax = plt.subplots(figsize=(side, side))
    title_font = _load_local_font(logger, Path.home() / 'fonts' / 'Volkhov')
    axis_font = _load_local_font(logger, Path.home() / 'fonts' / 'Ubuntu_Mono')
    if axis_font:
        plt.rcParams['font.family'] = axis_font
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap='coolwarm', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(metric_labels, fontsize=10)
    if title_font:
        ax.set_title(title, fontsize=16, pad=14, fontname=title_font)
    else:
        ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='#dddddd', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    if annotate:
        for i in range(n):
            for j in range(n):
                val = corr[i, j]
                if not np.isfinite(val):
                    continue
                text_color = 'black'
                if abs(val) > 0.6:
                    text_color = 'white'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=9, color=text_color)
    if rim_significant:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                p = pvals[i, j]
                if np.isfinite(p) and p < alpha:
                    ax.add_patch(
                        mpatches.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor='black',
                            linewidth=2.0,
                        )
                    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160)
    plt.close(fig)
    return buf.getvalue()
def _render_pedagogical_svg(*, curves: list, title: str = 'Pedagogical Utility', width: int = 1200, height: int = 600,
                             x_min: Optional[float] = None, x_max: Optional[float] = None,
                             y_min: Optional[float] = None, y_max: Optional[float] = None,
                             markers: bool = True, grid: bool = True) -> bytes:
    """Render pedagogical utility curves into an SVG and return raw bytes.

    This is a best-effort renderer that is robust to several possible point key names
    used in the frontend cache (x/pct/percentage and y/mean/tu/value).
    """
    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    def _css_hsl_to_hex(s: str) -> str:
        """Convert CSS-style 'hsl(h, s%, l%)' to hex string usable by matplotlib.

        Returns original string if it doesn't match expected HSL pattern.
        """
        try:
            if not s or not isinstance(s, str):
                return s
            s = s.strip()
            if not s.startswith('hsl'):
                return s
            inside = s[s.find('(') + 1: s.rfind(')')]
            parts = [p.strip() for p in inside.replace(',', ' ').split() if p.strip()]
            if len(parts) < 3:
                return s
            h = float(parts[0])
            sat = parts[1].rstrip('%')
            light = parts[2].rstrip('%')
            s_frac = float(sat) / 100.0
            l_frac = float(light) / 100.0
            import colorsys
            r, g, b = colorsys.hls_to_rgb(h / 360.0, l_frac, s_frac)
            return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
        except Exception:
            return s
    for curve in curves:
        pts = curve.get('points', []) or []
        xs = []
        ys = []
        for p in pts:
            x = None
            for k in ('x', 'pct', 'percentage', 'percent', 'step_bin_center'):
                if k in p:
                    x = p[k]
                    break
            y = None
            for k in ('y', 'mean', 'mean_score', 'tu', 'value', 'mean_correct_smooth'):
                if k in p:
                    y = p[k]
                    break
            try:
                if x is not None and y is not None:
                    xs.append(float(x))
                    ys.append(float(y))
            except Exception:
                continue
        if not xs:
            continue
        raw_color = curve.get('color') or get_model_color(curve.get('teacher_model', ''))
        color = _css_hsl_to_hex(raw_color)
        label = curve.get('teacher_model') or curve.get('teacher', curve.get('teacher_name', None)) or None
        ax.plot(xs, ys, label=label, color=color, linewidth=1.4, alpha=0.95)
        if markers:
            try:
                ax.scatter(xs, ys, s=24, facecolors=color, edgecolors='white', linewidths=0.6, zorder=3)
            except Exception:
                ax.scatter(xs, ys, s=24, facecolors='none', edgecolors=color, linewidths=0.6, zorder=3)
    ax.set_xlabel('% Through Trace')
    ax.set_ylabel('Transfer Utility (TU) Score')
    ax.set_title(title)
    try:
        if x_min is not None and x_max is not None:
            ax.set_xlim(float(x_min), float(x_max))
        else:
            ax.set_xlim(0, 100)
    except Exception:
        ax.set_xlim(0, 100)
    try:
        if y_min is not None and y_max is not None:
            ax.set_ylim(float(y_min), float(y_max))
        else:
            ax.set_ylim(0.0, 1.0)
    except Exception:
        ax.set_ylim(0.0, 1.0)
    if grid:
        try:
            ax.set_axisbelow(True)
            ax.grid(True, color='#d4d4d4', linewidth=0.8, zorder=0)
        except Exception:
            ax.grid(True, alpha=0.25)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='svg')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
MODEL_COLOR_FAMILIES = {
    'qwen': {'base_hue': 15, 'range': 25},
    'deepseek': {'base_hue': 230, 'range': 25},
    'gemma': {'base_hue': 170, 'range': 20},
    'gpt': {'base_hue': 285, 'range': 30},
    'llama': {'base_hue': 200, 'range': 20},
    'mistral': {'base_hue': 45, 'range': 15},
    'nvidia': {'base_hue': 135, 'range': 20},
    'kimi': {'base_hue': 320, 'range': 15},
}
MODEL_COLORS = {
    'Qwen/Qwen3-0.6B': 'hsl(15, 85%, 45%)',
    'Qwen/Qwen3-4B': 'hsl(22, 85%, 40%)',
    'Qwen/Qwen3-8B': 'hsl(30, 85%, 35%)',
    'Qwen/QwQ-32B': 'hsl(38, 85%, 38%)',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B': 'hsl(230, 85%, 40%)',
    'deepseek-ai/deepseek-r1-0528': 'hsl(245, 85%, 45%)',
    'google/gemma-3-12b-it': 'hsl(172, 85%, 35%)',
    'google/gemma-3-27b-it': 'hsl(184, 85%, 32%)',
    'gpt-5': 'hsl(285, 85%, 45%)',
    'openai/gpt-oss-20b': 'hsl(300, 85%, 42%)',
    'openai/gpt-oss-120b': 'hsl(315, 85%, 40%)',
    'meta-llama/Meta-Llama-3.1-8B-Instruct': 'hsl(203, 85%, 42%)',
    'meta-llama/Meta-Llama-3.1-70B-Instruct': 'hsl(215, 85%, 38%)',
    'mistralai/Magistral-Small-2509': 'hsl(48, 80%, 40%)',
    'nvidia/Llama-3.1-Nemotron-Nano-8B-v1': 'hsl(138, 85%, 35%)',
    'nvidia/OpenReasoning-Nemotron-32B': 'hsl(150, 85%, 32%)',
    'math_ground_truth': 'hsl(145, 85%, 42%)',
}
def get_model_family(model_name):
    """Determine model family from model name"""
    model_lower = model_name.lower()
    if 'qwen' in model_lower or 'qwq' in model_lower:
        return 'qwen'
    elif 'deepseek' in model_lower:
        return 'deepseek'
    elif 'gemma' in model_lower:
        return 'gemma'
    elif 'gpt' in model_lower:
        return 'gpt'
    elif 'llama' in model_lower:
        return 'llama'
    elif 'mistral' in model_lower or 'magistral' in model_lower:
        return 'mistral'
    elif 'nvidia' in model_lower or 'nemotron' in model_lower:
        return 'nvidia'
    return None
def get_model_color(model_name):
    """Get consistent color for a model"""
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    normalized_name = model_name.replace('/', '_')
    if normalized_name in MODEL_COLORS:
        return MODEL_COLORS[normalized_name]
    alt_name = model_name.replace('_', '/')
    if alt_name in MODEL_COLORS:
        return MODEL_COLORS[alt_name]
    family = get_model_family(model_name)
    if family and family in MODEL_COLOR_FAMILIES:
        family_config = MODEL_COLOR_FAMILIES[family]
        base_hue = family_config['base_hue']
        hue_range = family_config['range']
        hash_val = hash(model_name)
        hue_offset = (hash_val % (hue_range * 2)) - hue_range
        hue = base_hue + hue_offset
        lightness = 30 + (abs(hash_val) % 20)
        return f'hsl({hue}, 85%, {lightness}%)'
    hash_val = hash(model_name)
    hue = (hash_val % 360)
    lightness = 35 + (abs(hash_val) % 15)
    return f'hsl({hue}, 80%, {lightness}%)'
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')
@app.route('/api/datasets')
def get_datasets():
    """API endpoint to get available datasets"""
    datasets = db.get_datasets()
    return jsonify(datasets)
@app.route('/api/dataset-metadata/<dataset>')
def get_dataset_metadata(dataset):
    """Get dataset-specific metadata configuration (placeholder)"""
    config = {'filters': [], 'score_range': [0, 1]}
    return jsonify(config)
@app.route('/api/model-colors')
def get_model_colors_endpoint():
    """Get standardized model color palette"""
    return jsonify(MODEL_COLORS)
@app.route('/api/models/<dataset>')
def get_models(dataset):
    """API endpoint to get available models for a dataset"""
    models = db.get_all_models(dataset)
    models_with_colors = [{'name': m['model'], 'color': get_model_color(m['model'])} for m in models]
    return jsonify(models_with_colors)
@app.route('/api/traces/<dataset>/<model>')
def get_traces(dataset, model):
    """API endpoint to get traces for a specific dataset and model"""
    page = request.args.get('page', 0, type=int)
    page_size = request.args.get('page_size', 5000, type=int)
    include_text = request.args.get('include_text', 'true').lower() == 'true'
    offset = page * page_size
    traces = db.get_traces(dataset, model, offset=offset, limit=page_size, include_text=include_text)
    total = db.count_traces(dataset, model)
    summary = db.get_model_summary(dataset, model)
    traces_list = []
    for trace in traces:
        traces_list.append({
            'index': trace['index'],
            'question': trace.get('question', ''),
            'trace': trace.get('trace', ''),
            'extracted_answer': trace.get('extracted_answer', ''),
            'score': trace.get('score', 0),
            'raw_score': trace.get('raw_score', 0),
            'ground_truth': trace.get('ground_truth', ''),
            'token_count': trace.get('token_count'),
            'sentence_count': trace.get('sentence_count'),
            'redundancy_score': trace.get('redundancy_score'),
        })
    return jsonify({
        'metadata': summary.get('metadata', {}) if summary else {},
        'traces': traces_list,
        'total': total,
        'has_efficiency': any(t.get('token_count') is not None for t in traces),
        'has_redundancy': any(t.get('redundancy_score') is not None for t in traces),
    })
@app.route('/api/trace-detail/<dataset>/<model>/<int:index>')
def get_trace_detail(dataset, model, index):
    """Get full details for a single trace"""
    trace = db.get_trace_by_index(dataset, model, index)
    if trace is None:
        return jsonify({'error': 'Trace not found'}), 404
    return jsonify(trace)
@app.route('/api/models-summary/<dataset>')
def get_models_summary(dataset):
    """API endpoint to get summary of all models for a dataset (for bar chart)"""
    models = db.get_all_models(dataset)
    summaries = []
    for model in models:
        summaries.append({
            'model_name': model['model'],
            'total_traces': model['total_traces'],
            'accuracy': model['accuracy'],
            'correct': int(model['accuracy'] * model['total_traces']),
            'incorrect': model['total_traces'] - int(model['accuracy'] * model['total_traces']),
            'color': get_model_color(model['model'])
        })
    return jsonify(summaries)
@app.route('/api/efficiency-analysis/<dataset>')
def get_efficiency_analysis(dataset):
    """API endpoint to get efficiency analysis across all models"""
    models = db.get_all_models(dataset)
    analysis_data = {}
    for model in models:
        if model.get('avg_tokens') is not None:
            traces = db.get_traces(dataset, model['model'], offset=0, limit=10000, include_text=False)
            scores = []
            tokens = []
            sentences = []
            for t in traces:
                if t.get('score') is not None:
                    scores.append(t.get('score', 0))
                if 'efficiency' in t:
                    eff = t['efficiency']
                    tokens.append(eff.get('token_count', 0))
                    sentences.append(eff.get('sentence_count', 0))
            if not tokens or not sentences:
                continue
            analysis_data[model['model']] = {
                'accuracy': model.get('accuracy', 0),
                'scores': scores,
                'token_counts': tokens,
                'sentence_counts': sentences,
                'avg_tokens': float(np.mean(tokens)),
                'median_tokens': float(np.median(tokens)),
                'avg_sentences': float(np.mean(sentences)),
                'median_sentences': float(np.median(sentences)),
                'color': get_model_color(model['model'])
            }
    return jsonify(analysis_data)
@app.route('/api/redundancy-analysis/<dataset>')
def get_redundancy_analysis(dataset):
    """API endpoint to get redundancy analysis across all models"""
    models = db.get_all_models(dataset)
    analysis_data = {}
    for model in models:
        if model.get('avg_redundancy') is not None:
            traces = db.get_traces(dataset, model['model'], offset=0, limit=10000, include_text=False)
            scores = []
            redundancies = []
            for t in traces:
                if t.get('score') is not None:
                    scores.append(t.get('score', 0))
                if 'redundancy' in t:
                    red = t['redundancy']
                    redundancies.append(red.get('redundancy_score', 0))
            if not redundancies:
                continue
            analysis_data[model['model']] = {
                'accuracy': model.get('accuracy', 0),
                'scores': scores,
                'redundancy_scores': redundancies,
                'avg_redundancy': float(np.mean(redundancies)),
                'median_redundancy': float(np.median(redundancies)),
                'color': get_model_color(model['model'])
            }
    return jsonify(analysis_data)
@app.route('/api/redundancy-trace/<dataset>/<model>/<int:index>')
def get_redundancy_trace_detail(dataset, model, index):
    """Get detailed redundancy information for a specific trace"""
    trace = db.get_trace_by_index(dataset, model, index)
    if not trace:
        return jsonify({'error': 'Trace not found'}), 404
    redundancy_data = trace.get('redundancy', {})
    redundancy_per_step = redundancy_data.get('redundancy_per_step', [])
    trace_text = trace.get('trace', '')
    try:
        import nltk
        steps = nltk.sent_tokenize(trace_text)
    except Exception as e:
        logger.warning(f"NLTK sentence tokenization failed: {e}, falling back to simple split")
        import re
        steps = [s.strip() for s in re.split(r'\n+', trace_text) if s.strip()]
    return jsonify({
        'index': index,
        'question': trace.get('question', ''),
        'ground_truth': trace.get('ground_truth', ''),
        'extracted_answer': trace.get('extracted_answer', ''),
        'is_correct': trace.get('score', 0) > 0,
        'redundancy_score': redundancy_data.get('redundancy_score', 0),
        'redundancy_per_step': redundancy_per_step,
        'redundancy_match_indices': redundancy_data.get('redundancy_match_indices', []),
        'steps': steps,
        'num_steps': len(redundancy_per_step)
    })
@app.route('/api/backtracking/datasets')
def get_backtracking_datasets():
    """Get available datasets for backtracking analysis."""
    return jsonify(db.get_backtracking_datasets())
@app.route('/api/backtracking/models/<dataset_name>')
def get_backtracking_models(dataset_name):
    """Get available models for backtracking analysis within a dataset."""
    return jsonify(db.get_backtracking_models(dataset_name))
@app.route('/api/backtracking/<dataset_name>/summary')
def get_backtracking_summary(dataset_name):
    """Get model-level backtracking summary for a dataset."""
    rows = db.get_backtracking_model_summary(dataset_name)
    for r in rows:
        r['color'] = get_model_color(r.get('model', ''))
    return jsonify({'dataset': dataset_name, 'models': rows})
@app.route('/api/backtracking/<dataset_name>/trace/<model>/<int:index>')
def get_backtracking_trace(dataset_name, model, index):
    """Get backtracking JSON + trace context for a specific trace."""
    detail = db.get_backtracking_trace_detail(dataset_name, model, index)
    if not detail:
        return jsonify({'error': 'Trace not found'}), 404
    return jsonify(detail)
@app.route('/api/backtracking/<dataset_name>/traces/<model>')
def get_backtracking_trace_list(dataset_name, model):
    """List trace indices for a model/dataset for backtracking drilldown."""
    try:
        limit = int(request.args.get('limit', 200))
        offset = int(request.args.get('offset', 0))
        detected_only_raw = request.args.get('detected_only', None)
        correctness = request.args.get('correctness', 'all')
        order = request.args.get('order', 'trace_index_asc')
        detected_only = None
        if detected_only_raw is not None:
            if str(detected_only_raw).lower() in {'1', 'true', 'yes', 'y'}:
                detected_only = True
            elif str(detected_only_raw).lower() in {'0', 'false', 'no', 'n'}:
                detected_only = False
            else:
                return jsonify({'error': f"Invalid detected_only: {detected_only_raw}"}), 400
        payload = db.get_backtracking_traces(
            dataset=dataset_name,
            model=model,
            limit=limit,
            offset=offset,
            detected_only=detected_only,
            correctness=correctness,
            order=order,
        )
        return jsonify(payload)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f"Failed backtracking trace list: dataset={dataset_name} model={model}")
        return jsonify({'error': str(e)}), 500
@app.route('/api/leaderboard')
def get_leaderboard():
    """Return a combined leaderboard table across datasets/models."""
    rows = db.get_leaderboard_rows()
    cursor = db.conn.cursor()
    cursor.execute("SELECT DISTINCT dataset FROM model_metadata ORDER BY dataset")
    datasets = [r['dataset'] for r in cursor.fetchall()]
    models_by_dataset = {}
    for ds in datasets:
        cursor.execute(
            """
            SELECT model
            FROM model_metadata
            WHERE dataset = ?
            ORDER BY model
            """,
            (ds,),
        )
        models_by_dataset[ds] = [r['model'] for r in cursor.fetchall()]
    for r in rows:
        r['color'] = get_model_color(r.get('model', ''))
    return jsonify({
        'rows': rows,
        'datasets': datasets,
        'models_by_dataset': models_by_dataset,
    })
@app.route('/api/pedagogical-utility/datasets')
def get_pu_datasets():
    """Get available pedagogical utility datasets"""
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE 'pu_%'
        AND name NOT LIKE 'pu_%_bins'
    """)
    datasets = set()
    for row in cursor.fetchall():
        table_name = row['name']
        parts = table_name.split('_', 2)
        if len(parts) >= 2:
            datasets.add(parts[1])
    return jsonify(sorted(list(datasets)))
@app.route('/api/pedagogical-utility/students')
def get_pu_students():
    """Get available student models"""
    dataset = request.args.get('dataset')
    students = db.get_pu_students(dataset)
    return jsonify(students)
@app.route('/api/pedagogical-utility/<dataset_name>')
def get_pu_data(dataset_name):
    """Get pedagogical utility dashboard data"""
    mode = request.args.get('mode', 'percentage')
    student_model = request.args.get('student')
    if mode != 'percentage':
        return jsonify({'error': "Only mode=percentage is supported"}), 400
    data = get_percentage_bins_from_db(dataset_name, student_model)
    if not data or not data.get('curves'):
        return jsonify({'error': 'Dataset not found or no data available'}), 404
    return jsonify(data)
def get_percentage_bins_from_db(dataset_name, student_model=None):
    """Get PU data with percentage-based bins from pre-computed cache
    
    Args:
        dataset_name: Dataset to query
        student_model: Filter by student model, or None for all students
    """
    from scipy.ndimage import gaussian_filter1d
    logger.info(f"Fetching cached percentage bins for {dataset_name} (student: {student_model or 'all'}) from database")
    teacher_data = db.get_cached_percentage_bins(dataset_name, student_model)
    if not teacher_data:
        logger.warning(f"No cached percentage bins found for {dataset_name}")
        return None
    logger.info(f"Fetched cached bins for {len(teacher_data)} teachers")
    curves = []
    student_names = set()
    for teacher_key, data in teacher_data.items():
        bins = data['bins']
        teacher_short_name = data['teacher_short_name']
        student = data.get('student_model', 'all_students')
        student_names.add(student)
        display_name = teacher_short_name
        if not student_model:
            student_short = student.split('/')[-1]
            display_name = f"{teacher_short_name} ({student_short})"
        points = []
        for bin_data in bins:
            points.append({
                'step_bin_center': bin_data['bin_center'],
                'mean_correct': bin_data['mean_correct'],
                'mean_correct_smooth': bin_data['mean_correct'],
                'std_correct': bin_data['std_correct'],
                'count': bin_data['count'],
                'hazard_prob': bin_data.get('hazard_prob', 0.0),
                'hazard_count': bin_data.get('hazard_count', 0),
            })
        if len(points) > 2:
            smoothed = gaussian_filter1d([p['mean_correct'] for p in points], sigma=1.5)
            for i, p in enumerate(points):
                p['mean_correct_smooth'] = smoothed[i]
        curves.append({
            'teacher_model': teacher_key,
            'teacher_short_name': display_name,
            'student_model': student,
            'points': points,
            'color': get_model_color(data['teacher_model'])
        })
    logger.info(f"Fetched percentage data for {len(curves)} teachers")
    total_datapoints = sum(sum(p['count'] for p in curve['points']) for curve in curves)
    metrics = []
    overall_performance = []
    for curve in curves:
        teacher = curve['teacher_model']
        teacher_short = curve['teacher_short_name']
        points = curve['points']
        if len(points) >= 2:
            x_vals = np.array([p['step_bin_center'] for p in points], dtype=float)
            y_vals = np.array([p['mean_correct'] for p in points], dtype=float)
            x_mean = float(np.mean(x_vals))
            y_mean = float(np.mean(y_vals))
            denom = float(np.sum((x_vals - x_mean) ** 2))
            if denom > 0:
                slope = float(np.sum((x_vals - x_mean) * (y_vals - y_mean)) / denom)
                intercept = y_mean - slope * x_mean
                y_pred = slope * x_vals + intercept
                mse = float(np.mean((y_vals - y_pred) ** 2))
            else:
                slope = 0.0
                intercept = y_mean
                mse = 0.0
            masses = np.array([float(p.get('hazard_prob') or 0.0) for p in points], dtype=float)
            total = float(np.sum(masses))
            if total > 0 and len(masses) > 1:
                q = masses / total
                eps = 1e-12
                h = -float(np.sum(q * np.log(q + eps)))
                r2 = float(h / np.log(len(q)))
            else:
                r2 = 0.0
            auc = float(np.trapz(y_vals, x_vals) / (x_vals[-1] - x_vals[0])) if x_vals[-1] != x_vals[0] else 0.0
        else:
            slope = 0.0
            mse = 0.0
            r2 = 0.0
            auc = 0.0
        total_count = sum(p['count'] for p in points)
        mean_score = float(np.mean([p['mean_correct'] for p in points])) if points else 0.0
        metrics.append({
            'teacher_model': teacher,
            'teacher_short_name': teacher_short,
            'linear_slope': slope,
            'mse_linear': mse,
            'r2_linear': r2,
            'penalized_auc': auc
        })
        overall_performance.append({
            'teacher_model': teacher,
            'mean_accuracy': mean_score,
            'total_count': total_count
        })
    try:
        raw_regs = db.get_cached_pu_regressions(dataset_name, student_model)
    except Exception:
        raw_regs = []
    regression_stats = []
    for r in raw_regs:
        teacher = r.get('teacher_model')
        student_short = r.get('student_model')
        display_name = r.get('teacher_short_name')
        teacher_key = teacher
        if not student_model and student_short:
            teacher_key = f"{teacher}_{student_short}"
            display_name = f"{display_name} ({student_short})"
        regression_stats.append({
            'teacher_model': teacher_key,
            'teacher_short_name': display_name,
            'total_questions': int(r.get('total_questions') or 0),
            'regressed_questions': int(r.get('regressed_questions') or 0),
            'regression_percentage': float(r.get('regression_percentage') or 0.0),
            'total_regressions': int(r.get('total_regressions') or 0),
            'avg_regressions_per_question': float(r.get('avg_regressions_per_question') or 0.0),
        })
    return {
        'curves': curves,
        'x_label': '% Through Trace',
        'y_label': 'Transfer Utility (TU) Score',
        'student_models': list(student_names),
        'num_teachers': len(curves),
        'total_datapoints': total_datapoints,
        'metrics': metrics,
        'overall_performance': overall_performance,
        'regression_stats': regression_stats
    }
@app.route('/api/pedagogical-utility/<dataset_name>/samples')
def get_pu_samples_endpoint(dataset_name):
    """
    Get sample data for a specific teacher and step bin
    Uses database backend for efficient querying
    """
    teacher_model = request.args.get('teacher')
    student_model = request.args.get('student')
    step_bin = request.args.get('step_bin', type=float)
    num_samples = request.args.get('num_samples', default=5, type=int)
    mode = request.args.get('mode', 'percentage')
    if mode != 'percentage':
        return jsonify({'error': 'Only mode=percentage is supported'}), 400
    if not teacher_model:
        return jsonify({'error': 'teacher parameter required'}), 400
    logger.info(f"Fetching PU samples from database for {teacher_model} (student: {student_model}) in {dataset_name}, step_bin={step_bin}, mode={mode}, n={num_samples}")
    try:
        samples = db.get_pu_samples(
            dataset=dataset_name,
            teacher_model=teacher_model,
            student_model=student_model,
            step_bin=step_bin,
            mode=mode,
            limit=num_samples
        )
        logger.info(f"Retrieved {len(samples)} PU samples from database")
        student_model = samples[0].get('student_model', 'N/A') if samples else 'N/A'
        return jsonify({
            'teacher_model': teacher_model,
            'student_model': student_model,
            'step_bin': step_bin,
            'num_samples': len(samples),
            'samples': samples
        })
    except Exception as e:
        logger.error(f"Error fetching PU samples: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'message': 'Could not fetch PU samples from database',
            'samples': []
        }), 500
@app.route('/api/pedagogical-utility/<dataset_name>/svg')
def get_pu_svg(dataset_name):
    """Render and return an SVG of the pedagogical-utility curves for the dataset.

    Query params:
      - teachers: optional, comma-separated list of teacher_model identifiers (DB keys)
      - student: optional student model filter (passed to cache fetch)
      - width, height: optional int image size in pixels
    """
    teachers_param = request.args.get('teachers', '')
    student_model = request.args.get('student')
    width = request.args.get('width', default=1200, type=int)
    height = request.args.get('height', default=600, type=int)
    data = get_percentage_bins_from_db(dataset_name, student_model)
    if not data or not data.get('curves'):
        return jsonify({'error': 'Dataset not found or no PU data available'}), 404
    curves = data.get('curves', [])
    if teachers_param:
        try:
            raw_teachers = [unquote_plus(t) for t in teachers_param.split(',') if t]
            def _norm(s: Optional[str]) -> str:
                if not s:
                    return ''
                ns = str(s).lower()
                ns = ns.replace('/', '_').replace(' ', '_').replace('-', '_')
                ns = ns.replace('.', '_').replace(':', '_')
                return ns
            teacher_norm = set(_norm(t) for t in raw_teachers)
            def matches(curve):
                tm = curve.get('teacher_model') or ''
                short = tm.split('/')[-1] if '/' in tm else tm
                candidates = {tm, short, curve.get('teacher_short_name') or ''}
                for c in candidates:
                    if _norm(c) in teacher_norm:
                        return True
                for t in teacher_norm:
                    if t and t in _norm(short):
                        return True
                return False
            curves = [c for c in curves if matches(c)]
        except Exception:
            pass
    if not curves:
        return jsonify({'error': 'No matching teacher curves found'}), 404
    title = f'Pedagogical Utility - {dataset_name}'
    try:
        x_min = request.args.get('x_min', type=float)
        x_max = request.args.get('x_max', type=float)
        y_min = request.args.get('y_min', type=float)
        y_max = request.args.get('y_max', type=float)
        markers = request.args.get('markers', default='1') in ('1', 'true', 'True')
        grid = request.args.get('grid', default='1') in ('1', 'true', 'True')
        svg_bytes = _render_pedagogical_svg(
            curves=curves, title=title, width=width, height=height,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            markers=markers, grid=grid
        )
        return send_file(io.BytesIO(svg_bytes), mimetype='image/svg+xml', download_name=f'pu_{dataset_name}.svg', as_attachment=True)
    except Exception as e:
        logger.error(f"Failed to render PU svg: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to render SVG', 'detail': str(e)}), 500
@app.route('/api/correlations/metadata')
def correlations_metadata():
    data = get_correlation_results()
    if data is None:
        return jsonify({
            'available': False,
            'error': f"Missing correlation results JSON at {str(CORRELATIONS_JSON_PATH)}",
        }), 404
    datasets = sorted(list((data.get('conditions', {}).get('by_dataset', {}) or {}).keys()))
    models = sorted(list((data.get('conditions', {}).get('by_model', {}) or {}).keys()))
    return jsonify({
        'available': True,
        'generated_on': data.get('generated_on'),
        'n_rows': data.get('n_rows'),
        'metrics': data.get('metrics', []),
        'datasets': datasets,
        'models': models,
        'methods': ['pearson', 'spearman'],
        'correctness': ['all', 'correct_only', 'incorrect_only'],
        'p_kinds': ['p', 'p_holm'],
        'metric_presets': ['all', 'core', 'rm_pu', 'rm_only', 'pu_only'],
        'default': {
            'dataset': 'all',
            'model': 'all',
            'correctness': 'all',
            'method': 'pearson',
            'p_kind': 'p_holm',
            'alpha': 0.05,
            'metric_preset': 'all',
            'annotate': 1,
            'rim': 1,
        }
    })
@app.route('/api/correlations/heatmap.png')
def correlations_heatmap_png():
    data = get_correlation_results()
    if data is None:
        return jsonify({
            'error': f"Missing correlation results JSON at {str(CORRELATIONS_JSON_PATH)}",
        }), 404
    dataset = request.args.get('dataset', 'all')
    model = request.args.get('model', 'all')
    correctness = request.args.get('correctness', 'all')
    method = request.args.get('method', 'pearson')
    metric_preset = request.args.get('metric_preset', 'all')
    p_kind = request.args.get('p_kind', 'p_holm')
    alpha = float(request.args.get('alpha', '0.05'))
    annotate = request.args.get('annotate', '1') != '0'
    rim = request.args.get('rim', '1') != '0'
    if method not in {'pearson', 'spearman'}:
        return jsonify({'error': f"Unknown method: {method}"}), 400
    if p_kind not in {'p', 'p_holm'}:
        return jsonify({'error': f"Unknown p_kind: {p_kind}"}), 400
    try:
        cond_key, group_key = _corr_condition_key(dataset=dataset, model=model, correctness=correctness)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    cond = (data.get('conditions', {}) or {}).get(cond_key)
    if cond is None:
        return jsonify({'error': f"Condition not found: {cond_key}"}), 404
    group = (cond or {}).get(group_key)
    if group is None:
        return jsonify({'error': f"Group not found: {group_key}"}), 404
    all_metrics = data.get('metrics', [])
    try:
        metrics = _metrics_for_preset(all_metrics, metric_preset)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    pairs = group.get(method, [])
    corr, pvals = _build_square_matrices(pairs, metrics, p_kind)
    labels = [_metric_label(m) for m in metrics]
    title = (
        f"Metric Correlations ({method.title()}) | dataset={dataset} | model={model} | "
        f"correctness={correctness} | p={p_kind} | α={alpha}"
    )
    png = _render_correlation_heatmap_png(
        corr=corr,
        pvals=pvals,
        metric_labels=labels,
        title=title,
        alpha=alpha,
        annotate=annotate,
        rim_significant=rim,
    )
    return send_file(
        io.BytesIO(png),
        mimetype='image/png',
        as_attachment=False,
        download_name='correlation_heatmap.png',
        max_age=0,
    )
@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'version': 'database_backend',
        'timestamp': datetime.now().isoformat(),
        'database_exists': DB_PATH.exists(),
    })
if __name__ == '__main__':
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    logger.info("="*80)
    logger.info("Server Configuration:")
    logger.info(f"  Host: 0.0.0.0 (all interfaces)")
    logger.info(f"  Port: 8080")
    logger.info(f"  Hostname: {hostname}")
    logger.info(f"  IP Address: {ip_address}")
    logger.info("="*80)
    logger.info("Access URLs:")
    logger.info(f"  Local: http://localhost:8080")
    logger.info(f"  Network: http://{ip_address}:8080")
    logger.info("="*80)
    app.run(host='0.0.0.0', port=8080, debug=False)
