"""
Pedagogical Utility Analysis Script

Analyzes pedagogical utility experiment results and produces visualization plots.
Includes metrics for:
- Smoothness of reasoning progression
- Area Under Curve (AUC) with pedagogical penalties
- Correctness vs reasoning depth
- Student model comparisons

"""
import argparse
import pickle
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from collections import defaultdict
import os.path as osp
import json
sns.set_style("whitegrid", {
    'grid.linestyle': ':',
    'grid.linewidth': 0.5,
    'grid.color': '#E5E5E5'
})
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 16
COLORS = ['#FF6B35', '#004E89', '#1B998B', '#C5A880', '#9B59B6', '#E74C3C', '#3498DB']
def load_pedagogical_data(pattern, load_full_data=False):
    """
    Load pedagogical utility data from pickle files matching pattern.
    
    Args:
        pattern: Glob pattern for input files
        load_full_data: If True, also return the full pickle data for detailed analysis
    
    Returns:
        If load_full_data=False: pd.DataFrame with basic columns
        If load_full_data=True: (pd.DataFrame, dict) where dict maps teacher_model to full pickle data
    """
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No files found matching pattern: {pattern}")
    print(f"Found {len(files)} result files")
    all_data = []
    full_pickle_data = {}
    for fpath in files:
        print(f"  Loading: {osp.basename(fpath)}")
        with open(fpath, 'rb') as f:
            results = pickle.load(f)
        metadata = results['metadata']
        student_model = metadata['student_model']
        teacher_model = metadata['teacher_model']
        dataset = metadata['dataset_name']
        if load_full_data:
            full_pickle_data[teacher_model] = results
        for dp in results['data']:
            row = {
                'student_model': student_model,
                'teacher_model': teacher_model,
                'dataset': dataset,
                'index': dp.get('index'),
                'num_steps': dp.get('num_steps'),
                'score': dp.get('score'),
                'extracted_answer': dp.get('extracted_answer'),
                'grading_error': dp.get('grading_error'),
                'ran_second_pass': dp.get('ran_second_pass', False),
            }
            all_data.append(row)
    df = pd.DataFrame(all_data)
    print(f"\nLoaded {len(df)} datapoints")
    print(f"  Student models: {df['student_model'].nunique()}")
    print(f"  Teacher models: {df['teacher_model'].nunique()}")
    if load_full_data:
        return df, full_pickle_data
    return df
def compute_smoothness_metrics(df_student, bin_col='step_bin_center', value_col='mean_correct'):
    """
    Compute smoothness metrics for a student model's pedagogical utility curve.
    
    Metrics:
    1. MSE to linear fit (lower = smoother, more linear progression)
    2. MSE to logarithmic fit (lower = smoother log progression)
    3. Total variation (sum of absolute differences between consecutive points)
    4. Normalized total variation (total variation / data range)
    
    Args:
        df_student: DataFrame with binned data for one student model
        bin_col: Column name for x-axis (step bins)
        value_col: Column name for y-axis (correctness)
    
    Returns:
        dict of smoothness metrics
    """
    data = df_student.copy()
    data[bin_col] = pd.to_numeric(data[bin_col], errors='coerce')
    data[value_col] = pd.to_numeric(data[value_col], errors='coerce')
    data = data.dropna(subset=[bin_col, value_col])
    data = data.sort_values(bin_col)
    x = data[bin_col].to_numpy(dtype=float)
    y = data[value_col].to_numpy(dtype=float)
    if len(x) < 3:
        return {
            'mse_linear': np.nan,
            'mse_log': np.nan,
            'total_variation': np.nan,
            'normalized_variation': np.nan,
            'r2_linear': np.nan,
            'r2_log': np.nan,
        }
    linear_coeffs = np.polyfit(x, y, 1)
    linear_slope = linear_coeffs[0]
    linear_intercept = linear_coeffs[1]
    linear_fit = np.polyval(linear_coeffs, x)
    mse_linear = np.mean((y - linear_fit) ** 2)
    ss_res_linear = np.sum((y - linear_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_linear = 1 - (ss_res_linear / ss_tot) if ss_tot > 0 else 0
    try:
        def log_func(x, a, b):
            return a * np.log(x + 1) + b
        popt, _ = curve_fit(log_func, x, y, p0=[0.1, 0.1], maxfev=10000)
        log_fit = log_func(x, *popt)
        mse_log = np.mean((y - log_fit) ** 2)
        ss_res_log = np.sum((y - log_fit) ** 2)
        r2_log = 1 - (ss_res_log / ss_tot) if ss_tot > 0 else 0
    except (RuntimeError, ValueError):
        mse_log = np.nan
        r2_log = np.nan
    diffs = np.abs(np.diff(y))
    total_variation = np.sum(diffs)
    y_range = y.max() - y.min()
    normalized_variation = total_variation / y_range if y_range > 0 else 0
    return {
        'mse_linear': mse_linear,
        'mse_log': mse_log,
        'total_variation': total_variation,
        'normalized_variation': normalized_variation,
        'r2_linear': r2_linear,
        'r2_log': r2_log,
        'linear_slope': linear_slope,
        'linear_intercept': linear_intercept,
        'num_points': len(x),
    }
def compute_auc_with_penalties(df_student, bin_col='step_bin_center', value_col='mean_correct',
                                length_weight=0.1, concentration_weight=0.1, smoothness_weight=0.1):
    """
    Compute Area Under Curve with pedagogical penalties.
    
    Penalties:
    - Length: penalizes models requiring more reasoning steps
    - Concentration: penalizes dispersed/inconsistent step distributions  
    - Smoothness: penalizes noisy, non-monotonic curves
    
    Args:
        df_student: DataFrame for one student model
        bin_col: X-axis column (reasoning steps)
        value_col: Y-axis column (correctness probability)
        length_weight, concentration_weight, smoothness_weight: Penalty weights
    
    Returns:
        dict of AUC metrics
    """
    data = df_student.copy()
    data[bin_col] = pd.to_numeric(data[bin_col], errors='coerce')
    data[value_col] = pd.to_numeric(data[value_col], errors='coerce')
    data = data.dropna(subset=[bin_col, value_col])
    data = data.sort_values(bin_col)
    x = data[bin_col].to_numpy(dtype=float)
    y = data[value_col].to_numpy(dtype=float)
    if len(x) < 2:
        return {
            'raw_auc': np.nan,
            'normalized_auc': np.nan,
            'length_penalty': np.nan,
            'concentration_penalty': np.nan,
            'smoothness_penalty': np.nan,
            'total_penalties': np.nan,
            'penalized_auc': np.nan,
        }
    raw_auc = integrate.trapezoid(y, x)
    x_range = x.max() - x.min()
    normalized_auc = raw_auc / x_range if x_range > 0 else raw_auc
    mean_steps = x.mean()
    step_std = x.std()
    if len(y) > 1:
        diffs = np.abs(np.diff(y))
        smoothness_penalty_raw = np.mean(diffs)
    else:
        smoothness_penalty_raw = 0
    return {
        'raw_auc': raw_auc,
        'normalized_auc': normalized_auc,
        'mean_steps': mean_steps,
        'step_std': step_std,
        'smoothness_raw': smoothness_penalty_raw,
        'length_penalty': np.nan,
        'concentration_penalty': np.nan,
        'smoothness_penalty': np.nan,
        'total_penalties': np.nan,
        'penalized_auc': np.nan,
    }
def plot_pedagogical_curves(df, output_dir, dataset_name):
    """
    Create 2x2 grid of pedagogical utility curves:
    - Raw steps (log & linear)
    - Percentage of steps (log & linear)
    
    Plots are grouped by TEACHER model (different reasoning sources)
    for the SAME student model.
    """
    bin_size = 2
    bins = np.arange(0, df['num_steps'].max() + bin_size, bin_size)
    df['step_bin'] = pd.cut(df['num_steps'], bins, right=False)
    df['step_bin_center'] = df['step_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan).astype(float)
    max_steps_per_teacher = df.groupby('teacher_model')['num_steps'].transform('max')
    df['step_percentage'] = (df['num_steps'] / max_steps_per_teacher) * 100
    percent_bin_size = 2
    percent_bins = np.arange(0, 102, percent_bin_size)
    df['percent_bin'] = pd.cut(df['step_percentage'], percent_bins, right=False)
    df['percent_bin_center'] = df['percent_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan).astype(float)
    grouped_binned = df.groupby(['teacher_model', 'step_bin_center']).agg({
        'score': ['mean', 'std', 'count']
    }).reset_index()
    grouped_binned.columns = ['teacher_model', 'step_bin_center', 'mean_correct', 'std_correct', 'count']
    grouped_binned = grouped_binned[grouped_binned['count'] >= 10]
    grouped_percent = df.groupby(['teacher_model', 'percent_bin_center']).agg({
        'score': ['mean', 'std', 'count']
    }).reset_index()
    grouped_percent.columns = ['teacher_model', 'percent_bin_center', 'mean_correct', 'std_correct', 'count']
    grouped_percent = grouped_percent[grouped_percent['count'] >= 10]
    all_counts = grouped_binned['count'].values
    min_count, max_count = all_counts.min(), all_counts.max()
    min_linewidth, max_linewidth = 1.0, 4.5
    teachers = grouped_binned['teacher_model'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    def plot_on_axis(ax, grouped_data, x_col, y_col, x_scale, x_label, title, x_max=None):
        from matplotlib.ticker import LogLocator, ScalarFormatter, MaxNLocator
        for i, teacher in enumerate(teachers):
            data = grouped_data[grouped_data['teacher_model'] == teacher].copy()
            data = data.sort_values(x_col)
            color = COLORS[i % len(COLORS)]
            data[x_col] = pd.to_numeric(data[x_col], errors='coerce')
            data[y_col] = pd.to_numeric(data[y_col], errors='coerce')
            data['std_correct'] = pd.to_numeric(data.get('std_correct', pd.Series()), errors='coerce')
            data['count'] = pd.to_numeric(data.get('count', pd.Series()), errors='coerce')
            data = data.dropna(subset=[x_col, y_col])
            if len(data) < 2:
                continue
            cur_min = data['count'].min()
            cur_max = data['count'].max()
            if cur_max == cur_min:
                data['linewidth'] = (min_linewidth + max_linewidth) / 2.0
            else:
                data['linewidth'] = min_linewidth + (max_linewidth - min_linewidth) * \
                                    (data['count'] - cur_min) / (cur_max - cur_min)
            x_vals = data[x_col].to_numpy(dtype=float)
            y_vals = data[y_col].to_numpy(dtype=float)
            lw_vals = data['linewidth'].to_numpy()
            for j in range(len(x_vals) - 1):
                ax.plot(x_vals[j:j+2],
                       y_vals[j:j+2],
                       linewidth=float(lw_vals[j]),
                       color=color, alpha=0.9)
            teacher_label = teacher.split('/')[-1] if '/' in teacher else teacher
            ax.scatter(x_vals, y_vals,
                      s=60, color=color, label=teacher_label, zorder=5,
                      edgecolors='white', linewidths=0.5)
            counts = data['count'].to_numpy(dtype=float)
            stds = data['std_correct'].to_numpy(dtype=float)
            stderr = stds / np.sqrt(np.clip(counts, a_min=1, a_max=None))
            lower = (y_vals - stderr)
            upper = (y_vals + stderr)
            lower = np.asarray(lower, dtype=float)
            upper = np.asarray(upper, dtype=float)
            ax.fill_between(x_vals, lower, upper, alpha=0.15, color=color)
        ax.set_xscale(x_scale)
        if x_scale == 'log':
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.set_xlabel(x_label, fontweight='500')
        ax.set_ylabel('Probability of Correct Answer', fontweight='500')
        ax.set_title(title, fontweight='600', pad=15)
        ax.legend(title='Teacher Model', title_fontsize=10, frameon=True,
                 fancybox=True, shadow=True, loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
        if x_max is not None:
            ax.set_xlim([0, x_max])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plot_on_axis(axes[0, 0], grouped_binned, 'step_bin_center', 'mean_correct', 'log',
                'Number of Reasoning Steps', 'Raw Steps (Log Scale)')
    plot_on_axis(axes[0, 1], grouped_binned, 'step_bin_center', 'mean_correct', 'linear',
                'Number of Reasoning Steps', 'Raw Steps (Linear Scale)')
    plot_on_axis(axes[1, 0], grouped_percent, 'percent_bin_center', 'mean_correct', 'log',
                'Percentage of Steps (%)', 'Percentage of Steps (Log Scale)', x_max=100)
    plot_on_axis(axes[1, 1], grouped_percent, 'percent_bin_center', 'mean_correct', 'linear',
                'Percentage of Steps (%)', 'Percentage of Steps (Linear Scale)', x_max=100)
    plt.tight_layout()
    output_path = osp.join(output_dir, f'{dataset_name}_pedagogical_utility_2x2.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    return grouped_binned, grouped_percent
def plot_sample_count_vs_correctness(df, grouped_binned, output_dir, dataset_name):
    """Plot correctness vs number of samples."""
    teachers = grouped_binned['teacher_model'].unique()
    fig, ax = plt.subplots(figsize=(12, 7))
    all_counts = grouped_binned['count'].values
    min_count, max_count = all_counts.min(), all_counts.max()
    min_linewidth, max_linewidth = 1.0, 4.5
    for i, teacher in enumerate(teachers):
        data = grouped_binned[grouped_binned['teacher_model'] == teacher].copy()
        data = data.sort_values('count')
        color = COLORS[i % len(COLORS)]
        data['count'] = pd.to_numeric(data['count'], errors='coerce')
        data['mean_correct'] = pd.to_numeric(data['mean_correct'], errors='coerce')
        data['std_correct'] = pd.to_numeric(data['std_correct'], errors='coerce')
        data = data.dropna(subset=['count', 'mean_correct'])
        if len(data) < 2:
            continue
        cur_min = data['count'].min()
        cur_max = data['count'].max()
        if cur_max == cur_min:
            data['linewidth'] = (min_linewidth + max_linewidth) / 2.0
        else:
            data['linewidth'] = min_linewidth + (max_linewidth - min_linewidth) * \
                                (data['count'] - cur_min) / (cur_max - cur_min)
        count_vals = data['count'].to_numpy(dtype=float)
        mean_vals = data['mean_correct'].to_numpy(dtype=float)
        std_vals = data['std_correct'].to_numpy(dtype=float)
        lw_vals = data['linewidth'].to_numpy()
        for j in range(len(count_vals) - 1):
            ax.plot(count_vals[j:j+2],
                   mean_vals[j:j+2],
                   linewidth=float(lw_vals[j]),
                   color=color, alpha=0.9)
        teacher_label = teacher.split('/')[-1] if '/' in teacher else teacher
        ax.scatter(count_vals, mean_vals,
                  s=60, color=color, label=teacher_label, zorder=5,
                  edgecolors='white', linewidths=0.5)
        stderr = std_vals / np.sqrt(np.clip(count_vals, a_min=1, a_max=None))
        lower = np.asarray(mean_vals - stderr, dtype=float)
        upper = np.asarray(mean_vals + stderr, dtype=float)
        ax.fill_between(count_vals, lower, upper, alpha=0.15, color=color)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.set_xlabel('Number of Samples', fontweight='500')
    ax.set_ylabel('Probability of Correct Answer', fontweight='500')
    ax.set_title('Correctness vs. Sample Count', fontweight='600', pad=20)
    ax.legend(title='Teacher Model', title_fontsize=11, frameon=True,
             fancybox=True, shadow=True, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    output_path = osp.join(output_dir, f'{dataset_name}_sample_count_vs_correctness.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
def compute_all_metrics(df, grouped_binned):
    """
    Compute comprehensive metrics for all teacher models.
    
    Returns:
        pd.DataFrame with metrics per teacher
    """
    teachers = df['teacher_model'].unique()
    metrics_list = []
    for teacher in teachers:
        teacher_data = grouped_binned[grouped_binned['teacher_model'] == teacher]
        auc_metrics = compute_auc_with_penalties(teacher_data)
        smoothness_metrics = compute_smoothness_metrics(teacher_data)
        teacher_df = df[df['teacher_model'] == teacher]
        overall_acc = teacher_df['score'].mean()
        total_samples = len(teacher_df)
        j0_data = teacher_df[teacher_df['num_steps'] == 0]
        j0_acc = j0_data['score'].mean() if len(j0_data) > 0 else np.nan
        metrics = {
            'teacher_model': teacher,
            'overall_accuracy': overall_acc,
            'j0_accuracy': j0_acc,
            'total_samples': total_samples,
            **auc_metrics,
            **smoothness_metrics,
        }
        metrics_list.append(metrics)
    metrics_df = pd.DataFrame(metrics_list)
    if len(metrics_df) > 1:
        max_mean_steps = metrics_df['mean_steps'].max()
        max_step_std = metrics_df['step_std'].max()
        for idx in metrics_df.index:
            mean_steps = metrics_df.loc[idx, 'mean_steps']
            step_std = metrics_df.loc[idx, 'step_std']
            smoothness_raw = metrics_df.loc[idx, 'smoothness_raw']
            length_penalty = 0.1 * (mean_steps / max_mean_steps) if max_mean_steps > 0 else 0
            concentration_penalty = 0.1 * (step_std / max_step_std) if max_step_std > 0 else 0
            smoothness_penalty = 0.1 * smoothness_raw
            total_penalties = length_penalty + concentration_penalty + smoothness_penalty
            normalized_auc = metrics_df.loc[idx, 'normalized_auc']
            penalized_auc = normalized_auc - total_penalties
            metrics_df.loc[idx, 'length_penalty'] = length_penalty
            metrics_df.loc[idx, 'concentration_penalty'] = concentration_penalty
            metrics_df.loc[idx, 'smoothness_penalty'] = smoothness_penalty
            metrics_df.loc[idx, 'total_penalties'] = total_penalties
            metrics_df.loc[idx, 'penalized_auc'] = penalized_auc
    return metrics_df
def print_summary(df, metrics_df):
    """Print summary statistics and tables."""
    print("\n" + "="*80)
    print("PEDAGOGICAL UTILITY ANALYSIS SUMMARY")
    print("="*80)
    students = df['student_model'].unique()
    print(f"\n=== Student Model ===")
    for student in students:
        print(f"  {student}")
    print("\n=== Overall Performance by Teacher Model ===")
    summary = df.groupby('teacher_model')['score'].agg(['mean', 'std', 'count'])
    print(summary.round(4))
    print("\n=== Smoothness Metrics ===")
    print("Lower MSE = smoother, more predictable progression")
    smoothness_cols = ['teacher_model', 'mse_linear', 'mse_log', 'r2_linear', 'r2_log',
                       'total_variation', 'normalized_variation']
    print(metrics_df[smoothness_cols].round(4).to_string(index=False))
    print("\n=== AUC Metrics with Pedagogical Penalties ===")
    auc_cols = ['teacher_model', 'normalized_auc', 'length_penalty', 'concentration_penalty',
                'smoothness_penalty', 'total_penalties', 'penalized_auc']
    print(metrics_df[auc_cols].round(4).to_string(index=False))
    print("\n=== Rankings ===")
    print("\nBy Penalized AUC (higher is better):")
    ranking_auc = metrics_df[['teacher_model', 'penalized_auc']].sort_values('penalized_auc', ascending=False)
    for idx, row in ranking_auc.iterrows():
        teacher_name = row['teacher_model'].split('/')[-1] if '/' in row['teacher_model'] else row['teacher_model']
        print(f"  {teacher_name}: {row['penalized_auc']:.4f}")
    print("\nBy Smoothness (lower MSE to linear fit is better):")
    ranking_smooth = metrics_df[['teacher_model', 'mse_linear']].sort_values('mse_linear')
    for idx, row in ranking_smooth.iterrows():
        teacher_name = row['teacher_model'].split('/')[-1] if '/' in row['teacher_model'] else row['teacher_model']
        print(f"  {teacher_name}: {row['mse_linear']:.4f}")
    print("\nBy R² to linear fit (higher is better - more linear progression):")
    ranking_r2 = metrics_df[['teacher_model', 'r2_linear']].sort_values('r2_linear', ascending=False)
    for idx, row in ranking_r2.iterrows():
        teacher_name = row['teacher_model'].split('/')[-1] if '/' in row['teacher_model'] else row['teacher_model']
        print(f"  {teacher_name}: {row['r2_linear']:.4f}")
def compute_regression_stats(full_pickle_data):
    """
    Compute regression statistics: cases where student gets correct answer at some point,
    then fails on later steps.
    
    Args:
        full_pickle_data: dict mapping teacher_model to full pickle results
    
    Returns:
        dict mapping teacher_model to regression stats
    """
    regression_stats = {}
    for teacher_model, results in full_pickle_data.items():
        question_data = defaultdict(list)
        for dp in results['data']:
            question_data[dp['index']].append({
                'num_steps': dp['num_steps'],
                'score': dp['score']
            })
        total_questions = len(question_data)
        regressed_questions = 0
        total_regressions = 0
        for q_idx, datapoints in question_data.items():
            datapoints.sort(key=lambda x: x['num_steps'])
            got_correct = False
            regressed_this_question = False
            for i, dp in enumerate(datapoints):
                if dp['score'] == 1:
                    got_correct = True
                elif got_correct and dp['score'] == 0:
                    if not regressed_this_question:
                        regressed_questions += 1
                        regressed_this_question = True
                    total_regressions += 1
        regression_stats[teacher_model] = {
            'total_questions': total_questions,
            'regressed_questions': regressed_questions,
            'regression_percentage': (regressed_questions / total_questions * 100) if total_questions > 0 else 0,
            'total_regressions': total_regressions,
            'avg_regressions_per_question': (total_regressions / total_questions) if total_questions > 0 else 0
        }
    return regression_stats
def smooth_curve(x, y, sigma=1.5):
    """
    Apply Gaussian smoothing to a curve for visualization.
    Returns smoothed y values.
    """
    if len(x) < 3:
        return y
    return gaussian_filter1d(y, sigma=sigma)
def generate_json_output(df, grouped_binned, metrics_df, regression_stats, output_dir, dataset_name, original_data_files):
    """
    Generate JSON output for the web dashboard.
    Includes:
    - Smoothed curves for visualization
    - Raw aggregated data points
    - Metrics for each teacher
    - Regression statistics
    - Original data for sample viewing
    """
    print("\nGenerating JSON output for dashboard...")
    teachers = grouped_binned['teacher_model'].unique()
    curves_data = []
    for teacher in teachers:
        teacher_data = grouped_binned[grouped_binned['teacher_model'] == teacher].copy()
        teacher_data['step_bin_center'] = pd.to_numeric(teacher_data['step_bin_center'], errors='coerce')
        teacher_data['mean_correct'] = pd.to_numeric(teacher_data['mean_correct'], errors='coerce')
        teacher_data['std_correct'] = pd.to_numeric(teacher_data['std_correct'], errors='coerce')
        teacher_data['count'] = pd.to_numeric(teacher_data['count'], errors='coerce')
        teacher_data = teacher_data.dropna(subset=['step_bin_center', 'mean_correct'])
        teacher_data = teacher_data.sort_values('step_bin_center')
        if len(teacher_data) < 2:
            continue
        x = teacher_data['step_bin_center'].to_numpy(dtype=float)
        y = teacher_data['mean_correct'].to_numpy(dtype=float)
        std = teacher_data['std_correct'].to_numpy(dtype=float)
        counts = teacher_data['count'].to_numpy(dtype=float)
        y_smooth = smooth_curve(x, y, sigma=1.5)
        curves_data.append({
            'teacher_model': teacher,
            'teacher_short_name': teacher.split('/')[-1] if '/' in teacher else teacher,
            'points': [
                {
                    'step_bin_center': float(x[i]),
                    'mean_correct': float(y[i]),
                    'mean_correct_smooth': float(y_smooth[i]),
                    'std_correct': float(std[i]) if not np.isnan(std[i]) else 0.0,
                    'count': int(counts[i])
                }
                for i in range(len(x))
            ]
        })
    metrics_data = []
    for _, row in metrics_df.iterrows():
        metrics_data.append({
            'teacher_model': row['teacher_model'],
            'teacher_short_name': row['teacher_model'].split('/')[-1] if '/' in row['teacher_model'] else row['teacher_model'],
            'mse_linear': float(row['mse_linear']),
            'mse_log': float(row['mse_log']),
            'r2_linear': float(row['r2_linear']),
            'r2_log': float(row['r2_log']),
            'linear_slope': float(row['linear_slope']),
            'linear_intercept': float(row['linear_intercept']),
            'total_variation': float(row['total_variation']),
            'normalized_variation': float(row['normalized_variation']),
            'normalized_auc': float(row['normalized_auc']),
            'length_penalty': float(row['length_penalty']),
            'concentration_penalty': float(row['concentration_penalty']),
            'smoothness_penalty': float(row['smoothness_penalty']),
            'total_penalties': float(row['total_penalties']),
            'penalized_auc': float(row['penalized_auc'])
        })
    students = df['student_model'].unique()
    student_model = students[0] if len(students) > 0 else "unknown"
    overall_perf = df.groupby('teacher_model')['score'].agg(['mean', 'std', 'count']).reset_index()
    overall_perf_data = [
        {
            'teacher_model': row['teacher_model'],
            'teacher_short_name': row['teacher_model'].split('/')[-1] if '/' in row['teacher_model'] else row['teacher_model'],
            'mean_accuracy': float(row['mean']),
            'std_accuracy': float(row['std']),
            'total_count': int(row['count'])
        }
        for _, row in overall_perf.iterrows()
    ]
    regression_data = [
        {
            'teacher_model': teacher,
            'teacher_short_name': teacher.split('/')[-1] if '/' in teacher else teacher,
            'total_questions': regression_stats[teacher]['total_questions'],
            'regressed_questions': regression_stats[teacher]['regressed_questions'],
            'regression_percentage': regression_stats[teacher]['regression_percentage'],
            'total_regressions': regression_stats[teacher]['total_regressions'],
            'avg_regressions_per_question': regression_stats[teacher]['avg_regressions_per_question']
        }
        for teacher in regression_stats.keys()
    ]
    output_data = {
        'dataset': dataset_name,
        'student_model': student_model,
        'student_short_name': student_model.split('/')[-1] if '/' in student_model else student_model,
        'num_teachers': len(teachers),
        'total_datapoints': len(df),
        'curves': curves_data,
        'metrics': metrics_data,
        'overall_performance': overall_perf_data,
        'regression_stats': regression_data,
        'data_files': original_data_files
    }
    json_path = osp.join(output_dir, f'{dataset_name}_dashboard.json')
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved dashboard JSON to: {json_path}")
    sample_info = {
        'data_files': original_data_files,
        'note': 'Load original pickle files for detailed sample data'
    }
    samples_path = osp.join(output_dir, f'{dataset_name}_sample_info.json')
    with open(samples_path, 'w') as f:
        json.dump(sample_info, f, indent=2)
    print(f"Saved sample info to: {samples_path}")
    return output_data
def main():
    parser = argparse.ArgumentParser(description="Analyze pedagogical utility experiments")
    parser.add_argument('--input_pattern', type=str, required=True,
                       help='Glob pattern for input pickle files (e.g., "outputs/pu_debug/math/*/pu_*.pkl")')
    parser.add_argument('--output_dir', type=str, default='outputs/pu_analysis',
                       help='Output directory for plots and results')
    parser.add_argument('--dataset', type=str, default='math',
                       help='Dataset name for output files')
    parser.add_argument('--save_csv', action='store_true',
                       help='Save metrics to CSV file')
    parser.add_argument('--save_json', action='store_true', default=True,
                       help='Save JSON output for dashboard (default: True)')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading pedagogical utility data...")
    df, full_pickle_data = load_pedagogical_data(args.input_pattern, load_full_data=True)
    original_files = glob.glob(args.input_pattern)
    print("\nPreparing binned data for metrics...")
    bin_size = 2
    bins = np.arange(0, df['num_steps'].max() + bin_size, bin_size)
    df['step_bin'] = pd.cut(df['num_steps'], bins, right=False)
    df['step_bin_center'] = df['step_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan).astype(float)
    grouped_binned = df.groupby(['teacher_model', 'step_bin_center']).agg({
        'score': ['mean', 'std', 'count']
    }).reset_index()
    grouped_binned.columns = ['teacher_model', 'step_bin_center', 'mean_correct', 'std_correct', 'count']
    grouped_binned = grouped_binned[grouped_binned['count'] >= 10]
    print("\nComputing comprehensive metrics...")
    metrics_df = compute_all_metrics(df, grouped_binned)
    print("\nComputing regression statistics...")
    regression_stats = compute_regression_stats(full_pickle_data)
    print_summary(df, metrics_df)
    print("\n=== Regression Analysis ===")
    print("Questions where student got correct answer then failed on later steps:")
    for teacher, stats in regression_stats.items():
        teacher_name = teacher.split('/')[-1] if '/' in teacher else teacher
        print(f"  {teacher_name}: {stats['regressed_questions']}/{stats['total_questions']} ({stats['regression_percentage']:.1f}%)")
    if args.save_csv:
        csv_path = osp.join(args.output_dir, f'{args.dataset}_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        print(f"\nSaved metrics to: {csv_path}")
    if args.save_json:
        generate_json_output(df, grouped_binned, metrics_df, regression_stats, args.output_dir, args.dataset, original_files)
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
if __name__ == "__main__":
    import os
    main()
