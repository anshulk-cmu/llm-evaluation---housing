# -*- coding: utf-8 -*-
"""
Step 9: Validate with Logistic Regression Comparison (CPU-only)

Compares the LLM's behavioral feature weights (from flip experiments) against
the logistic regression model trained on raw feature differences.
Answers: is the LLM doing implicit linear regression, or something more complex?

Also compares against probing accuracy rankings, giving three independent
signals about feature importance.

Inputs:
  - macro_mapping/data/exchange_rate_comparison_{model}.csv (Step 6)
  - macro_mapping/data/selected_samples_{model}.csv         (Step 1)
  - data/raw_feature_logit_results/raw_feature_logit_summary.json
  - data/probe_results/probe_best_layers_{model}_{condition}.csv

Outputs:
  - macro_mapping/outputs/tables/regression_comparison_{model}.csv
  - macro_mapping/outputs/graphs/weight_comparison_{model}.png
  - macro_mapping/outputs/tables/linearity_verdict_{model}.txt

Usage:
    python macro_mapping/validate_regression.py --model llama-3.2-3b
    python macro_mapping/validate_regression.py --model qwen3-4b
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mm_config import TARGET_FEATURES, CONDITION, setup_logger

logger = setup_logger("Step9_ValidateRegression")

# ============================================================================
# PATHS
# ============================================================================

MM_DATA_DIR = Path(__file__).resolve().parent / "data"
MM_GRAPHS_DIR = Path(__file__).resolve().parent / "outputs" / "graphs"
MM_TABLES_DIR = Path(__file__).resolve().parent / "outputs" / "tables"
MM_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
MM_TABLES_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBE_DIR = PROJECT_ROOT / "data" / "probe_results"
LOGIT_DIR = PROJECT_ROOT / "data" / "raw_feature_logit_results"

# Feature name mappings
REGRESSION_COEFF_MAP = {
    'bathrooms_diff': 'bathrooms',
    'bedrooms_diff': 'bedrooms',
    'lot_diff': 'lot',
    'year_diff': 'yearBuilt',
}

PROBE_FEATURE_MAP = {
    'bathrooms_p1_more': 'bathrooms',
    'bedrooms_p1_more': 'bedrooms',
    'lot_p1_larger': 'lot',
    'year_p1_newer': 'yearBuilt',
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_behavioral_weights(model: str) -> pd.DataFrame:
    """
    Load behavioral weights from Step 6 exchange rate comparison.
    Returns DataFrame with feature, behavioral_weight_correct, _incorrect, _all.
    """
    comparison_file = MM_DATA_DIR / f"exchange_rate_comparison_{model}.csv"
    if not comparison_file.exists():
        logger.error(f" {comparison_file} not found. Run Step 6 first.")
        sys.exit(1)

    comp_df = pd.read_csv(comparison_file)

    # Ensure normalized weights exist
    for group in ['correct', 'incorrect']:
        col = f'behavioral_weight_{group}'
        norm_col = f'{col}_norm'
        if norm_col not in comp_df.columns:
            total = comp_df[col].sum()
            if total > 0:
                comp_df[norm_col] = comp_df[col] / total

    return comp_df


def load_regression_coefficients() -> dict:
    """Load logistic regression coefficients from existing results."""
    logit_file = LOGIT_DIR / "raw_feature_logit_summary.json"
    if not logit_file.exists():
        logger.warning(f" {logit_file} not found. Regression comparison skipped.")
        return {}

    with open(logit_file) as f:
        summary = json.load(f)

    coeffs = summary.get('coefficients', {})

    # Map to our feature names and take absolute values
    mapped = {}
    for coeff_name, feat_name in REGRESSION_COEFF_MAP.items():
        if coeff_name in coeffs:
            mapped[feat_name] = abs(coeffs[coeff_name])

    # Normalize to sum to 1
    total = sum(mapped.values())
    if total > 0:
        mapped = {k: v / total for k, v in mapped.items()}

    return mapped


def load_probe_accuracies(model: str) -> dict:
    """Load probing accuracies for each feature."""
    probe_file = PROBE_DIR / f"probe_best_layers_{model}_{CONDITION}.csv"
    if not probe_file.exists():
        logger.warning(f" {probe_file} not found. Probe comparison skipped.")
        return {}

    probe_df = pd.read_csv(probe_file)
    mapped = {}
    for _, row in probe_df.iterrows():
        feat_name = PROBE_FEATURE_MAP.get(row['feature'])
        if feat_name:
            mapped[feat_name] = row['test_accuracy']

    return mapped


# ============================================================================
# COMPARISON
# ============================================================================

def build_comparison_table(comp_df: pd.DataFrame,
                           regression_weights: dict,
                           probe_accuracies: dict) -> pd.DataFrame:
    """
    Build the master comparison table.

    Columns: feature, bw_correct, bw_incorrect, regression_weight,
             probe_accuracy, rank_bw_correct, rank_regression, rank_match
    """
    rows = []
    for feat in TARGET_FEATURES:
        row_data = {'feature': feat}

        # Behavioral weights from flip experiments
        feat_row = comp_df[comp_df['feature'] == feat]
        if len(feat_row) > 0:
            fr = feat_row.iloc[0]
            row_data['bw_correct'] = fr.get('behavioral_weight_correct_norm', np.nan)
            row_data['bw_incorrect'] = fr.get('behavioral_weight_incorrect_norm', np.nan)
            row_data['mean_flip_correct'] = fr.get('mean_flip_correct', np.nan)
            row_data['mean_flip_incorrect'] = fr.get('mean_flip_incorrect', np.nan)
            row_data['divergence'] = fr.get('divergence', np.nan)
        else:
            row_data['bw_correct'] = np.nan
            row_data['bw_incorrect'] = np.nan

        # Regression coefficients
        row_data['regression_weight'] = regression_weights.get(feat, np.nan)

        # Probe accuracy
        row_data['probe_accuracy'] = probe_accuracies.get(feat, np.nan)

        rows.append(row_data)

    df = pd.DataFrame(rows)

    # Compute ranks (1 = most important)
    for col, rank_col in [('bw_correct', 'rank_bw_correct'),
                           ('bw_incorrect', 'rank_bw_incorrect'),
                           ('regression_weight', 'rank_regression'),
                           ('probe_accuracy', 'rank_probe')]:
        if col in df.columns and df[col].notna().any():
            df[rank_col] = df[col].rank(ascending=False, method='min').astype(int)

    # Check rank matches
    if 'rank_bw_correct' in df.columns and 'rank_regression' in df.columns:
        df['rank_match_bw_vs_reg'] = df['rank_bw_correct'] == df['rank_regression']

    return df


def compute_correlations(comparison_df: pd.DataFrame) -> dict:
    """
    Compute Spearman rank correlations between all pairs of weight signals.

    NOTE: With n=4 features, Spearman rho is descriptive only — not
    statistically testable (rho=1.0 barely reaches p=0.08 with n=4).
    We report rho for completeness but emphasize rank agreement counts.
    """
    results = {}
    weight_cols = {
        'bw_correct': 'LLM behavioral (correct)',
        'bw_incorrect': 'LLM behavioral (incorrect)',
        'regression_weight': 'Logistic regression',
        'probe_accuracy': 'Probe accuracy',
    }

    available = {k: v for k, v in weight_cols.items()
                 if k in comparison_df.columns and comparison_df[k].notna().all()}

    cols = list(available.keys())
    n = len(comparison_df)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = comparison_df[cols[i]].values
            b = comparison_df[cols[j]].values
            if len(a) >= 3:
                rho, _ = stats.spearmanr(a, b)
                # Count how many rank positions match
                rank_a = stats.rankdata(-a)  # descending
                rank_b = stats.rankdata(-b)
                rank_matches = int((rank_a == rank_b).sum())
                # Does top-1 feature agree?
                top1_match = (np.argmax(a) == np.argmax(b))
                key = f"{available[cols[i]]} vs {available[cols[j]]}"
                results[key] = {
                    'rho': rho,
                    'rank_matches': rank_matches,
                    'n_features': n,
                    'top1_match': top1_match,
                }

    return results


def determine_verdict(comparison_df: pd.DataFrame, correlations: dict) -> str:
    """
    Determine linearity verdict based on rank agreement and weight divergence.

    With only n=4 features, we use rank agreement counts (not p-values)
    as the primary signal. Rho is reported descriptively.

    Returns (verdict_text, verdict_short).
    """
    # Primary signal: behavioral weights (correct) vs regression
    key_bw_reg = None
    for k in correlations:
        if 'behavioral (correct)' in k and 'regression' in k.lower():
            key_bw_reg = k
            break

    rho_bw_reg = correlations[key_bw_reg]['rho'] if key_bw_reg else np.nan
    rank_matches = correlations[key_bw_reg]['rank_matches'] if key_bw_reg else 0
    n_features = correlations[key_bw_reg]['n_features'] if key_bw_reg else 4
    top1_match = correlations[key_bw_reg]['top1_match'] if key_bw_reg else False

    # Weight concentration: coefficient of variation (std/mean) — robust to tiny min
    bw = comparison_df['bw_correct'].dropna()
    rw = comparison_df['regression_weight'].dropna()

    if len(bw) > 0 and bw.mean() > 0:
        bw_cv = float(bw.std() / bw.mean())
    else:
        bw_cv = np.nan
    if len(rw) > 0 and rw.mean() > 0:
        rw_cv = float(rw.std() / rw.mean())
    else:
        rw_cv = np.nan

    # Correct vs incorrect divergence
    if 'divergence' in comparison_df.columns:
        max_divergence = comparison_df['divergence'].max()
        mean_divergence = comparison_df['divergence'].mean()
    else:
        max_divergence = np.nan
        mean_divergence = np.nan

    # Decision rules — based on rank agreement, not rho thresholds
    lines = []

    if key_bw_reg:
        if rank_matches == n_features:
            verdict = "approximately linear"
            lines.append(f"VERDICT: {verdict}")
            lines.append(f"  Rank agreement: {rank_matches}/{n_features} features match exactly")
            lines.append(f"  Spearman rho = {rho_bw_reg:.3f} (descriptive, n={n_features})")
            lines.append(f"  The LLM's implicit feature weighting closely matches")
            lines.append(f"  a logistic regression trained on raw feature differences.")
        elif rank_matches >= n_features - 1 and top1_match:
            verdict = "partially linear"
            lines.append(f"VERDICT: {verdict}")
            lines.append(f"  Rank agreement: {rank_matches}/{n_features} features match")
            lines.append(f"  Top-1 feature agrees. Spearman rho = {rho_bw_reg:.3f} (descriptive, n={n_features})")
            lines.append(f"  Feature rankings mostly agree, suggesting the LLM uses")
            lines.append(f"  similar priorities with context-dependent adjustments.")
        elif top1_match:
            verdict = "partially linear"
            lines.append(f"VERDICT: {verdict}")
            lines.append(f"  Rank agreement: {rank_matches}/{n_features} features match")
            lines.append(f"  Top-1 feature agrees. Spearman rho = {rho_bw_reg:.3f} (descriptive, n={n_features})")
            lines.append(f"  Top feature priority matches but lower-ranked features diverge.")
        else:
            verdict = "non-linear / context-dependent"
            lines.append(f"VERDICT: {verdict}")
            lines.append(f"  Rank agreement: {rank_matches}/{n_features} features match")
            lines.append(f"  Top-1 feature disagrees. Spearman rho = {rho_bw_reg:.3f} (descriptive, n={n_features})")
            lines.append(f"  The LLM uses a substantially different weighting strategy")
            lines.append(f"  than a linear model — suggesting non-linear or contextual reasoning.")
    else:
        verdict = "insufficient data"
        lines.append(f"VERDICT: {verdict}")
        lines.append(f"  Could not compute behavioral vs regression correlation.")

    lines.append("")

    # Supplementary evidence: coefficient of variation
    if pd.notna(bw_cv) and pd.notna(rw_cv):
        lines.append(f"  Weight concentration (CV): LLM={bw_cv:.2f}, Regression={rw_cv:.2f}")
        if bw_cv > rw_cv * 2:
            lines.append(f"  → LLM concentrates weight more unevenly than regression")
        elif rw_cv > bw_cv * 2:
            lines.append(f"  → Regression concentrates weight more unevenly than LLM")
        else:
            lines.append(f"  → Similar concentration of weights across features")

    if pd.notna(mean_divergence):
        lines.append(f"  Correct/incorrect divergence: mean={mean_divergence:.2f}, max={max_divergence:.2f}")
        if max_divergence > 0.5:
            lines.append(f"  → Model applies different weights when wrong — context-dependent")
        else:
            lines.append(f"  → Weights are stable across correct/incorrect predictions")

    # Add all correlation results (descriptive only — no p-values with n=4)
    lines.append("")
    lines.append(f"RANK COMPARISONS (n={n_features} features, descriptive only):")
    for key, vals in correlations.items():
        lines.append(f"  {key}: rho={vals['rho']:.3f}, "
                     f"rank matches={vals['rank_matches']}/{vals['n_features']}, "
                     f"top-1 {'agrees' if vals['top1_match'] else 'disagrees'}")

    return "\n".join(lines), verdict


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_weight_comparison(comparison_df: pd.DataFrame, model: str):
    """
    Side-by-side bar chart: LLM behavioral weights vs regression coefficients.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    features = comparison_df['feature'].values
    x = np.arange(len(features))
    bar_width = 0.25

    # Panel 1: Weight magnitudes
    ax = axes[0]
    cols_to_plot = []
    labels = []
    colors = []

    if 'bw_correct' in comparison_df.columns:
        cols_to_plot.append('bw_correct')
        labels.append('LLM (correct)')
        colors.append('#2196F3')
    if 'bw_incorrect' in comparison_df.columns:
        cols_to_plot.append('bw_incorrect')
        labels.append('LLM (incorrect)')
        colors.append('#F44336')
    if 'regression_weight' in comparison_df.columns:
        cols_to_plot.append('regression_weight')
        labels.append('Logistic Regression')
        colors.append('#4CAF50')

    for i, (col, label, color) in enumerate(zip(cols_to_plot, labels, colors)):
        vals = comparison_df[col].fillna(0).values
        offset = (i - len(cols_to_plot) / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=label, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=10)
    ax.set_ylabel('Normalized Weight', fontsize=11)
    ax.set_title(f'{model} — Feature Importance Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0)

    # Panel 2: Rankings
    ax2 = axes[1]
    rank_cols = []
    rank_labels = []
    rank_colors = []

    for col, label, color in [('rank_bw_correct', 'LLM (correct)', '#2196F3'),
                               ('rank_regression', 'Regression', '#4CAF50'),
                               ('rank_probe', 'Probe Accuracy', '#FF9800')]:
        if col in comparison_df.columns:
            rank_cols.append(col)
            rank_labels.append(label)
            rank_colors.append(color)

    for i, (col, label, color) in enumerate(zip(rank_cols, rank_labels, rank_colors)):
        vals = comparison_df[col].fillna(0).values
        offset = (i - len(rank_cols) / 2 + 0.5) * bar_width
        ax2.bar(x + offset, vals, bar_width, label=label, color=color, alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(features, fontsize=10)
    ax2.set_ylabel('Rank (1 = most important)', fontsize=11)
    ax2.set_title(f'{model} — Feature Importance Rankings', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.invert_yaxis()
    ax2.set_ylim(len(features) + 0.5, 0.5)

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main(model: str = "llama-3.2-3b"):
    logger.info(f"Step 9: Validate Regression Comparison for {model}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    comp_df = load_behavioral_weights(model)
    regression_weights = load_regression_coefficients()
    probe_accuracies = load_probe_accuracies(model)

    logger.info(f"  Loaded behavioral weights: {len(comp_df)} features")
    logger.info(f"  Regression coefficients: {regression_weights}")
    logger.info(f"  Probe accuracies: {probe_accuracies}")

    # ------------------------------------------------------------------
    # Build comparison table
    # ------------------------------------------------------------------
    comparison_df = build_comparison_table(comp_df, regression_weights, probe_accuracies)

    # ------------------------------------------------------------------
    # Compute correlations
    # ------------------------------------------------------------------
    correlations = compute_correlations(comparison_df)

    # ------------------------------------------------------------------
    # Determine verdict
    # ------------------------------------------------------------------
    verdict_text, verdict_short = determine_verdict(comparison_df, correlations)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"  REGRESSION COMPARISON")
    logger.info(f"{'='*60}")

    logger.info(f"\n  Feature weight comparison:")
    logger.info(f"  {'Feature':12s} {'BW(correct)':>12s} {'BW(incorrect)':>14s} "
          f"{'Regression':>12s} {'Probe':>8s}")
    for _, row in comparison_df.iterrows():
        bw_c = f"{row['bw_correct']:.3f}" if pd.notna(row.get('bw_correct')) else 'N/A'
        bw_i = f"{row['bw_incorrect']:.3f}" if pd.notna(row.get('bw_incorrect')) else 'N/A'
        reg = f"{row['regression_weight']:.3f}" if pd.notna(row.get('regression_weight')) else 'N/A'
        probe = f"{row['probe_accuracy']:.1%}" if pd.notna(row.get('probe_accuracy')) else 'N/A'
        logger.info(f"  {row['feature']:12s} {bw_c:>12s} {bw_i:>14s} {reg:>12s} {probe:>8s}")

    logger.info(f"\n  Rankings:")
    for col in ['rank_bw_correct', 'rank_regression', 'rank_probe']:
        if col in comparison_df.columns:
            ranking = comparison_df.sort_values(col)[['feature', col]].values
            ordered = [f"{f}({int(r)})" for f, r in ranking]
            logger.info(f"    {col:20s}: {' > '.join(ordered)}")

    logger.info(f"\n  Rank comparisons (n={len(comparison_df)} features, descriptive):")
    for key, vals in correlations.items():
        logger.info(f"    {key}: rho={vals['rho']:.3f}, "
                    f"rank matches={vals['rank_matches']}/{vals['n_features']}, "
                    f"top-1 {'agrees' if vals['top1_match'] else 'disagrees'}")

    logger.info(f"\n{verdict_text}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    fig = plot_weight_comparison(comparison_df, model)

    outfile_png = MM_GRAPHS_DIR / f"weight_comparison_{model}.png"
    fig.savefig(outfile_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"\nSaved: {outfile_png}")

    # ------------------------------------------------------------------
    # Save tables
    # ------------------------------------------------------------------
    comp_file = MM_TABLES_DIR / f"regression_comparison_{model}.csv"
    comparison_df.to_csv(comp_file, index=False)
    logger.info(f"Saved: {comp_file}")

    verdict_file = MM_TABLES_DIR / f"linearity_verdict_{model}.txt"
    with open(verdict_file, 'w') as f:
        f.write(verdict_text)
    logger.info(f"Saved: {verdict_file}")

    logger.info(f"\n  Step 9 complete for {model}")
    return comparison_df, verdict_short


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 9: Validate regression comparison")
    parser.add_argument('--model', type=str, default='llama-3.2-3b',
                        choices=['llama-3.2-3b', 'qwen3-4b'],
                        help='Model to validate')
    args = parser.parse_args()
    main(model=args.model)
