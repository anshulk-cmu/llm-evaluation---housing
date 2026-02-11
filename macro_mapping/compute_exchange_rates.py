# -*- coding: utf-8 -*-
"""
Step 6: Compute Feature Exchange Rates from Flip Distances (CPU-only)

Derives the model's implicit "exchange rates" between features from flip
distances. If bathrooms flips at +1.2 avg and bedrooms at +2.8 avg, the
model values 1 bathroom ~ 2.3 bedrooms.

Computes separately for correct/incorrect groups to detect divergence.
Also compares behavioral weights against probing accuracy and logistic
regression coefficients.

Inputs:
  - macro_mapping/data/flip_results_{model}.csv     (Step 4)
  - data/probe_results/probe_best_layers_*.csv       (existing probing)
  - data/raw_feature_logit_results/raw_feature_logit_summary.json

Outputs:
  - macro_mapping/data/exchange_rates_{model}.csv
  - macro_mapping/data/pairwise_exchange_rates_{model}.csv
  - macro_mapping/data/exchange_rate_comparison_{model}.csv

Usage:
    python macro_mapping/compute_exchange_rates.py --model llama-3.2-3b
    python macro_mapping/compute_exchange_rates.py --model qwen3-4b
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mm_config import TARGET_FEATURES, PERTURBATION_STEPS, CONDITION, setup_logger

logger = setup_logger("Step6_ExchangeRates")

# ============================================================================
# PATHS
# ============================================================================

MM_DATA_DIR = Path(__file__).resolve().parent / "data"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBE_DIR = PROJECT_ROOT / "data" / "probe_results"
LOGIT_DIR = PROJECT_ROOT / "data" / "raw_feature_logit_results"


# ============================================================================
# HELPERS
# ============================================================================

def flip_increment_to_natural(feature: str, step_index: int) -> float:
    """Convert step_index to natural-unit increment from perturbation schedule."""
    schedule = PERTURBATION_STEPS[feature]
    if step_index < len(schedule):
        return float(schedule[step_index])
    return float(schedule[-1])  # fallback to max


def compute_flip_stats(flip_df: pd.DataFrame, group_label: str) -> pd.DataFrame:
    """
    Compute flip statistics for one group of flip results.

    Returns DataFrame with one row per feature.
    """
    rows = []
    for feat in TARGET_FEATURES:
        feat_df = flip_df[flip_df['feature'] == feat]
        n_total = len(feat_df)
        if n_total == 0:
            continue

        flipped = feat_df[feat_df['flipped'] == True]
        n_flipped = len(flipped)
        no_flip_rate = 1.0 - n_flipped / n_total

        if n_flipped > 0:
            # Convert step_index to natural-unit increment
            increments = flipped['flip_step_index'].apply(
                lambda s: flip_increment_to_natural(feat, int(s))
            )
            mean_inc = increments.mean()
            median_inc = increments.median()
            std_inc = increments.std()
            min_inc = increments.min()
            max_inc = increments.max()
        else:
            # No flips — assign max increment as censored value
            mean_inc = float(PERTURBATION_STEPS[feat][-1])
            median_inc = mean_inc
            std_inc = 0.0
            min_inc = np.nan
            max_inc = np.nan

        rows.append({
            'feature': feat,
            'group': group_label,
            'n_total': n_total,
            'n_flipped': n_flipped,
            'no_flip_rate': no_flip_rate,
            'mean_flip_increment': mean_inc,
            'median_flip_increment': median_inc,
            'std_flip_increment': std_inc,
            'min_flip_increment': min_inc,
            'max_flip_increment': max_inc,
        })

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main(model: str = "llama-3.2-3b"):
    logger.info(f"Step 6: Compute Exchange Rates for {model}")

    # ------------------------------------------------------------------
    # Load flip results
    # ------------------------------------------------------------------
    flip_file = MM_DATA_DIR / f"flip_results_{model}.csv"
    if not flip_file.exists():
        logger.error(f" {flip_file} not found. Run Step 4 first.")
        sys.exit(1)

    flip_df = pd.read_csv(flip_file)
    logger.info(f"  Loaded {len(flip_df)} flip results")

    # ------------------------------------------------------------------
    # Compute flip stats per group
    # ------------------------------------------------------------------
    correct_df = flip_df[flip_df['is_correct'] == True]
    incorrect_df = flip_df[flip_df['is_correct'] == False]

    stats_all = compute_flip_stats(flip_df, 'all')
    stats_correct = compute_flip_stats(correct_df, 'correct')
    stats_incorrect = compute_flip_stats(incorrect_df, 'incorrect')

    exchange_df = pd.concat([stats_all, stats_correct, stats_incorrect], ignore_index=True)

    # ------------------------------------------------------------------
    # Compute pairwise exchange rates
    # ------------------------------------------------------------------
    logger.info(f"\nComputing pairwise exchange rates...")

    pairwise_rows = []
    for group in ['all', 'correct', 'incorrect']:
        group_stats = exchange_df[exchange_df['group'] == group]

        for feat_a, feat_b in combinations(TARGET_FEATURES, 2):
            a_stats = group_stats[group_stats['feature'] == feat_a]
            b_stats = group_stats[group_stats['feature'] == feat_b]

            if len(a_stats) == 0 or len(b_stats) == 0:
                continue

            mean_a = a_stats.iloc[0]['mean_flip_increment']
            mean_b = b_stats.iloc[0]['mean_flip_increment']

            # Exchange rate: how many units of A equal 1 unit of B
            # If B flips faster (lower increment), B is more important
            rate_a_per_b = mean_a / mean_b if mean_b > 0 else np.inf
            rate_b_per_a = mean_b / mean_a if mean_a > 0 else np.inf

            pairwise_rows.append({
                'feature_a': feat_a,
                'feature_b': feat_b,
                'mean_flip_a': mean_a,
                'mean_flip_b': mean_b,
                'rate_a_per_b': rate_a_per_b,
                'rate_b_per_a': rate_b_per_a,
                'interpretation': f"1 {feat_b} ~ {rate_a_per_b:.1f} {feat_a}",
                'group': group,
            })

    pairwise_df = pd.DataFrame(pairwise_rows)

    # ------------------------------------------------------------------
    # Correct vs incorrect comparison
    # ------------------------------------------------------------------
    logger.info(f"\nComparing correct vs incorrect exchange rates...")

    comparison_rows = []
    for feat in TARGET_FEATURES:
        c = exchange_df[(exchange_df['feature'] == feat) & (exchange_df['group'] == 'correct')]
        ic = exchange_df[(exchange_df['feature'] == feat) & (exchange_df['group'] == 'incorrect')]

        if len(c) == 0 or len(ic) == 0:
            continue

        mean_c = c.iloc[0]['mean_flip_increment']
        mean_ic = ic.iloc[0]['mean_flip_increment']
        divergence = abs(mean_c - mean_ic) / mean_c if mean_c > 0 else np.inf

        # Behavioral weight: 1 / mean_flip (higher = more important)
        bw_c = 1.0 / mean_c if mean_c > 0 else 0
        bw_ic = 1.0 / mean_ic if mean_ic > 0 else 0

        comparison_rows.append({
            'feature': feat,
            'mean_flip_correct': mean_c,
            'mean_flip_incorrect': mean_ic,
            'no_flip_rate_correct': c.iloc[0]['no_flip_rate'],
            'no_flip_rate_incorrect': ic.iloc[0]['no_flip_rate'],
            'behavioral_weight_correct': bw_c,
            'behavioral_weight_incorrect': bw_ic,
            'divergence': divergence,
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # Normalize behavioral weights to sum to 1
    for col in ['behavioral_weight_correct', 'behavioral_weight_incorrect']:
        total = comparison_df[col].sum()
        if total > 0:
            comparison_df[f'{col}_norm'] = comparison_df[col] / total

    # ------------------------------------------------------------------
    # Load probing + regression for comparison
    # ------------------------------------------------------------------
    probe_file = PROBE_DIR / f"probe_best_layers_{model}_{CONDITION}.csv"
    logit_file = LOGIT_DIR / "raw_feature_logit_summary.json"

    if probe_file.exists():
        probe_df = pd.read_csv(probe_file)
        # Map probe features to our names
        probe_map = {
            'bathrooms_p1_more': 'bathrooms',
            'bedrooms_p1_more': 'bedrooms',
            'lot_p1_larger': 'lot',
            'year_p1_newer': 'yearBuilt',
        }
        for _, row in probe_df.iterrows():
            feat_name = probe_map.get(row['feature'], None)
            if feat_name and feat_name in comparison_df['feature'].values:
                comparison_df.loc[comparison_df['feature'] == feat_name, 'probe_accuracy'] = row['test_accuracy']
                comparison_df.loc[comparison_df['feature'] == feat_name, 'probe_auc'] = row['test_auc']

    if logit_file.exists():
        with open(logit_file) as f:
            logit_summary = json.load(f)
        coeffs = logit_summary.get('coefficients', {})
        coeff_map = {
            'bathrooms_diff': 'bathrooms',
            'bedrooms_diff': 'bedrooms',
            'lot_diff': 'lot',
            'year_diff': 'yearBuilt',
        }
        for coeff_name, feat_name in coeff_map.items():
            if coeff_name in coeffs and feat_name in comparison_df['feature'].values:
                comparison_df.loc[comparison_df['feature'] == feat_name, 'regression_coeff'] = abs(coeffs[coeff_name])

        # Normalize regression coefficients
        total_coeff = comparison_df['regression_coeff'].sum()
        if total_coeff > 0:
            comparison_df['regression_weight_norm'] = comparison_df['regression_coeff'] / total_coeff

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"  EXCHANGE RATE RESULTS")
    logger.info(f"{'='*60}")

    logger.info(f"\n  Mean flip increments (all pairs):")
    for _, row in exchange_df[exchange_df['group'] == 'all'].iterrows():
        logger.info(f"    {row['feature']:12s}: {row['mean_flip_increment']:.2f} "
              f"(no-flip rate: {row['no_flip_rate']:.0%})")

    logger.info(f"\n  Pairwise exchange rates (all):")
    for _, row in pairwise_df[pairwise_df['group'] == 'all'].iterrows():
        logger.info(f"    {row['interpretation']}")

    logger.info(f"\n  Correct vs Incorrect comparison:")
    for _, row in comparison_df.iterrows():
        logger.info(f"    {row['feature']:12s}: correct flip={row['mean_flip_correct']:.2f}, "
              f"incorrect flip={row['mean_flip_incorrect']:.2f}, "
              f"divergence={row['divergence']:.2f}")

    if 'behavioral_weight_correct_norm' in comparison_df.columns:
        logger.info(f"\n  Normalized behavioral weights:")
        logger.info(f"    {'Feature':12s} {'BW(correct)':>12s} {'BW(incorrect)':>14s} "
              f"{'Regression':>12s} {'Probe Acc':>10s}")
        for _, row in comparison_df.iterrows():
            reg = f"{row.get('regression_weight_norm', 0):.3f}" if pd.notna(row.get('regression_weight_norm')) else 'N/A'
            probe = f"{row.get('probe_accuracy', 0):.1%}" if pd.notna(row.get('probe_accuracy')) else 'N/A'
            logger.info(f"    {row['feature']:12s} {row['behavioral_weight_correct_norm']:12.3f} "
                  f"{row['behavioral_weight_incorrect_norm']:14.3f} {reg:>12s} {probe:>10s}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    exchange_file = MM_DATA_DIR / f"exchange_rates_{model}.csv"
    exchange_df.to_csv(exchange_file, index=False)
    logger.info(f"\nSaved: {exchange_file}")

    pairwise_file = MM_DATA_DIR / f"pairwise_exchange_rates_{model}.csv"
    pairwise_df.to_csv(pairwise_file, index=False)
    logger.info(f"Saved: {pairwise_file}")

    comparison_file = MM_DATA_DIR / f"exchange_rate_comparison_{model}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"Saved: {comparison_file}")

    return exchange_df, pairwise_df, comparison_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6: Compute exchange rates")
    parser.add_argument('--model', type=str, default='llama-3.2-3b',
                        choices=['llama-3.2-3b', 'qwen3-4b'],
                        help='Model to analyze')
    args = parser.parse_args()
    main(model=args.model)
