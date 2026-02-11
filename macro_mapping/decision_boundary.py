# -*- coding: utf-8 -*-
"""
Step 8: Decision Boundary Map and Stubbornness Ratios (CPU-only)

Visualizes the model's tipping points for each feature. For each feature,
draws a number line showing flip distances, with correct vs incorrect groups
overlaid. Computes "stubbornness ratio" — does the model require more evidence
to change its mind when it's already wrong?

Inputs:
  - macro_mapping/data/flip_results_{model}.csv     (Step 4)
  - macro_mapping/data/exchange_rates_{model}.csv   (Step 6)

Outputs:
  - macro_mapping/outputs/graphs/decision_boundary_{model}.png
  - macro_mapping/outputs/tables/boundary_stats_{model}.csv

Usage:
    python macro_mapping/decision_boundary.py --model llama-3.2-3b
    python macro_mapping/decision_boundary.py --model qwen3-4b
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mm_config import TARGET_FEATURES, PERTURBATION_STEPS, setup_logger

logger = setup_logger("Step8_DecisionBoundary")

# ============================================================================
# PATHS
# ============================================================================

MM_DATA_DIR = Path(__file__).resolve().parent / "data"
MM_GRAPHS_DIR = Path(__file__).resolve().parent / "outputs" / "graphs"
MM_TABLES_DIR = Path(__file__).resolve().parent / "outputs" / "tables"
MM_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
MM_TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# HELPERS
# ============================================================================

def flip_step_to_increment(feature: str, step_index: int) -> float:
    """Convert step_index to natural-unit increment."""
    schedule = PERTURBATION_STEPS[feature]
    if step_index < len(schedule):
        return float(schedule[step_index])
    return float(schedule[-1])


# Unit labels for each feature
UNIT_LABELS = {
    'bedrooms': 'bedrooms',
    'bathrooms': 'bathrooms',
    'yearBuilt': 'years newer',
    'lot': 'sqft added',
}


# ============================================================================
# MAIN
# ============================================================================

def main(model: str = "llama-3.2-3b"):
    logger.info(f"Step 8: Decision Boundary Map for {model}")

    # Load flip results
    flip_file = MM_DATA_DIR / f"flip_results_{model}.csv"
    if not flip_file.exists():
        logger.error(f" {flip_file} not found. Run Step 4 first.")
        sys.exit(1)

    flip_df = pd.read_csv(flip_file)
    logger.info(f"  Loaded {len(flip_df)} flip results")

    # ------------------------------------------------------------------
    # Compute boundary stats
    # ------------------------------------------------------------------
    stats_rows = []

    for feat in TARGET_FEATURES:
        for group_label, is_correct in [('correct', True), ('incorrect', False)]:
            sub = flip_df[(flip_df['feature'] == feat) & (flip_df['is_correct'] == is_correct)]

            n_total = len(sub)
            if n_total == 0:
                continue

            flipped = sub[sub['flipped'] == True]
            n_flipped = len(flipped)
            no_flip_rate = 1 - n_flipped / n_total

            if n_flipped > 0:
                increments = flipped['flip_step_index'].apply(
                    lambda s: flip_step_to_increment(feat, int(s))
                )
                mean_flip = increments.mean()
                median_flip = increments.median()
                std_flip = increments.std()
            else:
                mean_flip = np.nan
                median_flip = np.nan
                std_flip = np.nan

            stats_rows.append({
                'feature': feat,
                'group': group_label,
                'n_total': n_total,
                'n_flipped': n_flipped,
                'no_flip_rate': no_flip_rate,
                'mean_flip': mean_flip,
                'median_flip': median_flip,
                'std_flip': std_flip,
            })

    stats_df = pd.DataFrame(stats_rows)

    # Compute stubbornness ratio
    stubbornness = []
    for feat in TARGET_FEATURES:
        c = stats_df[(stats_df['feature'] == feat) & (stats_df['group'] == 'correct')]
        ic = stats_df[(stats_df['feature'] == feat) & (stats_df['group'] == 'incorrect')]

        if len(c) > 0 and len(ic) > 0:
            mean_c = c.iloc[0]['mean_flip']
            mean_ic = ic.iloc[0]['mean_flip']
            if pd.notna(mean_c) and pd.notna(mean_ic) and mean_c > 0:
                ratio = mean_ic / mean_c
            else:
                ratio = np.nan
        else:
            ratio = np.nan

        stubbornness.append({'feature': feat, 'stubbornness_ratio': ratio})

    stubb_df = pd.DataFrame(stubbornness)

    # Merge stubbornness into stats
    stats_df = stats_df.merge(stubb_df, on='feature', how='left')

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"  DECISION BOUNDARY STATS")
    logger.info(f"{'='*60}")

    for feat in TARGET_FEATURES:
        feat_stats = stats_df[stats_df['feature'] == feat]
        stubb = stubb_df[stubb_df['feature'] == feat].iloc[0]['stubbornness_ratio']
        stubb_str = f"{stubb:.2f}" if pd.notna(stubb) else "N/A"
        logger.info(f"\n  {feat} (stubbornness ratio: {stubb_str}):")
        for _, row in feat_stats.iterrows():
            mean_str = f"{row['mean_flip']:.1f}" if pd.notna(row['mean_flip']) else "N/A"
            logger.info(f"    {row['group']:12s}: mean flip = {mean_str} {UNIT_LABELS[feat]}, "
                  f"no-flip rate = {row['no_flip_rate']:.0%}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(len(TARGET_FEATURES), 1, figsize=(14, 3 * len(TARGET_FEATURES)))

    for i, feat in enumerate(TARGET_FEATURES):
        ax = axes[i]
        schedule = PERTURBATION_STEPS[feat]

        # Get flip data for correct and incorrect
        for group_label, is_correct, color, marker in [
            ('correct', True, '#2196F3', 'o'),
            ('incorrect', False, '#F44336', 's')
        ]:
            sub = flip_df[(flip_df['feature'] == feat) & (flip_df['is_correct'] == is_correct)]
            flipped = sub[sub['flipped'] == True]

            if len(flipped) > 0:
                increments = flipped['flip_step_index'].apply(
                    lambda s: flip_step_to_increment(feat, int(s))
                )
                # Histogram / strip plot of flip increments
                jitter = np.random.uniform(-0.15, 0.15, size=len(increments))
                y_pos = 0.3 if group_label == 'correct' else -0.3
                ax.scatter(increments, [y_pos] * len(increments) + jitter,
                           c=color, marker=marker, s=25, alpha=0.6,
                           label=f'{group_label} (n={len(flipped)})')

                # Mean marker
                mean_val = increments.mean()
                ax.axvline(mean_val, color=color, linestyle='--', alpha=0.7, linewidth=2)
                ax.text(mean_val, y_pos + 0.5, f'avg={mean_val:.1f}',
                        ha='center', va='bottom', fontsize=8, color=color,
                        fontweight='bold')

            # No-flip indicator
            no_flip_count = (sub['flipped'] == False).sum()
            if no_flip_count > 0:
                max_val = schedule[-1]
                ax.annotate(f'{no_flip_count} no-flip →',
                            xy=(max_val * 1.05, 0.3 if group_label == 'correct' else -0.3),
                            fontsize=7, color=color, alpha=0.7,
                            ha='left', va='center')

        # Formatting
        ax.set_xlim(0, schedule[-1] * 1.2)
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([])
        ax.set_xlabel(f'{UNIT_LABELS[feat]} added to loser', fontsize=10)

        # Stubbornness annotation
        stubb = stubb_df[stubb_df['feature'] == feat].iloc[0]['stubbornness_ratio']
        if pd.notna(stubb):
            stubb_label = f"stubbornness: {stubb:.2f}x"
            ax.text(0.98, 0.95, stubb_label, transform=ax.transAxes,
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title(f'{feat}', fontsize=12, fontweight='bold', loc='left')
        ax.legend(loc='upper right', fontsize=8)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    # Save
    outfile = MM_GRAPHS_DIR / f"decision_boundary_{model}.png"
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"\nSaved: {outfile}")

    # Save stats table
    stats_file = MM_TABLES_DIR / f"boundary_stats_{model}.csv"
    stats_df.to_csv(stats_file, index=False)
    logger.info(f"Saved: {stats_file}")

    logger.info(f"  Step 8 complete for {model}")

    return stats_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 8: Decision boundary map")
    parser.add_argument('--model', type=str, default='llama-3.2-3b',
                        choices=['llama-3.2-3b', 'qwen3-4b'],
                        help='Model to visualize')
    args = parser.parse_args()
    main(model=args.model)
