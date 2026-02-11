# -*- coding: utf-8 -*-
"""
Step 3: Set Up Perturbation Plan (CPU-only, no inference)

For each of the 100 selected pairs, determines:
  1. Which property the model said is cheaper (the "loser")
     — based on the RE-INFERRED prediction from Step 2
  2. The loser's current feature values
  3. The full perturbation schedule (valid increments that respect caps)

Produces a flat CSV where each row is one perturbation query for Step 4.

Inputs:
  - macro_mapping/data/selected_samples_{model}.csv   (Step 1 — feature data)
  - macro_mapping/data/cot_responses_{model}.csv       (Step 2 — new predictions)

Outputs:
  - macro_mapping/data/perturbation_plan_{model}.csv
  - macro_mapping/data/perturbation_summary_{model}.txt

Usage:
    python macro_mapping/setup_perturbations.py --model llama-3.2-3b
    python macro_mapping/setup_perturbations.py --model qwen3-4b
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mm_utils import get_perturbation_steps, PERTURBATION_SCHEDULE, FEATURE_CAPS
from mm_config import TARGET_FEATURES, setup_logger

logger = setup_logger("Step3_SetupPerturbations")

# ============================================================================
# PATHS
# ============================================================================

MM_DATA_DIR = Path(__file__).resolve().parent / "data"


# ============================================================================
# MAIN
# ============================================================================

def main(model: str = "llama-3.2-3b"):
    logger.info(f"Step 3: Setup Perturbation Plan for {model}")

    # ------------------------------------------------------------------
    # Load Step 1 (feature data) and Step 2 (re-inferred predictions)
    # ------------------------------------------------------------------
    samples_file = MM_DATA_DIR / f"selected_samples_{model}.csv"
    cot_file = MM_DATA_DIR / f"cot_responses_{model}.csv"

    if not samples_file.exists():
        logger.error(f" {samples_file} not found. Run Step 1 first.")
        sys.exit(1)
    if not cot_file.exists():
        logger.error(f" {cot_file} not found. Run Step 2 first.")
        sys.exit(1)

    samples_df = pd.read_csv(samples_file)
    cot_df = pd.read_csv(cot_file)

    logger.info(f"  Loaded {len(samples_df)} samples, {len(cot_df)} CoT responses")

    # Merge: samples_df has raw features, cot_df has re-inferred predictions.
    # They are aligned by row order (pair_index 0..99), but merge on zpids
    # for safety.
    merged = samples_df.merge(
        cot_df[['zpid_1', 'zpid_2', 'pair_index', 'new_pred_choice', 'new_is_correct']],
        on=['zpid_1', 'zpid_2'],
        how='inner'
    )
    logger.info(f"  Merged: {len(merged)} pairs")

    # Drop pairs with unparseable predictions
    valid = merged[merged['new_pred_choice'].notna()].copy()
    valid = valid.reset_index(drop=True)
    valid['new_pred_choice'] = valid['new_pred_choice'].astype(int)
    skipped = len(merged) - len(valid)
    if skipped > 0:
        logger.warning(f" Skipping {skipped} pairs with unparseable predictions")
    logger.info(f"  Valid pairs for perturbation: {len(valid)}")

    # ------------------------------------------------------------------
    # Build perturbation plan
    # ------------------------------------------------------------------
    logger.info(f"\nBuilding perturbation plan...")
    logger.info(f"  Features: {TARGET_FEATURES}")
    logger.info(f"  Perturbation schedules:")
    for feat, steps in PERTURBATION_SCHEDULE.items():
        logger.info(f"    {feat}: {steps}  (cap: {FEATURE_CAPS[feat]})")

    plan_rows = []

    for idx, row in valid.iterrows():
        pair_index = row.get('pair_index', idx)
        pred_choice = int(row['new_pred_choice'])
        is_correct = bool(row['new_is_correct'])

        # Loser = property NOT chosen by the model
        # If model chose Property 2, loser is Property 1 (suffix "_1")
        loser_property = 1 if pred_choice == 2 else 2
        loser_suffix = f"_{loser_property}"

        for feature in TARGET_FEATURES:
            # Get valid perturbation steps (respects caps)
            steps = get_perturbation_steps(row, loser_property, feature)

            if not steps:
                continue

            # Original value
            original_col = f"{feature}{loser_suffix}"
            original_value = row[original_col]
            if pd.isna(original_value):
                continue
            original_value = float(original_value)

            for step_index, (perturbed_value, increment_label) in enumerate(steps):
                plan_rows.append({
                    'pair_index': pair_index,
                    'zpid_1': row['zpid_1'],
                    'zpid_2': row['zpid_2'],
                    'is_correct': is_correct,
                    'pred_choice': pred_choice,
                    'loser_property': loser_property,
                    'feature': feature,
                    'step_index': step_index,
                    'increment_label': increment_label,
                    'original_value': original_value,
                    'perturbed_value': perturbed_value,
                })

    plan_df = pd.DataFrame(plan_rows)
    logger.info(f"\nPerturbation plan: {len(plan_df)} total queries")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    n_pairs = plan_df['pair_index'].nunique()
    n_correct = plan_df[plan_df['is_correct']]['pair_index'].nunique()
    n_incorrect = plan_df[~plan_df['is_correct']]['pair_index'].nunique()

    logger.info(f"\n{'='*60}")
    logger.info(f"  PERTURBATION PLAN SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Pairs with perturbations: {n_pairs}")
    logger.info(f"    - Correct predictions:   {n_correct}")
    logger.info(f"    - Incorrect predictions:  {n_incorrect}")
    logger.info(f"  Total queries to run:      {len(plan_df)}")

    # Per-feature breakdown
    logger.info(f"\n  Queries per feature:")
    for feat in TARGET_FEATURES:
        feat_df = plan_df[plan_df['feature'] == feat]
        n_q = len(feat_df)
        n_p = feat_df['pair_index'].nunique()
        avg_steps = n_q / n_p if n_p > 0 else 0
        logger.info(f"    {feat:12s}: {n_q:4d} queries across {n_p:3d} pairs "
              f"(avg {avg_steps:.1f} steps/pair)")

    # Per-feature breakdown by correctness
    logger.info(f"\n  Queries by correctness x feature:")
    for correct_label, correct_val in [("correct", True), ("incorrect", False)]:
        sub = plan_df[plan_df['is_correct'] == correct_val]
        logger.info(f"    {correct_label}:")
        for feat in TARGET_FEATURES:
            feat_sub = sub[sub['feature'] == feat]
            logger.info(f"      {feat:12s}: {len(feat_sub):4d} queries")

    # Loser distribution
    loser_counts = plan_df.groupby('pair_index')['loser_property'].first().value_counts()
    logger.info(f"\n  Loser distribution:")
    logger.info(f"    Property 1 is loser: {loser_counts.get(1, 0)} pairs")
    logger.info(f"    Property 2 is loser: {loser_counts.get(2, 0)} pairs")

    # Estimate runtime (assuming ~2s per query with dual-GPU batching)
    est_seconds = len(plan_df) * 2
    est_minutes = est_seconds / 60
    logger.info(f"\n  Estimated Step 4 runtime: ~{est_minutes:.0f} min "
          f"({len(plan_df)} queries x ~2s each)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_file = MM_DATA_DIR / f"perturbation_plan_{model}.csv"
    plan_df.to_csv(output_file, index=False)
    logger.info(f"\nSaved: {output_file}")

    # Save summary text
    summary_file = MM_DATA_DIR / f"perturbation_summary_{model}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Perturbation Plan Summary for {model}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Total pairs: {n_pairs}\n")
        f.write(f"  Correct: {n_correct}\n")
        f.write(f"  Incorrect: {n_incorrect}\n")
        f.write(f"Total queries: {len(plan_df)}\n\n")

        f.write(f"Queries per feature:\n")
        for feat in TARGET_FEATURES:
            feat_df = plan_df[plan_df['feature'] == feat]
            f.write(f"  {feat}: {len(feat_df)}\n")

        f.write(f"\nPerturbation schedules:\n")
        for feat in TARGET_FEATURES:
            f.write(f"  {feat}: {PERTURBATION_SCHEDULE[feat]} (cap: {FEATURE_CAPS[feat]})\n")

        # Show a few example rows
        f.write(f"\nExample rows (first 20):\n")
        f.write(plan_df.head(20).to_string(index=False))
        f.write("\n")

    logger.info(f"Saved summary: {summary_file}")

    return plan_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Setup perturbation plan")
    parser.add_argument('--model', type=str, default='llama-3.2-3b',
                        choices=['llama-3.2-3b', 'qwen3-4b'],
                        help='Model to setup perturbations for')
    args = parser.parse_args()
    main(model=args.model)
