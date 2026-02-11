# -*- coding: utf-8 -*-
"""
Step 1: Select 100 Property Pairs (60 Correct, 40 Incorrect)

Stratified selection ensuring diversity across:
  - Difficulty level (price gap: easy/medium/hard)
  - Feature profile (which feature dominates the pair)

Usage:
    python macro_mapping/select_samples.py --model llama-3.2-3b
    python macro_mapping/select_samples.py --model qwen3-4b
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse

from mm_config import (
    DATA_PATH, RESULTS_DIR, MM_DATA_DIR,
    N_TOTAL, N_CORRECT, N_INCORRECT, RANDOM_STATE,
    DIFFICULTY_BINS, CONDITION, TARGET_FEATURES,
    setup_logger,
)

logger = setup_logger("Step1_SelectSamples")


# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PAIRS_FILE = Path(__file__).resolve().parent.parent / DATA_PATH
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_FILES = {
    'llama-3.2-3b': DATA_DIR / "results" / f"results_Llama-3.2-3B-Instruct_{CONDITION}.csv",
    'qwen3-4b': DATA_DIR / "results" / f"results_Qwen3-4B-Instruct-2507_{CONDITION}.csv",
}


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_price_ratio(row):
    """Compute relative price difference as (hi - lo) / lo."""
    p1, p2 = row['price.y_1'], row['price.y_2']
    hi, lo = max(p1, p2), min(p1, p2)
    return (hi - lo) / lo if lo > 0 else 0


def assign_difficulty(ratio):
    """Assign difficulty bin based on price ratio."""
    for label, (lo, hi) in DIFFICULTY_BINS.items():
        if lo <= ratio < hi:
            return label
    # Ratio >= 1.0 is still "easy"
    return "easy"


def assign_feature_profile(row):
    """
    Assign a feature profile tag based on which feature differs most
    between the two properties.

    Tags:
      - "bathrooms_dominant": bathrooms differ, bedrooms equal
      - "bedrooms_dominant":  bedrooms differ, bathrooms equal
      - "year_dominant":      yearBuilt gap > 20 years, beds/baths similar
      - "lot_dominant":       lot gap > 2000 sqft, beds/baths similar
      - "mixed":              multiple features differ meaningfully
    """
    bed_diff = abs(row['bedrooms_diff'])
    bath_diff = abs(row['bathrooms_diff'])
    year_diff = abs(row['year_diff'])
    lot_diff = abs(row['lot_diff'])

    beds_equal = bed_diff == 0
    baths_equal = bath_diff == 0

    if bath_diff >= 1 and beds_equal:
        return "bathrooms_dominant"
    if bed_diff >= 1 and baths_equal:
        return "bedrooms_dominant"
    if year_diff >= 20 and bed_diff <= 1 and bath_diff <= 1:
        return "year_dominant"
    if lot_diff >= 2000 and bed_diff <= 1 and bath_diff <= 1:
        return "lot_dominant"
    return "mixed"


def compute_features(df):
    """Add difficulty, feature diffs, and profile columns."""
    df['price_ratio'] = df.apply(compute_price_ratio, axis=1)
    df['difficulty'] = df['price_ratio'].apply(assign_difficulty)

    df['bedrooms_diff'] = df['bedrooms_1'] - df['bedrooms_2']
    df['bathrooms_diff'] = df['bathrooms_1'] - df['bathrooms_2']
    df['year_diff'] = df['yearBuilt_1'] - df['yearBuilt_2']
    df['lot_diff'] = df['lot_1'] - df['lot_2']

    df['feature_profile'] = df.apply(assign_feature_profile, axis=1)
    return df


# ============================================================================
# STRATIFIED SAMPLING
# ============================================================================

def stratified_sample(df, n_target, random_state=42):
    """
    Sample n_target rows from df, stratified by difficulty.
    Within each difficulty bin, ensure feature profile diversity.
    """
    if n_target >= len(df):
        logger.info(f"  Pool ({len(df)}) <= target ({n_target}), using all.")
        return df.copy()

    rng = np.random.RandomState(random_state)

    # Step 1: proportional allocation across difficulty bins
    difficulty_counts = df['difficulty'].value_counts()
    allocation = {}
    for diff_bin in difficulty_counts.index:
        proportion = difficulty_counts[diff_bin] / len(df)
        allocation[diff_bin] = max(1, int(round(proportion * n_target)))

    # Adjust to hit exact target
    total_allocated = sum(allocation.values())
    if total_allocated != n_target:
        largest_bin = max(allocation, key=allocation.get)
        allocation[largest_bin] += (n_target - total_allocated)

    # Step 2: within each difficulty bin, sample with feature profile diversity
    selected = []
    for diff_bin, n_needed in allocation.items():
        pool = df[df['difficulty'] == diff_bin]

        if len(pool) <= n_needed:
            selected.append(pool)
            continue

        # Try to get at least 1 from each feature profile present
        profiles = pool['feature_profile'].value_counts()
        guaranteed = []
        remaining_n = n_needed

        for profile in profiles.index:
            if remaining_n <= 0:
                break
            profile_pool = pool[pool['feature_profile'] == profile]
            pick = profile_pool.sample(n=1, random_state=rng)
            guaranteed.append(pick)
            remaining_n -= 1

        # Fill the rest randomly from the remaining pool
        if remaining_n > 0:
            picked_idx = pd.concat(guaranteed).index
            leftover = pool.drop(picked_idx)
            if len(leftover) >= remaining_n:
                fill = leftover.sample(n=remaining_n, random_state=rng)
            else:
                fill = leftover
            guaranteed.append(fill)

        selected.append(pd.concat(guaranteed))

    result = pd.concat(selected, ignore_index=True)

    # Final safety: trim or pad to exact target
    if len(result) > n_target:
        result = result.sample(n=n_target, random_state=rng).reset_index(drop=True)
    return result


# ============================================================================
# ANALYSIS
# ============================================================================

def print_analysis(selected, original, label):
    """Print distribution comparison."""
    logger.info(f"{'='*60}")
    logger.info(f"  {label}: {len(selected)} samples")
    logger.info(f"{'='*60}")

    logger.info(f"  Difficulty distribution:")
    for d in ['easy', 'medium', 'hard']:
        orig_pct = 100 * (original['difficulty'] == d).mean()
        sel_pct = 100 * (selected['difficulty'] == d).mean()
        n = (selected['difficulty'] == d).sum()
        logger.info(f"    {d:8s}: {n:3d} ({sel_pct:5.1f}%)  [pool: {orig_pct:5.1f}%]")

    logger.info(f"  Feature profile distribution:")
    for p in ['bathrooms_dominant', 'bedrooms_dominant', 'year_dominant', 'lot_dominant', 'mixed']:
        n = (selected['feature_profile'] == p).sum()
        if n > 0:
            logger.info(f"    {p:22s}: {n:3d}")

    logger.info(f"  Price ratio: mean={selected['price_ratio'].mean():.3f}, "
          f"median={selected['price_ratio'].median():.3f}")


# ============================================================================
# MAIN
# ============================================================================

def main(model: str = "llama-3.2-3b"):
    logger.info(f"Step 1: Sample Selection for {model}")
    logger.info(f"Target: {N_CORRECT} correct + {N_INCORRECT} incorrect = {N_TOTAL}")

    # Load data
    pairs_df = pd.read_csv(PAIRS_FILE)
    results_df = pd.read_csv(RESULTS_FILES[model])
    logger.info(f"Loaded {len(pairs_df)} pairs, {len(results_df)} predictions")

    # Merge
    df = pairs_df.merge(
        results_df[['zpid_1', 'zpid_2', 'pred_choice', 'model_text',
                     'true_choice', 'correct']],
        on=['zpid_1', 'zpid_2']
    )
    logger.info(f"Merged: {len(df)} rows")

    # Compute features
    df = compute_features(df)

    # Split
    correct_pool = df[df['correct'] == True].copy()
    incorrect_pool = df[df['correct'] == False].copy()
    logger.info(f"Correct pool: {len(correct_pool)}, Incorrect pool: {len(incorrect_pool)}")

    # Stratified sample
    correct_sel = stratified_sample(correct_pool, N_CORRECT, RANDOM_STATE)
    incorrect_sel = stratified_sample(incorrect_pool, N_INCORRECT, RANDOM_STATE)

    # Combine
    correct_sel['is_correct'] = True
    incorrect_sel['is_correct'] = False
    all_selected = pd.concat([correct_sel, incorrect_sel], ignore_index=True)

    # Analysis
    print_analysis(correct_sel, correct_pool, "CORRECT")
    print_analysis(incorrect_sel, incorrect_pool, "INCORRECT")

    logger.info(f"{'='*60}")
    logger.info(f"  TOTAL SELECTED: {len(all_selected)}")
    logger.info(f"  Correct: {len(correct_sel)}, Incorrect: {len(incorrect_sel)}")
    logger.info(f"{'='*60}")

    # Save
    output_file = OUTPUT_DIR / f"selected_samples_{model}.csv"
    all_selected.to_csv(output_file, index=False)
    logger.info(f"Saved: {output_file}")

    # Save metadata
    meta_file = OUTPUT_DIR / f"selection_meta_{model}.json"
    meta = {
        'model': model,
        'condition': CONDITION,
        'n_correct': len(correct_sel),
        'n_incorrect': len(incorrect_sel),
        'n_total': len(all_selected),
        'random_state': RANDOM_STATE,
        'correct_pool_size': len(correct_pool),
        'incorrect_pool_size': len(incorrect_pool),
        'difficulty_distribution': all_selected['difficulty'].value_counts().to_dict(),
        'feature_profile_distribution': all_selected['feature_profile'].value_counts().to_dict(),
        'zpid_pairs': all_selected[['zpid_1', 'zpid_2']].values.tolist(),
    }
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Saved metadata: {meta_file}")

    return all_selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Select samples for macro mapping")
    parser.add_argument('--model', type=str, default='llama-3.2-3b',
                        choices=['llama-3.2-3b', 'qwen3-4b'],
                        help='Model to select samples for')
    args = parser.parse_args()
    main(model=args.model)
