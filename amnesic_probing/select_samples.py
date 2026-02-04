"""
Balanced Sample Selection for Amnesic Probing Experiment

With cuML GPU acceleration, we can use ALL 5,130 samples!

Selection includes:
1. All prediction outcomes (natural ratio ~60% correct)
2. Price difference difficulty categorization (easy/medium/hard)
3. Feature value distribution analysis

Usage:
    python select_samples.py --model llama-3.2-3b
    python select_samples.py --model qwen3-4b
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse

# Configuration
RANDOM_SEED = 42
USE_ALL_SAMPLES = True  # With cuML, we can use all samples!
N_CORRECT = None  # Will use all if USE_ALL_SAMPLES=True
N_INCORRECT = None
EXPERIMENT = "2_fewshot_cot_temp0"

# Paths
DATA_DIR = Path("/home/anshulk/Housing/data")
PAIRS_FILE = DATA_DIR / "pairs_20pct_price_diff.csv"
OUTPUT_DIR = DATA_DIR / "amnesic_samples"
OUTPUT_DIR.mkdir(exist_ok=True)

# Model to results file mapping
RESULTS_FILES = {
    'llama-3.2-3b': DATA_DIR / "results" / f"results_Llama-3.2-3B-Instruct_{EXPERIMENT}.csv",
    'qwen3-4b': DATA_DIR / "results" / f"results_Qwen3-4B-Instruct-2507_{EXPERIMENT}.csv"
}


def compute_price_diff_pct(row):
    """Compute absolute percentage difference between prices."""
    p1, p2 = row['price.y_1'], row['price.y_2']
    return abs(p1 - p2) / max(p1, p2) * 100


def compute_feature_diffs(df):
    """Compute feature differences for stratification."""
    df['price_diff_pct'] = df.apply(compute_price_diff_pct, axis=1)
    
    # Categorize difficulty based on price difference
    # Easy: >40% diff, Medium: 25-40%, Hard: 20-25%
    df['difficulty'] = pd.cut(
        df['price_diff_pct'],
        bins=[0, 25, 40, 100],
        labels=['hard', 'medium', 'easy']
    )
    
    # Feature differences (which property has more)
    df['bedrooms_diff'] = df['bedrooms_1'] - df['bedrooms_2']
    df['bathrooms_diff'] = df['bathrooms_1'] - df['bathrooms_2']
    df['year_diff'] = df['yearBuilt_1'] - df['yearBuilt_2']
    df['lot_diff'] = df['lot_1'] - df['lot_2']
    
    # Categorize feature differences for stratification
    df['bedroom_category'] = pd.cut(
        df['bedrooms_diff'],
        bins=[-np.inf, -1, 0, 1, np.inf],
        labels=['p2_more', 'equal', 'p1_slight', 'p1_much']
    )
    
    df['year_category'] = pd.cut(
        df['year_diff'],
        bins=[-np.inf, -20, 0, 20, np.inf],
        labels=['p2_newer', 'similar', 'p1_slight', 'p1_much']
    )
    
    return df


def create_stratification_key(df):
    """Create composite key for stratified sampling."""
    # Combine difficulty + bedroom category for stratification
    df['strat_key'] = (
        df['difficulty'].astype(str) + '_' + 
        df['bedroom_category'].astype(str)
    )
    return df


def select_stratified_samples(df, n_samples, random_state=42):
    """Select stratified samples maintaining feature distribution.
    
    If n_samples is None, return all samples (for cuML full-data mode).
    """
    if n_samples is None or n_samples >= len(df):
        return df.copy()
    
    # Get stratification key distribution
    strat_counts = df['strat_key'].value_counts()
    
    # Calculate proportional samples per stratum
    total = len(df)
    samples_per_stratum = (strat_counts / total * n_samples).round().astype(int)
    
    # Adjust to hit exact target
    diff = n_samples - samples_per_stratum.sum()
    if diff != 0:
        # Add/remove from largest strata
        largest = samples_per_stratum.idxmax()
        samples_per_stratum[largest] += diff
    
    # Sample from each stratum
    selected = []
    np.random.seed(random_state)
    
    for stratum, n in samples_per_stratum.items():
        stratum_df = df[df['strat_key'] == stratum]
        if len(stratum_df) <= n:
            selected.append(stratum_df)
        else:
            selected.append(stratum_df.sample(n=n, random_state=random_state))
    
    return pd.concat(selected, ignore_index=True)


def analyze_selection(selected_df, original_df, label):
    """Print analysis of selected samples vs original."""
    print(f"\n{'='*60}")
    print(f"Analysis: {label}")
    print(f"{'='*60}")
    print(f"Selected: {len(selected_df)} samples")
    
    # Difficulty distribution
    print(f"\nDifficulty distribution:")
    orig_diff = original_df['difficulty'].value_counts(normalize=True)
    sel_diff = selected_df['difficulty'].value_counts(normalize=True)
    for d in ['easy', 'medium', 'hard']:
        orig_pct = orig_diff.get(d, 0) * 100
        sel_pct = sel_diff.get(d, 0) * 100
        print(f"  {d}: {sel_pct:.1f}% (original: {orig_pct:.1f}%)")
    
    # Feature distributions
    print(f"\nBedroom difference distribution:")
    orig_bed = original_df['bedroom_category'].value_counts(normalize=True)
    sel_bed = selected_df['bedroom_category'].value_counts(normalize=True)
    for cat in orig_bed.index:
        orig_pct = orig_bed.get(cat, 0) * 100
        sel_pct = sel_bed.get(cat, 0) * 100
        print(f"  {cat}: {sel_pct:.1f}% (original: {orig_pct:.1f}%)")
    
    # Price difference stats
    print(f"\nPrice difference stats:")
    print(f"  Mean: {selected_df['price_diff_pct'].mean():.1f}% (orig: {original_df['price_diff_pct'].mean():.1f}%)")
    print(f"  Median: {selected_df['price_diff_pct'].median():.1f}% (orig: {original_df['price_diff_pct'].median():.1f}%)")
    print(f"  Std: {selected_df['price_diff_pct'].std():.1f}% (orig: {original_df['price_diff_pct'].std():.1f}%)")


def main(model: str = "llama-3.2-3b"):
    """Run sample selection for specified model."""
    print(f"Sample selection for model: {model}")
    print("Loading data...")
    
    # Get the results file for this model
    if model not in RESULTS_FILES:
        raise ValueError(f"Unknown model: {model}. Valid: {list(RESULTS_FILES.keys())}")
    
    results_file = RESULTS_FILES[model]
    print(f"Results file: {results_file}")
    
    # Load pairs data
    pairs_df = pd.read_csv(PAIRS_FILE)
    print(f"Total pairs: {len(pairs_df)}")
    
    # Load results (predictions)
    results_df = pd.read_csv(results_file)
    print(f"Total predictions: {len(results_df)}")
    
    # Merge pairs with results
    df = pairs_df.merge(
        results_df[['zpid_1', 'zpid_2', 'pred_choice', 'true_choice', 'correct']],
        on=['zpid_1', 'zpid_2']
    )
    print(f"Merged: {len(df)} samples")
    
    # Compute features for stratification
    df = compute_feature_diffs(df)
    df = create_stratification_key(df)
    
    # Split by correctness
    correct_df = df[df['correct'] == True].copy()
    incorrect_df = df[df['correct'] == False].copy()
    
    print(f"\nCorrect predictions: {len(correct_df)}")
    print(f"Incorrect predictions: {len(incorrect_df)}")
    
    # Select samples (all if USE_ALL_SAMPLES, otherwise stratified subset)
    if USE_ALL_SAMPLES:
        print(f"\n[cuML MODE] Using ALL samples for maximum power!")
        correct_selected = correct_df.copy()
        incorrect_selected = incorrect_df.copy()
    else:
        print(f"\nSelecting {N_CORRECT} correct samples...")
        correct_selected = select_stratified_samples(correct_df, N_CORRECT, RANDOM_SEED)
        
        print(f"Selecting {N_INCORRECT} incorrect samples...")
        incorrect_selected = select_stratified_samples(incorrect_df, N_INCORRECT, RANDOM_SEED)
    
    # Analyze selections
    analyze_selection(correct_selected, correct_df, "CORRECT Predictions")
    analyze_selection(incorrect_selected, incorrect_df, "INCORRECT Predictions")
    
    # Combine all selected samples
    all_selected = pd.concat([correct_selected, incorrect_selected], ignore_index=True)
    all_selected['sample_group'] = ['correct'] * len(correct_selected) + ['incorrect'] * len(incorrect_selected)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SELECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_selected)}")
    print(f"  - Correct: {len(correct_selected)} ({len(correct_selected)/len(all_selected)*100:.1f}%)")
    print(f"  - Incorrect: {len(incorrect_selected)} ({len(incorrect_selected)/len(all_selected)*100:.1f}%)")
    
    # Save selected samples
    output_file = OUTPUT_DIR / f"selected_samples_{model}_{EXPERIMENT}.csv"
    all_selected.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    
    # Save selection indices for activation extraction
    indices_file = OUTPUT_DIR / f"selected_indices_{model}_{EXPERIMENT}.json"
    selection_info = {
        'model': model,
        'experiment': EXPERIMENT,
        'n_correct': N_CORRECT,
        'n_incorrect': N_INCORRECT,
        'total': len(all_selected),
        'random_seed': RANDOM_SEED,
        'pair_indices': all_selected.index.tolist(),
        'zpid_pairs': all_selected[['zpid_1', 'zpid_2']].values.tolist()
    }
    with open(indices_file, 'w') as f:
        json.dump(selection_info, f, indent=2)
    print(f"Saved indices to: {indices_file}")
    
    # Also save separate files for correct/incorrect for analysis
    correct_selected.to_csv(OUTPUT_DIR / f"correct_samples_{model}_{EXPERIMENT}.csv", index=False)
    incorrect_selected.to_csv(OUTPUT_DIR / f"incorrect_samples_{model}_{EXPERIMENT}.csv", index=False)
    
    return all_selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select samples for amnesic probing")
    parser.add_argument('--model', type=str, default='llama-3.2-3b',
                       choices=['llama-3.2-3b', 'qwen3-4b'],
                       help='Model to select samples for')
    args = parser.parse_args()
    
    selected = main(model=args.model)
