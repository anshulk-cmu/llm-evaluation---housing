# -*- coding: utf-8 -*-
"""
Results processing and saving for LLM Real Estate Valuation Experiment
"""

import os
import numpy as np
import pandas as pd
from config import RESULTS_DIR


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def calculate_accuracy(results_df):
    """
    Calculate accuracy of predictions vs ground truth.
    
    Args:
        results_df: DataFrame with pred_choice, price.y_1, price.y_2 columns
        
    Returns:
        dict with accuracy, n_valid, n_total
    """
    # Compute true choice from prices
    results_df["true_choice"] = np.where(
        results_df["price.y_1"] > results_df["price.y_2"], 1,
        np.where(results_df["price.y_2"] > results_df["price.y_1"], 2, np.nan)
    )
    
    # Check correctness
    results_df["correct"] = results_df["pred_choice"] == results_df["true_choice"]
    
    # Calculate accuracy on valid predictions
    valid = results_df.dropna(subset=["pred_choice", "true_choice"])
    accuracy = valid["correct"].mean() if len(valid) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "n_valid": len(valid),
        "n_total": len(results_df),
    }


def save_results(results_df, model_name, condition_name):
    """
    Save results DataFrame to CSV.
    
    Args:
        results_df: DataFrame with experiment results
        model_name: Name of the model
        condition_name: Name of the condition
    """
    ensure_results_dir()
    filename = f"results_{model_name}_{condition_name}.csv"
    filepath = os.path.join(RESULTS_DIR, filename)
    results_df.to_csv(filepath, index=False)
    print(f"  Saved: {filename}")


def save_experiment_summary(summary_rows, model_name):
    """
    Save experiment summary CSV.
    
    Args:
        summary_rows: List of dicts with condition results
        model_name: Name of the model
    """
    ensure_results_dir()
    summary_df = pd.DataFrame(summary_rows)
    filename = f"experiment_summary_{model_name}.csv"
    filepath = os.path.join(RESULTS_DIR, filename)
    summary_df.to_csv(filepath, index=False)
    print(f"\nSaved: {filename}")


def compute_zestimate_baseline(pairs_df):
    """
    Compute Zestimate baseline accuracy.
    
    Args:
        pairs_df: Full pairs DataFrame with zestimate columns
        
    Returns:
        float: Zestimate accuracy
    """
    p1 = pairs_df["price.y_1"]
    p2 = pairs_df["price.y_2"]
    true_choice = np.where(p1 > p2, 1, np.where(p2 > p1, 2, np.nan))
    
    z1 = pairs_df["zestimate_linear_1"]
    z2 = pairs_df["zestimate_linear_2"]
    zestimate_choice = np.where(z1 > z2, 1, np.where(z2 > z1, 2, np.nan))
    
    # Calculate accuracy
    valid_mask = ~np.isnan(true_choice) & ~np.isnan(zestimate_choice)
    if valid_mask.sum() == 0:
        return 0.0
    
    accuracy = (true_choice[valid_mask] == zestimate_choice[valid_mask]).mean()
    return accuracy


def print_experiment_summary(model_results, model_name, sample_size, zestimate_acc=None):
    """
    Print formatted experiment summary table.
    
    Args:
        model_results: Dict of condition_name -> result dict
        model_name: Name of the model
        sample_size: Number of samples per condition
        zestimate_acc: Zestimate baseline accuracy (optional)
    """
    baseline_acc = model_results.get("0_baseline_temp0", {}).get("accuracy", 0)
    
    print("\n" + "="*70)
    print(f"EXPERIMENT SUMMARY: {model_name.upper()} (n={sample_size} pairs per condition)")
    print("="*70)
    
    if zestimate_acc is not None:
        print(f"\nZestimate Benchmark: {zestimate_acc:.3f}")
    
    print("-"*70)
    print(f"{'Condition':<35} {'Accuracy':>10} {'vs Baseline':>12} {'Time(s)':>10}")
    print("-"*70)
    
    for cond_name, result in model_results.items():
        acc = result["accuracy"]
        delta = (acc - baseline_acc) * 100 if baseline_acc > 0 else 0
        time_s = result["time_sec"]
        print(f"{cond_name:<35} {acc:>10.3f} {delta:>+11.1f}% {time_s:>10.1f}")
    
    print("="*70)
