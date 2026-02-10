#!/usr/bin/env python3
"""
Logistic Regression on Raw Input Features

This script trains a logistic regression model to predict which property
in a pair is more expensive, using only the raw input features:
- bedrooms difference
- bathrooms difference  
- lot size difference
- year built difference

This establishes a baseline for whether the features themselves are
linearly predictive of price.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import json
from pathlib import Path
from datetime import datetime

# Configuration
DATA_PATH = Path("data/pairs_20pct_price_diff.csv")
OUTPUT_DIR = Path("data/raw_feature_logit_results")
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Hyperparameter search space
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def load_and_prepare_data(data_path: Path) -> tuple:
    """Load CSV and extract features + target."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} property pairs")
    
    # Extract raw features
    features = pd.DataFrame({
        'bedrooms_diff': df['bedrooms_1'] - df['bedrooms_2'],
        'bathrooms_diff': df['bathrooms_1'] - df['bathrooms_2'],
        'lot_diff': df['lot_1'] - df['lot_2'],
        'year_diff': df['yearBuilt_1'] - df['yearBuilt_2'],
    })
    
    # Target: 1 if property 1 is more expensive
    target = (df['price.y_1'] > df['price.y_2']).astype(int)
    
    print(f"\nFeature statistics:")
    print(features.describe())
    print(f"\nTarget distribution: {target.mean():.1%} property 1 more expensive")
    
    return features, target, df


def split_data(features: pd.DataFrame, target: pd.Series, 
               train_ratio: float, val_ratio: float, seed: int) -> dict:
    """Split data into train/val/test with fixed seed."""
    n = len(features)
    indices = np.random.RandomState(seed).permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    splits = {
        'train': {
            'X': features.iloc[train_idx],
            'y': target.iloc[train_idx],
            'n': len(train_idx)
        },
        'val': {
            'X': features.iloc[val_idx],
            'y': target.iloc[val_idx],
            'n': len(val_idx)
        },
        'test': {
            'X': features.iloc[test_idx],
            'y': target.iloc[test_idx],
            'n': len(test_idx)
        }
    }
    
    print(f"\nData split (seed={seed}):")
    print(f"  Train: {splits['train']['n']} ({splits['train']['n']/n:.1%})")
    print(f"  Val:   {splits['val']['n']} ({splits['val']['n']/n:.1%})")
    print(f"  Test:  {splits['test']['n']} ({splits['test']['n']/n:.1%})")
    
    return splits


def train_and_tune(splits: dict, c_values: list) -> dict:
    """Train logistic regression with hyperparameter tuning on validation set."""
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(splits['train']['X'])
    X_val_scaled = scaler.transform(splits['val']['X'])
    X_test_scaled = scaler.transform(splits['test']['X'])
    
    y_train = splits['train']['y']
    y_val = splits['val']['y']
    y_test = splits['test']['y']
    
    # Hyperparameter tuning
    print(f"\nTuning regularization (C) on validation set...")
    print("-" * 50)
    
    best_c = None
    best_val_acc = 0
    results_by_c = []
    
    for c in c_values:
        model = LogisticRegression(C=c, solver='lbfgs', max_iter=1000, random_state=RANDOM_SEED)
        model.fit(X_train_scaled, y_train)
        
        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        val_acc = accuracy_score(y_val, model.predict(X_val_scaled))
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])
        
        results_by_c.append({
            'C': c,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_auc': val_auc
        })
        
        print(f"  C={c:>6}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_auc={val_auc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_c = c
    
    print(f"\nBest C: {best_c} (val_acc={best_val_acc:.4f})")
    
    # Retrain with best C
    final_model = LogisticRegression(C=best_c, solver='lbfgs', max_iter=1000, random_state=RANDOM_SEED)
    final_model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    test_pred = final_model.predict(X_test_scaled)
    test_prob = final_model.predict_proba(X_test_scaled)[:, 1]
    
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_prob)
    
    # Bootstrap confidence interval for test accuracy
    n_bootstrap = 1000
    rng = np.random.RandomState(RANDOM_SEED)
    bootstrap_accs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_test), size=len(y_test), replace=True)
        boot_acc = accuracy_score(y_test.iloc[idx], test_pred[idx])
        bootstrap_accs.append(boot_acc)
    
    ci_lower = np.percentile(bootstrap_accs, 2.5)
    ci_upper = np.percentile(bootstrap_accs, 97.5)
    
    # Feature importance (coefficients)
    feature_names = splits['train']['X'].columns.tolist()
    coefficients = dict(zip(feature_names, final_model.coef_[0]))
    
    return {
        'best_c': best_c,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'coefficients': coefficients,
        'intercept': final_model.intercept_[0],
        'scaler_mean': dict(zip(feature_names, scaler.mean_)),
        'scaler_std': dict(zip(feature_names, scaler.scale_)),
        'results_by_c': results_by_c,
        'model': final_model,
        'scaler': scaler
    }


def print_results(results: dict, splits: dict):
    """Print final results summary."""
    print("\n" + "=" * 60)
    print("FINAL RESULTS: Raw Feature Logistic Regression")
    print("=" * 60)
    
    print(f"\nTest Set Performance (n={splits['test']['n']}):")
    print(f"  Accuracy: {results['test_acc']:.4f} ({results['test_acc']*100:.1f}%)")
    print(f"  AUC-ROC:  {results['test_auc']:.4f}")
    print(f"  95% CI:   [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
    
    print(f"\nFeature Coefficients (standardized):")
    for feat, coef in sorted(results['coefficients'].items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "+" if coef > 0 else ""
        print(f"  {feat:>15}: {direction}{coef:.4f}")
    print(f"  {'intercept':>15}: {results['intercept']:.4f}")
    
    print("\n" + "-" * 60)
    print("COMPARISON WITH LLM RESULTS:")
    print("-" * 60)
    print(f"  Raw Feature Logit:     {results['test_acc']*100:.1f}% accuracy")
    print(f"  Llama-3.2-3B (fewshot): 59.3% accuracy")
    print(f"  Qwen3-4B (fewshot):     ~59% accuracy")
    print(f"  Activation Probe:       61.3% accuracy (Llama best layer)")
    
    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION:")
    print("-" * 60)
    if results['test_acc'] > 0.70:
        print("  → Features ARE linearly predictive of price!")
        print("  → LLM significantly underperforms this simple baseline")
    elif results['test_acc'] > 0.63:
        print("  → Features have SOME linear predictive power")
        print("  → LLM performance is reasonable but could be better")
    else:
        print("  → Features are NOT strongly linearly predictive")
        print("  → Task is inherently hard; LLM does reasonably well")


def save_results(results: dict, splits: dict, output_dir: Path):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary JSON
    summary = {
        'experiment': 'raw_feature_logit',
        'timestamp': datetime.now().isoformat(),
        'data_split': {
            'train': splits['train']['n'],
            'val': splits['val']['n'],
            'test': splits['test']['n'],
            'seed': RANDOM_SEED
        },
        'best_c': results['best_c'],
        'test_accuracy': results['test_acc'],
        'test_auc': results['test_auc'],
        'ci_lower': results['ci_lower'],
        'ci_upper': results['ci_upper'],
        'coefficients': results['coefficients'],
        'intercept': results['intercept']
    }
    
    summary_path = output_dir / "raw_feature_logit_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")
    
    # Tuning results CSV
    tuning_df = pd.DataFrame(results['results_by_c'])
    tuning_path = output_dir / "raw_feature_logit_tuning.csv"
    tuning_df.to_csv(tuning_path, index=False)
    print(f"Saved tuning results to {tuning_path}")


def main():
    print("=" * 60)
    print("Raw Feature Logistic Regression Experiment")
    print("=" * 60)
    
    # Load data
    features, target, df = load_and_prepare_data(DATA_PATH)
    
    # Split data
    splits = split_data(features, target, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
    
    # Train and tune
    results = train_and_tune(splits, C_VALUES)
    
    # Print results
    print_results(results, splits)
    
    # Save results
    save_results(results, splits, OUTPUT_DIR)
    
    print("\nDone!")
    return results


class Tee:
    """Write to both stdout and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()


if __name__ == "__main__":
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"raw_feature_logit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = Tee(log_path)
    print(f"Logging to {log_path}")
    main()
