#!/usr/bin/env python3
"""
Logistic Regression on Raw Input Features - V2 (Enhanced)

Improvements over V1:
- Visualization: feature distributions, correlations, coefficients
- Better feature engineering: ratios, log transforms, interactions
- Multiple model variants compared
- Enhanced analysis of counterintuitive coefficients
"""

import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime

# Configuration
DATA_PATH = Path("data/pairs_20pct_price_diff.csv")
OUTPUT_DIR = Path("data/raw_feature_logit_results")
PLOTS_DIR = OUTPUT_DIR / "plots"
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Hyperparameter search space
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}


def load_and_prepare_data(data_path: Path) -> tuple:
    """Load CSV and extract features + target."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} property pairs")
    
    # Store raw values for feature engineering
    raw_data = {
        'bedrooms_1': df['bedrooms_1'],
        'bedrooms_2': df['bedrooms_2'],
        'bathrooms_1': df['bathrooms_1'],
        'bathrooms_2': df['bathrooms_2'],
        'lot_1': df['lot_1'],
        'lot_2': df['lot_2'],
        'year_1': df['yearBuilt_1'],
        'year_2': df['yearBuilt_2'],
        'price_1': df['price.y_1'],
        'price_2': df['price.y_2'],
    }
    
    # Target: 1 if property 1 is more expensive
    target = (df['price.y_1'] > df['price.y_2']).astype(int)
    
    print(f"\nTarget distribution: {target.mean():.1%} property 1 more expensive")
    
    return raw_data, target, df


def create_feature_sets(raw_data: dict) -> dict:
    """Create multiple feature set variants for comparison."""
    
    feature_sets = {}
    
    # V1: Basic differences (original)
    feature_sets['v1_basic_diff'] = pd.DataFrame({
        'bedrooms_diff': raw_data['bedrooms_1'] - raw_data['bedrooms_2'],
        'bathrooms_diff': raw_data['bathrooms_1'] - raw_data['bathrooms_2'],
        'lot_diff': raw_data['lot_1'] - raw_data['lot_2'],
        'year_diff': raw_data['year_1'] - raw_data['year_2'],
    })

    # V2: Percentage/ratio differences (better for different scales)
    avg_lot = (raw_data['lot_1'] + raw_data['lot_2']) / 2
    avg_year = (raw_data['year_1'] + raw_data['year_2']) / 2

    feature_sets['v2_pct_diff'] = pd.DataFrame({
        'bedrooms_diff': raw_data['bedrooms_1'] - raw_data['bedrooms_2'],
        'bathrooms_diff': raw_data['bathrooms_1'] - raw_data['bathrooms_2'],
        'lot_pct_diff': (raw_data['lot_1'] - raw_data['lot_2']) / avg_lot.replace(0, 1),
        'year_diff': raw_data['year_1'] - raw_data['year_2'],
    })

    # V3: Log-transformed lot size (common in real estate)
    log_lot_1 = np.log1p(raw_data['lot_1'].clip(lower=1))
    log_lot_2 = np.log1p(raw_data['lot_2'].clip(lower=1))

    feature_sets['v3_log_lot'] = pd.DataFrame({
        'bedrooms_diff': raw_data['bedrooms_1'] - raw_data['bedrooms_2'],
        'bathrooms_diff': raw_data['bathrooms_1'] - raw_data['bathrooms_2'],
        'log_lot_diff': log_lot_1 - log_lot_2,
        'year_diff': raw_data['year_1'] - raw_data['year_2'],
    })

    # V4: Binary indicators (simpler, more robust)
    feature_sets['v4_binary'] = pd.DataFrame({
        'bed_1_more': (raw_data['bedrooms_1'] > raw_data['bedrooms_2']).astype(int),
        'bath_1_more': (raw_data['bathrooms_1'] > raw_data['bathrooms_2']).astype(int),
        'lot_1_larger': (raw_data['lot_1'] > raw_data['lot_2']).astype(int),
        'year_1_newer': (raw_data['year_1'] > raw_data['year_2']).astype(int),
    })

    # V5: With interaction terms
    beds_diff = raw_data['bedrooms_1'] - raw_data['bedrooms_2']
    baths_diff = raw_data['bathrooms_1'] - raw_data['bathrooms_2']
    lot_diff = raw_data['lot_1'] - raw_data['lot_2']
    year_diff = raw_data['year_1'] - raw_data['year_2']

    feature_sets['v5_interactions'] = pd.DataFrame({
        'bedrooms_diff': beds_diff,
        'bathrooms_diff': baths_diff,
        'lot_diff': lot_diff,
        'year_diff': year_diff,
        'beds_x_baths': beds_diff * baths_diff,
        'lot_x_year': lot_diff * year_diff / 1000,  # scaled
        'baths_x_lot': baths_diff * lot_diff / 1000,
    })

    # V6: Separate features (not differences)
    feature_sets['v6_separate'] = pd.DataFrame({
        'bedrooms_1': raw_data['bedrooms_1'],
        'bedrooms_2': raw_data['bedrooms_2'],
        'bathrooms_1': raw_data['bathrooms_1'],
        'bathrooms_2': raw_data['bathrooms_2'],
        'lot_1': raw_data['lot_1'] / 1000,  # scale to thousands
        'lot_2': raw_data['lot_2'] / 1000,
        'year_1': (raw_data['year_1'] - 1900) / 100,  # normalize years
        'year_2': (raw_data['year_2'] - 1900) / 100,
    })
    
    return feature_sets


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
    
    return {
        'train': {'X': features.iloc[train_idx], 'y': target.iloc[train_idx], 'idx': train_idx},
        'val': {'X': features.iloc[val_idx], 'y': target.iloc[val_idx], 'idx': val_idx},
        'test': {'X': features.iloc[test_idx], 'y': target.iloc[test_idx], 'idx': test_idx}
    }


def train_single_model(splits: dict, c_values: list, verbose: bool = True) -> dict:
    """Train and tune a single logistic regression model."""
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(splits['train']['X'])
    X_val = scaler.transform(splits['val']['X'])
    X_test = scaler.transform(splits['test']['X'])
    
    y_train = splits['train']['y']
    y_val = splits['val']['y']
    y_test = splits['test']['y']
    
    # Tune C
    best_c, best_val_acc = None, 0
    for c in c_values:
        model = LogisticRegression(C=c, solver='lbfgs', max_iter=1000, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_c = c
    
    # Final model
    final_model = LogisticRegression(C=best_c, solver='lbfgs', max_iter=1000, random_state=RANDOM_SEED)
    final_model.fit(X_train, y_train)
    
    # Test evaluation
    test_pred = final_model.predict(X_test)
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_prob)
    
    # Bootstrap CI (use same indices for y and pred)
    rng = np.random.RandomState(RANDOM_SEED)
    bootstrap_accs = []
    for _ in range(1000):
        idx = rng.choice(len(y_test), len(y_test), replace=True)
        bootstrap_accs.append(accuracy_score(y_test.iloc[idx], test_pred[idx]))
    
    feature_names = splits['train']['X'].columns.tolist()
    
    return {
        'best_c': best_c,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'ci_lower': np.percentile(bootstrap_accs, 2.5),
        'ci_upper': np.percentile(bootstrap_accs, 97.5),
        'coefficients': dict(zip(feature_names, final_model.coef_[0])),
        'intercept': final_model.intercept_[0],
        'model': final_model,
        'scaler': scaler,
        'test_pred': test_pred,
        'test_prob': test_prob
    }


def compare_feature_sets(feature_sets: dict, target: pd.Series) -> pd.DataFrame:
    """Compare all feature set variants."""
    
    print("\n" + "=" * 70)
    print("COMPARING FEATURE SET VARIANTS")
    print("=" * 70)
    
    results = []
    
    for name, features in feature_sets.items():
        splits = split_data(features, target, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
        result = train_single_model(splits, C_VALUES, verbose=False)
        
        results.append({
            'variant': name,
            'n_features': len(features.columns),
            'val_acc': result['val_acc'],
            'test_acc': result['test_acc'],
            'test_auc': result['test_auc'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'best_c': result['best_c']
        })
        
        print(f"\n{name}:")
        print(f"  Features: {list(features.columns)}")
        print(f"  Val Acc: {result['val_acc']:.4f}, Test Acc: {result['test_acc']:.4f}, AUC: {result['test_auc']:.4f}")
    
    return pd.DataFrame(results).sort_values('test_acc', ascending=False)


def plot_feature_distributions(raw_data: dict, target: pd.Series, output_dir: Path):
    """Plot feature distributions by target class."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    features_to_plot = [
        ('bedrooms_diff', raw_data['bedrooms_1'] - raw_data['bedrooms_2']),
        ('bathrooms_diff', raw_data['bathrooms_1'] - raw_data['bathrooms_2']),
        ('lot_diff', (raw_data['lot_1'] - raw_data['lot_2']) / 1000),  # in thousands
        ('year_diff', raw_data['year_1'] - raw_data['year_2']),
    ]
    
    for ax, (name, values) in zip(axes.flatten(), features_to_plot):
        # Split by target
        vals_0 = values[target == 0]
        vals_1 = values[target == 1]
        
        ax.hist(vals_0, bins=30, alpha=0.6, label='P2 more expensive', color=COLORS['negative'])
        ax.hist(vals_1, bins=30, alpha=0.6, label='P1 more expensive', color=COLORS['positive'])
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel(name + (' (thousands)' if 'lot' in name else ''))
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {name}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'feature_distributions.png'}")


def plot_correlation_matrix(raw_data: dict, target: pd.Series, output_dir: Path):
    """Plot correlation matrix of features with target."""
    
    df = pd.DataFrame({
        'bedrooms_diff': raw_data['bedrooms_1'] - raw_data['bedrooms_2'],
        'bathrooms_diff': raw_data['bathrooms_1'] - raw_data['bathrooms_2'],
        'lot_diff': raw_data['lot_1'] - raw_data['lot_2'],
        'year_diff': raw_data['year_1'] - raw_data['year_2'],
        'price_1_higher': target
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'correlation_matrix.png'}")


def plot_coefficient_comparison(comparison_df: pd.DataFrame, output_dir: Path):
    """Plot model comparison bar chart."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(comparison_df))
    width = 0.6
    
    colors = [COLORS['positive'] if acc >= 0.63 else COLORS['neutral'] 
              for acc in comparison_df['test_acc']]
    
    bars = ax.bar(x, comparison_df['test_acc'], width, color=colors, edgecolor='black', alpha=0.8)
    
    # Error bars
    yerr_lower = comparison_df['test_acc'] - comparison_df['ci_lower']
    yerr_upper = comparison_df['ci_upper'] - comparison_df['test_acc']
    ax.errorbar(x, comparison_df['test_acc'], yerr=[yerr_lower, yerr_upper], 
                fmt='none', color='black', capsize=5)
    
    # Add LLM baseline
    ax.axhline(0.593, color='red', linestyle='--', linewidth=2, label='Llama-3.2-3B (59.3%)')
    ax.axhline(0.613, color='orange', linestyle='--', linewidth=2, label='Activation Probe (61.3%)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['variant'], rotation=45, ha='right')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Feature Set Comparison: Test Accuracy')
    ax.set_ylim(0.5, 0.75)
    ax.legend(loc='upper right')
    
    # Add value labels
    for bar, acc in zip(bars, comparison_df['test_acc']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'model_comparison.png'}")


def plot_best_model_coefficients(result: dict, variant_name: str, output_dir: Path):
    """Plot coefficients of the best model."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    coefs = result['coefficients']
    features = list(coefs.keys())
    values = list(coefs.values())
    
    colors = [COLORS['positive'] if v > 0 else COLORS['negative'] for v in values]
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Coefficient (standardized)')
    ax.set_title(f'Feature Coefficients: {variant_name}\n(positive = P1 more expensive when feature is higher)')
    
    # Add value labels
    for bar, val in zip(bars, values):
        x_pos = val + 0.02 if val > 0 else val - 0.02
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                ha=ha, va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_model_coefficients.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'best_model_coefficients.png'}")


def plot_feature_importance_by_accuracy(raw_data: dict, target: pd.Series, output_dir: Path):
    """Train single-feature models to show individual feature predictive power."""
    
    single_feature_results = []
    
    features = {
        'bedrooms_diff': raw_data['bedrooms_1'] - raw_data['bedrooms_2'],
        'bathrooms_diff': raw_data['bathrooms_1'] - raw_data['bathrooms_2'],
        'lot_diff': raw_data['lot_1'] - raw_data['lot_2'],
        'year_diff': raw_data['year_1'] - raw_data['year_2'],
    }

    for name, values in features.items():
        df = pd.DataFrame({name: values})
        splits = split_data(df, target, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
        result = train_single_model(splits, C_VALUES, verbose=False)
        single_feature_results.append({
            'feature': name,
            'test_acc': result['test_acc'],
            'test_auc': result['test_auc']
        })
    
    single_df = pd.DataFrame(single_feature_results).sort_values('test_acc', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    y_pos = np.arange(len(single_df))
    bars = ax.barh(y_pos, single_df['test_acc'], color=COLORS['neutral'], edgecolor='black', alpha=0.8)
    
    ax.axvline(0.5, color='gray', linestyle=':', label='Random (50%)')
    ax.axvline(0.593, color='red', linestyle='--', label='LLM (59.3%)')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(single_df['feature'])
    ax.set_xlabel('Test Accuracy (single feature only)')
    ax.set_title('Individual Feature Predictive Power')
    ax.set_xlim(0.45, 0.7)
    ax.legend(loc='lower right')
    
    for bar, acc in zip(bars, single_df['test_acc']):
        ax.text(acc + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{acc:.1%}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'single_feature_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'single_feature_accuracy.png'}")
    
    return single_df


def analyze_counterintuitive_coefficients(raw_data: dict, target: pd.Series):
    """Analyze why year and lot size have negative coefficients."""

    print("\n" + "=" * 70)
    print("ANALYZING COUNTERINTUITIVE COEFFICIENTS")
    print("=" * 70)

    # Create dataframe for analysis
    df = pd.DataFrame({
        'year_diff': raw_data['year_1'] - raw_data['year_2'],
        'lot_diff': raw_data['lot_1'] - raw_data['lot_2'],
        'baths_diff': raw_data['bathrooms_1'] - raw_data['bathrooms_2'],
        'beds_diff': raw_data['bedrooms_1'] - raw_data['bedrooms_2'],
        'price_1_higher': target
    })

    # Check: When P1 is newer, is it more expensive?
    newer = df[df['year_diff'] > 0]
    older = df[df['year_diff'] < 0]

    print(f"\nYear analysis:")
    print(f"  When P1 is NEWER: {newer['price_1_higher'].mean():.1%} of time P1 is more expensive (n={len(newer)})")
    print(f"  When P1 is OLDER: {older['price_1_higher'].mean():.1%} of time P1 is more expensive (n={len(older)})")

    # Check: When P1 has larger lot, is it more expensive?
    larger = df[df['lot_diff'] > 100]
    smaller = df[df['lot_diff'] < -100]

    print(f"\nLot size analysis:")
    print(f"  When P1 is LARGER: {larger['price_1_higher'].mean():.1%} of time P1 is more expensive (n={len(larger)})")
    print(f"  When P1 is SMALLER: {smaller['price_1_higher'].mean():.1%} of time P1 is more expensive (n={len(smaller)})")

    # Check correlations between features
    print(f"\nFeature correlations (explain confounding):")
    print(f"  year_diff vs baths_diff: {df['year_diff'].corr(df['baths_diff']):.3f}")
    print(f"  year_diff vs beds_diff:  {df['year_diff'].corr(df['beds_diff']):.3f}")
    print(f"  lot_diff vs baths_diff: {df['lot_diff'].corr(df['baths_diff']):.3f}")


def save_all_results(comparison_df: pd.DataFrame, best_result: dict, 
                     best_variant: str, output_dir: Path):
    """Save all results to files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Comparison CSV
    comparison_df.to_csv(output_dir / 'feature_set_comparison.csv', index=False)
    
    # Best model summary
    summary = {
        'experiment': 'raw_feature_logit_v2',
        'timestamp': datetime.now().isoformat(),
        'best_variant': best_variant,
        'test_accuracy': best_result['test_acc'],
        'test_auc': best_result['test_auc'],
        'ci_95': [best_result['ci_lower'], best_result['ci_upper']],
        'coefficients': best_result['coefficients'],
        'best_c': best_result['best_c'],
        'comparison': comparison_df.to_dict(orient='records')
    }
    
    with open(output_dir / 'raw_feature_logit_v2_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved comparison to {output_dir / 'feature_set_comparison.csv'}")
    print(f"Saved summary to {output_dir / 'raw_feature_logit_v2_summary.json'}")


def main():
    print("=" * 70)
    print("Raw Feature Logistic Regression - V2 (Enhanced)")
    print("=" * 70)
    
    # Create output directories
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    raw_data, target, df = load_and_prepare_data(DATA_PATH)
    
    # Create feature set variants
    feature_sets = create_feature_sets(raw_data)
    
    # Compare all variants
    comparison_df = compare_feature_sets(feature_sets, target)
    
    print("\n" + "=" * 70)
    print("FEATURE SET RANKING")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    
    # Get best variant
    best_variant = comparison_df.iloc[0]['variant']
    best_features = feature_sets[best_variant]
    best_splits = split_data(best_features, target, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
    best_result = train_single_model(best_splits, C_VALUES, verbose=False)
    
    # Analyze counterintuitive coefficients
    analyze_counterintuitive_coefficients(raw_data, target)
    
    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plot_feature_distributions(raw_data, target, PLOTS_DIR)
    plot_correlation_matrix(raw_data, target, PLOTS_DIR)
    plot_coefficient_comparison(comparison_df, PLOTS_DIR)
    plot_best_model_coefficients(best_result, best_variant, PLOTS_DIR)
    single_feat_df = plot_feature_importance_by_accuracy(raw_data, target, PLOTS_DIR)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nBest Model: {best_variant}")
    print(f"  Test Accuracy: {best_result['test_acc']:.4f} ({best_result['test_acc']*100:.1f}%)")
    print(f"  Test AUC:      {best_result['test_auc']:.4f}")
    print(f"  95% CI:        [{best_result['ci_lower']:.4f}, {best_result['ci_upper']:.4f}]")
    
    print(f"\nSingle Feature Ranking:")
    for _, row in single_feat_df.iterrows():
        print(f"  {row['feature']:>15}: {row['test_acc']:.1%}")
    
    print("\n" + "-" * 70)
    print("COMPARISON WITH LLM:")
    print("-" * 70)
    print(f"  Best Raw Feature Logit: {best_result['test_acc']*100:.1f}%")
    print(f"  Llama-3.2-3B (fewshot): 59.3%")
    print(f"  Activation Probe:       61.3%")
    print(f"  Difference:             {(best_result['test_acc'] - 0.593)*100:+.1f} percentage points")
    
    # Save results
    save_all_results(comparison_df, best_result, best_variant, OUTPUT_DIR)
    
    print("\nDone! Check plots in:", PLOTS_DIR)
    return comparison_df, best_result


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
    log_path = log_dir / f"raw_feature_logit_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = Tee(log_path)
    print(f"Logging to {log_path}")
    main()
