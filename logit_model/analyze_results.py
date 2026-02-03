# -*- coding: utf-8 -*-
"""
Analyze and Plot Logistic Regression Price Predictor Results

Generates layer-wise accuracy curves with baseline comparisons.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import config
import importlib.util
_THIS_DIR = Path(__file__).parent.resolve()
config_spec = importlib.util.spec_from_file_location("logit_config", str(_THIS_DIR / "logit_config.py"))
config = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config)


def load_results(model: str, condition: str) -> pd.DataFrame:
    """Load results CSV for a model-condition pair."""
    path = config.RESULTS_DIR / f"logit_results_{model}_{condition}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    return pd.read_csv(path)


def plot_accuracy_vs_layer(results: dict, output_path: Path):
    """
    Plot layer-wise accuracy curves for all models.

    Args:
        results: Dict mapping model names to DataFrames
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'llama-3.2-3b': '#2ecc71', 'qwen3-4b': '#3498db'}
    markers = {'llama-3.2-3b': 'o', 'qwen3-4b': 's'}

    for model, df in results.items():
        color = colors.get(model, '#95a5a6')
        marker = markers.get(model, 'o')

        ax.plot(df['layer'], df['test_accuracy'],
                marker=marker, markersize=6, linewidth=2,
                color=color, label=f'{model}', alpha=0.9)

        # Confidence band
        ax.fill_between(df['layer'], df['ci_lower'], df['ci_upper'],
                       alpha=0.15, color=color)

        # Mark best layer
        best_idx = df['test_accuracy'].idxmax()
        best_row = df.loc[best_idx]
        ax.scatter([best_row['layer']], [best_row['test_accuracy']],
                  s=150, color=color, marker='*', zorder=5, edgecolors='white', linewidth=1.5)
        ax.annotate(f"L{int(best_row['layer'])}: {best_row['test_accuracy']:.1%}",
                   (best_row['layer'], best_row['test_accuracy']),
                   textcoords="offset points", xytext=(10, 10),
                   fontsize=10, fontweight='bold', color=color)

    # Baseline reference lines
    ax.axhline(y=config.BASELINE_CHANCE, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.axhline(y=config.BASELINE_LLM_OUTPUT, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)
    ax.axhline(y=config.BASELINE_ZESTIMATE, color='#9b59b6', linestyle='-.', linewidth=2, alpha=0.8)

    # Labels for baselines
    max_layer = max(df['layer'].max() for df in results.values())
    ax.text(max_layer + 0.5, config.BASELINE_CHANCE, 'Chance (50%)',
            va='center', fontsize=9, color='gray')
    ax.text(max_layer + 0.5, config.BASELINE_LLM_OUTPUT, 'LLM Output (~60%)',
            va='center', fontsize=9, color='#e74c3c')
    ax.text(max_layer + 0.5, config.BASELINE_ZESTIMATE, 'Zestimate (77.6%)',
            va='center', fontsize=9, color='#9b59b6')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Price Prediction Accuracy by Layer\n(Logistic Regression on LLM Activations)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0.45, 0.85)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_method_comparison(results: dict, output_path: Path):
    """
    Bar chart comparing: Chance, LLM Output, Best Probe, Zestimate.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect best accuracies per model
    probe_results = {}
    for model, df in results.items():
        best_idx = df['test_accuracy'].idxmax()
        best_row = df.loc[best_idx]
        probe_results[model] = {
            'accuracy': best_row['test_accuracy'],
            'ci_lower': best_row['ci_lower'],
            'ci_upper': best_row['ci_upper'],
            'layer': int(best_row['layer'])
        }

    # Categories
    categories = ['Chance', 'LLM Output'] + [f'Probe ({m})' for m in results.keys()] + ['Zestimate']
    values = [config.BASELINE_CHANCE, config.BASELINE_LLM_OUTPUT]
    errors_lower = [0, 0]
    errors_upper = [0, 0]
    colors = ['#bdc3c7', '#e74c3c']

    model_colors = {'llama-3.2-3b': '#2ecc71', 'qwen3-4b': '#3498db'}
    for model in results.keys():
        values.append(probe_results[model]['accuracy'])
        errors_lower.append(probe_results[model]['accuracy'] - probe_results[model]['ci_lower'])
        errors_upper.append(probe_results[model]['ci_upper'] - probe_results[model]['accuracy'])
        colors.append(model_colors.get(model, '#95a5a6'))

    values.append(config.BASELINE_ZESTIMATE)
    errors_lower.append(0)
    errors_upper.append(0)
    colors.append('#9b59b6')

    x = np.arange(len(categories))
    bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=1.5)

    # Error bars for probes
    for i, (el, eu) in enumerate(zip(errors_lower, errors_upper)):
        if el > 0 or eu > 0:
            ax.errorbar(x[i], values[i], yerr=[[el], [eu]],
                       fmt='none', color='black', capsize=5, capthick=2)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Price Prediction: Method Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.95)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_summary(results: dict):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY: LOGISTIC REGRESSION PRICE PREDICTOR")
    print("=" * 60)

    for model, df in results.items():
        best_idx = df['test_accuracy'].idxmax()
        best = df.loc[best_idx]
        print(f"\n{model}:")
        print(f"  Best Layer: {int(best['layer'])}")
        print(f"  Test Accuracy: {best['test_accuracy']:.4f}")
        print(f"  95% CI: [{best['ci_lower']:.4f}, {best['ci_upper']:.4f}]")
        print(f"  AUC: {best['test_auc']:.4f}")
        print(f"  Mean Accuracy (all layers): {df['test_accuracy'].mean():.4f}")

    print("\n" + "-" * 60)
    print("BASELINES:")
    print(f"  Chance: {config.BASELINE_CHANCE:.1%}")
    print(f"  LLM Output: {config.BASELINE_LLM_OUTPUT:.1%}")
    print(f"  Zestimate: {config.BASELINE_ZESTIMATE:.1%}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default=config.DEFAULT_CONDITION)
    parser.add_argument("--models", nargs="+", default=list(config.MODELS.keys()))
    args = parser.parse_args()

    # Load results for each model
    results = {}
    for model in args.models:
        try:
            results[model] = load_results(model, args.condition)
            print(f"Loaded results for {model}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if not results:
        print("No results found. Run train_price_probe.py first.")
        return

    # Generate plots
    plots_dir = config.PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_vs_layer(results, plots_dir / f"accuracy_vs_layer_{args.condition}.png")
    plot_method_comparison(results, plots_dir / f"method_comparison_{args.condition}.png")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
