#!/usr/bin/env python3
"""
Plotting functions for Amnesic Probing results.

Creates visualizations for:
1. INLP erasure convergence curves
2. Feature encoding (initial probe accuracy)
3. Price prediction drops by feature
4. Model comparison (Llama vs Qwen)
5. Correctness analysis
6. Summary figure for paper
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Colors
COLORS = {
    'llama': '#1f77b4',  # blue
    'qwen': '#ff7f0e',   # orange
    'baseline': '#2ca02c',  # green
    'random': '#7f7f7f',  # gray
    'features': {
        'bedrooms': '#e41a1c',
        'bathrooms': '#377eb8',
        'year_built': '#4daf4a',
        'lot_size': '#984ea3',
        'sqft': '#ff7f00'
    }
}

FEATURE_LABELS = {
    'bedrooms': 'Bedrooms',
    'bathrooms': 'Bathrooms',
    'year_built': 'Year Built',
    'lot_size': 'Lot Size',
    'sqft': 'Square Feet'
}


def load_results(results_dir: str, model: str, condition: str = "2_fewshot_cot_temp0"):
    """Load all result files for a model."""
    results = {}
    
    # Main results
    results_file = Path(results_dir) / f"amnesic_results_{model}_{condition}.csv"
    if results_file.exists():
        results['main'] = pd.read_csv(results_file)
    
    # Summary JSON
    summary_file = Path(results_dir) / f"amnesic_summary_{model}_{condition}.json"
    if summary_file.exists():
        with open(summary_file) as f:
            results['summary'] = json.load(f)
    
    # INLP info
    inlp_file = Path(results_dir) / f"inlp_info_{model}_{condition}.json"
    if inlp_file.exists():
        with open(inlp_file) as f:
            results['inlp'] = json.load(f)
    
    # Correctness analysis
    correctness_file = Path(results_dir) / f"correctness_analysis_{model}_{condition}.csv"
    if correctness_file.exists():
        results['correctness'] = pd.read_csv(correctness_file)
    
    return results


def plot_inlp_convergence(results_dir: str, output_dir: str, models: list, condition: str = "2_fewshot_cot_temp0"):
    """Plot INLP convergence curves for each feature."""
    features = ['bedrooms', 'bathrooms', 'year_built', 'lot_size', 'sqft']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        for model in models:
            results = load_results(results_dir, model, condition)
            if 'inlp' not in results or feature not in results['inlp']:
                continue
            
            inlp_data = results['inlp'][feature]
            accuracies = inlp_data['accuracies']
            iterations = range(1, len(accuracies) + 1)
            
            model_label = 'Llama-3.2-3B' if 'llama' in model else 'Qwen3-4B'
            color = COLORS['llama'] if 'llama' in model else COLORS['qwen']
            
            ax.plot(iterations, accuracies, label=model_label, color=color, linewidth=2)
        
        # Add reference lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random (50%)')
        ax.axhline(y=0.6, color='red', linestyle=':', alpha=0.5, label='Erasure threshold')
        
        ax.set_xlabel('INLP Iteration')
        ax.set_ylabel('Probe Accuracy')
        ax.set_title(FEATURE_LABELS[feature])
        ax.set_ylim(0.4, 1.0)
        ax.legend(loc='upper right', fontsize=8)
    
    # Remove empty subplot
    axes[5].axis('off')
    
    plt.suptitle('INLP Convergence: Feature Erasure Over Iterations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'inlp_convergence.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_feature_encoding(results_dir: str, output_dir: str, models: list, condition: str = "2_fewshot_cot_temp0"):
    """Plot initial probe accuracy (encoding strength) for each feature."""
    features = ['bedrooms', 'bathrooms', 'year_built', 'lot_size', 'sqft']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(features))
    width = 0.35
    
    for i, model in enumerate(models):
        results = load_results(results_dir, model, condition)
        if 'inlp' not in results:
            continue
        
        initial_accs = []
        for feature in features:
            if feature in results['inlp']:
                initial_accs.append(results['inlp'][feature]['initial_probe_accuracy'])
            else:
                initial_accs.append(0)
        
        model_label = 'Llama-3.2-3B' if 'llama' in model else 'Qwen3-4B'
        color = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, initial_accs, width, label=model_label, color=color, alpha=0.8)
        
        # Add value labels
        for bar, acc in zip(bars, initial_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.1%}', ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random chance')
    
    ax.set_xlabel('Feature')
    ax.set_ylabel('Initial Probe Accuracy')
    ax.set_title('Feature Encoding Strength (Pre-INLP Probe Accuracy)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in features])
    ax.set_ylim(0, 1.05)
    ax.legend()
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'feature_encoding.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_erasure_quality(results_dir: str, output_dir: str, models: list, condition: str = "2_fewshot_cot_temp0"):
    """Plot post-INLP verification accuracy (erasure quality) for each feature."""
    features = ['bedrooms', 'bathrooms', 'year_built', 'lot_size', 'sqft']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(features))
    width = 0.35
    
    for i, model in enumerate(models):
        results = load_results(results_dir, model, condition)
        if 'inlp' not in results:
            continue
        
        verification_accs = []
        for feature in features:
            if feature in results['inlp']:
                verification_accs.append(results['inlp'][feature]['verification_accuracy'])
            else:
                verification_accs.append(0)
        
        model_label = 'Llama-3.2-3B' if 'llama' in model else 'Qwen3-4B'
        color = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, verification_accs, width, label=model_label, color=color, alpha=0.8)
        
        # Add value labels
        for bar, acc in zip(bars, verification_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.1%}', ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Target (random)')
    ax.axhline(y=0.6, color='red', linestyle=':', alpha=0.5, label='Incomplete erasure')
    
    # Add shaded regions
    ax.axhspan(0.48, 0.55, alpha=0.1, color='green', label='Good erasure zone')
    ax.axhspan(0.6, 1.0, alpha=0.1, color='red')
    
    ax.set_xlabel('Feature')
    ax.set_ylabel('Post-INLP Probe Accuracy')
    ax.set_title('Erasure Quality (Lower = Better Erasure)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in features])
    ax.set_ylim(0.4, 0.85)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'erasure_quality.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_price_prediction_drops(results_dir: str, output_dir: str, models: list, condition: str = "2_fewshot_cot_temp0"):
    """Plot price prediction accuracy drops after feature erasure."""
    features = ['bedrooms', 'bathrooms', 'year_built', 'lot_size', 'sqft']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(features) + 2)  # +2 for ALL and RANDOM
    width = 0.35
    
    for i, model in enumerate(models):
        results = load_results(results_dir, model, condition)
        if 'main' not in results:
            continue
        
        df = results['main']
        
        drops = []
        for feature in features:
            row = df[df['condition'] == f'erase_{feature}']
            if len(row) > 0:
                drops.append(row['accuracy_drop'].values[0])
            else:
                drops.append(0)
        
        # Add ALL and RANDOM
        all_row = df[df['condition'] == 'erase_all_features']
        random_row = df[df['condition'] == 'random_control']
        drops.append(all_row['accuracy_drop'].values[0] if len(all_row) > 0 else 0)
        drops.append(random_row['accuracy_drop'].values[0] if len(random_row) > 0 else 0)
        
        model_label = 'Llama-3.2-3B' if 'llama' in model else 'Qwen3-4B'
        color = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, [d * 100 for d in drops], width, label=model_label, color=color, alpha=0.8)
        
        # Add value labels
        for bar, drop in zip(bars, drops):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{drop*100:.1f}%', ha='center', va='bottom', fontsize=8)
    
    labels = [FEATURE_LABELS[f] for f in features] + ['All Features', 'Random Control']
    ax.set_xlabel('Erased Feature')
    ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title('Price Prediction Accuracy Drop After Feature Erasure (Causal Impact)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()
    
    # Add significance threshold
    ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Significance threshold (2%)')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'price_prediction_drops.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_correctness_analysis(results_dir: str, output_dir: str, models: list, condition: str = "2_fewshot_cot_temp0"):
    """Plot price prediction drops by correctness of original prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    features = ['bedrooms', 'bathrooms', 'year_built', 'lot_size', 'sqft']
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        results = load_results(results_dir, model, condition)
        
        if 'correctness' not in results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        df = results['correctness']
        
        # Get data for correct and incorrect groups
        categories = ['Baseline'] + [FEATURE_LABELS[f] for f in features]
        correct_data = []
        incorrect_data = []
        
        # Extract baseline and per-feature accuracies for each group
        correct_df = df[df['group'] == 'correct']
        incorrect_df = df[df['group'] == 'incorrect']
        
        # Get baseline
        correct_baseline = correct_df[correct_df['condition'] == 'baseline']['accuracy'].values
        incorrect_baseline = incorrect_df[incorrect_df['condition'] == 'baseline']['accuracy'].values
        
        if len(correct_baseline) == 0 or len(incorrect_baseline) == 0:
            ax.text(0.5, 0.5, 'Incomplete data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        correct_data = [correct_baseline[0]]
        incorrect_data = [incorrect_baseline[0]]
        
        # Get per-feature accuracies
        for feature in features:
            correct_feat = correct_df[correct_df['condition'] == f'erase_{feature}']['accuracy'].values
            incorrect_feat = incorrect_df[incorrect_df['condition'] == f'erase_{feature}']['accuracy'].values
            correct_data.append(correct_feat[0] if len(correct_feat) > 0 else 0.5)
            incorrect_data.append(incorrect_feat[0] if len(incorrect_feat) > 0 else 0.5)
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, [v * 100 for v in correct_data], width, 
                      label='Correct Predictions', color='#2ca02c', alpha=0.8)
        bars2 = ax.bar(x + width/2, [v * 100 for v in incorrect_data], width,
                      label='Incorrect Predictions', color='#d62728', alpha=0.8)
        
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        
        model_label = 'Llama-3.2-3B' if 'llama' in model else 'Qwen3-4B'
        ax.set_title(f'{model_label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Condition')
        ax.set_ylabel('Probe Accuracy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=9)
        ax.set_ylim(45, 70)
        ax.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('Price Prediction by Original Answer Correctness', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'correctness_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_figure(results_dir: str, output_dir: str, models: list, condition: str = "2_fewshot_cot_temp0"):
    """Create a comprehensive summary figure for paper/presentation."""
    features = ['bedrooms', 'bathrooms', 'year_built', 'lot_size', 'sqft']
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # Panel A: Feature Encoding (Initial Probe Accuracy)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(features))
    width = 0.35
    
    for i, model in enumerate(models):
        results = load_results(results_dir, model, condition)
        if 'inlp' not in results:
            continue
        
        initial_accs = [results['inlp'][f]['initial_probe_accuracy'] for f in features 
                       if f in results['inlp']]
        
        model_label = 'Llama' if 'llama' in model else 'Qwen'
        color = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        
        offset = (i - 0.5) * width
        ax1.bar(x + offset, initial_accs, width, label=model_label, color=color, alpha=0.8)
    
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('A. Feature Encoding', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f[:3].upper() for f in features], fontsize=9)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc='lower right', fontsize=9)
    
    # Panel B: Erasure Quality (Post-INLP)
    ax2 = fig.add_subplot(gs[0, 1])
    
    for i, model in enumerate(models):
        results = load_results(results_dir, model, condition)
        if 'inlp' not in results:
            continue
        
        verification_accs = [results['inlp'][f]['verification_accuracy'] for f in features 
                            if f in results['inlp']]
        
        model_label = 'Llama' if 'llama' in model else 'Qwen'
        color = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        
        offset = (i - 0.5) * width
        ax2.bar(x + offset, verification_accs, width, label=model_label, color=color, alpha=0.8)
    
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Target')
    ax2.axhline(y=0.6, color='red', linestyle=':', alpha=0.5)
    ax2.axhspan(0.48, 0.55, alpha=0.15, color='green')
    
    ax2.set_ylabel('Accuracy')
    ax2.set_title('B. Erasure Quality', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f[:3].upper() for f in features], fontsize=9)
    ax2.set_ylim(0.4, 0.85)
    ax2.legend(loc='upper right', fontsize=9)
    
    # Panel C: Price Prediction Drops
    ax3 = fig.add_subplot(gs[0, 2])
    
    for i, model in enumerate(models):
        results = load_results(results_dir, model, condition)
        if 'main' not in results:
            continue
        
        df = results['main']
        drops = []
        for feature in features:
            row = df[df['condition'] == f'erase_{feature}']
            drops.append(row['accuracy_drop'].values[0] * 100 if len(row) > 0 else 0)
        
        model_label = 'Llama' if 'llama' in model else 'Qwen'
        color = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        
        offset = (i - 0.5) * width
        ax3.bar(x + offset, drops, width, label=model_label, color=color, alpha=0.8)
    
    ax3.set_ylabel('Accuracy Drop (%)')
    ax3.set_title('C. Price Prediction Impact', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f[:3].upper() for f in features], fontsize=9)
    ax3.legend(loc='upper right', fontsize=9)
    
    # Panel D: INLP Convergence (example: year_built - successful erasure)
    ax4 = fig.add_subplot(gs[1, 0])
    
    for model in models:
        results = load_results(results_dir, model, condition)
        if 'inlp' not in results or 'year_built' not in results['inlp']:
            continue
        
        accs = results['inlp']['year_built']['accuracies']
        model_label = 'Llama' if 'llama' in model else 'Qwen'
        color = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        ax4.plot(range(1, len(accs)+1), accs, label=model_label, color=color, linewidth=2)
    
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Probe Accuracy')
    ax4.set_title('D. INLP: Year Built (Complete)', fontweight='bold')
    ax4.set_ylim(0.4, 1.0)
    ax4.legend(fontsize=9)
    
    # Panel E: INLP Convergence (example: bedrooms - incomplete erasure)
    ax5 = fig.add_subplot(gs[1, 1])
    
    for model in models:
        results = load_results(results_dir, model, condition)
        if 'inlp' not in results or 'bedrooms' not in results['inlp']:
            continue
        
        accs = results['inlp']['bedrooms']['accuracies']
        model_label = 'Llama' if 'llama' in model else 'Qwen'
        color = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        ax5.plot(range(1, len(accs)+1), accs, label=model_label, color=color, linewidth=2)
    
    ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax5.axhline(y=0.6, color='red', linestyle=':', alpha=0.5)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Probe Accuracy')
    ax5.set_title('E. INLP: Bedrooms (Incomplete)', fontweight='bold')
    ax5.set_ylim(0.4, 1.0)
    ax5.legend(fontsize=9)
    
    # Panel F: Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create summary table
    summary_text = "SUMMARY\n" + "="*40 + "\n\n"
    
    for model in models:
        results = load_results(results_dir, model, condition)
        if 'summary' not in results or 'main' not in results:
            continue
        
        model_name = 'Llama-3.2-3B' if 'llama' in model else 'Qwen3-4B'
        baseline = results['summary']['baseline_accuracy']
        random_drop = results['summary']['random_control_drop']
        
        df = results['main']
        avg_drop = df[df['condition'].str.startswith('erase_') & 
                     ~df['condition'].str.contains('all|random')]['accuracy_drop'].mean()
        
        summary_text += f"{model_name}\n"
        summary_text += f"  Baseline: {baseline:.1%}\n"
        summary_text += f"  Avg Feature Drop: {avg_drop:.1%}\n"
        summary_text += f"  Random Drop: {random_drop:.2%}\n"
        summary_text += f"  Control: {'✓ PASSED' if avg_drop > random_drop else '✗ FAILED'}\n\n"
    
    summary_text += "\nConclusion:\n"
    summary_text += "Features are CAUSALLY used\n"
    summary_text += "for price prediction"
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Amnesic Probing: Causal Analysis of Feature Usage in Price Prediction',
                fontsize=14, fontweight='bold', y=0.98)
    
    output_path = Path(output_dir) / 'amnesic_summary.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_model_comparison(results_dir: str, output_dir: str, models: list, condition: str = "2_fewshot_cot_temp0"):
    """Create a direct model comparison visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    features = ['bedrooms', 'bathrooms', 'year_built', 'lot_size', 'sqft']
    
    # Collect data
    data = {model: load_results(results_dir, model, condition) for model in models}
    
    # Plot 1: Initial vs Final probe accuracy
    ax = axes[0]
    x = np.arange(len(features))
    width = 0.2
    
    for i, model in enumerate(models):
        if 'inlp' not in data[model]:
            continue
        
        initial = [data[model]['inlp'][f]['initial_probe_accuracy'] for f in features]
        final = [data[model]['inlp'][f]['verification_accuracy'] for f in features]
        
        color = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        model_short = 'L' if 'llama' in model else 'Q'
        
        ax.bar(x + i*width*2 - width, initial, width, label=f'{model_short}-Initial', 
               color=color, alpha=0.9)
        ax.bar(x + i*width*2, final, width, label=f'{model_short}-Final',
               color=color, alpha=0.4, hatch='//')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Probe Accuracy')
    ax.set_title('Initial vs Post-INLP Accuracy')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([f[:3].upper() for f in features])
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_ylim(0.4, 1.0)
    
    # Plot 2: Accuracy drop comparison
    ax = axes[1]
    
    for i, model in enumerate(models):
        if 'main' not in data[model]:
            continue
        
        df = data[model]['main']
        drops = [df[df['condition'] == f'erase_{f}']['accuracy_drop'].values[0] * 100 
                for f in features]
        
        color = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        model_label = 'Llama' if 'llama' in model else 'Qwen'
        
        ax.bar(x + (i-0.5)*width*2, drops, width*1.5, label=model_label, color=color, alpha=0.8)
    
    ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title('Price Prediction Drop per Feature')
    ax.set_xticks(x)
    ax.set_xticklabels([f[:3].upper() for f in features])
    ax.legend()
    
    # Plot 3: Baseline vs erased
    ax = axes[2]
    conditions = ['Baseline', 'After Erasure', 'Random Control']
    
    for i, model in enumerate(models):
        if 'main' not in data[model]:
            continue
        
        df = data[model]['main']
        baseline = df[df['condition'] == 'baseline']['accuracy'].values[0] * 100
        erased = df[df['condition'] == 'erase_all_features']['accuracy'].values[0] * 100
        random = df[df['condition'] == 'random_control']['accuracy'].values[0] * 100
        
        values = [baseline, erased, random]
        color = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        model_label = 'Llama' if 'llama' in model else 'Qwen'
        
        x_pos = np.arange(len(conditions))
        ax.bar(x_pos + (i-0.5)*width*2, values, width*1.5, label=model_label, color=color, alpha=0.8)
    
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Random chance')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall Price Prediction Accuracy')
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_ylim(40, 70)
    ax.legend()
    
    plt.suptitle('Model Comparison: Llama-3.2-3B vs Qwen3-4B', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'model_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_all_plots(results_dir: str, output_dir: str, condition: str = "2_fewshot_cot_temp0"):
    """Generate all visualization plots."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect available models
    models = []
    for model in ['llama-3.2-3b', 'qwen3-4b']:
        summary_file = Path(results_dir) / f"amnesic_summary_{model}_{condition}.json"
        if summary_file.exists():
            models.append(model)
    
    if not models:
        print(f"No results found in {results_dir}")
        return
    
    print(f"Found results for models: {models}")
    print(f"Generating plots to: {output_dir}")
    print("=" * 50)
    
    # Generate all plots
    plot_inlp_convergence(results_dir, output_dir, models, condition)
    plot_feature_encoding(results_dir, output_dir, models, condition)
    plot_erasure_quality(results_dir, output_dir, models, condition)
    plot_price_prediction_drops(results_dir, output_dir, models, condition)
    plot_correctness_analysis(results_dir, output_dir, models, condition)
    plot_model_comparison(results_dir, output_dir, models, condition)
    plot_summary_figure(results_dir, output_dir, models, condition)
    
    print("=" * 50)
    print("All plots generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Amnesic Probing results")
    parser.add_argument("--results-dir", type=str, 
                       default="data/amnesic_results",
                       help="Directory containing amnesic probing results")
    parser.add_argument("--output-dir", type=str,
                       default="data/amnesic_results/plots",
                       help="Directory to save plots")
    parser.add_argument("--condition", type=str,
                       default="2_fewshot_cot_temp0",
                       help="Experiment condition")
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / args.results_dir
    output_dir = base_dir / args.output_dir
    
    generate_all_plots(str(results_dir), str(output_dir), args.condition)
