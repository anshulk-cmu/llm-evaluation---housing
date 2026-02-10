#!/usr/bin/env python3
"""
Plotting for MP Amnesic Probing results — All Layers.

Generates layer-sweep plots for the 3-step Dobrzeniecka et al. (2025) framework
using Mean Projection (MP) across all transformer layers.

Plots:
  1. Feature Encoding by Layer  (probe accuracy across layers)
  2. Price Prediction Drops by Layer  (causal impact across layers — KEY RESULT)
  3. Cosine Similarity by Layer  (representation distortion)
  4. Information Control at Best Layer  (target vs random bar chart)
  5. Selectivity Heatmap  (gold label recovery across layers)
  6. Correctness Analysis at Best Layer
  7. Summary Figure  (multi-panel)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

COLORS = {
    'llama': '#1f77b4',
    'qwen': '#ff7f0e',
    'baseline': '#333333',
    'random': '#999999',
    'features': {
        'bedrooms': '#e41a1c',
        'bathrooms': '#377eb8',
        'year_built': '#4daf4a',
        'lot_size': '#984ea3',
    }
}

FEATURE_LABELS = {
    'bedrooms': 'Bedrooms',
    'bathrooms': 'Bathrooms',
    'year_built': 'Year Built',
    'lot_size': 'Lot Size',
}

FEATURES = ['bedrooms', 'bathrooms', 'year_built', 'lot_size']

MODEL_LABELS = {
    'llama-3.2-3b': 'Llama-3.2-3B',
    'qwen3-4b': 'Qwen3-4B',
}


# =============================================================================
# Data Loading — All Layers
# =============================================================================

def load_all_layers(results_dir: str, model: str) -> Dict[int, Dict]:
    """
    Load results across all layers for a model.

    Expects directory structure: {results_dir}/{model}/layer_{N}/

    Returns:
        {layer_num: {'main': DataFrame, 'erasure': dict, 'selectivity': DataFrame,
                     'correctness': DataFrame, 'summary': dict}}
    """
    model_dir = Path(results_dir) / model
    layers = {}

    if not model_dir.is_dir():
        return layers

    for layer_dir in sorted(model_dir.glob("layer_*")):
        if not layer_dir.is_dir():
            continue
        try:
            layer = int(layer_dir.name.split('_')[1])
        except (IndexError, ValueError):
            continue

        layer_data = {}

        f = layer_dir / "results.csv"
        if f.exists():
            layer_data['main'] = pd.read_csv(f)

        f = layer_dir / "erasure_info.json"
        if f.exists():
            with open(f) as fh:
                layer_data['erasure'] = json.load(fh)

        f = layer_dir / "selectivity.csv"
        if f.exists():
            layer_data['selectivity'] = pd.read_csv(f)

        f = layer_dir / "correctness.csv"
        if f.exists():
            layer_data['correctness'] = pd.read_csv(f)

        f = layer_dir / "summary.json"
        if f.exists():
            with open(f) as fh:
                layer_data['summary'] = json.load(fh)

        if layer_data:
            layers[layer] = layer_data

    return layers


def extract_layer_series(layers: Dict[int, Dict], feature: str, field: str,
                         source: str = 'erasure', method: str = 'mp') -> Tuple[List[int], List[float]]:
    """Extract a value for a feature across all layers.

    Args:
        source: 'erasure' (from erasure_info JSON), 'main' (from results CSV),
                'selectivity' (from selectivity CSV)
        field: the key/column to extract
    """
    xs, ys = [], []
    for layer in sorted(layers.keys()):
        data = layers[layer]

        if source == 'erasure':
            val = data.get('erasure', {}).get(method, {}).get(feature, {}).get(field)
        elif source == 'main':
            df = data.get('main')
            if df is not None:
                mdf = df[df['method'] == method] if 'method' in df.columns else df
                row = mdf[mdf['condition'] == f'erase_{feature}']
                val = row[field].values[0] if len(row) > 0 else None
            else:
                val = None
        elif source == 'selectivity':
            df = data.get('selectivity')
            if df is not None:
                mdf = df[df['method'] == method] if 'method' in df.columns else df
                row = mdf[mdf['feature'] == feature]
                val = row[field].values[0] if len(row) > 0 else None
            else:
                val = None
        else:
            val = None

        if val is not None:
            xs.append(layer)
            ys.append(float(val))

    return xs, ys


def get_baseline_by_layer(layers: Dict[int, Dict], method: str = 'mp') -> Tuple[List[int], List[float]]:
    """Extract baseline price prediction accuracy across layers."""
    xs, ys = [], []
    for layer in sorted(layers.keys()):
        df = layers[layer].get('main')
        if df is None:
            continue
        mdf = df[df['method'] == method] if 'method' in df.columns else df
        row = mdf[mdf['condition'] == 'baseline']
        if len(row) > 0:
            xs.append(layer)
            ys.append(row['accuracy'].values[0])
    return xs, ys


def get_random_drop_by_layer(layers: Dict[int, Dict], method: str = 'mp') -> Tuple[List[int], List[float], List[float]]:
    """Extract random control drop across layers. Returns (layers, means, stds)."""
    xs, means, stds = [], [], []
    for layer in sorted(layers.keys()):
        df = layers[layer].get('main')
        if df is None:
            continue
        mdf = df[df['method'] == method] if 'method' in df.columns else df
        row = mdf[mdf['condition'] == 'random_control']
        if len(row) > 0:
            xs.append(layer)
            means.append(row['accuracy_drop'].values[0] * 100)
            stds.append(row['random_std_drop'].values[0] * 100 if 'random_std_drop' in row.columns else 0)
    return xs, means, stds


def find_best_layer(layers: Dict[int, Dict], method: str = 'mp') -> int:
    """Find the layer with highest total causal impact (sum of feature drops)."""
    best_layer, best_total = 0, -1
    for layer in sorted(layers.keys()):
        df = layers[layer].get('main')
        if df is None:
            continue
        mdf = df[df['method'] == method] if 'method' in df.columns else df
        total_drop = 0
        for f in FEATURES:
            row = mdf[mdf['condition'] == f'erase_{f}']
            if len(row) > 0:
                total_drop += row['accuracy_drop'].values[0]
        if total_drop > best_total:
            best_total = total_drop
            best_layer = layer
    return best_layer


# =============================================================================
# Plot 1: Feature Encoding by Layer
# =============================================================================

def plot_feature_encoding_by_layer(results_dir, output_dir, models):
    """Line plot of probe accuracy per feature across all layers, one panel per model."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 5), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        layers = load_all_layers(results_dir, model)
        if not layers:
            continue

        for feature in FEATURES:
            xs, ys = extract_layer_series(layers, feature, 'initial_probe_accuracy',
                                          source='erasure', method='mp')
            if xs:
                ax.plot(xs, ys, '-o', markersize=3, color=COLORS['features'][feature],
                        label=FEATURE_LABELS[feature], linewidth=1.5)

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Chance (50%)')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Probe Accuracy')
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=12, fontweight='bold')
        ax.set_ylim(0.45, 1.0)
        ax.legend(fontsize=8, loc='upper left')

    plt.suptitle('Feature Encoding Strength by Layer (Pre-Erasure Probe Accuracy)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'feature_encoding_by_layer.png', dpi=150, bbox_inches='tight')
    print("Saved: feature_encoding_by_layer.png")
    plt.close()


# =============================================================================
# Plot 2: Price Prediction Drops by Layer (KEY RESULT)
# =============================================================================

def plot_price_drops_by_layer(results_dir, output_dir, models):
    """Line plot of accuracy drop per feature across layers — the key causal result."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 5), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        layers = load_all_layers(results_dir, model)
        if not layers:
            continue

        # Feature drops
        for feature in FEATURES:
            xs, ys = extract_layer_series(layers, feature, 'accuracy_drop',
                                          source='main', method='mp')
            if xs:
                ax.plot(xs, [y * 100 for y in ys], '-o', markersize=3,
                        color=COLORS['features'][feature],
                        label=FEATURE_LABELS[feature], linewidth=1.5)

        # Random control band
        rx, rmeans, rstds = get_random_drop_by_layer(layers, method='mp')
        if rx:
            rmeans = np.array(rmeans)
            rstds = np.array(rstds)
            ax.fill_between(rx, rmeans - rstds, rmeans + rstds,
                            alpha=0.15, color='gray')
            ax.plot(rx, rmeans, '--', color='gray', linewidth=1.5, label='Random control')

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Price Prediction Drop (%)')
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')

    plt.suptitle('Causal Impact by Layer: Price Prediction Drop After Feature Erasure (MP)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'price_drops_by_layer.png', dpi=150, bbox_inches='tight')
    print("Saved: price_drops_by_layer.png")
    plt.close()


# =============================================================================
# Plot 3: Cosine Similarity by Layer
# =============================================================================

def plot_cosine_similarity_by_layer(results_dir, output_dir, models):
    """Line plot of cosine similarity across layers — representation distortion."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 5), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        layers = load_all_layers(results_dir, model)
        if not layers:
            continue

        for feature in FEATURES:
            xs, ys = extract_layer_series(layers, feature, 'cosine_similarity',
                                          source='erasure', method='mp')
            if xs:
                ax.plot(xs, ys, '-o', markersize=3, color=COLORS['features'][feature],
                        label=FEATURE_LABELS[feature], linewidth=1.5)

        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.90, color='orange', linestyle=':', alpha=0.5, label='0.90 threshold')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Cosine Similarity (original vs erased)')
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=12, fontweight='bold')
        ax.set_ylim(0.85, 1.005)
        ax.legend(fontsize=8, loc='lower left')

    plt.suptitle('Representation Preservation by Layer (Higher = Less Distortion)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'cosine_similarity_by_layer.png', dpi=150, bbox_inches='tight')
    print("Saved: cosine_similarity_by_layer.png")
    plt.close()


# =============================================================================
# Plot 4: Information Control at Best Layer
# =============================================================================

def plot_information_control_best_layer(results_dir, output_dir, models):
    """Bar chart: target vs random erasure at the best layer per model."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        layers = load_all_layers(results_dir, model)
        if not layers:
            continue

        best_layer = find_best_layer(layers, method='mp')
        data = layers[best_layer]
        df = data.get('main')
        if df is None:
            continue

        mdf = df[df['method'] == 'mp'] if 'method' in df.columns else df

        drops = []
        for f in FEATURES:
            row = mdf[mdf['condition'] == f'erase_{f}']
            drops.append(row['accuracy_drop'].values[0] * 100 if len(row) > 0 else 0)

        random_row = mdf[mdf['condition'] == 'random_control']
        random_drop = random_row['accuracy_drop'].values[0] * 100 if len(random_row) > 0 else 0
        random_std = random_row['random_std_drop'].values[0] * 100 if len(random_row) > 0 else 0

        x = np.arange(len(FEATURES))
        colors = [COLORS['features'][f] for f in FEATURES]
        bars = ax.bar(x, drops, 0.6, color=colors, alpha=0.8)

        # Random baseline band
        ax.axhspan(random_drop - random_std, random_drop + random_std,
                   alpha=0.2, color='gray')
        ax.axhline(y=random_drop, color='gray', linestyle='--', linewidth=2,
                   label=f'Random ({random_drop:.2f}%)')

        for bar, drop in zip(bars, drops):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{drop:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('Erased Feature')
        ax.set_ylabel('Price Prediction Drop (%)')
        label = MODEL_LABELS.get(model, model)
        ax.set_title(f'{label} — Layer {best_layer}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES], rotation=15, ha='right')
        ax.legend()

    plt.suptitle('Information Control: Target vs Random Erasure at Best Layer (MP)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'information_control_best_layer.png', dpi=150, bbox_inches='tight')
    print("Saved: information_control_best_layer.png")
    plt.close()


# =============================================================================
# Plot 5: Selectivity Heatmap
# =============================================================================

def plot_selectivity_heatmap(results_dir, output_dir, models):
    """Heatmap of gold-label recovery % across layers and features."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(max(10, 6 * n_models), 4), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        layers = load_all_layers(results_dir, model)
        if not layers:
            continue

        sorted_layers = sorted(layers.keys())
        matrix = np.full((len(FEATURES), len(sorted_layers)), np.nan)

        for j, layer in enumerate(sorted_layers):
            for i, feature in enumerate(FEATURES):
                df = layers[layer].get('selectivity')
                if df is not None:
                    mdf = df[df['method'] == 'mp'] if 'method' in df.columns else df
                    row = mdf[mdf['feature'] == feature]
                    if len(row) > 0:
                        matrix[i, j] = row['recovery_pct'].values[0]

        sns.heatmap(matrix, ax=ax, cmap='RdYlGn', center=100, vmin=0, vmax=200,
                    xticklabels=sorted_layers, yticklabels=[FEATURE_LABELS[f] for f in FEATURES],
                    annot=False, fmt='.0f', linewidths=0.5)
        ax.set_xlabel('Layer')
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=12, fontweight='bold')

        # Thin out x-tick labels for readability
        n_ticks = len(sorted_layers)
        if n_ticks > 20:
            step = max(n_ticks // 10, 1)
            ax.set_xticks(range(0, n_ticks, step))
            ax.set_xticklabels([sorted_layers[i] for i in range(0, n_ticks, step)])

    plt.suptitle('Selectivity Control: Gold Label Recovery (%) by Layer\n'
                 '(green ~100% = clean causal evidence)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'selectivity_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved: selectivity_heatmap.png")
    plt.close()


# =============================================================================
# Plot 6: Correctness Analysis at Best Layer
# =============================================================================

def plot_correctness_best_layer(results_dir, output_dir, models):
    """Price prediction by correct/incorrect at the best layer."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        layers = load_all_layers(results_dir, model)
        if not layers:
            continue

        best_layer = find_best_layer(layers, method='mp')
        data = layers[best_layer]
        df = data.get('correctness')
        if df is None:
            continue

        mdf = df[df['method'] == 'mp'] if 'method' in df.columns else df

        correct_df = mdf[mdf['group'] == 'correct']
        incorrect_df = mdf[mdf['group'] == 'incorrect']

        cats = ['Baseline'] + [FEATURE_LABELS[f] for f in FEATURES]
        correct_vals, incorrect_vals = [], []

        c_base = correct_df[correct_df['condition'] == 'baseline']['accuracy'].values
        i_base = incorrect_df[incorrect_df['condition'] == 'baseline']['accuracy'].values
        correct_vals.append(c_base[0] * 100 if len(c_base) > 0 else 50)
        incorrect_vals.append(i_base[0] * 100 if len(i_base) > 0 else 50)

        for f in FEATURES:
            c_row = correct_df[correct_df['condition'] == f'erase_{f}']['accuracy'].values
            i_row = incorrect_df[incorrect_df['condition'] == f'erase_{f}']['accuracy'].values
            correct_vals.append(c_row[0] * 100 if len(c_row) > 0 else 50)
            incorrect_vals.append(i_row[0] * 100 if len(i_row) > 0 else 50)

        x = np.arange(len(cats))
        width = 0.35
        ax.bar(x - width / 2, correct_vals, width, label='Correct', color='#2ca02c', alpha=0.8)
        ax.bar(x + width / 2, incorrect_vals, width, label='Incorrect', color='#d62728', alpha=0.8)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)

        label = MODEL_LABELS.get(model, model)
        ax.set_title(f'{label} — Layer {best_layer}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price Prediction Accuracy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=30, ha='right', fontsize=9)
        ax.set_ylim(45, 70)
        ax.legend()

    plt.suptitle('Correctness Analysis at Best Layer (MP)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'correctness_best_layer.png', dpi=150, bbox_inches='tight')
    print("Saved: correctness_best_layer.png")
    plt.close()


# =============================================================================
# Plot 7: Summary Figure
# =============================================================================

def plot_summary(results_dir, output_dir, models):
    """Combined 2x3 summary figure with layer-sweep highlights."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

    # Panel A: Feature Encoding by Layer (overlay both models)
    ax = fig.add_subplot(gs[0, 0])
    for model in models:
        layers = load_all_layers(results_dir, model)
        if not layers:
            continue
        mcolor = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        mlabel = 'L' if 'llama' in model else 'Q'
        # Plot average encoding across features
        sorted_lyrs = sorted(layers.keys())
        avg_enc = []
        for lyr in sorted_lyrs:
            accs = []
            edata = layers[lyr].get('erasure', {}).get('mp', {})
            for f in FEATURES:
                v = edata.get(f, {}).get('initial_probe_accuracy')
                if v is not None:
                    accs.append(v)
            avg_enc.append(np.mean(accs) if accs else 0.5)
        ax.plot(sorted_lyrs, avg_enc, '-', color=mcolor, linewidth=2, label=mlabel)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Avg Probe Accuracy')
    ax.set_xlabel('Layer')
    ax.set_title('A. Feature Encoding', fontweight='bold')
    ax.legend(fontsize=9)

    # Panel B: Price Drops by Layer (overlay both models — average across features)
    ax = fig.add_subplot(gs[0, 1])
    for model in models:
        layers = load_all_layers(results_dir, model)
        if not layers:
            continue
        mcolor = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        mlabel = 'L' if 'llama' in model else 'Q'
        sorted_lyrs = sorted(layers.keys())
        avg_drops = []
        for lyr in sorted_lyrs:
            df = layers[lyr].get('main')
            if df is None:
                avg_drops.append(0)
                continue
            mdf = df[df['method'] == 'mp'] if 'method' in df.columns else df
            drops = []
            for f in FEATURES:
                row = mdf[mdf['condition'] == f'erase_{f}']
                if len(row) > 0:
                    drops.append(row['accuracy_drop'].values[0] * 100)
            avg_drops.append(np.mean(drops) if drops else 0)
        ax.plot(sorted_lyrs, avg_drops, '-', color=mcolor, linewidth=2, label=mlabel)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.set_ylabel('Avg Drop (%)')
    ax.set_xlabel('Layer')
    ax.set_title('B. Causal Impact', fontweight='bold')
    ax.legend(fontsize=9)

    # Panel C: Best Layer Bar Chart for first model
    ax = fig.add_subplot(gs[0, 2])
    for i, model in enumerate(models):
        layers = load_all_layers(results_dir, model)
        if not layers:
            continue
        best_layer = find_best_layer(layers, method='mp')
        data = layers[best_layer]
        df = data.get('main')
        if df is None:
            continue
        mdf = df[df['method'] == 'mp'] if 'method' in df.columns else df
        drops = []
        for f in FEATURES:
            row = mdf[mdf['condition'] == f'erase_{f}']
            drops.append(row['accuracy_drop'].values[0] * 100 if len(row) > 0 else 0)

        x = np.arange(len(FEATURES))
        width = 0.35
        mcolor = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        offset = (i - 0.5) * width
        mlabel = f"{'Llama' if 'llama' in model else 'Qwen'} (L{best_layer})"
        ax.bar(x + offset, drops, width, label=mlabel, color=mcolor, alpha=0.8)

    ax.set_ylabel('Drop (%)')
    ax.set_title('C. Best Layer Detail', fontweight='bold')
    ax.set_xticks(np.arange(len(FEATURES)))
    ax.set_xticklabels([f[:3].upper() for f in FEATURES], fontsize=9)
    ax.legend(fontsize=8)

    # Panel D: Cosine Similarity by Layer
    ax = fig.add_subplot(gs[1, 0])
    for model in models:
        layers = load_all_layers(results_dir, model)
        if not layers:
            continue
        mcolor = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        mlabel = 'L' if 'llama' in model else 'Q'
        sorted_lyrs = sorted(layers.keys())
        avg_sims = []
        for lyr in sorted_lyrs:
            sims = []
            edata = layers[lyr].get('erasure', {}).get('mp', {})
            for f in FEATURES:
                v = edata.get(f, {}).get('cosine_similarity')
                if v is not None:
                    sims.append(v)
            avg_sims.append(np.mean(sims) if sims else 1.0)
        ax.plot(sorted_lyrs, avg_sims, '-', color=mcolor, linewidth=2, label=mlabel)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Avg Cosine Similarity')
    ax.set_xlabel('Layer')
    ax.set_title('D. Representation Preservation', fontweight='bold')
    ax.set_ylim(0.9, 1.005)
    ax.legend(fontsize=9)

    # Panel E: Selectivity at Best Layer
    ax = fig.add_subplot(gs[1, 1])
    for i, model in enumerate(models):
        layers = load_all_layers(results_dir, model)
        if not layers:
            continue
        best_layer = find_best_layer(layers, method='mp')
        data = layers[best_layer]
        sdf = data.get('selectivity')
        if sdf is None:
            continue
        mdf = sdf[sdf['method'] == 'mp'] if 'method' in sdf.columns else sdf
        recovery = []
        for f in FEATURES:
            row = mdf[mdf['feature'] == f]
            recovery.append(row['recovery_pct'].values[0] if len(row) > 0 else 0)
        x = np.arange(len(FEATURES))
        width = 0.35
        mcolor = COLORS['llama'] if 'llama' in model else COLORS['qwen']
        offset = (i - 0.5) * width
        mlabel = 'Llama' if 'llama' in model else 'Qwen'
        ax.bar(x + offset, recovery, width, label=mlabel, color=mcolor, alpha=0.8)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Full recovery')
    ax.set_ylabel('Recovery (%)')
    ax.set_title('E. Selectivity (Best Layer)', fontweight='bold')
    ax.set_xticks(np.arange(len(FEATURES)))
    ax.set_xticklabels([f[:3].upper() for f in FEATURES], fontsize=9)
    ax.legend(fontsize=8)

    # Panel F: Summary text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    text = "SUMMARY\n" + "=" * 40 + "\n\n"
    for model in models:
        layers = load_all_layers(results_dir, model)
        if not layers:
            continue
        best_layer = find_best_layer(layers, method='mp')
        data = layers[best_layer]
        mname = MODEL_LABELS.get(model, model)
        summary = data.get('summary', {})
        mp_data = summary.get('methods', {}).get('mp', {})
        if mp_data:
            text += f"{mname} (Layer {best_layer})\n"
            text += f"  Baseline: {mp_data['baseline_accuracy']:.1%}\n"
            text += f"  Random drop: {mp_data['random_control_drop']:.2%}\n"
            for feat, fdata in mp_data.get('features', {}).items():
                v = fdata.get('verdict', '?')
                d = fdata.get('price_drop', 0)
                text += f"  {feat}: {v} ({d:.2%})\n"
            text += "\n"

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Amnesic Probing: MP Causal Analysis (All Layers)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(Path(output_dir) / 'mp_summary.png', dpi=200, bbox_inches='tight')
    print("Saved: mp_summary.png")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def generate_all_plots(results_dir: str, output_dir: str):
    """Generate all plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Discover available models by checking for model subdirectories with layer results
    models = []
    for model in ['llama-3.2-3b', 'qwen3-4b']:
        model_dir = Path(results_dir) / model
        if model_dir.is_dir() and list(model_dir.glob("layer_*")):
            models.append(model)

    if not models:
        print(f"No results found in {results_dir}")
        return

    print(f"Models: {models}")
    for model in models:
        layers = load_all_layers(results_dir, model)
        print(f"  {model}: {len(layers)} layers ({sorted(layers.keys())[0]}-{sorted(layers.keys())[-1]})")
    print(f"Output: {output_dir}")
    print("=" * 50)

    plot_feature_encoding_by_layer(results_dir, output_dir, models)
    plot_price_drops_by_layer(results_dir, output_dir, models)
    plot_cosine_similarity_by_layer(results_dir, output_dir, models)
    plot_information_control_best_layer(results_dir, output_dir, models)
    plot_selectivity_heatmap(results_dir, output_dir, models)
    plot_correctness_best_layer(results_dir, output_dir, models)
    plot_summary(results_dir, output_dir, models)

    print("=" * 50)
    print("All plots generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MP Amnesic Probing results (all layers)")
    parser.add_argument("--results-dir", type=str, default="data/mp_results")
    parser.add_argument("--output-dir", type=str, default="data/mp_results/plots")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    generate_all_plots(
        str(base_dir / args.results_dir),
        str(base_dir / args.output_dir),
    )
