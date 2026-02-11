# -*- coding: utf-8 -*-
"""
Step 7: Build the Reasoning Graph (CPU-only, matplotlib)

Constructs a visual flowchart showing the model's decision process:
  Layer 0: Input feature nodes (with probe encoding accuracy)
  Layer 1: Comparison nodes (with CoT mention frequency)
  Layer 2: Weighing node (with exchange rate labels)
  Layer 3: Decision node (CHOICE)

Builds separate graphs for correct and incorrect predictions,
plus a side-by-side comparison view.

Inputs:
  - macro_mapping/data/stage_summary_{model}.csv        (Step 5)
  - macro_mapping/data/exchange_rates_{model}.csv       (Step 6)
  - macro_mapping/data/exchange_rate_comparison_{model}.csv (Step 6)
  - macro_mapping/data/transition_patterns_{model}.csv  (Step 5)
  - data/probe_results/probe_best_layers_*.csv          (existing probing)

Outputs:
  - macro_mapping/outputs/graphs/reasoning_graph_{model}_correct.png
  - macro_mapping/outputs/graphs/reasoning_graph_{model}_incorrect.png
  - macro_mapping/outputs/graphs/reasoning_graph_{model}_comparison.png

Usage:
    python macro_mapping/build_reasoning_graph.py --model llama-3.2-3b
    python macro_mapping/build_reasoning_graph.py --model qwen3-4b
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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from mm_config import TARGET_FEATURES, CONDITION, setup_logger

logger = setup_logger("Step7_ReasoningGraph")

# ============================================================================
# PATHS
# ============================================================================

MM_DATA_DIR = Path(__file__).resolve().parent / "data"
MM_GRAPHS_DIR = Path(__file__).resolve().parent / "outputs" / "graphs"
MM_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBE_DIR = PROJECT_ROOT / "data" / "probe_results"

# All 8 features in the prompt (4 numeric + 4 categorical)
ALL_FEATURES = ['bedrooms', 'bathrooms', 'yearBuilt', 'lot',
                'type', 'heating', 'cooling', 'parkingType']


# ============================================================================
# DATA LOADING
# ============================================================================

def load_graph_data(model: str):
    """Load all data needed for graph construction."""
    data = {}

    # Stage summary (CoT mention frequencies)
    stage_file = MM_DATA_DIR / f"stage_summary_{model}.csv"
    if stage_file.exists():
        data['stages'] = pd.read_csv(stage_file)

    # Exchange rates
    exchange_file = MM_DATA_DIR / f"exchange_rates_{model}.csv"
    if exchange_file.exists():
        data['exchange'] = pd.read_csv(exchange_file)

    comparison_file = MM_DATA_DIR / f"exchange_rate_comparison_{model}.csv"
    if comparison_file.exists():
        data['comparison'] = pd.read_csv(comparison_file)

    # Transition patterns
    trans_file = MM_DATA_DIR / f"transition_patterns_{model}.csv"
    if trans_file.exists():
        data['transitions'] = pd.read_csv(trans_file)

    # Probe results
    probe_file = PROBE_DIR / f"probe_best_layers_{model}_{CONDITION}.csv"
    if probe_file.exists():
        data['probes'] = pd.read_csv(probe_file)

    return data


def get_probe_accuracy(data: dict, feature: str) -> str:
    """Get probe accuracy for a feature, return as percentage string."""
    if 'probes' not in data:
        return '?%'
    probe_map = {
        'bathrooms': 'bathrooms_p1_more',
        'bedrooms': 'bedrooms_p1_more',
        'lot': 'lot_p1_larger',
        'yearBuilt': 'year_p1_newer',
    }
    probe_name = probe_map.get(feature, None)
    if probe_name is None:
        return 'N/A'
    row = data['probes'][data['probes']['feature'] == probe_name]
    if len(row) == 0:
        return '?%'
    return f"{row.iloc[0]['test_accuracy']:.0%}"


def get_behavioral_weight(data: dict, feature: str, group: str) -> float:
    """Get normalized behavioral weight for a feature."""
    if 'comparison' not in data:
        return 0.0
    col = f'behavioral_weight_{group}_norm'
    row = data['comparison'][data['comparison']['feature'] == feature]
    if len(row) == 0 or col not in row.columns:
        return 0.0
    val = row.iloc[0].get(col, 0)
    return float(val) if pd.notna(val) else 0.0


def get_cot_frequency(data: dict, feature: str, group: str) -> float:
    """Get CoT mention frequency for pairwise_comparison stage.

    Returns the fraction of samples (0-1) where this feature was mentioned
    during the pairwise comparison reasoning stage.
    """
    if 'stages' not in data:
        return 0.0
    stages = data['stages']
    sub = stages[(stages['stage_name'] == 'pairwise_comparison') &
                 (stages['group'] == group)]
    if len(sub) == 0:
        return 0.0
    top_str = str(sub.iloc[0].get('top_features', ''))
    if feature in top_str:
        import re
        match = re.search(rf"'{feature}',\s*(\d+)", top_str)
        if match:
            count = int(match.group(1))
            n_samples = sub.iloc[0].get('n_samples', 1)
            return min(count / n_samples, 1.0)
    return 0.0


def get_dominant_transition(data: dict, group_label: str) -> str:
    """Get the dominant transition pattern for a group (correct/incorrect)."""
    if 'transitions' not in data:
        return '?'
    trans = data['transitions']
    is_correct_val = (group_label == 'correct')
    # Filter by group if is_correct column exists
    if 'is_correct' in trans.columns:
        trans = trans[trans['is_correct'] == is_correct_val]
    flipped = trans[trans['did_flip'] == True]
    if len(flipped) == 0:
        return 'no data'
    return flipped['shift_type'].mode().iloc[0]


# ============================================================================
# GRAPH RENDERING
# ============================================================================

def draw_reasoning_graph(ax, data: dict, group: str, title: str,
                         color_scheme: str = 'blue'):
    """
    Draw one reasoning graph on a matplotlib axis.

    Args:
        ax: matplotlib axis
        data: dict of loaded DataFrames
        group: 'correct' or 'incorrect'
        title: graph title
        color_scheme: 'blue' for correct, 'red' for incorrect
    """
    main_color = '#2196F3' if color_scheme == 'blue' else '#F44336'
    light_color = '#BBDEFB' if color_scheme == 'blue' else '#FFCDD2'
    edge_color = '#1565C0' if color_scheme == 'blue' else '#C62828'
    gray = '#BDBDBD'

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    # Layer 0: Input feature nodes (y=4)
    y_input = 4.0
    feature_positions = {}
    for i, feat in enumerate(ALL_FEATURES):
        x = i + 0.5
        feature_positions[feat] = (x, y_input)

        # Get probe accuracy
        probe_acc = get_probe_accuracy(data, feat)

        # Node
        box = FancyBboxPatch((x - 0.4, y_input - 0.25), 0.8, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor=light_color, edgecolor=main_color,
                              linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y_input + 0.05, feat, ha='center', va='center',
                fontsize=6, fontweight='bold')
        if feat in TARGET_FEATURES:
            ax.text(x, y_input - 0.12, probe_acc, ha='center', va='center',
                    fontsize=5, color='gray')

    # Layer 1: Comparison nodes (y=2.8)
    # Layer 0→1 edges: thickness from CoT mention frequency (how often mentioned)
    # Layer 1→2 edges: thickness from behavioral weight (flip-based importance)
    y_compare = 2.8
    for feat in TARGET_FEATURES:
        x = ALL_FEATURES.index(feat) + 0.5

        # CoT mention frequency for input→comparison edge
        cot_freq = get_cot_frequency(data, feat, group)
        edge_width_01 = max(0.5, cot_freq * 6)

        # Edge from input to comparison
        ax.annotate('', xy=(x, y_compare + 0.25), xytext=(x, y_input - 0.25),
                     arrowprops=dict(arrowstyle='->', color=edge_color,
                                    lw=edge_width_01, alpha=0.7))

        # Comparison node — show behavioral weight
        weight = get_behavioral_weight(data, feat, group)
        box = FancyBboxPatch((x - 0.35, y_compare - 0.2), 0.7, 0.4,
                              boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor=main_color,
                              linewidth=1)
        ax.add_patch(box)
        ax.text(x, y_compare, f'w={weight:.2f}', ha='center', va='center',
                fontsize=5)

    # Layer 2: Weighing node (y=1.5)
    y_weigh = 1.5
    x_weigh = 4.0

    # Edges from comparison to weighing
    for feat in TARGET_FEATURES:
        x = ALL_FEATURES.index(feat) + 0.5
        weight = get_behavioral_weight(data, feat, group)
        edge_width = max(0.5, weight * 8)
        ax.annotate('', xy=(x_weigh, y_weigh + 0.3), xytext=(x, y_compare - 0.2),
                     arrowprops=dict(arrowstyle='->', color=edge_color,
                                    lw=edge_width, alpha=0.5))

    # Weighing node
    box = FancyBboxPatch((x_weigh - 1.2, y_weigh - 0.3), 2.4, 0.6,
                          boxstyle="round,pad=0.1",
                          facecolor=light_color, edgecolor=main_color,
                          linewidth=2)
    ax.add_patch(box)
    ax.text(x_weigh, y_weigh + 0.05, 'WEIGHING', ha='center', va='center',
            fontsize=8, fontweight='bold')

    # Add exchange rate label if available
    if 'comparison' in data:
        comp = data['comparison']
        # Find the two most important features
        col = f'behavioral_weight_{group}_norm'
        if col in comp.columns:
            sorted_feats = comp.sort_values(col, ascending=False)
            if len(sorted_feats) >= 2:
                f1 = sorted_feats.iloc[0]['feature']
                f2 = sorted_feats.iloc[1]['feature']
                w1 = sorted_feats.iloc[0][col]
                w2 = sorted_feats.iloc[1][col]
                if pd.notna(w1) and pd.notna(w2) and w2 > 0:
                    ratio = w1 / w2
                    ax.text(x_weigh, y_weigh - 0.15,
                            f'{f1} {ratio:.1f}x > {f2}',
                            ha='center', va='center', fontsize=5, style='italic')

    # Layer 3: Decision node (y=0.5)
    y_decision = 0.5

    # Transition label
    transition = get_dominant_transition(data, group)

    ax.annotate('', xy=(x_weigh, y_decision + 0.25), xytext=(x_weigh, y_weigh - 0.3),
                 arrowprops=dict(arrowstyle='->', color=edge_color, lw=2))
    ax.text(x_weigh + 1.3, (y_weigh + y_decision) / 2, transition,
            ha='center', va='center', fontsize=6, color='gray', style='italic')

    # Decision node
    box = FancyBboxPatch((x_weigh - 0.8, y_decision - 0.25), 1.6, 0.5,
                          boxstyle="round,pad=0.1",
                          facecolor=main_color, edgecolor=edge_color,
                          linewidth=2)
    ax.add_patch(box)
    ax.text(x_weigh, y_decision, 'CHOICE', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')


# ============================================================================
# MAIN
# ============================================================================

def main(model: str = "llama-3.2-3b"):
    logger.info(f"Step 7: Build Reasoning Graph for {model}")

    data = load_graph_data(model)
    logger.info(f"  Loaded data: {list(data.keys())}")

    # Individual graphs
    for group, color, label in [('correct', 'blue', 'Correct Predictions'),
                                 ('incorrect', 'red', 'Incorrect Predictions')]:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        draw_reasoning_graph(ax, data, group, f"{model} — {label}", color)
        outfile = MM_GRAPHS_DIR / f"reasoning_graph_{model}_{group}.png"
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  Saved: {outfile}")

    # Side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    draw_reasoning_graph(ax1, data, 'correct', f"{model} — Correct (60)", 'blue')
    draw_reasoning_graph(ax2, data, 'incorrect', f"{model} — Incorrect (40)", 'red')
    fig.suptitle(f"Reasoning Graph Comparison — {model}", fontsize=16, fontweight='bold')

    outfile = MM_GRAPHS_DIR / f"reasoning_graph_{model}_comparison.png"
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {outfile}")

    logger.info(f"  Step 7 complete for {model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7: Build reasoning graph")
    parser.add_argument('--model', type=str, default='llama-3.2-3b',
                        choices=['llama-3.2-3b', 'qwen3-4b'],
                        help='Model to visualize')
    args = parser.parse_args()
    main(model=args.model)
