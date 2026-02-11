# -*- coding: utf-8 -*-
"""
Step 10: Produce the Final Reasoning Graph with Annotations (CPU-only)

Combines everything from Steps 1-9 into one annotated multi-panel figure.
This is the main deliverable for the paper/presentation.

Layout:
  Row 1: Correct reasoning graph (left) + Incorrect reasoning graph (right)
  Row 2: Decision boundary sub-graph (left) + Summary evidence table (right)

Inputs:
  - macro_mapping/data/exchange_rate_comparison_{model}.csv  (Step 6)
  - macro_mapping/data/pairwise_exchange_rates_{model}.csv   (Step 6)
  - macro_mapping/data/stage_summary_{model}.csv             (Step 5)
  - macro_mapping/data/transition_patterns_{model}.csv       (Step 5)
  - macro_mapping/data/flip_results_{model}.csv              (Step 4)
  - macro_mapping/outputs/tables/regression_comparison_{model}.csv (Step 9)
  - macro_mapping/outputs/tables/linearity_verdict_{model}.txt     (Step 9)
  - data/probe_results/probe_best_layers_{model}_{condition}.csv

Outputs:
  - macro_mapping/outputs/graphs/final_reasoning_graph_{model}.png
  - macro_mapping/outputs/tables/final_summary_{model}.csv
  - macro_mapping/outputs/tables/full_evidence_table_{model}.csv

Usage:
    python macro_mapping/produce_final_graph.py --model llama-3.2-3b
    python macro_mapping/produce_final_graph.py --model qwen3-4b
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
from matplotlib.patches import FancyBboxPatch

from mm_config import TARGET_FEATURES, PERTURBATION_STEPS, CONDITION, setup_logger

logger = setup_logger("Step10_FinalGraph")

# ============================================================================
# PATHS
# ============================================================================

MM_DATA_DIR = Path(__file__).resolve().parent / "data"
MM_GRAPHS_DIR = Path(__file__).resolve().parent / "outputs" / "graphs"
MM_TABLES_DIR = Path(__file__).resolve().parent / "outputs" / "tables"
MM_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
MM_TABLES_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBE_DIR = PROJECT_ROOT / "data" / "probe_results"

# All 8 features in the prompt
ALL_FEATURES = ['bedrooms', 'bathrooms', 'yearBuilt', 'lot',
                'type', 'heating', 'cooling', 'parkingType']

UNIT_LABELS = {
    'bedrooms': 'beds',
    'bathrooms': 'baths',
    'yearBuilt': 'yrs',
    'lot': 'sqft',
}

PROBE_FEATURE_MAP = {
    'bathrooms_p1_more': 'bathrooms',
    'bedrooms_p1_more': 'bedrooms',
    'lot_p1_larger': 'lot',
    'year_p1_newer': 'yearBuilt',
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data(model: str) -> dict:
    """Load all intermediate outputs needed for the final figure."""
    data = {}

    # Exchange rate comparison (Step 6)
    f = MM_DATA_DIR / f"exchange_rate_comparison_{model}.csv"
    if f.exists():
        data['comparison'] = pd.read_csv(f)

    # Pairwise exchange rates (Step 6)
    f = MM_DATA_DIR / f"pairwise_exchange_rates_{model}.csv"
    if f.exists():
        data['pairwise'] = pd.read_csv(f)

    # Stage summary (Step 5)
    f = MM_DATA_DIR / f"stage_summary_{model}.csv"
    if f.exists():
        data['stages'] = pd.read_csv(f)

    # Transition patterns (Step 5)
    f = MM_DATA_DIR / f"transition_patterns_{model}.csv"
    if f.exists():
        data['transitions'] = pd.read_csv(f)

    # Flip results (Step 4)
    f = MM_DATA_DIR / f"flip_results_{model}.csv"
    if f.exists():
        data['flips'] = pd.read_csv(f)

    # Regression comparison (Step 9)
    f = MM_TABLES_DIR / f"regression_comparison_{model}.csv"
    if f.exists():
        data['regression'] = pd.read_csv(f)

    # Linearity verdict (Step 9)
    f = MM_TABLES_DIR / f"linearity_verdict_{model}.txt"
    if f.exists():
        with open(f) as fh:
            data['verdict_text'] = fh.read()
        # Extract short verdict from first line
        first_line = data['verdict_text'].split('\n')[0]
        data['verdict_short'] = first_line.replace('VERDICT: ', '').strip()

    # Probe results
    f = PROBE_DIR / f"probe_best_layers_{model}_{CONDITION}.csv"
    if f.exists():
        probe_df = pd.read_csv(f)
        data['probes'] = {}
        for _, row in probe_df.iterrows():
            feat_name = PROBE_FEATURE_MAP.get(row['feature'])
            if feat_name:
                data['probes'][feat_name] = row['test_accuracy']

    return data


# ============================================================================
# HELPERS
# ============================================================================

def get_weight(data: dict, feature: str, group: str) -> float:
    """Get normalized behavioral weight for a feature."""
    if 'comparison' not in data:
        return 0.0
    col = f'behavioral_weight_{group}_norm'
    comp = data['comparison']
    row = comp[comp['feature'] == feature]
    if len(row) == 0 or col not in row.columns:
        return 0.0
    val = row.iloc[0].get(col, 0)
    return float(val) if pd.notna(val) else 0.0


def get_cot_frequency(data: dict, feature: str, group: str) -> float:
    """Get CoT mention frequency for pairwise_comparison stage."""
    if 'stages' not in data:
        return 0.0
    stages = data['stages']
    sub = stages[(stages['stage_name'] == 'pairwise_comparison') &
                 (stages['group'] == group)]
    if len(sub) == 0:
        return 0.0
    # Check if this feature appears in top_features
    top_str = str(sub.iloc[0].get('top_features', ''))
    if feature in top_str:
        # Extract count if possible
        import re
        match = re.search(rf"'{feature}',\s*(\d+)", top_str)
        if match:
            count = int(match.group(1))
            n_samples = sub.iloc[0].get('n_samples', 1)
            return min(count / n_samples, 1.0)
    return 0.0


def get_dominant_transition(data: dict, group: str = 'all') -> str:
    """Get the most common transition pattern for flipped pairs in a group."""
    if 'transitions' not in data:
        return '?'
    trans = data['transitions']
    # Filter by group if is_correct column exists
    if group != 'all' and 'is_correct' in trans.columns:
        is_correct_val = (group == 'correct')
        trans = trans[trans['is_correct'] == is_correct_val]
    flipped = trans[trans['did_flip'] == True]
    if len(flipped) == 0:
        return 'no flips'
    return flipped['shift_type'].mode().iloc[0]


def get_exchange_rate_label(data: dict, group: str) -> str:
    """Build a human-readable exchange rate label from pairwise data."""
    if 'pairwise' not in data:
        return ''
    pw = data['pairwise']
    pw_group = pw[pw['group'] == group] if group != 'all' else pw[pw['group'] == 'all']
    if len(pw_group) == 0:
        pw_group = pw[pw['group'] == 'all']
    if len(pw_group) == 0:
        return ''

    # Find the pair with bathrooms as reference (anchor feature)
    labels = []
    for _, row in pw_group.iterrows():
        if row['feature_b'] == 'bathrooms':
            labels.append(f"1 bath ~ {row['rate_a_per_b']:.1f} {row['feature_a']}")
        elif row['feature_a'] == 'bathrooms':
            labels.append(f"1 bath ~ {row['rate_b_per_a']:.1f} {row['feature_b']}")

    return ' | '.join(labels[:3]) if labels else ''


# ============================================================================
# PANEL: ANNOTATED REASONING GRAPH
# ============================================================================

def draw_annotated_graph(ax, data: dict, group: str, title: str,
                         color_scheme: str = 'blue'):
    """
    Draw one annotated reasoning graph with all data overlaid.
    """
    main_color = '#2196F3' if color_scheme == 'blue' else '#F44336'
    light_color = '#BBDEFB' if color_scheme == 'blue' else '#FFCDD2'
    edge_color = '#1565C0' if color_scheme == 'blue' else '#C62828'

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1.0, 5.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # --- Layer 0: Input feature nodes (y=4.2) ---
    y_input = 4.2
    feature_positions = {}
    for i, feat in enumerate(ALL_FEATURES):
        x = i + 0.5
        feature_positions[feat] = (x, y_input)

        is_target = feat in TARGET_FEATURES
        fc = light_color if is_target else '#F5F5F5'
        ec = main_color if is_target else '#BDBDBD'
        lw = 1.5 if is_target else 0.8

        box = FancyBboxPatch((x - 0.4, y_input - 0.22), 0.8, 0.44,
                              boxstyle="round,pad=0.05",
                              facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(box)
        ax.text(x, y_input + 0.05, feat, ha='center', va='center',
                fontsize=5.5, fontweight='bold')

        # Probe accuracy annotation
        if is_target and 'probes' in data:
            acc = data['probes'].get(feat, None)
            if acc is not None:
                ax.text(x, y_input - 0.12, f'{acc:.0%} enc',
                        ha='center', va='center', fontsize=4.5, color='gray')

    # --- Layer 1: Comparison nodes (y=2.8) ---
    y_compare = 2.8
    for feat in TARGET_FEATURES:
        x = ALL_FEATURES.index(feat) + 0.5

        # Behavioral weight for edge thickness
        weight = get_weight(data, feat, group)
        edge_width = max(0.5, weight * 12)

        # CoT mention frequency as edge label
        cot_freq = get_cot_frequency(data, feat, group)

        # Edge from input to comparison
        ax.annotate('', xy=(x, y_compare + 0.22), xytext=(x, y_input - 0.22),
                     arrowprops=dict(arrowstyle='->', color=edge_color,
                                    lw=edge_width, alpha=0.7))

        # CoT frequency label on edge
        if cot_freq > 0:
            ax.text(x + 0.25, (y_input + y_compare) / 2,
                    f'{cot_freq:.0%}', fontsize=4, color=edge_color, alpha=0.7)

        # Comparison node
        box = FancyBboxPatch((x - 0.35, y_compare - 0.18), 0.7, 0.36,
                              boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor=main_color, linewidth=1)
        ax.add_patch(box)
        ax.text(x, y_compare, f'w={weight:.2f}', ha='center', va='center',
                fontsize=5)

    # --- Layer 2: Weighing node (y=1.5) ---
    y_weigh = 1.5
    x_weigh = 4.0

    for feat in TARGET_FEATURES:
        x = ALL_FEATURES.index(feat) + 0.5
        weight = get_weight(data, feat, group)
        edge_width = max(0.5, weight * 8)
        ax.annotate('', xy=(x_weigh, y_weigh + 0.25), xytext=(x, y_compare - 0.18),
                     arrowprops=dict(arrowstyle='->', color=edge_color,
                                    lw=edge_width, alpha=0.5))

    # Weighing box
    box = FancyBboxPatch((x_weigh - 1.5, y_weigh - 0.35), 3.0, 0.7,
                          boxstyle="round,pad=0.1",
                          facecolor=light_color, edgecolor=main_color, linewidth=2)
    ax.add_patch(box)
    ax.text(x_weigh, y_weigh + 0.1, 'WEIGHING', ha='center', va='center',
            fontsize=8, fontweight='bold')

    # Verdict label on weighing node
    verdict = data.get('verdict_short', '')
    if verdict:
        ax.text(x_weigh, y_weigh - 0.12, f'"{verdict}"',
                ha='center', va='center', fontsize=5, style='italic', color='gray')

    # Exchange rate label below weighing
    rate_label = get_exchange_rate_label(data, group)
    if rate_label:
        ax.text(x_weigh, y_weigh - 0.55, rate_label,
                ha='center', va='center', fontsize=4.5, color=edge_color,
                style='italic')

    # --- Layer 3: Decision node (y=0.2) ---
    y_decision = 0.2

    # Transition label on edge
    transition = get_dominant_transition(data, group)
    ax.annotate('', xy=(x_weigh, y_decision + 0.22), xytext=(x_weigh, y_weigh - 0.35),
                 arrowprops=dict(arrowstyle='->', color=edge_color, lw=2))
    ax.text(x_weigh + 1.5, (y_weigh + y_decision) / 2, transition,
            ha='center', va='center', fontsize=5, color='gray', style='italic')

    # Decision node
    box = FancyBboxPatch((x_weigh - 0.8, y_decision - 0.22), 1.6, 0.44,
                          boxstyle="round,pad=0.1",
                          facecolor=main_color, edgecolor=edge_color, linewidth=2)
    ax.add_patch(box)
    ax.text(x_weigh, y_decision, 'CHOICE', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')


# ============================================================================
# PANEL: DECISION BOUNDARY
# ============================================================================

def draw_boundary_panel(ax, data: dict):
    """Draw compact decision boundary strip plots."""
    ax.set_title('Decision Boundary Map', fontsize=12, fontweight='bold', loc='left')

    if 'flips' not in data:
        ax.text(0.5, 0.5, 'No flip data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')
        ax.axis('off')
        return

    flip_df = data['flips']
    n_feats = len(TARGET_FEATURES)

    for i, feat in enumerate(TARGET_FEATURES):
        y_base = n_feats - 1 - i
        schedule = PERTURBATION_STEPS[feat]

        for group_label, is_correct, color, marker in [
            ('correct', True, '#2196F3', 'o'),
            ('incorrect', False, '#F44336', 's')
        ]:
            sub = flip_df[(flip_df['feature'] == feat) & (flip_df['is_correct'] == is_correct)]
            flipped = sub[sub['flipped'] == True]

            if len(flipped) > 0:
                increments = flipped['flip_step_index'].apply(
                    lambda s, f=feat: float(PERTURBATION_STEPS[f][min(int(s), len(PERTURBATION_STEPS[f])-1)])
                )
                jitter = np.random.uniform(-0.12, 0.12, size=len(increments))
                y_offset = 0.15 if group_label == 'correct' else -0.15
                ax.scatter(increments, [y_base + y_offset] * len(increments) + jitter,
                           c=color, marker=marker, s=12, alpha=0.5,
                           label=f'{group_label}' if i == 0 else '')

                mean_val = increments.mean()
                ax.plot([mean_val], [y_base + y_offset], marker='|', color=color,
                        markersize=10, markeredgewidth=2)

            # No-flip count
            no_flip = (sub['flipped'] == False).sum()
            if no_flip > 0:
                ax.text(schedule[-1] * 1.02, y_base + (0.15 if group_label == 'correct' else -0.15),
                        f'{no_flip}x', fontsize=5, color=color, alpha=0.6,
                        va='center')

        ax.text(-0.02, y_base, feat, ha='right', va='center', fontsize=8,
                fontweight='bold', transform=ax.get_yaxis_transform())

    ax.set_yticks(range(n_feats))
    ax.set_yticklabels([])
    ax.set_ylim(-0.8, n_feats - 0.2)
    ax.set_xlabel('Perturbation increment (natural units)', fontsize=9)
    ax.legend(fontsize=7, loc='lower right')


# ============================================================================
# PANEL: SUMMARY TABLE
# ============================================================================

def draw_summary_table(ax, data: dict, model: str):
    """Draw the evidence summary table as a matplotlib table."""
    ax.axis('off')
    ax.set_title(f'Evidence Summary — {model}', fontsize=12, fontweight='bold', loc='left')

    # Build table data
    headers = ['Feature', 'Probe\nAccuracy', 'BW\n(correct)', 'BW\n(incorrect)',
               'Regression\nWeight', 'Flip(C)', 'Flip(I)']

    table_data = []
    for feat in TARGET_FEATURES:
        row = [feat]

        # Probe accuracy
        if 'probes' in data and feat in data['probes']:
            row.append(f"{data['probes'][feat]:.0%}")
        else:
            row.append('N/A')

        # Behavioral weights
        bw_c = get_weight(data, feat, 'correct')
        bw_i = get_weight(data, feat, 'incorrect')
        row.append(f"{bw_c:.3f}")
        row.append(f"{bw_i:.3f}")

        # Regression weight
        if 'regression' in data:
            reg_row = data['regression'][data['regression']['feature'] == feat]
            if len(reg_row) > 0:
                val = reg_row.iloc[0].get('regression_weight', np.nan)
                row.append(f"{val:.3f}" if pd.notna(val) else 'N/A')
            else:
                row.append('N/A')
        else:
            row.append('N/A')

        # Mean flip distances
        if 'comparison' in data:
            comp_row = data['comparison'][data['comparison']['feature'] == feat]
            if len(comp_row) > 0:
                fc = comp_row.iloc[0].get('mean_flip_correct', np.nan)
                fi = comp_row.iloc[0].get('mean_flip_incorrect', np.nan)
                row.append(f"{fc:.1f}" if pd.notna(fc) else 'N/A')
                row.append(f"{fi:.1f}" if pd.notna(fi) else 'N/A')
            else:
                row.extend(['N/A', 'N/A'])
        else:
            row.extend(['N/A', 'N/A'])

        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor('#E3F2FD')
        cell.set_text_props(fontweight='bold')

    # Add verdict annotation below table
    verdict = data.get('verdict_short', 'N/A')
    ax.text(0.5, 0.05, f'Linearity verdict: {verdict}',
            ha='center', va='center', fontsize=9, fontweight='bold',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# ============================================================================
# MAIN
# ============================================================================

def main(model: str = "llama-3.2-3b"):
    logger.info(f"Step 10: Produce Final Reasoning Graph for {model}")

    # ------------------------------------------------------------------
    # Load all data
    # ------------------------------------------------------------------
    data = load_all_data(model)
    logger.info(f"  Loaded data keys: {list(data.keys())}")

    # ------------------------------------------------------------------
    # Create the composite figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(22, 24))

    # GridSpec: 3 rows
    # Row 0: Correct graph (left) + Incorrect graph (right)
    # Row 1: Decision boundary (full width)
    # Row 2: Summary table (full width)
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.5, 1.2],
                          hspace=0.3, wspace=0.15)

    # Row 0: Reasoning graphs side by side
    ax_correct = fig.add_subplot(gs[0, 0])
    ax_incorrect = fig.add_subplot(gs[0, 1])

    draw_annotated_graph(ax_correct, data, 'correct',
                         f'{model} — Correct Predictions', 'blue')
    draw_annotated_graph(ax_incorrect, data, 'incorrect',
                         f'{model} — Incorrect Predictions', 'red')

    # Row 1: Decision boundary (spans both columns)
    ax_boundary = fig.add_subplot(gs[1, :])
    draw_boundary_panel(ax_boundary, data)

    # Row 2: Summary table (spans both columns)
    ax_table = fig.add_subplot(gs[2, :])
    draw_summary_table(ax_table, data, model)

    # Super title
    fig.suptitle(f'Behavioral Macro Mapping — {model}',
                 fontsize=18, fontweight='bold', y=0.98)

    # ------------------------------------------------------------------
    # Save PNG
    # ------------------------------------------------------------------
    outfile_png = MM_GRAPHS_DIR / f"final_reasoning_graph_{model}.png"
    fig.savefig(outfile_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  Saved: {outfile_png}")

    # ------------------------------------------------------------------
    # Save summary CSV
    # ------------------------------------------------------------------
    summary_rows = []
    for feat in TARGET_FEATURES:
        row = {'feature': feat}

        if 'probes' in data:
            row['probe_accuracy'] = data['probes'].get(feat, np.nan)

        row['bw_correct'] = get_weight(data, feat, 'correct')
        row['bw_incorrect'] = get_weight(data, feat, 'incorrect')

        if 'regression' in data:
            reg_row = data['regression'][data['regression']['feature'] == feat]
            if len(reg_row) > 0:
                row['regression_weight'] = reg_row.iloc[0].get('regression_weight', np.nan)

        if 'comparison' in data:
            comp_row = data['comparison'][data['comparison']['feature'] == feat]
            if len(comp_row) > 0:
                row['mean_flip_correct'] = comp_row.iloc[0].get('mean_flip_correct', np.nan)
                row['mean_flip_incorrect'] = comp_row.iloc[0].get('mean_flip_incorrect', np.nan)
                row['divergence'] = comp_row.iloc[0].get('divergence', np.nan)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df['linearity_verdict'] = data.get('verdict_short', 'N/A')

    summary_file = MM_TABLES_DIR / f"final_summary_{model}.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"  Saved: {summary_file}")

    # Full evidence table (all data combined)
    evidence_rows = []
    for feat in TARGET_FEATURES:
        row = {'feature': feat}

        # Probing
        if 'probes' in data:
            row['probe_accuracy'] = data['probes'].get(feat, np.nan)

        # Behavioral weights
        row['bw_correct'] = get_weight(data, feat, 'correct')
        row['bw_incorrect'] = get_weight(data, feat, 'incorrect')

        # Regression
        if 'regression' in data:
            reg_row = data['regression'][data['regression']['feature'] == feat]
            if len(reg_row) > 0:
                for col in ['regression_weight', 'rank_bw_correct', 'rank_regression',
                            'rank_match_bw_vs_reg']:
                    if col in reg_row.columns:
                        row[col] = reg_row.iloc[0].get(col, np.nan)

        # Flip stats
        if 'comparison' in data:
            comp_row = data['comparison'][data['comparison']['feature'] == feat]
            if len(comp_row) > 0:
                for col in ['mean_flip_correct', 'mean_flip_incorrect',
                            'no_flip_rate_correct', 'no_flip_rate_incorrect', 'divergence']:
                    if col in comp_row.columns:
                        row[col] = comp_row.iloc[0].get(col, np.nan)

        # CoT frequency
        row['cot_freq_correct'] = get_cot_frequency(data, feat, 'correct')
        row['cot_freq_incorrect'] = get_cot_frequency(data, feat, 'incorrect')

        evidence_rows.append(row)

    evidence_df = pd.DataFrame(evidence_rows)
    evidence_df['linearity_verdict'] = data.get('verdict_short', 'N/A')
    evidence_df['model'] = model

    evidence_file = MM_TABLES_DIR / f"full_evidence_table_{model}.csv"
    evidence_df.to_csv(evidence_file, index=False)
    logger.info(f"  Saved: {evidence_file}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"  FINAL SUMMARY — {model}")
    logger.info(f"{'='*60}")

    logger.info(f"\n  {'Feature':12s} {'Probe':>8s} {'BW(C)':>8s} {'BW(I)':>8s} "
          f"{'Regr':>8s} {'Flip(C)':>8s} {'Flip(I)':>8s}")
    for _, row in summary_df.iterrows():
        probe = f"{row.get('probe_accuracy', 0):.0%}" if pd.notna(row.get('probe_accuracy')) else 'N/A'
        bw_c = f"{row['bw_correct']:.3f}"
        bw_i = f"{row['bw_incorrect']:.3f}"
        reg = f"{row.get('regression_weight', 0):.3f}" if pd.notna(row.get('regression_weight')) else 'N/A'
        fc = f"{row.get('mean_flip_correct', 0):.1f}" if pd.notna(row.get('mean_flip_correct')) else 'N/A'
        fi = f"{row.get('mean_flip_incorrect', 0):.1f}" if pd.notna(row.get('mean_flip_incorrect')) else 'N/A'
        logger.info(f"  {row['feature']:12s} {probe:>8s} {bw_c:>8s} {bw_i:>8s} "
              f"{reg:>8s} {fc:>8s} {fi:>8s}")

    verdict = data.get('verdict_short', 'N/A')
    logger.info(f"\n  Linearity verdict: {verdict}")
    logger.info(f"\n  Step 10 complete for {model}")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 10: Produce final reasoning graph")
    parser.add_argument('--model', type=str, default='llama-3.2-3b',
                        choices=['llama-3.2-3b', 'qwen3-4b'],
                        help='Model to visualize')
    args = parser.parse_args()
    main(model=args.model)
