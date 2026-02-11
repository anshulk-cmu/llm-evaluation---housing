# -*- coding: utf-8 -*-
"""
Step 5: Map the Model's Reasoning Stages (CPU-only)

Parses every CoT response (baseline + perturbation) into structured reasoning
stages, then compares patterns between correct and incorrect predictions.

Reasoning stages (matching the few-shot prompt format):
  1. Feature Reading   — model lists/reads feature values
  2. Pairwise Comparison — model compares features head-to-head ("P1 has 3 vs 2")
  3. Weighing/Aggregation — model weighs which advantages matter more (REASON field)
  4. Final Decision    — model commits to CHOICE

Also analyzes how reasoning shifts during perturbation:
  - Gradual: loser's advantages grow in the CoT before flip
  - Sudden: CoT unchanged until abrupt flip
  - Non-monotonic: CoT oscillates

Inputs:
  - macro_mapping/data/cot_responses_{model}.csv          (Step 2 baseline CoT)
  - macro_mapping/data/perturbation_responses_{model}.csv  (Step 4 perturbation CoT)

Outputs:
  - macro_mapping/data/reasoning_stages_{model}.csv
  - macro_mapping/data/stage_summary_{model}.csv
  - macro_mapping/data/transition_patterns_{model}.csv

Usage:
    python macro_mapping/map_reasoning_stages.py --model llama-3.2-3b
    python macro_mapping/map_reasoning_stages.py --model qwen3-4b
"""

import sys
import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mm_config import TARGET_FEATURES, REASONING_STAGES, setup_logger

logger = setup_logger("Step5_ReasoningStages")


# ============================================================================
# PATHS
# ============================================================================

MM_DATA_DIR = Path(__file__).resolve().parent / "data"


# ============================================================================
# FEATURE DETECTION PATTERNS
# ============================================================================

# Map feature keywords in CoT text to canonical feature names
FEATURE_PATTERNS = {
    'bedrooms': re.compile(
        r'\b(bed(?:room)?s?|beds?)\b', re.IGNORECASE
    ),
    'bathrooms': re.compile(
        r'\b(bath(?:room)?s?|baths?)\b', re.IGNORECASE
    ),
    'yearBuilt': re.compile(
        r'\b(year\s*built|year\s+(?:\w+\s+){1,3}built|age|newer|older|built\s+in\s+\d{4})\b',
        re.IGNORECASE
    ),
    'lot': re.compile(
        r'\b(lot\s*(?:size)?|sqft|square\s*feet?|sq\s*ft|acreage|acres?)\b', re.IGNORECASE
    ),
    'type': re.compile(
        r'\b(type|single\s*family|townhouse|condo(?:minium)?|multi\s*family|duplex)\b',
        re.IGNORECASE
    ),
    'heating': re.compile(
        r'\b(heat(?:ing)?|heat\s*pump|forced\s*air|radiant|boiler|furnace)\b', re.IGNORECASE
    ),
    'cooling': re.compile(
        r'\b(cool(?:ing)?|air\s*condition(?:ing|er)?|central\s*air|window\s*(?:ac|unit)|AC|HVAC)\b',
        re.IGNORECASE
    ),
    'parkingType': re.compile(
        r'\b(park(?:ing)?|garage|driveway|carport|street\s*parking|attached\s*garage)\b',
        re.IGNORECASE
    ),
}


# ============================================================================
# STAGE CLASSIFICATION
# ============================================================================

# Stage indicators
WEIGHING_KEYWORDS = re.compile(
    r'\b(outweigh|more\s+important|significant|despite|overall|'
    r'advantage|favor|edge|offset|compensate|overcome|'
    r'however|although|while|but|nevertheless|on\s+balance)\b',
    re.IGNORECASE
)

CONFIDENCE_KEYWORDS = re.compile(
    r'\b(clearly|obviously|definitely|likely|slight|close\s+call|'
    r'marginal|toss-up|barely|edge|strong|weak|uncertain)\b',
    re.IGNORECASE
)

COMPARISON_KEYWORDS = re.compile(
    r'\b(vs\.?|versus|compared\s+to|more\s+than|less\s+than|'
    r'has\s+\d|larger|smaller|bigger|newer|older|better|worse|'
    r'Property\s+[12]\s+has|P[12]\s+has|\+\d+\s+for\s+P[12])\b',
    re.IGNORECASE
)


def detect_features_in_text(text: str) -> list:
    """Find which features are mentioned in a text segment."""
    found = []
    for feat, pattern in FEATURE_PATTERNS.items():
        if pattern.search(text):
            found.append(feat)
    return found


def segment_cot(text: str) -> list:
    """
    Split CoT text into reasoning segments.

    Returns list of dicts: {'text': str, 'segment_type': str}
    where segment_type is 'bullet', 'reason', 'choice', 'header', or 'freeform'.
    """
    if not text or not isinstance(text, str):
        return []

    segments = []

    # Split on lines
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.upper().startswith('CHOICE:'):
            segments.append({'text': line, 'segment_type': 'choice'})
        elif line.upper().startswith('REASON:'):
            segments.append({'text': line, 'segment_type': 'reason'})
        elif line.startswith('- ') or line.startswith('* '):
            segments.append({'text': line, 'segment_type': 'bullet'})
        elif line.lower().startswith("let's think") or line.lower().startswith("let us think"):
            segments.append({'text': line, 'segment_type': 'header'})
        else:
            segments.append({'text': line, 'segment_type': 'freeform'})

    return segments


def classify_segment_stage(segment: dict) -> str:
    """
    Classify a segment into one of the 4 reasoning stages.

    Returns: 'feature_reading', 'pairwise_comparison', 'weighing_aggregation',
             or 'final_decision'
    """
    stype = segment['segment_type']
    text = segment['text']

    if stype == 'choice':
        return 'final_decision'

    if stype == 'reason':
        return 'weighing_aggregation'

    if stype == 'header':
        return 'feature_reading'  # "Let's think step by step" starts reading

    # For bullets and freeform, classify based on content
    if stype == 'bullet':
        # Check if it's a comparison (e.g., "Property 1 has 3 vs 2")
        if COMPARISON_KEYWORDS.search(text):
            return 'pairwise_comparison'
        # If it just lists a feature without comparison, it's reading
        return 'feature_reading'

    # Freeform text
    if WEIGHING_KEYWORDS.search(text):
        return 'weighing_aggregation'
    if COMPARISON_KEYWORDS.search(text):
        return 'pairwise_comparison'

    return 'feature_reading'


def parse_cot_stages(text: str) -> list:
    """
    Full pipeline: segment → classify → extract features for each CoT response.

    Returns list of dicts with: stage, features, text_excerpt
    """
    segments = segment_cot(text)
    results = []

    for seg in segments:
        stage = classify_segment_stage(seg)
        features = detect_features_in_text(seg['text'])
        results.append({
            'stage': stage,
            'features': features,
            'text_excerpt': seg['text'][:200],
        })

    return results


# ============================================================================
# TRANSITION ANALYSIS (perturbation CoT shifts)
# ============================================================================

def analyze_transitions(pert_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (pair_index, feature), analyze how the CoT shifts across
    perturbation steps as we approach the flip point.

    Classifies each (pair, feature) sequence as:
      - 'gradual': loser mentions increase monotonically before flip
      - 'sudden': loser mentions flat until abrupt flip
      - 'non_monotonic': loser mentions oscillate
      - 'no_change': no flip occurred, reasoning stable
    """
    transition_rows = []

    for (pi, feat), group in pert_df.groupby(['pair_index', 'feature']):
        group_sorted = group.sort_values('step_index')

        # Count how many times the perturbed feature is mentioned
        # favorably for the loser across steps
        mention_counts = []
        for _, row in group_sorted.iterrows():
            cot = row.get('cot_text', '') or ''
            # Count mentions of the perturbed feature
            if feat in FEATURE_PATTERNS:
                count = len(FEATURE_PATTERNS[feat].findall(cot))
            else:
                count = 0
            mention_counts.append(count)

        # Classify transition pattern
        did_flip = group_sorted['did_flip'].any()

        if not did_flip:
            shift_type = 'no_change'
        elif len(mention_counts) <= 1:
            shift_type = 'sudden'
        else:
            # Check monotonicity of mention counts
            diffs = [mention_counts[i+1] - mention_counts[i]
                     for i in range(len(mention_counts)-1)]

            if all(d >= 0 for d in diffs) and any(d > 0 for d in diffs):
                shift_type = 'gradual'
            elif all(d == 0 for d in diffs[:-1]) and len(diffs) > 0:
                shift_type = 'sudden'
            else:
                shift_type = 'non_monotonic'

        transition_rows.append({
            'pair_index': pi,
            'feature': feat,
            'is_correct': group_sorted.iloc[0].get('is_correct', None),
            'shift_type': shift_type,
            'did_flip': did_flip,
            'n_steps': len(group_sorted),
            'mention_counts': str(mention_counts),
        })

    return pd.DataFrame(transition_rows)


# ============================================================================
# MAIN
# ============================================================================

def main(model: str = "llama-3.2-3b"):
    logger.info(f"Step 5: Map Reasoning Stages for {model}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    cot_file = MM_DATA_DIR / f"cot_responses_{model}.csv"
    pert_file = MM_DATA_DIR / f"perturbation_responses_{model}.csv"

    if not cot_file.exists():
        logger.error(f" {cot_file} not found. Run Step 2 first.")
        sys.exit(1)
    if not pert_file.exists():
        logger.error(f" {pert_file} not found. Run Step 4 first.")
        sys.exit(1)

    cot_df = pd.read_csv(cot_file)
    pert_df = pd.read_csv(pert_file)
    logger.info(f"  Loaded {len(cot_df)} baseline CoTs, {len(pert_df)} perturbation CoTs")

    # ------------------------------------------------------------------
    # Parse baseline CoT into stages
    # ------------------------------------------------------------------
    logger.info(f"\nParsing baseline CoT into reasoning stages...")

    stage_rows = []
    for _, row in cot_df.iterrows():
        cot_text = row.get('full_cot', '') or ''
        stages = parse_cot_stages(cot_text)

        # Which stages are present?
        stages_present = set(s['stage'] for s in stages)
        all_features = []
        for s in stages:
            all_features.extend(s['features'])

        for stage_name in REASONING_STAGES:
            stage_segments = [s for s in stages if s['stage'] == stage_name]
            stage_features = []
            stage_excerpts = []
            for s in stage_segments:
                stage_features.extend(s['features'])
                stage_excerpts.append(s['text_excerpt'])

            stage_rows.append({
                'pair_index': row.get('pair_index', None),
                'is_correct': row.get('new_is_correct', None),
                'stage_name': stage_name,
                'stage_present': stage_name in stages_present,
                'n_segments': len(stage_segments),
                'features_mentioned': ','.join(sorted(set(stage_features))) if stage_features else '',
                'n_features': len(set(stage_features)),
                'text_excerpt': ' | '.join(stage_excerpts)[:500] if stage_excerpts else '',
            })

    stages_df = pd.DataFrame(stage_rows)

    # ------------------------------------------------------------------
    # Stage summary: correct vs incorrect
    # ------------------------------------------------------------------
    logger.info(f"\nBuilding stage summary (correct vs incorrect)...")

    summary_rows = []
    for stage_name in REASONING_STAGES:
        stage_data = stages_df[stages_df['stage_name'] == stage_name]

        for label, is_correct_val in [('correct', True), ('incorrect', False)]:
            sub = stage_data[stage_data['is_correct'] == is_correct_val]

            if len(sub) == 0:
                continue

            freq = sub['stage_present'].mean()
            avg_features = sub['n_features'].mean()

            # Most common features mentioned in this stage
            all_feats = []
            for feats_str in sub['features_mentioned']:
                if feats_str:
                    all_feats.extend(feats_str.split(','))
            feat_counter = Counter(all_feats)
            top_features = feat_counter.most_common(5)

            summary_rows.append({
                'stage_name': stage_name,
                'group': label,
                'frequency': freq,
                'avg_features_mentioned': avg_features,
                'n_samples': len(sub),
                'top_features': str(top_features),
            })

    summary_df = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # Transition analysis from perturbation CoTs
    # ------------------------------------------------------------------
    logger.info(f"\nAnalyzing reasoning transitions during perturbation...")

    transitions_df = analyze_transitions(pert_df)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"  REASONING STAGE ANALYSIS")
    logger.info(f"{'='*60}")

    for stage_name in REASONING_STAGES:
        logger.info(f"\n  {stage_name}:")
        stage_sum = summary_df[summary_df['stage_name'] == stage_name]
        for _, row in stage_sum.iterrows():
            logger.info(f"    {row['group']:12s}: present in {row['frequency']:.0%} of responses, "
                  f"avg {row['avg_features_mentioned']:.1f} features")

    # Key finding: do incorrect predictions skip weighing?
    weighing_correct = summary_df[
        (summary_df['stage_name'] == 'weighing_aggregation') &
        (summary_df['group'] == 'correct')
    ]
    weighing_incorrect = summary_df[
        (summary_df['stage_name'] == 'weighing_aggregation') &
        (summary_df['group'] == 'incorrect')
    ]
    if len(weighing_correct) > 0 and len(weighing_incorrect) > 0:
        logger.info(f"\n  KEY FINDING — Weighing stage presence:")
        logger.info(f"    Correct predictions:   {weighing_correct.iloc[0]['frequency']:.0%}")
        logger.info(f"    Incorrect predictions:  {weighing_incorrect.iloc[0]['frequency']:.0%}")

    # Transition summary
    logger.info(f"\n{'='*60}")
    logger.info(f"  TRANSITION PATTERNS (perturbation)")
    logger.info(f"{'='*60}")
    shift_counts = transitions_df['shift_type'].value_counts()
    for shift_type, count in shift_counts.items():
        pct = 100 * count / len(transitions_df)
        logger.info(f"  {shift_type:15s}: {count:4d} ({pct:.1f}%)")

    # Per-feature transition patterns
    logger.info(f"\n  Transition patterns by feature:")
    for feat in TARGET_FEATURES:
        feat_trans = transitions_df[transitions_df['feature'] == feat]
        if len(feat_trans) > 0:
            shift_dist = feat_trans['shift_type'].value_counts()
            logger.info(f"    {feat:12s}: {dict(shift_dist)}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    stages_file = MM_DATA_DIR / f"reasoning_stages_{model}.csv"
    stages_df.to_csv(stages_file, index=False)
    logger.info(f"\nSaved: {stages_file}")

    summary_file = MM_DATA_DIR / f"stage_summary_{model}.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved: {summary_file}")

    transitions_file = MM_DATA_DIR / f"transition_patterns_{model}.csv"
    transitions_df.to_csv(transitions_file, index=False)
    logger.info(f"Saved: {transitions_file}")

    return stages_df, summary_df, transitions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Map reasoning stages")
    parser.add_argument('--model', type=str, default='llama-3.2-3b',
                        choices=['llama-3.2-3b', 'qwen3-4b'],
                        help='Model to analyze')
    args = parser.parse_args()
    main(model=args.model)
