# -*- coding: utf-8 -*-
"""
Perturbation Prompt and Utilities for Behavioral Macro Mapping Experiment

Based on the winning prompt strategy (2_fewshot_cot_temp0) from the initial analysis.
Key differences from original:
  1. Same system message, same property format, same few-shot examples
  2. Adds a REASON field to capture the model's stated logic for reasoning graph
  3. Property features can be perturbed one at a time for flip-point detection
  4. fmt_property_perturbed swaps exactly one feature, everything else byte-identical

Usage:
  - build_perturbation_prompt(row, perturb_property, perturb_feature, perturb_value)
  - perturb_property: 1 or 2 (which property to modify)
  - perturb_feature: "bedrooms", "bathrooms", "yearBuilt", "lot"
  - perturb_value: the new value to substitute (e.g., original bedrooms + increment)
  - If perturb_feature is None, returns the unperturbed baseline prompt
"""

import re
import numpy as np
import pandas as pd
from mm_config import PERTURBATION_STEPS, FEATURE_CAPS


# ============================================================================
# HELPERS (identical to original experiment)
# ============================================================================

def clean_str(x):
    """Clean and format string values, handle NaN."""
    if pd.isna(x):
        return "NA"
    s = str(x).strip()
    return s if s else "NA"


def fmt_property(row, suffix: str) -> str:
    """
    Create a human-readable property block using original text inputs.
    IDENTICAL to the original experiment — same field order, same formatting.
    """
    return (
        f"zpid: {clean_str(row['zpid' + suffix])}\n"
        f"type: {clean_str(row['type' + suffix])}\n"
        f"yearBuilt: {clean_str(row['yearBuilt' + suffix])}\n"
        f"bathrooms: {clean_str(row['bathrooms' + suffix])}\n"
        f"bedrooms: {clean_str(row['bedrooms' + suffix])}\n"
        f"lot: {clean_str(row['lot' + suffix])} {clean_str(row['lotUnit' + suffix])}\n"
        f"heating: {clean_str(row['heating' + suffix])}\n"
        f"cooling: {clean_str(row['cooling' + suffix])}\n"
        f"parkingType: {clean_str(row['parkingType' + suffix])}\n"
    )


def fmt_property_perturbed(row, suffix: str, perturb_feature: str, perturb_value) -> str:
    """
    Create a property block with ONE feature replaced by perturb_value.
    All other features remain identical to the original.

    Args:
        row: DataFrame row with property data
        suffix: "_1" or "_2"
        perturb_feature: which feature to change ("bedrooms", "bathrooms", "yearBuilt", "lot")
        perturb_value: the new value for that feature
    """
    # Start with original values
    fields = {
        'zpid': clean_str(row['zpid' + suffix]),
        'type': clean_str(row['type' + suffix]),
        'yearBuilt': clean_str(row['yearBuilt' + suffix]),
        'bathrooms': clean_str(row['bathrooms' + suffix]),
        'bedrooms': clean_str(row['bedrooms' + suffix]),
        'lot': f"{clean_str(row['lot' + suffix])} {clean_str(row['lotUnit' + suffix])}",
        'heating': clean_str(row['heating' + suffix]),
        'cooling': clean_str(row['cooling' + suffix]),
        'parkingType': clean_str(row['parkingType' + suffix]),
    }

    # Override the perturbed feature
    if perturb_feature == "lot":
        fields['lot'] = f"{perturb_value} {clean_str(row['lotUnit' + suffix])}"
    elif perturb_feature in fields:
        fields[perturb_feature] = str(perturb_value)

    return (
        f"zpid: {fields['zpid']}\n"
        f"type: {fields['type']}\n"
        f"yearBuilt: {fields['yearBuilt']}\n"
        f"bathrooms: {fields['bathrooms']}\n"
        f"bedrooms: {fields['bedrooms']}\n"
        f"lot: {fields['lot']}\n"
        f"heating: {fields['heating']}\n"
        f"cooling: {fields['cooling']}\n"
        f"parkingType: {fields['parkingType']}\n"
    )


# ============================================================================
# SYSTEM MESSAGE (identical to original experiment)
# ============================================================================

SYSTEM_MSG = (
    "You are a professional residential real-estate appraiser. "
    "You are given two home listings from the SAME ZIP CODE and the SAME MONTH. "
    "Your task is to determine which home is more likely to have a HIGHER CURRENT MARKET VALUE.\n"
    "You must rely only on the provided listing attributes. "
    "Do NOT use external knowledge, price indexes, or unstated assumptions. "
    "Be consistent, careful, and comparative."
)


# ============================================================================
# FEW-SHOT EXAMPLES (identical to original experiment)
# ============================================================================

FEW_SHOT_EXAMPLES = """EXAMPLE 1:
Property 1: Single Family, 1950, 3 bed/2 bath, 5000 sqft lot, Central Air, Garage
Property 2: Single Family, 1960, 2 bed/1 bath, 6000 sqft lot, Window AC, Street parking

Let's think step by step:
- Type: Both Single Family
- Beds: Property 1 has 3 vs 2 (+1 for P1)
- Baths: Property 1 has 2 vs 1 (+1 for P1)
- Lot: Property 2 has 6000 vs 5000 sqft (+1000 for P2)
- Age: Property 2 is 10 years newer
- Cooling: Property 1 has Central Air vs Window AC
- Parking: Property 1 has Garage vs Street

REASON: Property 1 has advantages in bedrooms, bathrooms, cooling, and parking. Property 2 has advantages in lot size and is newer. The structural advantages favor Property 1.

CHOICE: 1

EXAMPLE 2:
Property 1: Townhouse, 2005, 2 bed/2.5 bath, 1800 sqft lot, Heat Pump, Attached Garage
Property 2: Single Family, 1985, 3 bed/1 bath, 5000 sqft lot, Forced Air, Driveway

Let's think step by step:
- Type: Property 2 is Single Family, Property 1 is Townhouse
- Beds: Property 2 has 3 vs 2 (+1 for P2)
- Baths: Property 1 has 2.5 vs 1 (+1.5 for P1)
- Lot: Property 2 has 5000 vs 1800 sqft (+3200 for P2)
- Age: Property 1 is 20 years newer
- Heating: Property 1 has Heat Pump vs Forced Air
- Parking: Property 1 has Attached Garage vs Driveway

REASON: Property 2 is a single-family home with more lot space and an extra bedroom. Property 1 is newer with more bathrooms and better parking. The type and size advantages favor Property 2.

CHOICE: 2

"""


# ============================================================================
# PROMPT BUILDER
# ============================================================================

def build_perturbation_prompt(
    row,
    perturb_property: int = None,
    perturb_feature: str = None,
    perturb_value=None
) -> str:
    """
    Build the perturbation prompt for one property pair.

    If perturb_property/feature/value are None, returns the unperturbed baseline prompt.
    Otherwise, replaces one feature of the specified property with the new value.

    Args:
        row: DataFrame row from pairs_20pct_price_diff.csv
        perturb_property: 1 or 2 (which property to modify), or None for baseline
        perturb_feature: "bedrooms", "bathrooms", "yearBuilt", or "lot"
        perturb_value: the new value for that feature

    Returns:
        Formatted prompt string (user message only; system message applied separately)
    """
    z = clean_str(row['zipcode_1'])
    ym = clean_str(row['year_month.y_1'])

    # Build property blocks — perturb one if specified
    if perturb_property == 1 and perturb_feature is not None:
        p1 = fmt_property_perturbed(row, "_1", perturb_feature, perturb_value)
        p2 = fmt_property(row, "_2")
    elif perturb_property == 2 and perturb_feature is not None:
        p1 = fmt_property(row, "_1")
        p2 = fmt_property_perturbed(row, "_2", perturb_feature, perturb_value)
    else:
        # No perturbation — baseline prompt
        p1 = fmt_property(row, "_1")
        p2 = fmt_property(row, "_2")

    return (
        f"You are given two residential property listings from the same local market.\n"
        f"(zipcode: {z}, year_month: {ym}).\n\n"
        f"Your task is to decide which property is MORE VALUABLE in the current market.\n"
        f"IMPORTANT RULES:\n"
        f"- Use ONLY the information provided below.\n"
        f"- Do NOT infer neighborhood quality (they are in the same zipcode).\n"
        f"- Compare the two properties RELATIVELY, feature by feature.\n"
        f"- If the properties are close, still choose the more likely one.\n"
        f"- Do NOT output probabilities.\n\n"
        f"{FEW_SHOT_EXAMPLES}"
        f"NOW YOUR TURN:\n"
        f"Property 1:\n{p1}\n"
        f"Property 2:\n{p2}\n\n"
        f"Let's think step by step:\n\n"
        f"OUTPUT FORMAT (MUST FOLLOW EXACTLY):\n\n"
        f"REASON: <1-2 sentences explaining which features mattered most and why>\n"
        f"CHOICE: <1 or 2>\n"
    )


# ============================================================================
# PERTURBATION SCHEDULES (imported from mm_config — single source of truth)
# ============================================================================

# PERTURBATION_STEPS and FEATURE_CAPS imported at top from mm_config
PERTURBATION_SCHEDULE = PERTURBATION_STEPS  # alias for backward compat


# Conversion factor: 1 Acre = 43,560 sqft
SQFT_PER_ACRE = 43560.0


def get_perturbation_steps(row, perturb_property: int, feature: str):
    """
    Generate the list of (perturbed_value, increment_label) for a given feature.
    Respects realistic caps based on the data.

    For lot: the increment schedule is always in sqft. If the lot is measured
    in Acres, the increment is converted to acres (100 sqft ≈ 0.0023 acres)
    so the perturbed value stays in the original unit.

    Args:
        row: DataFrame row
        perturb_property: 1 or 2
        feature: "bedrooms", "bathrooms", "yearBuilt", or "lot"

    Returns:
        List of (new_absolute_value, increment_string) tuples
    """
    suffix = f"_{perturb_property}"
    original = row[f"{feature}{suffix}"]

    if pd.isna(original):
        return []

    original = float(original)
    steps = []

    if feature == "lot":
        # Determine lot unit for this property
        lot_unit = str(row.get(f'lotUnit{suffix}', 'sqft')).strip()
        is_acres = lot_unit.lower() == 'acres'

        # Cap in the property's native unit
        cap_sqft = FEATURE_CAPS[feature]  # 43560 sqft
        cap = cap_sqft / SQFT_PER_ACRE if is_acres else cap_sqft

        for increment_sqft in PERTURBATION_SCHEDULE[feature]:
            # Convert increment to property's native unit
            if is_acres:
                increment_native = increment_sqft / SQFT_PER_ACRE
                new_val = round(original + increment_native, 4)
            else:
                increment_native = increment_sqft
                new_val = int(original + increment_native)

            if new_val > cap:
                break

            label = f"+{increment_sqft}sqft"
            steps.append((new_val, label))

    elif feature in ("bedrooms", "bathrooms"):
        cap = FEATURE_CAPS[feature]
        for increment in PERTURBATION_SCHEDULE[feature]:
            new_val = int(original + increment)
            if new_val > cap:
                break
            label = f"+{increment}"
            steps.append((new_val, label))

    elif feature == "yearBuilt":
        cap = FEATURE_CAPS[feature]
        for increment in PERTURBATION_SCHEDULE[feature]:
            new_val = int(original + increment)
            if new_val > cap:
                break
            label = f"+{increment}yr"
            steps.append((new_val, label))

    return steps


# ============================================================================
# RESPONSE PARSING
# ============================================================================

CHOICE_RE = re.compile(r"CHOICE:\s*([12])", re.IGNORECASE)
REASON_RE = re.compile(r"REASON:\s*(.+?)(?:\n|CHOICE:)", re.IGNORECASE | re.DOTALL)

# Fallback patterns for models that don't use explicit REASON: prefix
# Matches freeform weighing paragraphs (Qwen often uses this style)
FREEFORM_REASON_RE = re.compile(
    r'(?:Key factors?|Overall|In (?:summary|conclusion)|'
    r'Considering|Taking (?:all|everything)|The (?:key|main|primary)|'
    r'Despite|Although|While)[:.]?\s*(.+?)(?:\n\n|CHOICE:)',
    re.IGNORECASE | re.DOTALL
)


def parse_response(text: str) -> dict:
    """
    Parse the model's response to extract choice and reasoning.

    Tries explicit REASON: pattern first (Llama follows this).
    Falls back to freeform weighing paragraph detection (Qwen style).

    Returns:
        dict with keys: 'choice' (1, 2, or NaN), 'reason' (str or None), 'full_text' (str)
    """
    if not text:
        return {'choice': np.nan, 'reason': None, 'full_text': ''}

    # Extract CHOICE
    m = CHOICE_RE.search(text)
    choice = int(m.group(1)) if m else np.nan

    # Extract REASON — try explicit pattern first
    m = REASON_RE.search(text)
    if m:
        reason = m.group(1).strip()
    else:
        # Fallback: freeform weighing paragraph
        m = FREEFORM_REASON_RE.search(text)
        reason = m.group(0).strip() if m else None

    return {
        'choice': choice,
        'reason': reason,
        'full_text': text
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Load data and show example prompts
    pairs = pd.read_csv("data/pairs_20pct_price_diff.csv")
    row = pairs.iloc[0]

    print("=" * 70)
    print("SYSTEM MESSAGE")
    print("=" * 70)
    print(SYSTEM_MSG)
    print()

    print("=" * 70)
    print("BASELINE PROMPT (no perturbation)")
    print("=" * 70)
    baseline = build_perturbation_prompt(row)
    print(baseline)
    print()

    print("=" * 70)
    print("PERTURBED PROMPT (Property 1 bedrooms +2)")
    print("=" * 70)
    original_beds = row['bedrooms_1']
    perturbed = build_perturbation_prompt(
        row,
        perturb_property=1,
        perturb_feature="bedrooms",
        perturb_value=int(original_beds + 2)
    )
    print(perturbed)
    print()

    # Show perturbation schedule for this pair
    print("=" * 70)
    print("PERTURBATION SCHEDULE FOR PROPERTY 1")
    print("=" * 70)
    for feature in PERTURBATION_SCHEDULE:
        steps = get_perturbation_steps(row, 1, feature)
        print(f"  {feature}: {steps}")
