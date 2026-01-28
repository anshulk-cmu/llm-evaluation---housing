# -*- coding: utf-8 -*-
"""
Prompt builders for LLM Real Estate Valuation Experiment
"""

from utils import clean_str

# ============================================================================
# PROPERTY FORMATTING
# ============================================================================

def fmt_property(row, suffix: str) -> str:
    """
    Create a human readable property block using original text inputs.
    
    Args:
        row: DataFrame row with property features
        suffix: "_1" or "_2" for the two listings
        
    Returns:
        Formatted property description string
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

# ============================================================================
# PROMPT BUILDERS
# ============================================================================

def build_prompt_baseline(row) -> str:
    """
    Baseline prompt (Condition 0) - Original structured approach.
    """
    z = clean_str(row['zipcode_1'])
    ym = clean_str(row['year_month.y_1'])
    p1 = fmt_property(row, "_1")
    p2 = fmt_property(row, "_2")
    
    return (
        f"You are given two residential property listings from the same local market\n"
        f"(zipcode: {z}, year_month: {ym}).\n\n"
        f"Your task is to decide which property is MORE VALUABLE in the current market.\n"
        f"IMPORTANT RULES:\n"
        f"- Use ONLY the information provided below.\n"
        f"- Do NOT infer neighborhood quality (they are in the same zipcode).\n"
        f"- Compare the two properties RELATIVELY, feature by feature.\n"
        f"- If the properties are close, still choose the more likely one.\n"
        f"- Do NOT output probabilities.\n\n"
        f"Focus especially on:\n"
        f"- Structural type (e.g., single family vs condo)\n"
        f"- Living space and lot size\n"
        f"- Bedrooms and bathrooms\n"
        f"- Heating, cooling, and parking characteristics\n"
        f"- Overall functional desirability implied by the features\n\n"
        f"Property 1:\n{p1}\n"
        f"Property 2:\n{p2}\n\n"
        f"STEP-BY-STEP INSTRUCTIONS (do not skip):\n"
        f"1. Compare the two properties feature by feature.\n"
        f"2. Identify which features clearly favor one property.\n"
        f"3. Weigh major structural differences more than minor amenities.\n"
        f"4. Decide which property would typically sell for more in the same market.\n\n"
        f"OUTPUT FORMAT (MUST FOLLOW EXACTLY):\n\n"
        f"CHOICE: <1 or 2>\n"
    )


def build_prompt_zero_shot_cot(row) -> str:
    """
    Zero-shot Chain-of-Thought prompt (Conditions 1 & 3).
    Adds "Let's think step by step:" to encourage reasoning.
    """
    z = clean_str(row['zipcode_1'])
    ym = clean_str(row['year_month.y_1'])
    p1 = fmt_property(row, "_1")
    p2 = fmt_property(row, "_2")
    
    return (
        f"You are given two residential property listings from the same local market\n"
        f"(zipcode: {z}, year_month: {ym}).\n\n"
        f"Your task is to decide which property is MORE VALUABLE in the current market.\n\n"
        f"IMPORTANT RULES:\n"
        f"- Use ONLY the information provided below.\n"
        f"- Do NOT infer neighborhood quality (they are in the same zipcode).\n"
        f"- Compare the two properties RELATIVELY, feature by feature.\n"
        f"- Do NOT output probabilities.\n"
        f"- If the properties are close, still choose the more likely one.\n\n"
        f"Focus especially on:\n"
        f"- Structural type (e.g., single family vs condo)\n"
        f"- Living space and lot size\n"
        f"- Bedrooms and bathrooms\n"
        f"- Heating, cooling, and parking characteristics\n"
        f"- Overall functional desirability implied by the features\n\n"
        f"Property 1:\n{p1}\n"
        f"Property 2:\n{p2}\n\n"
        f"Let's think step by step:\n\n"
        f"OUTPUT FORMAT (MUST FOLLOW EXACTLY):\n\n"
        f"CHOICE: <1 or 2>\n"
    )


def build_prompt_few_shot_cot(row) -> str:
    """
    Few-shot Chain-of-Thought prompt (Conditions 2 & 4).
    Includes examples of step-by-step reasoning.
    """
    z = clean_str(row['zipcode_1'])
    ym = clean_str(row['year_month.y_1'])
    p1 = fmt_property(row, "_1")
    p2 = fmt_property(row, "_2")
    
    few_shot = """EXAMPLE 1:
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

CHOICE: 2

"""
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
        f"{few_shot}"
        f"NOW YOUR TURN:\n"
        f"Property 1:\n{p1}\n"
        f"Property 2:\n{p2}\n\n"
        f"Let's think step by step:\n\n"
        f"OUTPUT FORMAT (MUST FOLLOW EXACTLY):\n\n"
        f"CHOICE: <1 or 2>\n"
    )


# ============================================================================
# PROMPT FUNCTION REGISTRY
# ============================================================================

PROMPT_BUILDERS = {
    "build_prompt_baseline": build_prompt_baseline,
    "build_prompt_zero_shot_cot": build_prompt_zero_shot_cot,
    "build_prompt_few_shot_cot": build_prompt_few_shot_cot,
}
