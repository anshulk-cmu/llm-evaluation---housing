# -*- coding: utf-8 -*-
"""
Utility functions for LLM Real Estate Valuation Experiment
"""

import re
import numpy as np
import pandas as pd

# ============================================================================
# STRING CLEANING
# ============================================================================

def clean_str(x):
    """Clean and format string values, handle NaN."""
    if pd.isna(x):
        return "NA"
    s = str(x).strip()
    return s if s else "NA"

# ============================================================================
# RESULT PARSING
# ============================================================================

# Regex patterns to extract choice from model output
CHOICE_RE = re.compile(r"CHOICE:\s*([12])", re.IGNORECASE)
ANSWER_RE = re.compile(r"ANSWER:\s*([12])", re.IGNORECASE)

def parse_choice(text: str):
    """
    Extract choice (1 or 2) from model response.
    
    Args:
        text: Model's text response
        
    Returns:
        int (1 or 2) or np.nan if unable to parse
    """
    if not text:
        return np.nan
    
    # Try CHOICE: pattern first
    m = CHOICE_RE.search(text)
    if m:
        return int(m.group(1))
    
    # Try ANSWER: pattern
    m = ANSWER_RE.search(text)
    if m:
        return int(m.group(1))
    
    # Fallback: look for standalone 1 or 2 at end
    last_part = text[-100:]
    # Look for "1" or "2" as standalone
    if re.search(r'\b1\b', last_part) and not re.search(r'\b2\b', last_part):
        return 1
    elif re.search(r'\b2\b', last_part) and not re.search(r'\b1\b', last_part):
        return 2
    
    return np.nan
