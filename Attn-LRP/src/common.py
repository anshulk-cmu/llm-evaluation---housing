# -*- coding: utf-8 -*-
"""
Small shared helpers: value cleaning and choice parsing.

`clean_str` and the CHOICE regex mirror the existing housing pipeline
(llm-evaluation-housing-reference/utils.py) so formatting matches the
behavioural runs exactly.
"""

from __future__ import annotations
import re
import math


def clean_str(x) -> str:
    """Clean a cell value to a prompt-ready string; NaN/empty -> 'NA'."""
    # pandas may hand us float NaN, None, numpy types, etc.
    if x is None:
        return "NA"
    try:
        if isinstance(x, float) and math.isnan(x):
            return "NA"
    except (TypeError, ValueError):
        pass
    s = str(x).strip()
    return s if s and s.lower() != "nan" else "NA"


CHOICE_RE = re.compile(r"CHOICE:\s*([12])", re.IGNORECASE)


def parse_choice(text: str):
    """Extract 1 or 2 from generated text; None if unparseable.

    Used only to sanity-check generation in the GPU path; the attribution
    decision itself is read from logits, not from decoded text.
    """
    if not text:
        return None
    m = CHOICE_RE.search(text)
    if m:
        return int(m.group(1))
    tail = text[-100:]
    has1 = re.search(r"\b1\b", tail) is not None
    has2 = re.search(r"\b2\b", tail) is not None
    if has1 and not has2:
        return 1
    if has2 and not has1:
        return 2
    return None
