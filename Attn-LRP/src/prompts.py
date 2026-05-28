# -*- coding: utf-8 -*-
"""
Simplified attribution prompt + feature-order permutations (PLAN §9.4, §11.3).

The prompt deliberately DROPS the behavioural instruction blocks, leaving only
the two property listings and a trailing `CHOICE:` cue — matching the housing
IG/SHAP/Occlusion setup so AttnLRP is protocol-comparable, and pushing relevance
onto feature tokens.

This module is tokenizer-free: `build_attribution_prompt` returns the raw user
text plus character spans of every feature VALUE (and the zpid control value),
measured against that user text. The attribution module maps those character
spans to token indices using the tokenizer's offset mapping (PLAN §7.1).
"""

from __future__ import annotations
from itertools import permutations
from typing import Dict, List, Tuple

from config import FEATURES, FEATURE_LABEL, FEATURE_COLUMNS
from common import clean_str


def _feature_value(row, fkey: str, listing: int) -> str:
    """Render a feature's value string for a given CSV listing (1 or 2)."""
    cols = FEATURE_COLUMNS[fkey]
    if fkey == "lot":
        lot = clean_str(row[f"{cols[0]}_{listing}"])
        unit = clean_str(row[f"{cols[1]}_{listing}"])
        return f"{lot} {unit}".strip()
    return clean_str(row[f"{cols[0]}_{listing}"])


def build_attribution_prompt(
    row,
    feature_order: List[str] | None = None,
    swap_properties: bool = False,
) -> Tuple[str, Dict[Tuple[int, str], Tuple[int, int]], Dict]:
    """Build the simplified attribution prompt.

    Returns
    -------
    user_text : str
        The user message (listings + 'CHOICE:').
    spans : dict[(slot, feature_key) -> (char_start, char_end)]
        Character span of each VALUE within `user_text`. `feature_key` is one of
        FEATURES or 'zpid'. `slot` is 1 or 2 (display position).
    meta : dict
        {'slot_to_listing': {slot: csv_listing}} so callers can aggregate by
        feature IDENTITY regardless of P1/P2 swap.
    """
    if feature_order is None:
        feature_order = list(FEATURES)
    assert sorted(feature_order) == sorted(FEATURES), "feature_order must be a permutation of FEATURES"

    slot_to_listing = {1: 2, 2: 1} if swap_properties else {1: 1, 2: 2}

    parts: List[str] = []
    spans: Dict[Tuple[int, str], Tuple[int, int]] = {}
    cursor = 0

    def emit(s: str) -> None:
        nonlocal cursor
        parts.append(s)
        cursor += len(s)

    def emit_kv(slot: int, key: str, label: str, value: str) -> None:
        emit(f"{label}: ")
        start = cursor
        emit(value)
        spans[(slot, key)] = (start, cursor)   # end is exclusive
        emit("\n")

    for slot in (1, 2):
        listing = slot_to_listing[slot]
        emit(f"Property {slot}:\n")
        for fkey in feature_order:
            emit_kv(slot, fkey, FEATURE_LABEL[fkey], _feature_value(row, fkey, listing))
        emit("\n")

    emit("CHOICE:")
    user_text = "".join(parts)
    return user_text, spans, {"slot_to_listing": slot_to_listing}


# ---------------------------------------------------------------------------
# Feature-order permutations (PLAN §11.3): for 5 features the full set is 5!=120.
# ---------------------------------------------------------------------------
def all_feature_permutations() -> List[List[str]]:
    """All 5! = 120 orderings of FEATURES (the complete housing control set)."""
    perms = [list(p) for p in permutations(FEATURES)]
    assert len(perms) == 120, f"expected 120 permutations, got {len(perms)}"
    return perms


def positional_coverage(perms: List[List[str]]) -> Dict[Tuple[str, int], int]:
    """Count how often each feature lands at each position across `perms`."""
    counts: Dict[Tuple[str, int], int] = {(f, p): 0 for f in FEATURES for p in range(len(FEATURES))}
    for order in perms:
        for pos, f in enumerate(order):
            counts[(f, pos)] += 1
    return counts


def assert_complete_coverage(perms: List[List[str]]) -> None:
    """For the full 120-set every (feature, position) count must equal 24."""
    counts = positional_coverage(perms)
    bad = {k: v for k, v in counts.items() if v != 24}
    assert not bad, f"positional coverage not uniform (expected 24 each): {bad}"


def locate_user_text_offset(full_text: str, user_text: str) -> int:
    """Char offset of `user_text` inside the chat-templated `full_text`.

    The chat template only wraps the user content, so it appears verbatim.
    Raises if not found (a template change we must notice, not silently mishandle).
    """
    idx = full_text.find(user_text)
    if idx < 0:
        raise ValueError("user_text not found verbatim in templated full_text; "
                         "chat template may be altering content — investigate before trusting spans.")
    return idx
