# -*- coding: utf-8 -*-
"""
Token relevance -> per-feature importance (PLAN §7).

Conventions matched to the housing methods:
  - granularity = feature line (we already know exact value spans),
  - aggregation = MEAN over the tokens in a feature's value segment
    (neutralises token-count bias; SUM is kept only for conservation/fraction),
  - per-sample simplex normalisation over the five features.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from config import FEATURES


def map_spans_to_tokens(offsets: List[Tuple[int, int]],
                        user_off: int,
                        spans: Dict[Tuple[int, str], Tuple[int, int]]) -> Dict[Tuple[int, str], List[int]]:
    """Map each (slot, feature) char span (in user_text coords) to token indices.

    A token [a,b) belongs to value span [s,e) (shifted into full-text coords) iff
    they overlap and the token is not a zero-width special token.
    """
    token_map: Dict[Tuple[int, str], List[int]] = {}
    for key, (s, e) in spans.items():
        s_full, e_full = s + user_off, e + user_off
        idxs = []
        for ti, (a, b) in enumerate(offsets):
            if b <= a:                      # (0,0) special / empty token
                continue
            if a < e_full and b > s_full:   # overlap
                idxs.append(ti)
        token_map[key] = idxs
    return token_map


def aggregate_features(R0: np.ndarray,
                       token_map: Dict[Tuple[int, str], List[int]],
                       mode: str = "mean") -> Dict[Tuple[int, str], float]:
    """Reduce per-token relevance to one value per (slot, feature) segment."""
    assert mode in ("mean", "sum")
    out: Dict[Tuple[int, str], float] = {}
    for key, idxs in token_map.items():
        if not idxs:
            out[key] = 0.0
            continue
        vals = R0[idxs]
        out[key] = float(vals.mean() if mode == "mean" else vals.sum())
    return out


def combine_two_properties(seg_vals: Dict[Tuple[int, str], float]) -> Dict[str, Dict[str, float]]:
    """Combine the two slots per feature IDENTITY (PLAN §7.3).

    Returns feature -> {abs_sum, signed_sum, slot1, slot2}. zpid included too.
    """
    feats = set(k[1] for k in seg_vals)
    out = {}
    for f in feats:
        s1 = seg_vals.get((1, f), 0.0)
        s2 = seg_vals.get((2, f), 0.0)
        out[f] = {"abs_sum": abs(s1) + abs(s2), "signed_sum": s1 + s2, "slot1": s1, "slot2": s2}
    return out


def simplex_normalize(feature_abs: Dict[str, float]) -> Dict[str, float]:
    """Normalise the five FEATURES' abs values to sum to 1 (PLAN §7.4).

    Only the five features participate (zpid is excluded from the simplex).
    """
    total = sum(feature_abs.get(f, 0.0) for f in FEATURES)
    if total <= 0:
        return {f: 0.0 for f in FEATURES}
    return {f: feature_abs.get(f, 0.0) / total for f in FEATURES}


def attribution_fraction(R0: np.ndarray,
                         token_map: Dict[Tuple[int, str], List[int]]) -> float:
    """Fraction of total |relevance| (SUM form) landing on the five feature segments."""
    total = float(np.abs(R0).sum())
    if total <= 0:
        return 0.0
    feat_idx = set()
    for (slot, f), idxs in token_map.items():
        if f in FEATURES:
            feat_idx.update(idxs)
    on_feats = float(np.abs(R0[list(feat_idx)]).sum()) if feat_idx else 0.0
    return on_feats / total


def feature_token_counts(token_map: Dict[Tuple[int, str], List[int]]) -> Dict[str, int]:
    """Total token count per feature identity (both slots) — for the §15.3 check."""
    counts = {f: 0 for f in FEATURES}
    for (slot, f), idxs in token_map.items():
        if f in FEATURES:
            counts[f] += len(idxs)
    return counts
