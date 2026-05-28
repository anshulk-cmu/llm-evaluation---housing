# -*- coding: utf-8 -*-
"""
Load and sample housing pairs (PLAN §10 step 1).

The pairs file already enforces a >=20% price gap, so pairs are non-tie by
construction. "Valid" additionally requires the model to emit a parseable
choice with both digit logits available — that filter happens in the GPU path,
so here we just draw a seeded sample (a few extra, to survive parse failures).
"""

from __future__ import annotations
import pandas as pd

from config import DATA_PATH, SEED, N_VALID_TARGET, FEATURE_COLUMNS


# Columns the prompt builder needs, for both listings.
def _required_columns():
    cols = []
    for srcs in FEATURE_COLUMNS.values():
        for s in srcs:
            cols += [f"{s}_1", f"{s}_2"]
    return cols


def load_pairs() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in _required_columns() if c not in df.columns]
    if missing:
        raise KeyError(f"pairs file is missing expected columns: {missing}")
    return df


def sample_pairs(df: pd.DataFrame | None = None,
                 n: int = N_VALID_TARGET,
                 oversample: float = 1.3,
                 seed: int = SEED) -> pd.DataFrame:
    """Seeded draw of ~n*oversample rows (oversample covers parse failures).

    The GPU runner trims to the first N_VALID_TARGET pairs that yield a valid
    decision, so we draw a margin here.
    """
    if df is None:
        df = load_pairs()
    take = min(len(df), int(round(n * oversample)))
    return df.sample(n=take, random_state=seed).reset_index(drop=True)
