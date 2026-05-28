# -*- coding: utf-8 -*-
"""Phase 2: full 5!=120 feature-order permutation control (PLAN 11). Batched."""

from __future__ import annotations
import argparse
import csv
import os
import numpy as np
import pandas as pd

from config import (FEATURES, TARGETS, RESULTS_DIR, N_VALID_TARGET, BATCH_SIZE,
                    F_ATTN_SINGLE, F_ATTN_LOGDIFF, F_PERM, banner)
from data import load_pairs, sample_pairs
from prompts import build_attribution_prompt, all_feature_permutations, assert_complete_coverage
from aggregate import map_spans_to_tokens, aggregate_features, combine_two_properties, simplex_normalize
import stats as st
from log import get_logger

logger = get_logger(__name__)


def _write_rows(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("wrote %s (%d rows)", os.path.basename(path), len(rows))


def _fixed_ranks():
    out = {}
    for t, fn in (("single_logit", F_ATTN_SINGLE), ("logit_diff", F_ATTN_LOGDIFF)):
        p = os.path.join(RESULTS_DIR, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{fn} not found - run Phase 1 first.")
        df = pd.read_csv(p).set_index("feature")
        out[t] = {f: int(df.loc[f, "rank"]) for f in FEATURES}
        out[t + "_mean"] = {f: float(df.loc[f, "mean"]) for f in FEATURES}
    return out


def run(n_target=N_VALID_TARGET, seed=42, batch_size=BATCH_SIZE):
    from attribution import load_model_and_tokenizer, prepare_batch, attribute_batch

    banner("Attn-LRP Phase 2 - full 120-permutation position control")
    pool = all_feature_permutations()
    assert_complete_coverage(pool)
    fixed = _fixed_ranks()
    model, tok, method = load_model_and_tokenizer()
    df = sample_pairs(load_pairs(), n=n_target, seed=seed)

    items = []
    for i, (_, row) in enumerate(df.iterrows()):
        order = pool[i % 120]
        ut, spans, _ = build_attribution_prompt(row, feature_order=order, swap_properties=(i % 2 == 1))
        items.append((ut, spans, order))

    persample = {t: [] for t in TARGETS}
    pos_sum = {t: np.zeros((len(FEATURES), len(FEATURES))) for t in TARGETS}
    pos_cnt = {t: np.zeros((len(FEATURES), len(FEATURES))) for t in TARGETS}
    fidx = {f: i for i, f in enumerate(FEATURES)}

    valid = 0
    for start in range(0, len(items), batch_size):
        if valid >= n_target:
            break
        chunk = items[start:start + batch_size]
        try:
            prep = prepare_batch(model, tok, [c[0] for c in chunk])
            res = attribute_batch(model, prep, TARGETS, capture_layers=False)
        except Exception as e:
            logger.warning("batch at %d failed: %r", start, e)
            continue
        for k, (ut, spans, order) in enumerate(chunk):
            if valid >= n_target:
                break
            r0 = res[TARGETS[0]][k]
            tm = map_spans_to_tokens(r0["offsets"], r0["user_off"], spans)
            if any(len(tm[(s, f)]) == 0 for s in (1, 2) for f in FEATURES):
                continue
            cache, ok = {}, True
            for t in TARGETS:
                comb = combine_two_properties(aggregate_features(res[t][k]["R0"], tm, mode="mean"))
                feat_abs = {f: comb[f]["abs_sum"] for f in FEATURES}
                if sum(feat_abs.values()) <= 0:
                    ok = False
                    break
                cache[t] = simplex_normalize(feat_abs)
            if not ok:
                continue
            for t in TARGETS:
                persample[t].append(cache[t])
                for pos, f in enumerate(order):
                    pos_sum[t][fidx[f], pos] += cache[t][f]
                    pos_cnt[t][fidx[f], pos] += 1
            valid += 1
            if valid % 25 == 0:
                logger.info("%d/%d valid", valid, n_target)

    banner(f"Phase 2 collected {valid} valid - writing stability + position sensitivity")
    stab_rows, pos_rows = [], []
    for t in TARGETS:
        X = np.array([[d[f] for f in FEATURES] for d in persample[t]], dtype=float)
        mean = X.mean(axis=0)
        prank = st.ranks_from_means(mean)
        for f in FEATURES:
            stab_rows.append(dict(feature=f, target=t, rank_fixed=fixed[t][f],
                                  rank_permuted=prank[f], mean_fixed=fixed[t + "_mean"][f],
                                  mean_permuted=float(mean[fidx[f]])))
        with np.errstate(invalid="ignore"):
            posmean = np.where(pos_cnt[t] > 0, pos_sum[t] / pos_cnt[t], np.nan)
        for f in FEATURES:
            for pos in range(len(FEATURES)):
                pos_rows.append(dict(feature=f, position=pos, target=t,
                                     mean_norm=float(posmean[fidx[f], pos]),
                                     n=int(pos_cnt[t][fidx[f], pos])))
        rho, p = st.spearman([fixed[t][f] for f in FEATURES], [prank[f] for f in FEATURES])
        logger.info("%s: fixed-vs-permuted Spearman rho=%.3f (p=%.3g)", t, rho, p)

    _write_rows(os.path.join(RESULTS_DIR, F_PERM),
                ["feature", "target", "rank_fixed", "rank_permuted", "mean_fixed", "mean_permuted"], stab_rows)
    _write_rows(os.path.join(RESULTS_DIR, "position_sensitivity.csv"),
                ["feature", "position", "target", "mean_norm", "n"], pos_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_VALID_TARGET)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    a = ap.parse_args()
    run(n_target=a.n, batch_size=a.batch_size)
