# -*- coding: utf-8 -*-
"""
Non-GPU smoke test — validates the whole pipeline EXCEPT the LXT backward.

It exercises: data loading, prompt building + value spans, 120-permutation
coverage, char-span -> token mapping (with a whitespace pseudo-tokenizer),
feature aggregation + simplex normalisation, the statistics layer, the sanity
helpers, and figure generation (on synthetic results written to results/_smoke).

Run:
    python smoke_test.py
Requires only numpy/pandas/scipy/matplotlib (no torch, no model).
"""

from __future__ import annotations
import os
import sys
import re
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from log import get_logger, banner
from config import FEATURES, RESULTS_DIR, FIGURES_DIR
from data import load_pairs, sample_pairs
from prompts import build_attribution_prompt, all_feature_permutations, assert_complete_coverage
from aggregate import (map_spans_to_tokens, aggregate_features, combine_two_properties,
                       simplex_normalize, attribution_fraction, feature_token_counts)
import stats as st
import sanity as sn
import plots

logger = get_logger(__name__)


def _pseudo_offsets(text: str):
    """Whitespace pseudo-tokeniser -> list of (start,end) char spans."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def test_prompt_and_spans():
    banner("SMOKE 1/6 — prompt build + value spans + token mapping")
    df = sample_pairs(load_pairs(), n=5, oversample=1.0)
    row = df.iloc[0]
    user_text, spans, meta = build_attribution_prompt(row)
    logger.info("sample prompt:\n%s", user_text)
    # every value span must slice out non-empty text
    for (slot, key), (s, e) in spans.items():
        assert e > s, f"empty span for {(slot, key)}"
        assert user_text[s:e].strip() != "", f"blank value for {(slot, key)}"
    offsets = _pseudo_offsets(user_text)
    tok_map = map_spans_to_tokens(offsets, 0, spans)
    for s in (1, 2):
        for f in FEATURES:
            assert len(tok_map[(s, f)]) >= 1, f"no tokens for {(s, f)}"
    logger.info("OK: %d spans mapped; all 5 features have tokens in both slots", len(spans))
    return user_text, spans, tok_map


def test_aggregation(user_text, spans, tok_map):
    banner("SMOKE 2/6 — aggregation + simplex normalisation + fraction")
    rng = np.random.default_rng(0)
    n_tok = len(_pseudo_offsets(user_text))
    R0 = rng.normal(size=n_tok)
    seg = aggregate_features(R0, tok_map, mode="mean")
    comb = combine_two_properties(seg)
    norm = simplex_normalize({f: comb[f]["abs_sum"] for f in FEATURES})
    total = sum(norm.values())
    assert abs(total - 1.0) < 1e-9, f"simplex must sum to 1, got {total}"
    frac = attribution_fraction(R0, tok_map)
    counts = feature_token_counts(tok_map)
    logger.info("normalized relevance: %s", {k: round(v, 3) for k, v in norm.items()})
    logger.info("attribution fraction=%.3f  token counts=%s", frac, counts)
    logger.info("OK: simplex sums to 1 over 5 features")


def test_permutations():
    banner("SMOKE 3/6 — 120 permutations + complete positional coverage")
    pool = all_feature_permutations()
    assert len(pool) == 120
    assert_complete_coverage(pool)               # each (feature,pos) == 24
    logger.info("OK: 5! = 120 permutations, every (feature,position) appears 24x")


def test_stats():
    banner("SMOKE 4/6 — statistics (bootstrap, friedman, wilcoxon, ranks, cross)")
    rng = np.random.default_rng(1)
    # synthetic per-sample simplex rows with a planted ordering
    base = np.array([0.40, 0.20, 0.16, 0.14, 0.10])    # lot>bath>bed>yr>type (arbitrary)
    X = rng.dirichlet(base * 50, size=200)
    mean, lo, hi = st.bootstrap_ci(X, B=500)
    assert np.all(hi >= mean) and np.all(mean >= lo)
    fstat, fp = st.friedman(X)
    wil = st.wilcoxon_posthoc(X, [("lot", "year_built"), ("bedrooms", "bathrooms")])
    cm = st.cross_method_table(mean)
    ct = st.cross_target_correlation(mean, mean[::-1].copy())
    logger.info("friedman stat=%.1f p=%.2e", fstat, fp)
    logger.info("wilcoxon: %s", [(w["pair"], round(w["p_fdr"], 4)) for w in wil])
    logger.info("cross-method rows=%d (cross_model flagged=%s)", len(cm), cm[0]["cross_model"])
    logger.info("cross-target rho=%.3f", ct["spearman_rho"])
    logger.info("OK: stats layer runs end-to-end")
    return X


def test_sanity():
    banner("SMOKE 5/6 — sanity helpers (conservation, neutrality, deletion AUC)")
    cs = sn.conservation_summary([0.005, 0.01, 0.008, 0.02])
    tn = sn.token_count_neutrality({f: [0.2, 0.3] for f in FEATURES},
                                   {f: [3, 4] for f in FEATURES})
    curve = sn.deletion_curve([0, 1, 2, 3], target_logit_fn=lambda m: 5.0 - len(m))
    auc = sn.normalized_auc(curve)
    assert curve[0] > curve[-1], "deletion curve should decrease for this synthetic fn"
    logger.info("conservation median=%.4f pass=%s", cs["median_rel_residual"], cs["pass"])
    logger.info("neutrality rho=%.3f ; deletion AUC=%.3f", tn["spearman_rho"], auc)
    logger.info("OK: sanity helpers run")


def test_plots(X):
    banner("SMOKE 6/6 — synthetic result CSVs + figure generation")
    rdir = os.path.join(RESULTS_DIR, "_smoke")
    fdir = os.path.join(FIGURES_DIR, "_smoke")
    os.makedirs(rdir, exist_ok=True); os.makedirs(fdir, exist_ok=True)

    def w(name, fields, rows):
        with open(os.path.join(rdir, name), "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields); wr.writeheader()
            for r in rows: wr.writerow(r)

    mean, lo, hi = st.bootstrap_ci(X, B=300)
    t1 = st.topk_frequency(X, 1); t3 = st.topk_frequency(X, 3); rk = st.ranks_from_means(mean)
    for nm in ("attnlrp_single.csv", "attnlrp_logitdiff.csv"):
        w(nm, ["feature", "mean", "std", "ci_lo", "ci_hi", "top1_freq", "top3_freq", "rank"],
          [dict(feature=f, mean=mean[i], std=X[:, i].std(), ci_lo=lo[i], ci_hi=hi[i],
                top1_freq=t1[i], top3_freq=t3[i], rank=rk[f]) for i, f in enumerate(FEATURES)])
    w("signed_relevance.csv", ["feature", "target", "mean_pos", "mean_neg", "pct_positive", "pct_negative", "net"],
      [dict(feature=f, target="single_logit", mean_pos=float(mean[i]), mean_neg=-0.02 * (i + 1),
            pct_positive=0.9, pct_negative=0.1, net=float(mean[i]) - 0.02 * (i + 1))
       for i, f in enumerate(FEATURES)])
    rng = np.random.default_rng(2)
    w("layer_feature_relevance.csv", ["target", "feature", "layer", "mean_norm"],
      [dict(target="single_logit", feature=f, layer=L, mean_norm=float(rng.random()))
       for L in range(28) for f in FEATURES] +
      [dict(target="single_logit", feature="__total__", layer=L, mean_norm=float(rng.random()))
       for L in range(28)])
    w("permutation_stability.csv", ["feature", "target", "rank_fixed", "rank_permuted", "mean_fixed", "mean_permuted"],
      [dict(feature=f, target="single_logit", rank_fixed=rk[f],
            rank_permuted=rk[f], mean_fixed=float(mean[i]), mean_permuted=float(mean[i]))
       for i, f in enumerate(FEATURES)])
    w("deletion_curve.csv", ["frac_removed", "attnlrp", "random"],
      [dict(frac_removed=g, attnlrp=5 - 5 * g, random=5 - 2 * g) for g in np.linspace(0, 1, 11)])

    plots.main(results_dir=rdir, figures_dir=fdir)
    made = [p for p in os.listdir(fdir) if p.endswith(".png")]
    assert len(made) >= 4, f"expected >=4 figures, got {made}"
    logger.info("OK: figures written to %s -> %s", fdir, sorted(made))


def main():
    banner("ATTN-LRP SMOKE TEST (no GPU, no model)")
    ut, spans, tm = test_prompt_and_spans()
    test_aggregation(ut, spans, tm)
    test_permutations()
    X = test_stats()
    test_sanity()
    test_plots(X)
    banner("SMOKE TEST PASSED (LXT backward is the only untested piece - needs the GPU box)")


if __name__ == "__main__":
    main()
