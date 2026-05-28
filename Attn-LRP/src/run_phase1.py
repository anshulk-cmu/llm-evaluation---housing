# -*- coding: utf-8 -*-
"""Phase 1: AttnLRP on Llama-3.2-3B, fixed feature order, N=500 (PLAN 10). Batched."""

from __future__ import annotations
import argparse
import csv
import os
import numpy as np

from config import (FEATURES, TARGETS, RESULTS_DIR, N_VALID_TARGET, BATCH_SIZE,
                    F_ATTN_SINGLE, F_ATTN_LOGDIFF, F_SIGNED, F_LAYER, F_CROSS,
                    F_SANITY, F_PERSAMPLE, banner)
from data import load_pairs, sample_pairs
from prompts import build_attribution_prompt
from aggregate import (map_spans_to_tokens, aggregate_features, combine_two_properties,
                       simplex_normalize, attribution_fraction, feature_token_counts)
import stats as st
import sanity as sn
from log import get_logger

logger = get_logger(__name__)


def _write_rows(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("wrote %s (%d rows)", os.path.basename(path), len(rows))


def run(n_target=N_VALID_TARGET, capture_layers=True, deletion_subset=25,
        randomization_subset=8, seed=42, batch_size=BATCH_SIZE):
    from attribution import (load_model_and_tokenizer, prepare_batch, attribute_batch,
                             prepare_inputs, target_logit_fn_factory)

    banner("Attn-LRP Phase 1 - Llama-3.2-3B (open discovery of feature drivers)")
    model, tok, method = load_model_and_tokenizer()
    df = sample_pairs(load_pairs(), n=n_target, seed=seed)

    persample = {t: [] for t in TARGETS}
    persample_signed = {t: [] for t in TARGETS}
    tokcount_rel = {t: {f: [] for f in FEATURES} for t in TARGETS}
    tokcount_cnt = {t: {f: [] for f in FEATURES} for t in TARGETS}
    cons_res = {t: [] for t in TARGETS}
    frac_on_feats = {t: [] for t in TARGETS}
    layer_acc = {t: None for t in TARGETS}
    layer_total = {t: None for t in TARGETS}
    layer_count = {t: 0 for t in TARGETS}
    persample_long = []
    deletion_samples = []

    cand = []
    for _, row in df.iterrows():
        ut, spans, _ = build_attribution_prompt(row)
        cand.append((row, ut, spans))

    rng = np.random.default_rng(seed)
    valid = 0
    for start in range(0, len(cand), batch_size):
        if valid >= n_target:
            break
        chunk = cand[start:start + batch_size]
        try:
            prep = prepare_batch(model, tok, [c[1] for c in chunk])
            res = attribute_batch(model, prep, TARGETS, capture_layers=capture_layers)
        except Exception as e:
            logger.warning("batch at %d failed: %r", start, e)
            continue

        for k, (row, ut, spans) in enumerate(chunk):
            if valid >= n_target:
                break
            r0 = res[TARGETS[0]][k]
            token_map = map_spans_to_tokens(r0["offsets"], r0["user_off"], spans)
            if any(len(token_map[(s, f)]) == 0 for s in (1, 2) for f in FEATURES):
                continue

            cache, ok = {}, True
            for t in TARGETS:
                r = res[t][k]
                seg = aggregate_features(r["R0"], token_map, mode="mean")
                comb = combine_two_properties(seg)
                feat_abs = {f: comb[f]["abs_sum"] for f in FEATURES}
                total = sum(feat_abs.values())
                if total <= 0:
                    ok = False
                    break
                cache[t] = dict(r=r, comb=comb,
                                norm=simplex_normalize(feat_abs),
                                signed={f: comb[f]["signed_sum"] / total for f in FEATURES},
                                total=total)
            if not ok:
                continue

            counts = feature_token_counts(token_map)
            for t in TARGETS:
                c = cache[t]
                persample[t].append(c["norm"])
                persample_signed[t].append(c["signed"])
                cons_res[t].append(c["r"]["conservation_rel"])
                frac_on_feats[t].append(attribution_fraction(c["r"]["R0"], token_map))
                for f in FEATURES:
                    tokcount_rel[t][f].append(c["norm"][f])
                    tokcount_cnt[t][f].append(counts[f])
                    persample_long.append(dict(sample_id=valid, target=t, feature=f,
                                               norm_relevance=c["norm"][f],
                                               signed_norm=c["signed"][f], token_count=counts[f]))
                lr = c["r"]["layer_rel"]
                if capture_layers and lr is not None:
                    _accum_layers(layer_acc, layer_total, layer_count, t, lr, token_map)

            if deletion_subset and valid < deletion_subset:
                _deletion_one(model, tok, ut, cache["single_logit"]["r"]["R0"],
                              build_token_map=token_map, rng=rng, out=deletion_samples,
                              prepare_inputs=prepare_inputs, factory=target_logit_fn_factory)

            valid += 1
            if valid % 25 == 0:
                logger.info("%d/%d valid (median conservation %.3f)",
                            valid, n_target, float(np.median(cons_res["single_logit"])))

    banner(f"Phase 1 collected {valid} valid samples - writing results")
    _write_outputs(persample, persample_signed, cons_res, frac_on_feats,
                   tokcount_rel, tokcount_cnt, layer_acc, layer_total, layer_count,
                   persample_long, deletion_samples, valid, method)
    if randomization_subset:
        _randomization_check(model, tok, df, randomization_subset)


def _accum_layers(layer_acc, layer_total, layer_count, t, lr, token_map):
    L = lr.shape[0]
    if layer_acc[t] is None:
        layer_acc[t] = np.zeros((L, len(FEATURES)))
        layer_total[t] = np.zeros(L)
    fidx = {f: token_map[(1, f)] + token_map[(2, f)] for f in FEATURES}
    all_tok = sorted({i for f in FEATURES for i in fidx[f]})
    per = np.zeros((L, len(FEATURES)))
    for li in range(L):
        vals = np.array([np.abs(lr[li, fidx[f]]).mean() if fidx[f] else 0.0 for f in FEATURES])
        s = vals.sum()
        per[li] = vals / s if s > 0 else vals
        layer_total[t][li] += float(np.abs(lr[li, all_tok]).sum()) if all_tok else 0.0
    layer_acc[t] += per
    layer_count[t] += 1


def _deletion_one(model, tok, ut, R0, build_token_map, rng, out, prepare_inputs, factory):
    feat_tokens = sorted({i for f in FEATURES for s in (1, 2) for i in build_token_map[(s, f)]})
    if not feat_tokens:
        return
    prep = prepare_inputs(model, tok, ut)
    fn = factory(model, prep, "single_logit")
    order_attn = sorted(feat_tokens, key=lambda i: -abs(R0[i]))
    order_rand = list(feat_tokens); rng.shuffle(order_rand)
    ca = sn.deletion_curve(order_attn, fn)
    cr = sn.deletion_curve(order_rand, fn)
    grid = np.linspace(0, 1, 11)
    out.append((np.interp(grid, np.linspace(0, 1, len(ca)), ca),
                np.interp(grid, np.linspace(0, 1, len(cr)), cr)))


def _matrix(persample, target):
    return np.array([[d[f] for f in FEATURES] for d in persample[target]], dtype=float)


def _write_outputs(persample, persample_signed, cons_res, frac_on_feats,
                   tokcount_rel, tokcount_cnt, layer_acc, layer_total, layer_count,
                   persample_long, deletion_samples, n_valid, method):
    fname = {"single_logit": F_ATTN_SINGLE, "logit_diff": F_ATTN_LOGDIFF}
    means, sig_rows = {}, []
    for t in TARGETS:
        X = _matrix(persample, t)
        mean, lo, hi = st.bootstrap_ci(X)
        std = X.std(axis=0, ddof=1)
        t1, t3 = st.topk_frequency(X, 1), st.topk_frequency(X, 3)
        rank = st.ranks_from_means(mean)
        _write_rows(os.path.join(RESULTS_DIR, fname[t]),
                    ["feature", "mean", "std", "ci_lo", "ci_hi", "top1_freq", "top3_freq", "rank"],
                    [dict(feature=f, mean=mean[i], std=std[i], ci_lo=lo[i], ci_hi=hi[i],
                          top1_freq=t1[i], top3_freq=t3[i], rank=rank[f]) for i, f in enumerate(FEATURES)])
        means[t] = mean
        fstat, fp = st.friedman(X)
        sig_rows.append(dict(test="friedman", detail=f"{t} omnibus", stat=fstat, p=fp, p_fdr=""))
        for r in st.wilcoxon_posthoc(X, [("lot", "year_built"), ("bedrooms", "bathrooms")]):
            sig_rows.append(dict(test="wilcoxon", detail=f"{t}: {r['pair']}",
                                 stat=r["stat"], p=r["p"], p_fdr=r["p_fdr"]))
    _write_rows(os.path.join(RESULTS_DIR, "significance.csv"),
                ["test", "detail", "stat", "p", "p_fdr"], sig_rows)

    srows = []
    for t in TARGETS:
        S = np.array([[d[f] for f in FEATURES] for d in persample_signed[t]], dtype=float)
        for i, f in enumerate(FEATURES):
            col = S[:, i]
            srows.append(dict(feature=f, target=t,
                              mean_pos=float(np.clip(col, 0, None).mean()),
                              mean_neg=float(np.clip(col, None, 0).mean()),
                              pct_positive=float((col > 0).mean()),
                              pct_negative=float((col < 0).mean()), net=float(col.mean())))
    _write_rows(os.path.join(RESULTS_DIR, F_SIGNED),
                ["feature", "target", "mean_pos", "mean_neg", "pct_positive", "pct_negative", "net"], srows)

    lrows = []
    for t in TARGETS:
        if layer_acc[t] is None or layer_count[t] == 0:
            continue
        M = layer_acc[t] / layer_count[t]
        tot = layer_total[t] / layer_count[t]
        for li in range(M.shape[0]):
            for i, f in enumerate(FEATURES):
                lrows.append(dict(target=t, feature=f, layer=li, mean_norm=float(M[li, i])))
            lrows.append(dict(target=t, feature="__total__", layer=li, mean_norm=float(tot[li])))
    if lrows:
        _write_rows(os.path.join(RESULTS_DIR, F_LAYER), ["target", "feature", "layer", "mean_norm"], lrows)

    cross = st.cross_method_table(means["single_logit"])
    cross.append(st.cross_target_correlation(means["single_logit"], means["logit_diff"]))
    _write_rows(os.path.join(RESULTS_DIR, F_CROSS),
                ["comparison", "spearman_rho", "p", "cross_model"], cross)

    _write_rows(os.path.join(RESULTS_DIR, F_PERSAMPLE),
                ["sample_id", "target", "feature", "norm_relevance", "signed_norm", "token_count"],
                persample_long)

    san = [dict(key="method", value=method), dict(key="n_valid", value=n_valid)]
    for t in TARGETS:
        cs = sn.conservation_summary(cons_res[t])
        san += [dict(key=f"conservation_median_rel[{t}]", value=cs["median_rel_residual"]),
                dict(key=f"conservation_pass[{t}]", value=cs["pass"]),
                dict(key=f"attribution_fraction_mean[{t}]", value=float(np.mean(frac_on_feats[t])))]
        tn = sn.token_count_neutrality(tokcount_rel[t], tokcount_cnt[t])
        san += [dict(key=f"tokencount_neutrality_rho[{t}]", value=tn["spearman_rho"]),
                dict(key=f"tokencount_neutrality_p[{t}]", value=tn["p"])]
    _write_rows(os.path.join(RESULTS_DIR, F_SANITY), ["key", "value"], san)

    if deletion_samples:
        grid = np.linspace(0, 1, 11)
        attn = np.mean([a for a, _ in deletion_samples], axis=0)
        rand = np.mean([r for _, r in deletion_samples], axis=0)
        _write_rows(os.path.join(RESULTS_DIR, "deletion_curve.csv"),
                    ["frac_removed", "attnlrp", "random"],
                    [dict(frac_removed=float(g), attnlrp=float(a), random=float(r))
                     for g, a, r in zip(grid, attn, rand)])


def _randomization_check(model, tok, df, k_samples):
    from attribution import prepare_inputs, attribute_target
    logger.info("Adebayo model-randomization sanity check")

    def vec(row):
        ut, sp, _ = build_attribution_prompt(row)
        prep = prepare_inputs(model, tok, ut)
        tm = map_spans_to_tokens(prep["offsets"], prep["user_off"], sp)
        if any(len(tm[(s, f)]) == 0 for s in (1, 2) for f in FEATURES):
            return None
        res = attribute_target(model, prep, "single_logit", capture_layers=False)
        comb = combine_two_properties(aggregate_features(res["R0"], tm, mode="mean"))
        return np.array(list(simplex_normalize({f: comb[f]["abs_sum"] for f in FEATURES}).values()))

    rows = [r for _, r in df.head(k_samples).iterrows()]
    base = [v for v in (vec(r) for r in rows) if v is not None]
    restore = sn.randomize_top_layers(model, k=4, seed=0)
    try:
        after = [v for v in (vec(r) for r in rows) if v is not None]
    finally:
        restore()
    if base and after:
        m = min(len(base), len(after))
        delta = float(np.mean(np.abs(np.array(base[:m]) - np.array(after[:m]))))
        with open(os.path.join(RESULTS_DIR, F_SANITY), "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["model_randomization_mean_abs_delta", delta])
        logger.info("randomization mean abs delta-relevance = %.3f (should be clearly > 0)", delta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_VALID_TARGET)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--no-layers", action="store_true")
    ap.add_argument("--deletion-subset", type=int, default=25)
    ap.add_argument("--randomization-subset", type=int, default=8)
    a = ap.parse_args()
    run(n_target=a.n, capture_layers=not a.no_layers, deletion_subset=a.deletion_subset,
        randomization_subset=a.randomization_subset, batch_size=a.batch_size)
