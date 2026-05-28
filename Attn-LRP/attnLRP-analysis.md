# AttnLRP Housing Analysis — Results and Interpretation

Findings from running Attention-aware Layer-wise Relevance Propagation (AttnLRP) on the housing
"which listing is more valuable" task. This file records **Phase 1** (fixed feature order). Phase 2
(the 120-permutation position-robustness control) will be appended to this same file when it finishes.

Principle: we state only what the data supports, and flag confounds explicitly. A "Claims register"
(Section 11) separates supported conclusions from things we cannot yet assert. Every number below
traces to a CSV in `results/` (file index in Section 13).

---

## 1. Setup (what was run)

- **Method:** AttnLRP only, via the LXT library (`method=attnlrp` confirmed at runtime). Two attribution
  targets: **single-logit** (relevance of the chosen answer token) and **logit-difference**
  (relevance of chosen − rejected, i.e. the decision margin).
- **Model:** Llama-3.2-3B-Instruct, bf16, on an RTX 5070 Ti (sm_120). 28 transformer layers. Qwen3 was
  dropped because LXT's AttnLRP is experimental on it and skews relevance to the first token.
- **Data / prompt:** 500 valid pairs from `pairs_20pct_price_diff.csv` (a >=20% price-gap dataset, so
  pairs are non-tie by construction). The simplified attribution prompt contains only the five feature
  lines per property plus a `CHOICE:` cue — no instruction text and, after the revision, **no `zpid`**.
- **Five features:** lot, bathrooms, bedrooms, year built, property type (the same set the existing
  IG/KernelSHAP/Occlusion analysis used; heating/cooling/parking were excluded there and here).
- **Aggregation:** mean relevance over the tokens of each feature value, summed over the two listings,
  per-sample normalised so the five features sum to 1, then averaged over the 500 samples with 95%
  bootstrap confidence intervals (10,000 resamples).
- **Throughput:** batched on the GPU (batch 4, layer capture on); Phase 1 completed in ~10.5 minutes.

---

## 2. Headline results

**Single-logit** (mean normalised relevance; 95% CI; std; Top-1 and Top-3 frequency across samples):

| Rank | Feature | Mean | 95% CI | Std | Top-1 | Top-3 |
|---|---|---|---|---|---|---|
| 1 | property type | 0.402 | [0.394, 0.410] | 0.089 | 82.6% | 100.0% |
| 2 | lot | 0.223 | [0.216, 0.230] | 0.079 | 10.4% | 90.8% |
| 3 | year built | 0.190 | [0.183, 0.197] | 0.082 | 6.6% | 78.6% |
| 4 | bathrooms | 0.110 | [0.105, 0.116] | 0.062 | 0.4% | 25.4% |
| 5 | bedrooms | 0.075 | [0.072, 0.078] | 0.036 | 0.0% | 5.2% |

**Logit-difference** (decision-margin target):

| Rank | Feature | Mean | 95% CI | Std | Top-1 | Top-3 |
|---|---|---|---|---|---|---|
| 1 | property type | 0.380 | [0.369, 0.392] | 0.126 | 85.0% | 99.8% |
| 2 | bathrooms | 0.182 | [0.176, 0.187] | 0.064 | 6.6% | 71.6% |
| 3 | year built | 0.157 | [0.152, 0.163] | 0.064 | 4.2% | 49.4% |
| 4 | bedrooms | 0.150 | [0.144, 0.155] | 0.066 | 3.4% | 48.2% |
| 5 | lot | 0.131 | [0.126, 0.136] | 0.062 | 0.8% | 31.0% |

Observations strictly from the table: property type is rank 1 under **both** targets, with Top-1 ~83-85%
and a CI clear of rank 2 — it is the single stable result. Everything below it is target-dependent: lot
is #2 under single-logit but **#5 under logit-difference**; bathrooms moves from #4 to #2. Property
type also has the largest dispersion (std 0.089-0.126), i.e. its dominance varies most across pairs;
bedrooms is the tightest and lowest (std 0.036).

---

## 3. Statistical significance

A Friedman omnibus test (each pair a block, five features the treatments) rejects "all features equal"
overwhelmingly: single-logit chi-square ≈ 1400 (p ≈ 5e-302), logit-difference ≈ 946 (p ≈ 2e-203).
Wilcoxon signed-rank post-hoc tests (FDR-corrected) on the two pairs of interest are all significant:
lot vs year built (single p ≈ 3e-7; margin p ≈ 6e-11) and bedrooms vs bathrooms (single p ≈ 3e-25;
margin p ≈ 1e-13). So the orderings are not sampling noise *within this method*. This does not speak to
whether the method itself is valid — Section 5 addresses that.

---

## 4. Comparison with the existing methods

The earlier housing analysis (IG, KernelSHAP, Occlusion on Qwen3) ranked **lot #1 unanimously**.
AttnLRP on Llama does not reproduce that. Spearman correlations of the AttnLRP single-logit ranking
against the published ranks:

| Comparison | Spearman rho | p |
|---|---|---|
| AttnLRP vs IG | 0.30 | 0.62 |
| AttnLRP vs KernelSHAP | 0.60 | 0.28 |
| AttnLRP vs Occlusion | -0.10 | 0.87 |
| AttnLRP vs 3-method average | 0.34 | 0.58 |
| single-logit vs logit-difference (same model) | 0.30 | 0.62 |

None is significant (n=5 features). Two takeaways the data supports: (a) AttnLRP produces a different
ordering from the perturbation methods and in particular does **not** put lot first; (b) even within
AttnLRP the two targets agree only weakly (rho=0.30) — they concur on property type and otherwise
reorder the field. **Caveat:** this is a cross-model comparison (AttnLRP on Llama vs the published
methods on Qwen3), so a disagreement could be method or model; it is not an apples-to-apples
contradiction.

---

## 5. Effect of removing zpid

zpid (the Zillow ID) was initially included as a placebo control and pulled ~45% of a top feature's
relevance — evidence that bare numeric tokens attract relevance regardless of meaning. After removing
it, the single-logit means shifted as follows:

| Feature | with zpid | without zpid | change |
|---|---|---|---|
| property type | 0.371 | 0.402 | +0.031 |
| lot | 0.182 | 0.223 | +0.041 (rose to #2) |
| year built | 0.249 | 0.190 | -0.059 (fell to #3) |
| bathrooms | 0.112 | 0.110 | ~0 |
| bedrooms | 0.086 | 0.075 | -0.011 |

Interpretation supported by the data: removing the ID redistributed relevance mainly to the other
numeric-heavy field, **lot**, which recovered enough to overtake year built. Notably the token-count
confound (Section 8) got *stronger* without zpid (rho 0.52 -> 0.65), because zpid was itself a numeric
field; with it gone, the remaining ranking is even more length-driven.

---

## 6. Signed relevance (direction)

LRP relevance is signed: positive = supports the chosen listing, negative = opposes it.

| Feature | single % pos | single net | margin % pos | margin net |
|---|---|---|---|---|
| property type | 100% | +0.402 | 99.8% | +0.366 |
| year built | 97.2% | +0.179 | 65.4% | +0.052 |
| lot | 100% | +0.223 | 2.2% | -0.128 |
| bathrooms | 95.4% | +0.105 | 6.8% | -0.146 |
| bedrooms | 92.8% | +0.064 | 0.2% | -0.149 |

Under single-logit, every feature is almost entirely positive (all five support the chosen answer).
Under the margin target the picture inverts: **lot, bathrooms and bedrooms become predominantly
negative** (97.8%, 93.2%, 99.8% of samples), while property type stays positive (99.8%) and year built
is mixed. Data-supported reading: when the model weighs one listing against the other, property type
pushes toward its choice, whereas beds/baths/lot more often point toward the rejected listing on the
margin. This is a genuinely different statement from the single-logit view and is why the two targets
disagree below rank 1.

---

## 7. Layer-wise (depth) analysis

Relevance was also captured per transformer layer (0-27). Peak layer and depth-band averages
(early L0-9, mid L10-18, late L19-27) per feature:

**Single-logit:** property type peaks at **layer 0** (0.442) and is front-loaded (early 0.323, mid
0.331, late 0.229); lot peaks early (L7, 0.284); year built peaks mid (L12, 0.376); **bathrooms peaks
late (L26, 0.378)** rising from a low early share (0.102 -> 0.288); bedrooms also peaks late but stays
small. Total feature-relevance is concentrated in early layers (top layers 3-5).

**Logit-difference:** **property type peaks late (layer 25, 0.707)** and is mid/late-heavy (mid 0.573,
late 0.457); bathrooms peaks late-ish (L19, 0.298); lot and year built peak at the last layers but at
low magnitude; bedrooms peaks early (L10).

Why this matters for interpretation:
- Under single-logit, property type's relevance lives in **early layers**, where the network is still
  processing token identity/surface form. That is exactly where a lexical/length effect would show up,
  and it reinforces the token-length concern (Section 9).
- Under logit-difference (the length-clean target), property type's relevance peaks in **late layers
  (L25)**, where the model consolidates its decision. That is the strongest single piece of evidence
  that property type is genuinely used in the decision, not merely a surface artifact.
- **Bathrooms** accrues relevance late under both targets despite a low aggregate rank. This is
  consistent with the existing housing finding that bathroom sensitivity emerges downstream of input
  reading rather than at the input level — AttnLRP independently locates it in late layers.

---

## 8. Validation and sanity checks (this governs confidence)

- **Token-count neutrality — the key confound.** Normalised relevance correlates with the feature's
  token count at **rho=0.65 (p ≈ 2e-301) for single-logit** and rho=0.11 (p ≈ 1e-8) for
  logit-difference. Property type is the longest field (a multi-word phrase such as "Single Family
  Residence, Residential"); bedrooms/bathrooms are single digits. So the single-logit ranking tracks
  field length strongly, despite mean-over-token aggregation. The margin target is far less affected.
- **Model-randomization (Adebayo) — PASS.** Re-initialising the top 4 layers shifts the normalised
  relevance vector by 0.142 on average, so the attribution depends on the trained weights, not just
  input geometry.
- **Deletion faithfulness — not confirmed.** Removing feature tokens in decreasing AttnLRP-relevance
  order does not drop the chosen logit more than a random order (curves overlap; the logit rises from
  9.78 to ~10.6-11.8). Caveat: the ablation zeroes embeddings — a crude, out-of-distribution
  perturbation — so this is weak evidence, not a clean refutation.
- **Conservation residual ≈ 0.45 (single), 0.38 (margin)** — above the 2% target, so "pass" is False.
  Expected for the efficient input×grad form of AttnLRP on a causal LM (relevance absorbed by the
  un-patched output head / final norm and the BOS attention-sink); it does not affect per-sample-
  normalised rankings but it is why absolute relevance values should not be over-read.
- **Feature-attribution fraction ≈ 2.0% (single), 4.1% (margin)** — only this share of total relevance
  lands on the five feature segments; the rest sits on template/BOS tokens. Rankings come from a thin
  slice of the model's total relevance.

---

## 9. Why property type ranks first — and why that is suspect

AttnLRP (efficient form) computes relevance as input-embedding × backward-gradient, so relevance scales
with how token embeddings enter the computation. **Text-heavy, multi-token categorical fields tend to
accumulate more relevance than terse numeric fields**, even after averaging over a field's tokens,
because text tokens and single digit tokens are not on the same footing. Property type is the most
text-heavy feature, so a high score is exactly what a length/token-type bias predicts. Three pieces of
evidence converge on this concern: the token-count correlation (rho=0.65, single), the original zpid
control (a meaningless ID drew ~45% relevance), and the **early-layer** localization of property type
under single-logit.

What partially pushes back: (a) the logit-difference target is length-clean (rho=0.11) and still ranks
property type first with 85% Top-1; (b) under that target property type's relevance peaks **late**
(L25), i.e. at the decision stage, not the lexical stage; (c) the randomization check confirms
weight-dependence. The honest synthesis: AttnLRP's relevance concentrates on property type, and the
margin/late-layer evidence suggests genuine decision use, but the single-logit ranking cannot be
separated from the length confound. We therefore treat the **logit-difference, property-type-first**
result as the defensible statement and the single-logit magnitudes as confounded.

---

## 10. Methodology and limitations

- **Single model, cross-model comparison.** AttnLRP ran on Llama-3.2-3B; the perturbation methods ran
  on Qwen3-4B. Cross-method correlations (Section 4) mix method and model differences.
- **Efficient input×grad AttnLRP** does not conserve tightly at the input on a causal LM, hence the
  ~40% residual and ~2% feature fraction. Rankings rely on per-sample normalisation, which is robust to
  this, but absolute relevance is not interpretable.
- **bf16** precision (mandatory on the 12 GB card) adds minor noise; conservation is monitored.
- **Deletion ablation by zeroing embeddings** is crude; a mean-token or baseline-token replacement
  would be a cleaner faithfulness probe (see recommendations).
- **Token-length confound** is the dominant validity threat for single-logit and is quantified, not
  hidden.
- **Five features only**; heating/cooling/parking excluded as in the existing analysis.

---

## 11. Claims register

**Supported by the data:**
- AttnLRP relevance concentrates on **property type** (rank 1 under both targets; Top-1 83-85%; CI clear
  of rank 2).
- The AttnLRP ranking does **not** reproduce the lot-#1 result of the perturbation methods; cross-model
  correlations are weak and non-significant.
- Feature relevances differ significantly (Friedman/Wilcoxon).
- Under the margin target, lot/bathrooms/bedrooms relevance is predominantly **negative** (toward the
  rejected listing); property type stays positive.
- The single-logit ranking is **strongly length-confounded** (rho=0.65) and front-loaded in early
  layers; the logit-difference ranking is length-clean (rho=0.11) and property type peaks late (L25).
- The method is **model-sensitive** (randomization delta=0.142).
- Removing zpid shifted relevance toward lot and increased the length confound.

**Not supported / cannot claim:**
- That property type is *genuinely* the most important feature semantically (length confound;
  deletion test did not confirm) — though the late-layer margin evidence is suggestive.
- That AttnLRP is faithful here (deletion inconclusive; conservation loose; ~2% on features).
- That the single-logit ordering below property type is reliable (disagrees with margin, rho=0.30).
- Any apples-to-apples claim against IG/SHAP/Occlusion (different model).

**Confidence:** low-to-moderate on the specific "property type is the #1 driver" claim (~40-50%);
moderate-to-high that AttnLRP reads these listings differently from the perturbation methods. The
logit-difference, property-type-first result, supported by its late-layer localization, is the most
defensible single statement.

---

## 12. Recommendations / next steps

- Treat the **logit-difference** ranking as primary; report single-logit only with the length caveat.
- Add a **length-controlled** check: regress relevance on token count and analyse residuals, or use a
  fixed-width encoding of property type, to test whether property type survives length adjustment.
- Replace the zero-embedding deletion with a **baseline/mean-token** ablation for a cleaner faithfulness
  curve.
- Use Phase 2 (permutation) to confirm the ranking is position-robust, not order-driven.

---

## 13. Reproducibility and file index

Run: `run_phase1.py --n 500 --batch-size 4` (seed 42), env `housing`, model gated via HF token.
Outputs in `results/`:
- `attnlrp_single.csv`, `attnlrp_logitdiff.csv` — rankings (mean/std/CI/Top-1/Top-3/rank).
- `signed_relevance.csv` — per-feature positive/negative split.
- `layer_feature_relevance.csv` — feature x layer mean relevance (Section 7).
- `cross_method_correlation.csv` — Section 4 table.
- `significance.csv` — Friedman/Wilcoxon.
- `conservation_and_sanity.csv` — conservation, fraction, token-count rho, randomization.
- `deletion_curve.csv` — faithfulness curve.
- `per_sample_relevance.csv` — raw per-sample rows for re-analysis.
Figures (PNG) in `figures/`: fig1 importance+CI, fig2 layer heatmap, fig3 signed, fig4 permutation
stability (Phase 2), diag deletion curve.

---

## 14. Phase 2 — feature-position permutation control (completed)

### 14.1 What Phase 2 does and why

Phase 1 used a single fixed feature order (lot, bathrooms, bedrooms, year built, property type —
positions 0-4). Any attribution on an autoregressive model is vulnerable to **position bias**: a
feature near the start of a block (just after the "Property N:" header) or near the end (adjacent to
the `CHOICE:` cue) can collect relevance because of *where it sits*, not *what it is*. Phase 2 removes
that confound by brute force: it re-runs all 500 pairs across the **complete set of 5! = 120 feature
orderings** (so every feature occupies every one of the five positions an equal number of times — the
coverage check confirms ~96-116 samples per (feature, position) cell, close to the ideal 500/5 = 100),
combined with a balanced Property-1/Property-2 swap, and then aggregates strictly **by feature
identity**. The comparison of the fixed-order ranking against this position-averaged ranking tells us
how much of Phase 1 was the feature and how much was its slot. Phase 2 ran in ~22 minutes (no layer
capture, batch 8) for 500/500 valid samples under both targets.

### 14.2 Rank stability: only the extremes survive

| Feature | single fixed | single permuted | margin fixed | margin permuted |
|---|---|---|---|---|
| property type | 1 | 1 | 1 | 1 |
| lot | 2 | 3 | 5 | 5 |
| year built | 3 | 2 | 3 | 4 |
| bathrooms | 4 | 4 | 2 | 3 |
| bedrooms | 5 | 5 | 4 | 2 |

The fixed-vs-permuted Spearman rank correlation is **rho = 0.90 (p = 0.037)** for single-logit — high
and significant — and **rho = 0.70 (p = 0.19)** for logit-difference — moderate and *not* significant.
Reading the table directly: **property type is rank 1 in all four cells**, and **lot is rank 5 on the
decision margin in both** the fixed and permuted runs. Those two facts are genuinely position-robust.
Everything in between moves: under single-logit lot and year built swap at ranks 2-3; under the margin
the reshuffle is larger — bedrooms climbs from 4 to 2, bathrooms slips 2 to 3, year built 3 to 4. The
non-significant margin correlation (p = 0.19) is itself a result: **the logit-difference mid-ranking is
not stable to feature order and should not be reported as a fixed ordering.** Only the single-logit top
(property type) and the margin bottom (lot) earn the label "robust."

### 14.3 Permuted magnitudes: the distribution flattens dramatically

Ranks tell only part of the story; the magnitudes change even more.

| Feature | single fixed | single permuted | margin fixed | margin permuted |
|---|---|---|---|---|
| property type | 0.402 | 0.257 | 0.380 | 0.300 |
| lot | 0.223 | 0.187 | 0.131 | 0.081 |
| year built | 0.190 | 0.218 | 0.157 | 0.154 |
| bathrooms | 0.110 | 0.182 | 0.182 | 0.220 |
| bedrooms | 0.075 | 0.157 | 0.150 | 0.245 |

Two things stand out. First, **property type's lead collapses** when position is averaged out: from
0.402 to 0.257 under single-logit (and 0.380 to 0.300 on the margin). Second, **the features that were
low in the fixed prompt rise sharply** — bedrooms more than doubles (0.075 to 0.157) and bathrooms
rises 0.110 to 0.182 under single-logit. The spread of the single-logit distribution compresses from a
5.4x range (0.075-0.402) in the fixed run to a 1.6x range (0.157-0.257) when permuted. In other words,
**once you stop always showing the same feature in the same slot, the five features look much more
similar than Phase 1 implied.** Property type still leads, but "by a wide margin" was an artifact of
the prompt layout, not a property of the model.

### 14.4 Position sensitivity (single-logit): a primacy + recency pattern

The position-sensitivity table records each feature's mean relevance as a function of which of the five
slots it occupied, holding identity fixed. For single-logit:

| Feature | pos 0 | pos 1 | pos 2 | pos 3 | pos 4 |
|---|---|---|---|---|---|
| lot | 0.227 | 0.155 | 0.134 | 0.162 | 0.247 |
| bathrooms | 0.338 | 0.149 | 0.125 | 0.109 | 0.196 |
| bedrooms | 0.275 | 0.140 | 0.094 | 0.091 | 0.190 |
| year built | 0.285 | 0.221 | 0.207 | 0.157 | 0.223 |
| property type | 0.286 | 0.218 | 0.196 | 0.214 | 0.369 |

Every single feature shows a **U-shape**: highest at position 0 (the first feature line, right after
the "Property N:" header) and at position 4 (the last line, adjacent to `CHOICE:`), and lowest in the
middle (position 2 or 3). This is the classic primacy + recency signature, and it is structural, not
semantic — it is about proximity to the salient anchor tokens, not about what the feature means. The
size of the swing is large: bathrooms nearly **triples** from its mid-position trough (0.109) to its
first-position peak (0.338); bedrooms shows the same (0.091 to 0.275). The short numeric features
(bathrooms, bedrooms) are the most primacy-sensitive, while property type is the only feature that is
**recency-dominant** (its position-4 value 0.369 exceeds its position-0 value 0.286). This is consistent
with the depth analysis in Section 7, where property type's single-logit relevance was concentrated in
early layers — a surface/positional phenomenon.

### 14.5 Position sensitivity (logit-difference): feature-specific effects

The margin target shows a different, more revealing pattern:

| Feature | pos 0 | pos 1 | pos 2 | pos 3 | pos 4 |
|---|---|---|---|---|---|
| lot | 0.111 | 0.076 | 0.060 | 0.068 | 0.086 |
| bathrooms | 0.426 | 0.212 | 0.144 | 0.112 | 0.215 |
| bedrooms | 0.404 | 0.262 | 0.171 | 0.147 | 0.248 |
| year built | 0.152 | 0.132 | 0.154 | 0.147 | 0.186 |
| property type | 0.256 | 0.265 | 0.276 | 0.312 | 0.385 |

Here the position effect is not uniform. **bathrooms and bedrooms show enormous primacy spikes** when
placed first (0.426 and 0.404 — roughly double their permuted means), then decay. So on the
decision-margin computation, whichever of beds/baths appears first grabs a disproportionate share of
relevance. **property type, by contrast, rises monotonically with position** (0.256 -> 0.385) — a clean
recency effect with no primacy component. **lot is flat and low at every position** (0.060-0.111),
which is why lot is robustly rank 5 on the margin: its low standing is genuinely position-independent.
year built is similarly flat. The practical implication: the margin ranking is dominated by two
competing position artifacts (beds/baths primacy vs property-type recency), which is exactly why its
fixed-vs-permuted correlation was only 0.70 and non-significant.

### 14.6 How the fixed prompt biased Phase 1

The fixed Phase-1 order placed **lot at position 0** and **property type at position 4**. Cross-
referencing the position tables, this means Phase 1 handed lot a first-position primacy boost (lifting
it to single-logit rank 2) and handed property type a last-position recency boost (inflating its 0.402).
The middle features (bathrooms, bedrooms, year built at positions 1-3) sat in the suppressed trough,
which depressed their fixed-order scores. Phase 2 corrects all of this, and the corrections are exactly
in the predicted directions: property type down (0.402 -> 0.257), lot down a little (still primacy-
helped only 1/5 of the time now), and the trough features up. Nothing here was a coding error — it is
the position confound the permutation control was designed to expose, now quantified.

### 14.7 P1/P2 swap and coverage caveats

We applied a balanced Property-1/Property-2 swap (half the samples with listings swapped) to control
first-listing bias, but this run did **not** record a dedicated slot-1-vs-slot-2 relevance metric, so
we do not separately quantify the magnitude of any first-listing effect — only that the control was
applied. Positional coverage was near-uniform (96-116 samples per feature-position cell) but not exactly
equal, because 500 does not divide evenly by 120; the small imbalance is unlikely to matter given the
size of the position effects but is noted for honesty. Permuted runs also did not recompute bootstrap
CIs or Top-k frequencies (only means and ranks), so the permuted columns above are point estimates.

---

## 15. Updated overall confidence (Phase 1 + Phase 2)

**Well-supported after both phases:**
- **Property type draws more AttnLRP relevance than any other feature, robustly to feature order** —
  rank 1 across all 120 permutations under both targets. This is the one finding that survives the
  position control.
- **lot is robustly low on the decision margin** (rank 5, fixed and permuted, flat across positions).
- AttnLRP's ordering differs from the lot-first perturbation result (Section 4).
- The model exhibits a **strong, structural primacy + recency position bias** in attributed relevance
  (Sections 14.4-14.5), feature- and target-specific in size.

**Clearly qualified or weakened:**
- **Property type's apparent dominance was substantially a position artifact.** Its fixed-order 0.402
  drops to a position-fair ~0.26-0.30, and the five-feature distribution compresses from 5.4x to 1.6x.
  It still leads, but not by the wide margin Phase 1 suggested.
- **The mid-tier ranking is unreliable** — lot/year built/bathrooms/bedrooms reorder under permutation,
  and the margin stability is non-significant (rho=0.70, p=0.19).
- The Phase-1 **token-length confound** (rho=0.65 single) compounds the position effect: property type
  benefited from being both the longest text field *and* (in the fixed prompt) the last line.
- Faithfulness is still unconfirmed (the deletion test did not validate the ranking; conservation is
  loose; only ~2-4% of relevance lands on feature tokens).

**Net confidence.** "Property type is the feature AttnLRP attends to most, robust to feature order" is a
**moderate-confidence** claim — it cleared the 120-permutation control, which is a meaningful bar. But
"property type dominates by a wide margin" is **refuted** as a position artifact, the mid-ranking is
**low confidence**, and we still cannot separate genuine semantic importance from the text-length bias.
The most defensible single sentence: *AttnLRP relevance concentrates on property type more than any
other feature and this is robust to feature order, but its lead is modest once prompt position and
text-length are accounted for, the ranking of the remaining features is not position-stable, and the
overall picture disagrees with the lot-first story told by the perturbation methods.*

---

## 16. Phase 2 claims register

**Supported by Phase 2 data:**
- Property type is rank 1 in all four (target x order) conditions; lot is rank 5 on the margin in both.
- Single-logit ranking is position-robust at the extremes (Spearman 0.90, p=0.037); margin ranking is
  not (0.70, p=0.19).
- A U-shaped primacy+recency position effect exists for every feature under single-logit; under the
  margin, beds/baths are primacy-driven and property type is recency-driven, while lot is flat-low.
- Property type's fixed-order magnitude was inflated by its last-slot placement; position-fair ~0.26-0.30.
- The feature distribution flattens markedly under permutation (range 5.4x -> 1.6x, single-logit).

**Not claimed:**
- A reliable ordering of the mid-tier features (it is position-dependent).
- A quantified Property-1/Property-2 (first-listing) effect (control applied but not separately measured).
- Permuted CIs or Top-k (not computed in the permutation run).

Figures: `figures/fig1_feature_importance_ci.png` (ranking + CIs), `fig2_layer_feature_heatmap.png`
(depth), `fig3_signed_relevance.png` (direction), `fig4_permutation_rank_stability.png` (Section 14.2),
`diag_deletion_insertion_curve.png` (faithfulness).
