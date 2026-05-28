# Attention LRP for the Housing Valuation Task — Validated Technical Plan

> **Status:** Validated technical plan. **No production code** — pseudocode only, kept minimal and
> explicitly non-runnable. Every mathematical claim has been re-derived by hand (see §14, the
> validation matrix) and every statistical claim has a stated estimator, test, and pass criterion.
> This file is the single source of truth for *why* we run Attention-aware Layer-wise Relevance
> Propagation (AttnLRP) on the housing track, *what* it computes, *how* the math works, *how* the
> mutual-fund track (**Alan**) did it, and *how* we adapt it to housing so the result is mathematically
> and statistically defensible — and directly comparable to the housing track's existing attribution
> methods.
>
> **Objective — open discovery (read this first).** The goal is to find **which of the five features
> actually drive the model's pairwise choice**. No feature is assumed to be the answer: lot's apparent
> dominance under the existing methods is a *hypothesis to test*, not a target to confirm. AttnLRP may
> confirm it, demote it, or surface a different driver — every one of those is a valid, reportable
> result. Predictions in §16 are hypotheses framed to be falsified, not conclusions.
>
> **Locked scope decisions (read first):**
> 1. **AttnLRP only** — not CP-LRP, not any other LRP rule variant.
> 2. **Two attribution targets** of that one method: single-logit and logit-difference (two *questions*, not two methods).
> 3. **One model: Llama-3.2-3B-Instruct** (sole model). **Qwen3 is dropped completely** — LXT marks it
>    🧪 experimental with a confirmed *"attribution skewed toward first token"* defect, while Llama-3 is
>    fully supported (✅). Llama-3.2-3B is also Alan's funds-LRP model and a housing probing/amnesic
>    model, so it is a legitimate housing model. Trade-off (accepted): we lose strict same-model
>    comparability with the Qwen3-based IG/SHAP/Occlusion runs — the cross-method check becomes
>    *cross-model* (§12.4).
> 4. **Five features: lot, bathrooms, bedrooms, year built, property type** — the exact set the other
>    housing attribution methods used (heating, cooling, parking type were dropped there; we match).
> 5. **N = 500 valid pairs** (locked) — a single run; **no** extra 183-pair subset.
> 6. **bfloat16** for the attribution backward (locked) — conservation is verified at runtime (§14–15).
> 7. **Phase 1** = fixed feature order; **Phase 2** = feature-position permutation as a scaling/robustness step.
> 8. **Four headline figures**, fully specified for programmatic generation with **no hand-editing** (§13).
>
> **Design principle:** wherever a choice exists, we mirror the housing track's existing attribution
> *protocol* (five features, simplified prompt, single-logit target, 120-permutation control —
> `main.tex` §"Real Estate Property", `sec:housing-internal-single`). The **one deliberate deviation is
> the model**: AttnLRP runs on **Llama-3.2-3B**, not the Qwen3-4B those methods used, because AttnLRP is
> not faithful on Qwen3. So AttnLRP is protocol-matched but cross-model; §12.4 treats the comparison
> accordingly.

---

## 0. Table of Contents

1. Executive summary (TL;DR)
2. What LRP is, conceptually
3. The general LRP math (conservation, ε-rule, Deep Taylor)
4. The AttnLRP math (matrix-multiplication rule, softmax rule, Q/K/V split)
5. AttnLRP — the two attribution targets we run
6. How LRP is computed (LXT, the gradient×input backward pass) — pseudocode
7. From token relevance to a feature-importance ranking
8. How the mutual-fund track (Alan) did it — recap and what we keep
9. Mapping the method onto the housing task
10. Phase 1 — single model, robust baseline (pseudocode)
11. Phase 2 — feature-position permutation as a scaling step (pseudocode)
12. Statistical-robustness protocol
13. Plot specifications — four figures, no hand-editing
14. Mathematical & statistical validation matrix
15. Runtime sanity checks
16. Expected results
17. Limitations, known issues, and mitigations
18. Deliverables and folder layout
19. Open clarifying questions for the team
20. References

---

## 1. Executive summary (TL;DR)

- **What:** AttnLRP distributes the model's output ("relevance") backward through every transformer
  layer — including the non-linear attention and softmax — until each *input token* holds a signed
  scalar saying how much it contributed to the decision. We aggregate those token scores into one
  number per **housing feature**: **lot, bathrooms, bedrooms, year built, property type** (the same
  five the housing IG/SHAP/Occlusion analysis used).

- **Scope:** **AttnLRP only** (no CP-LRP), run against **two targets** — the chosen-token logit
  (single-logit, the target the existing housing methods used) and the decision margin (logit-difference).

- **Why:** The housing track has three *perturbation/gradient* attribution methods (Integrated
  Gradients, KernelSHAP, Occlusion) that rank **lot first unanimously** but disagree below it. AttnLRP
  is a *different family* (internal relevance flow, signed, layer-resolved) and is the one lens the
  housing report explicitly flags as **not yet run** (`main.tex:757`). Running it as a matched fourth
  method is **open discovery of what actually drives the model's choice** — a different lens that may
  agree with the existing methods, disagree, or surface something they missed. Whether lot's apparent
  dominance holds under a relevance-flow method is a hypothesis to test, not a conclusion to confirm.

- **How (the trick):** With AttnLRP's rules, input relevance equals
  `input_embedding ⊙ (custom_backward_gradient)` — a **single modified backward pass**. The LXT
  library (`rachtibat/LRP-eXplains-Transformers`) implements those rules for Llama/Qwen by
  monkey-patching attention, softmax, and normalization. (The housing perturbation methods used the
  *Interpreto* library; AttnLRP needs LXT because Interpreto does not implement LRP.)

- **Robustness:** per-sample simplex normalization → average over **500** pairs → bootstrap 95% CIs on
  each feature's mean relevance → non-parametric rank tests (Friedman omnibus + Wilcoxon post-hoc,
  FDR-corrected) → report the fraction of total relevance that lands on feature tokens.

- **Scaling step (Phase 2):** re-run under **all 5! = 120 feature-order permutations** (this is exactly
  the 120-permutation control the housing methods used — for five features 120 is the *complete* set)
  plus a P1/P2 swap, so the ranking reflects **feature identity, not position**.

- **One model:** **Llama-3.2-3B-Instruct** (sole model; **Qwen3 dropped** — LXT's AttnLRP is 🧪 on
  Qwen3 with a confirmed *first-token-skew* defect). Llama-3 is fully supported, is Alan's funds-LRP
  model, and is a housing probing model. The cost is that the comparison to the Qwen3-based
  IG/SHAP/Occlusion becomes *cross-model* (§12.4).

---

## 2. What LRP is, conceptually

LRP answers: *"Of the scalar the model produced at the output (here, the logit of the chosen answer
token), how much can be traced back to each input element?"*

Picture the forward pass as water flowing **down** a river network from many input springs (tokens) to
one outlet (the decision logit). LRP runs it **backward**: it starts with a fixed amount of relevance
at the outlet and redistributes it upstream, splitting at each junction in proportion to how much each
branch contributed downstream. The discipline is **conservation** — relevance is redistributed, never
created or destroyed — so the total reaching the inputs equals what we injected at the outlet.

Two properties make AttnLRP attractive and distinct from what the housing track has:

1. **Signed.** A token can support (positive) or oppose (negative) the chosen answer. Perturbation
   methods give mostly magnitude; AttnLRP gives direction for free, letting us ask "which of the five
   features create *conflict* between the two listings?"
2. **Layer-resolved.** Relevance passes through every layer, so we can read off *where* in depth a
   feature accrues importance — connecting directly to the housing linear-probing results.

LRP is **not** ground truth; it is one principled lens whose faithfulness must be validated (§15). It
measures "how logit flows back through tokens," a genuinely different quantity from "what happens when
I delete this feature" (perturbation). The two can disagree, and the disagreement is informative.

---

## 3. The general LRP math

### 3.1 Conservation property

Network computes $f(\mathbf{x})$. Inject a scalar target $R^{L}$ at the last layer $L$. LRP defines a
relevance vector $R^{l}$ per layer such that relevance is **conserved**:

$$
\sum_i R_i^{\,l-1} \;=\; \sum_j R_j^{\,l} \;=\; \cdots \;=\; R^{L}.
$$

Sum the relevance on any layer's neurons and you get the injected total. The output is $R^{0}$, one
value per input dimension; summed over the embedding dimension it gives one relevance per **token**.

### 3.2 The ε-rule for linear layers

For an affine layer $z_j = \sum_i x_i W_{ij} + b_j$:

$$
R_{i\leftarrow j} \;=\; \frac{x_i\,W_{ij}}{z_j + \varepsilon\,\mathrm{sign}(z_j)}\, R_j,
\qquad
R_i \;=\; \sum_j R_{i\leftarrow j}.
$$

- Numerator $x_i W_{ij}$ = token $i$'s additive share of pre-activation $z_j$.
- Denominator = total pre-activation; dividing makes the shares sum to ≈ $R_j$ (conservation).
- $\varepsilon \approx 10^{-6}$ absorbs relevance when $z_j\approx 0$ (no division by zero). ε slightly
  and intentionally breaks strict conservation, suppressing noise from near-dead neurons.

### 3.3 Deep Taylor Decomposition (DTD) — the unifying view

AttnLRP frames each rule as a first-order Taylor expansion around a reference $\tilde{\mathbf{x}}$:

$$
f_j(\mathbf{x}) = f_j(\tilde{\mathbf{x}}) + \sum_i \mathbf{J}_{ji}(\tilde{\mathbf{x}})\,(x_i - \tilde{x}_i) + \mathcal{O}\!\left(\lVert \mathbf{x}-\tilde{\mathbf{x}}\rVert^2\right),
$$

$\mathbf{J}$ = Jacobian. Relevance is the **input × linearized-gradient** contracted with the incoming
relevance. The implementation consequence: **LRP at the input equals `input ⊙ gradient` under a
*modified* backward pass** where each non-linear op's backward is replaced by its DTD rule. This is why
LXT computes AttnLRP in one backward call (§6).

---

## 4. The AttnLRP math

Standard ε-LRP handles the linear parts (projections, MLP, embeddings). The hard parts are the two
**bilinear matrix multiplications** in attention and the **softmax**. AttnLRP's contribution (Achtibat
et al., ICML 2024) is faithful, conservation-respecting, stable rules for exactly these — which is why
we chose it over older "skip the softmax" approaches.

### 4.1 The bilinear matrix-multiplication rule

Attention has two products of two *variable* operands: the scores $\mathbf{S}=\mathbf{Q}\mathbf{K}^\top$
and the context $\mathbf{O}=\mathbf{A}\mathbf{V}$. For a bilinear product $O_{jp}=\sum_i A_{ji}V_{ip}$,
AttnLRP sends **half** the output relevance to each operand:

$$
R(A_{ji}) = \sum_p \frac{A_{ji} V_{ip}}{2\,O_{jp} + \varepsilon}\, R^{l}_{jp},
\qquad
R(V_{ip}) = \sum_j \frac{A_{ji} V_{ip}}{2\,O_{jp} + \varepsilon}\, R^{l}_{jp}.
$$

The $\tfrac12$ is the **uniform rule** for the symmetric Shapley decomposition of a product of two
operands (each factor gets half). No bias term absorbs relevance, so conservation across the matmul is
exact up to ε. **Hand-checked (§14):** summing $R(A)$ over all entries gives $\approx\tfrac12\sum R^l$,
summing $R(V)$ gives the other $\approx\tfrac12\sum R^l$, total $\approx\sum R^l$. ✓

### 4.2 Where "Q/4, K/4, V/2" comes from

A *derived* consequence of applying the bilinear rule twice. Trace one unit of relevance entering the
attention output:

1. It hits $\mathbf{O}=\mathbf{A}\mathbf{V}$ → **½ to $\mathbf{A}$**, **½ to $\mathbf{V}$**.
2. The ½ on $\mathbf{A}$ passes back through softmax (§4.3, conserving) to $\mathbf{S}=\mathbf{Q}\mathbf{K}^\top$.
3. $\mathbf{S}$ is bilinear → its ½ splits **¼ to $\mathbf{Q}$**, **¼ to $\mathbf{K}$**.

Net per attention block: $\mathbf{V}=\tfrac12$, $\mathbf{Q}=\tfrac14$, $\mathbf{K}=\tfrac14$ — hence
**"Q/4, K/4, V/2."** These are *path-level relevance budgets* (exact in aggregate under the uniform
rule); the within-path distribution across tokens follows the ε-weighted shares above. The value path
keeps half (content); the query/key path keeps half (the *attention pattern* — what the model attends
to). Routing relevance through the Q/K pattern is the defining behavior of AttnLRP.

### 4.3 The softmax rule (Proposition 3.1)

Softmax $s_i=\mathrm{softmax}(\mathbf{x})_i$ is non-linear; naive propagation is unstable. AttnLRP's
linearized, conserving rule:

$$
R_i^{\,l} \;=\; x_i\left(R_i^{\,l+1} - s_i \sum_j R_j^{\,l+1}\right).
$$

**Hand-derived (§14):** with the softmax Jacobian $\partial s_k/\partial x_i = s_k(\delta_{ki}-s_i)$
and the input×grad framework (treat incoming relevance on output $k$ as $R_k=s_k g_k$), the gradient to
input $i$ is $\sum_k g_k s_k(\delta_{ki}-s_i)=R_i - s_i\sum_k R_k$; multiply by $x_i$. ✓ The
$-s_i\sum_j R_j$ term re-centers rows to conserve. AttnLRP keeps a bias-like contribution because
softmax returns $1/N$ at zero input. **Propagating relevance through softmax is exactly what
distinguishes AttnLRP** from value-path-only variants — the piece we keep.

### 4.4 Other components

- **Residual add:** relevance splits between the two summands by their forward contributions (ε-rule, two inputs).
- **RMSNorm / LayerNorm:** identity rule (scale treated as constant in backprop) → relevance passes
  through. (LXT: `rules.IdentityRule` on the norm modules.)
- **RoPE:** parameter-free rotation → relevance passes through.
- **GQA (grouped-query attention):** Llama-3.2 (our model) uses GQA (24 query heads, 8 KV heads) + RoPE
  + RMSNorm; rules apply per head and aggregate relevance across shared KV groups (LXT supports Llama-3
  fully ✅). Crucially, Llama-3 has **no QK-Norm** — the extra in-attention RMSNorm behind the Qwen3
  first-token-skew defect — so the op that breaks AttnLRP on Qwen3 is simply absent here.

---

## 5. AttnLRP — the two attribution targets we run

One method (AttnLRP), two **targets**. A target is the scalar injected as relevance at the output
before back-propagating — it changes the *question*, not the *machinery*. Both use the identical
AttnLRP rules from §4.

### 5.1 The two targets

- **Single-logit:** inject at the **chosen answer token's logit** ($R^L$ = one-hot at the chosen
  token). Measures absolute support for the chosen answer. **This is the target the existing housing
  IG/SHAP/Occlusion runs used**, so our single-logit results are the like-for-like comparison.
- **Logit-difference:** inject at the **decision margin**
  $\;\mathrm{logit}_{\text{chosen}} - \mathrm{logit}_{\text{rejected}}\;$. Measures what
  *discriminates* the two options — the most natural target for a binary "which is more valuable" choice.

### 5.2 The two runs

| # | Method  | Target           | What it isolates                                       |
|---|---------|------------------|-------------------------------------------------------|
| 1 | AttnLRP | single-logit     | Full attention+content support for the chosen answer  |
| 2 | AttnLRP | logit-difference | Attention+content drivers of the decision margin      |

Two backward passes per pair. We report each separately and their **single-vs-margin correlation**
(§12.4) — the only within-AttnLRP comparison now that CP-LRP is out of scope.

### 5.3 Why CP-LRP is excluded (recorded)

CP-LRP (Ali et al. 2022) treats the attention matrix as constant and routes relevance only through the
value path, ignoring softmax and the Q/K pattern. It answers the narrower "what content flows through
V" — the thing AttnLRP was built to improve on. Per the team's decision we run **AttnLRP only**; CP-LRP
is a one-line rule swap if ever wanted, but **not** in this plan.

---

## 6. How LRP is computed (LXT) — pseudocode

### 6.1 The library

**LXT (LRP-eXplains-Transformers)**, `rachtibat/LRP-eXplains-Transformers`, the official AttnLRP
implementation. The funds track used **LXT v2.1**. Supports Llama 2/3, Qwen 2/3, Gemma 3, BERT, GPT-2,
ViT. **Llama 2/3 is fully supported (✅); Qwen3 is 🧪 with a confirmed first-token-skew defect, so we do
not use it** (§9.5). Note LXT is a *different* library from Interpreto (used by the housing
IG/SHAP/Occlusion runs), because Interpreto does not implement LRP.

### 6.2 The efficient path (recommended) — pseudocode

LXT's "efficient" mode monkey-patches the forward so a **standard backward pass** computes AttnLRP.

```
# Pseudocode — illustrative, not runnable.
install AttnLRP backward rules on the model           # lxt.efficient.monkey_patch
embeds ← embedding_lookup(input_ids); mark embeds as requiring gradient
logits ← model(inputs_embeds = embeds)

if target = SINGLE_LOGIT:    scalar ← logits[decision_pos, chosen_id]
if target = LOGIT_DIFF:      scalar ← logits[decision_pos, chosen_id] − logits[decision_pos, other_id]

backpropagate(scalar)                                  # AttnLRP rules active during backward
relevance_per_token ← sum over hidden_dim of (embeds ⊙ embeds.gradient)   # this is R^0
```

The last line **is** the DTD "input × linearized-gradient" identity (§3.3). The **only** difference
between our two runs is which `scalar` we backpropagate.

### 6.3 The explicit path (more control) — pseudocode

```
# Pseudocode — illustrative, not runnable.
Composite = { Linear: EpsilonRule,  RMSNorm: IdentityRule,
              Attention: AttnLRP_rule (bilinear + softmax decomposition) }
register(Composite, model)        # NB: AttnLRP attention rule only; no CP-LRP rule registered
```

### 6.5 Environment and hardware (local)

- **GPU:** local **NVIDIA RTX 5070 Ti, 12 GB** (Blackwell, sm_120). Sufficient for Llama-3.2-3B AttnLRP
  at batch size 1 + gradient checkpointing + bf16 (§6.4).
  - **Blackwell caveat:** the RTX 50-series needs **CUDA 12.8+ and a matching PyTorch build (cu128)**
    with sm_120 kernels. Before running, confirm `torch.cuda.is_available()` *and* that the installed
    torch is a Blackwell-capable build — an older torch will fail with a "no kernel image" error on this
    GPU. This is a setup check, not a code deliverable.
- **Python env:** use the existing **`housing`** environment (already has `transformers` and the housing
  data pipeline). Add **`lxt`** (LRP-eXplains-Transformers, v2.1): `pip install lxt`. Verify the env's
  `transformers` version is compatible with LXT v2.1's efficient monkey-patch and with Llama-3.2 (pin if
  needed); record the resolved `torch` / `transformers` / `lxt` versions in `conservation_and_sanity.csv`
  for reproducibility.
- **Throughput / parallelism:** the runners **batch** `BATCH_SIZE` prompts per forward+backward on the
  GPU (left-padded so per-sample grads stay independent) and use `CPU_THREADS` (default min(24, cores))
  for tokenisation and CPU tensor ops, exploiting the 24-core / 64 GB box. Phase 1 is ~1000 backward
  passes (500 × 2 targets) ÷ batch; Phase 2 re-runs across the 120 permutations. Run Phase 2 as a
  background job and checkpoint partials.

### 6.4 Cost (validated against the paper's "single backward pass" claim)

- **Compute:** ~1 forward + 1 backward per (sample × target). 500 × 2 = **~1000 backward passes** —
  tens of minutes to ~1 h on one GPU for Llama-3.2-3B.
- **Memory:** $\mathcal{O}(\sqrt{N})$ in layers with gradient checkpointing. Target hardware is the
  local **RTX 5070 Ti (12 GB)**: Llama-3.2-3B in **bf16** is ~6.4 GB of weights; with batch size 1, the
  short simplified prompt (a few hundred tokens), and checkpointing, the backward pass fits in ~9–10 GB.
- **Precision:** **bfloat16** for the attribution backward (team decision; matches Alan's funds setup
  and is **mandatory on the 12 GB card** — float32 weights alone would be ~12.8 GB and would not fit).
  bf16 can leak relevance, so we *verify conservation at runtime* (§14, §15) and report the residual;
  the float32 "escape hatch" (§15.1) would require CPU offload or a larger GPU, so we rely on
  conservation staying within tolerance instead.

---

## 7. From token relevance to a feature-importance ranking

The backward pass gives **per-token** relevance; we need **per-feature** importance over the five
features. We mirror the housing track's aggregation conventions for comparability.

1. **Segment / token-span mapping.** Use **PART_SENTENCE granularity** (segment on the punctuation that
   delimits feature lines), exactly as the housing methods did, and record each feature value's token
   indices from the tokenizer's **offset mappings** — never by heuristic.

2. **Per-feature aggregation = MEAN over tokens within the segment.** The housing methods aggregate by
   **MEAN over the tokens in each feature segment to neutralize token-count bias** (a multi-token value
   like "year built: 1998" must not outscore "bathrooms: 3" merely for having more tokens). We match:
   $\;r_f = \mathrm{mean}_{t \in \mathrm{span}(f)} R^0_t\;$ over the five features (× 2 properties).
   *(We also retain the SUM over tokens as a separate quantity used only for the conservation check and
   the feature-attribution fraction in step 6 — sum, not mean, is the conserved quantity.)*

3. **Two-property aggregation.** Each feature appears twice (P1, P2). Default magnitude score:
   $\;|r_{f,1}| + |r_{f,2}|\;$. Keep the **signed** mean separately for §12.5 / Plot 3.

4. **Per-sample simplex normalization (critical for statistics).**
   $\;\hat r_f = |r_f| \big/ \sum_{f'} |r_{f'}|\;$, so $\sum_f \hat r_f = 1$ over the five features.
   Stops high-magnitude prompts (big lot numbers) from dominating the average.

5. **Aggregate across samples**, per target: **mean normalized relevance** (headline), **std**,
   **Top-1 frequency**, **Top-3 frequency**, each with a bootstrap CI (§12.2).

6. **Feature-attribution fraction.** Report $\sum_f |\mathrm{sum}_t R^0_t| \big/ \sum_t |R^0_t|$. In the
   funds (full instruction prompt) this was only **4–9%**; because we use the **simplified attribution
   prompt** (§9.4) that strips instruction blocks, we expect a *higher* fraction landing on the five
   feature segments. We report it either way.

---

## 8. How the mutual-fund track (Alan) did it — recap and what we keep

The funds report (`mutual_fund/mutual_fund.tex`, §"Feature Attribution: LRP", lines ~600–896) is our
methodological template. The funds track (Alan's work) ran **four** variants (AttnLRP/CP-LRP ×
single/logit-diff) on **11** fund features; **we carry only the two AttnLRP runs forward on the five
housing features.** CP-LRP numbers below are context, not something we reproduce.

**Setup.** LXT v2.1 on **Llama-3.2-3B-Instruct, zero-shot**, **500 non-tie pairs**, bfloat16, 4-decimal
truncation of feature values, **100 feature-order permutations** (11 features → 100 sampled orderings).

**AttnLRP findings we mirror on housing.**
- **Sharpe rank-1** under AttnLRP single-logit (**59.4% Top-1**); Sharpe + Return 3Y = 35.7% of feature
  relevance — an **inversion** of the perturbation methods.
- **NTF** (rank-1 under perturbation) → **rank 10** under AttnLRP — evidence the perturbation result was
  partly a *last-feature/positional* artifact.
- **Single vs logit-difference (AttnLRP):** $\rho = 0.745$ — moderately stable; logit-diff lifted
  Expense Ratio (drives the *margin*).
- **Signed relevance:** mostly positive; Turnover (26.7% neg) and Expense Ratio (15.4% neg) carried the
  most opposing signal.
- **Layer structure:** bimodal — peaks in mid-layers (9–13) and late layers (20–23).
- **AttnLRP vs perturbation:** weak/negative correlations — they measure different things.
- **AttnLRP vs probe accuracy:** significantly positive at late layers (21, 22, 25).

*(Context only: CP-LRP gave a flatter distribution and lifted date-like features. Not reproduced.)*

**Caveats we inherit:** low feature-attribution fraction; thin-slice rankings; perturbation methods
didn't control feature order (LRP did).

**What we tighten / change for housing:** five features instead of eleven; the **simplified attribution
prompt** (Alan used the full zero-shot prompt; we strip instructions to match the housing methods and
lift the attribution fraction); **MEAN-over-segment** aggregation (to match the housing methods);
bootstrap CIs + formal rank tests (§12); conservation verification + float32 (§14–15); a
deletion/insertion faithfulness curve (§15); and a *differs-only* identity-dilution analysis (§12.5).

---

## 9. Mapping the method onto the housing task

### 9.1 The task
Pairwise: two listings, same ZIP and month; decide which is **more valuable**. The behavioral output is
`CHOICE: <1 or 2>`; the decision token is the digit `1`/`2`.

### 9.2 The five features
We attribute the **same five features the housing IG/SHAP/Occlusion analysis used**
(`main.tex`, `sec:housing-internal-single`):

`lot, bathrooms, bedrooms, year built, property type`.

Heating, cooling, and parking type are **dropped** — exactly as in the existing housing attribution —
because their multi-token free-text values produce noisy attributions, and parking type is identical
across both properties in 97.6% of pairs (no discriminating signal). The `zpid` identifier is **not
placed in the prompt at all** — it is meaningless metadata, and an early run showed bare numeric IDs
attract substantial relevance, contaminating the attribution; the prompt therefore contains only the
five feature lines per property. Two property blocks ⇒ 10 feature instances ⇒ 5 feature scores (§7.3).

### 9.3 Decision token and the two targets
- **Single-logit:** logit of the chosen digit (`1`/`2`) — the target the housing methods used.
- **Logit-difference:** $\mathrm{logit}(\text{chosen digit}) - \mathrm{logit}(\text{other digit})$.

Both are AttnLRP; only the back-propagated scalar differs.

### 9.4 Prompt — the **simplified attribution prompt** (match the housing methods)
We do **not** use the full behavioral prompt or a CoT prompt. We use the housing track's **simplified
attribution prompt**: it drops the instruction blocks, leaving **only the property listings and a
`CHOICE:` cue**. This is what the existing IG/SHAP/Occlusion runs used, and it has two benefits we want:
(a) it is the apples-to-apples input for cross-method comparison, and (b) it **pushes relevance onto
feature tokens** rather than instruction tokens, directly improving the attribution-fraction problem
that limited the funds analysis. The attribution target is the logit of the chosen token (single-logit),
plus our added logit-difference. LRP attributes a **single forward pass**, which this clean prompt makes
well-posed.

### 9.5 Model (single) — **Llama-3.2-3B-Instruct** (Qwen3 dropped)
We run AttnLRP on **Llama-3.2-3B-Instruct only**. **Qwen3-4B is dropped completely** — even though it
is the model the existing IG/SHAP/Occlusion runs used — because AttnLRP is not faithful on it (see the
confirmed-issue box below). Llama-3.2-3B is fully supported by LXT (✅), is the model Alan used for the
funds LRP, and is one of the two open-weight models the housing track already uses for probing, amnesic
probing, and macro mapping — so it is a legitimate housing model. It has 28 transformer layers, GQA
(24 query / 8 KV heads), RoPE, and RMSNorm, but **no QK-Norm** (§4.4). The accepted cost is that the
cross-method comparison (§12.4) is **cross-model** (AttnLRP on Llama vs IG/SHAP/Occlusion on Qwen3),
not same-model.

> **⚠️ WHY QWEN3 IS DROPPED — confirmed against the LXT repo (2026-05-28; team-confirmed).** LXT's
> model-support table marks **Qwen 3 as 🧪 "Attribution skewed toward first token,"** while **LLaMA 2/3,
> Gemma 3, Qwen 2, BERT, GPT-2, and ViT are ✅.** A closed issue (#29 "NaNs in relevance output") also
> exists. Probable cause: **Qwen3's QK-Norm** (an extra RMSNorm on Q and K *inside* attention, not in
> AttnLRP's published rule set) on top of RoPE + GQA; separately, LRP methods are known not to propagate
> relevance through positional encoding (RoPE) [Revisiting LRP, arXiv:2506.02138]. A first-token skew
> would (a) steal relevance budget from the five feature segments, deflating the attribution fraction
> (§7.6), and (b) confound the Phase-2 position-permutation analysis, whose entire purpose is to remove
> position bias. Both effects strike at the core of this study, so **Qwen3 is out of scope** — not used,
> not reported.

---

## 10. Phase 1 — single model, robust baseline (pseudocode)

Goal: defensible AttnLRP rankings (both targets) on Llama-3.2-3B over the five features, fixed feature
order, before permutation.

```
# Pseudocode — illustrative, not runnable.
sample ← seeded_draw(pairs_20pct_price_diff.csv, n=500, seed=42)      # N locked at 500
FEATURES ← {lot, bathrooms, bedrooms, year_built, property_type}      # 5, matching housing methods
for each pair in sample:
    prompt ← build_simplified_attribution_prompt(pair)               # listings + CHOICE: cue only
    logits, chosen_id, other_id ← run_model(prompt)                  # capture decision + both digit logits
    if not parseable(CHOICE): mark invalid; continue
    for target in {SINGLE_LOGIT, LOGIT_DIFF}:
        R0     ← attnlrp_backward(prompt, target)                    # §6.2; per-token relevance
        segs   ← part_sentence_segments(prompt)                      # §7.1, feature-line granularity
        r_feat ← mean_tokens_per_feature(R0, segs, FEATURES)         # §7.2 MEAN aggregation; exclude zpid
        r_hat  ← simplex_normalize(abs(r_feat))                      # §7.4 (over 5 features)
        store(per_sample[target], r_hat, signed=r_feat)
        store(layerwise[target]); store(attr_fraction[target])       # §7.6 + Plot 2

for target in {SINGLE_LOGIT, LOGIT_DIFF}:
    table[target] ← mean, std, top1_freq, top3_freq per feature      # §7.5
    ci[target]    ← bootstrap_CI(per_sample[target], B=10000)        # §12.2
report cross_method_correlation(table[SINGLE_LOGIT], housing_ranks)  # §12.4 (vs IG/SHAP/Occlusion)
```

**Deliverable of Phase 1:** two ranking tables (single, logit-diff), a signed-relevance table, layer
data, a cross-method correlation table (AttnLRP vs the published IG/SHAP/Occlusion ranks), and the
attribution-fraction number — fixed feature order, Llama-3.2-3B.

---

## 11. Phase 2 — feature-position permutation as a scaling step (pseudocode)

Phase 2 bolts onto the same pipeline — it changes prompts and bookkeeping, **not the math**.

### 11.1 Why
Autoregressive attention has **recency/position bias**: features nearer the decision can attract
inflated relevance regardless of identity. The housing perturbation methods control this with a
**120-permutation** feature-order control (`main.tex:751`); AttnLRP must match so the ranking reflects
**feature identity, not position**.

### 11.2 Two confounds to break
1. **Within-property feature order** — permute the five feature lines inside each block.
2. **Property order (P1 vs P2)** — swap which listing is first (a clean, separable confound in the
   pairwise framing).

### 11.3 Permutation design (five features = the complete set) — pseudocode
For five features there are exactly $5! = 120$ orderings — so the housing track's "120 permutations" is
the **complete** permutation set, and we use all of it (no sampling or dedup needed). Positional
coverage is automatically perfect: across the 120 orderings each feature appears at each of the five
positions exactly $120/5 = 24$ times.

```
# Pseudocode — illustrative, not runnable.
pool ← all_permutations(FEATURES)            # 5! = 120 orderings (the full housing control set)
assert len(pool) == 120
assert positional_coverage(pool, n_features=5) == 24 per (feature, position)   # exact, by construction
for i, pair in enumerate(sample):            # N = 500
    order   ← pool[i % 120]                   # cyclic assignment; ~4.17 full cycles over 500 samples
    swapP12 ← (i is odd)                      # balanced P1/P2 swap, 50/50
    prompt  ← build_simplified_attribution_prompt(pair, feature_order=order, swap_properties=swapP12)
    ... run AttnLRP exactly as Phase 1; aggregate by FEATURE IDENTITY (not position) ...
```

### 11.4 What Phase 2 produces
- **Permuted rankings** per target (aggregated by identity across positions).
- **Rank stability:** Spearman $\rho$ between Phase-1 (fixed) and Phase-2 (permuted) rankings → Plot 4.
- **Position-sensitivity curve:** mean relevance vs prompt position per feature → recency-bias check.
- **P1/P2 order effect:** chosen-side relevance with vs without swap.
- **Verdict:** does the **Phase-1 top feature — whatever it turns out to be** — survive permutation?
  A feature whose rank holds across all 120 orderings is identity-driven; one that collapses was
  position-inflated. We report this for every feature, not just the winner.

---

## 12. Statistical-robustness protocol

This makes the result *defensible*, not just plotted.

### 12.1 Normalization
Per-sample simplex normalization over the five features (§7.4); report mean normalized relevance plus
**rank-based** summaries (Top-1, Top-3 freq) which are scale-free and outlier-robust.

### 12.2 Uncertainty — bootstrap CIs
Per feature, per target, **bootstrap over the 500 samples** (resample with replacement, **B = 10,000**)
→ 95% CI on mean normalized relevance. **Overlapping CIs ⇒ ranks not distinguishable**; with only five
features and a known tight cluster below lot (the housing methods found the other four at avg ranks
3.0–3.7), this caveat is essential — we will not over-claim a rank-2-vs-rank-3 ordering whose CIs overlap.

### 12.3 Rank-difference significance
- **Friedman test** across the five features (each sample a block) — omnibus "do features differ at all?".
  Valid because per sample we have five paired normalized-relevance values.
- **Wilcoxon signed-rank** post-hoc for pairs of interest (lot vs year built; **bedrooms vs bathrooms**).
- **Multiple-comparison correction:** Benjamini–Hochberg FDR (report adjusted $p$). Holm–Bonferroni as a
  conservative alternative. With five features there are only $\binom{5}{2}=10$ pairwise tests, so
  correction is mild and the tests are well-powered at N = 500.

### 12.4 Cross-target and cross-method correlations
Spearman $\rho$ with permutation $p$-values for: **single-logit vs logit-difference** (target
sensitivity, same model), and **AttnLRP-on-Llama (single-logit) vs the published IG / SHAP / Occlusion
ranks on Qwen3** (lot 1.00, year built 3.00, bedrooms 3.67, bathrooms 3.67, property type 3.67). The
five features, the simplified prompt, and the single-logit target all match, **but the model differs**
(Llama-3.2-3B vs Qwen3-4B), so this is a deliberate **cross-model** comparison: agreement (e.g., lot
rank-1 under all four methods on two different models) is *strong* evidence the finding is
model-robust; disagreement could be a method difference *or* a model difference and must be reported as
such. We attach this caveat wherever the comparison appears, and we do **not** claim a same-model
contrast.

### 12.5 Identity-dilution control (housing-specific)
Bathrooms identical across both listings in ~40.1% of pairs, bedrooms in ~39.3%; those pairs add ~0
relevance and drag the mean toward zero — the inversion that can make a decisive feature look
unimportant in aggregate. (This is also why bathrooms-when-they-differ is the single most predictive
feature yet ranks low in aggregate under two of three existing methods.) Report **two views**:
(1) **all-pairs** mean (diluted, comparable to the existing methods); (2) **differs-only** mean
(restricted to pairs where the two listings differ on that feature) — the fair "when it matters, how
much does it matter." This directly engages the housing track's central bedrooms-vs-bathrooms question.

### 12.6 Power / sample size
**N = 500** valid pairs → tight bootstrap CIs for a 5-feature simplex. If a feature's differs-only
subset is small (e.g., property type rarely differs), flag reduced power for *that* feature specifically.

### 12.7 Reproducibility
Fix all seeds (sampling, permutation, bootstrap). Log model revision, LXT version, dtype, ε, and the
exact (complete) 120-permutation pool. Persist raw per-sample per-token relevance so every number is
recomputable.

---

## 13. Plot specifications — four figures, no hand-editing

**Global rules (all four):** generated programmatically from the result CSVs; **no manual annotation,
no post-hoc editing**. Save as **300-dpi PNG only** (no PDF, per request). Use
`bbox_inches="tight"`. Colorblind-safe palette (Okabe–Ito). Title, axis labels, units, $N$, and any
statistic come **from the data**, written by the routine. Fonts ≥ 9 pt. Feature order passed in, not
eyeballed. Fully reproducible from `results/*.csv` + a seed. Five features ⇒ compact, legible figures.

### Figure 1 — Feature importance with bootstrap 95% CIs (headline)
- **Type:** horizontal grouped bar chart. **Y:** the five features, sorted by single-logit mean
  (descending). **X:** mean normalized relevance (0–1). Two bars per feature (single-logit,
  logit-difference) with **horizontal error bars = 95% bootstrap CI**.
- **Reads from:** `attnlrp_single.csv`, `attnlrp_logitdiff.csv`.
- **Why:** the core result; CIs make over-/under-claiming impossible (overlap ⇒ tie).
- **Pseudocode:**
```
df ← join(single, logitdiff) on feature; sort by single.mean desc
hbar(y=feature, x=mean, xerr=[mean−ci_lo, ci_hi−mean], group∈{single, logitdiff})
xlabel "mean normalized relevance"; title f"AttnLRP feature importance (Llama-3.2-3B, N=500)"
legend; save PDF+PNG
```

### Figure 2 — Layer × feature relevance heatmap (mechanism)
- **Type:** heatmap. **Rows:** five features (same order as Fig 1). **Cols:** layer index $0..L-1$
  (Llama-3.2-3B has 28 layers). **Color:** mean normalized relevance at that layer (viridis) + labeled
  colorbar. **Top marginal strip:** line of total relevance per layer.
- **Reads from:** `layer_feature_relevance.csv`.
- **Why:** shows *where in depth* each feature accrues relevance; ties to the probing results.
- **Pseudocode:**
```
M ← pivot(layer_feature_relevance, rows=feature, cols=layer, val=mean_norm)
gridspec: top(line: total per layer)  +  main(imshow(M, cmap=viridis)); colorbar
xlabel "layer"; ylabel "feature"; title f"AttnLRP layer×feature relevance ({target})"; save PDF+PNG
```

### Figure 3 — Signed relevance, diverging bar (conflict analysis)
- **Type:** diverging horizontal bar. **Y:** five features, sorted by net signed relevance. **X:** mean
  signed normalized relevance — **positive (right)** = supports the chosen listing, **negative (left)** =
  opposes. Two colors (support vs oppose); zero line drawn.
- **Reads from:** `signed_relevance.csv`.
- **Why:** the unique LRP capability — which features create *conflict* between listings (e.g.,
  bedrooms vs bathrooms).
- **Pseudocode:**
```
df ← signed_relevance; sort by net desc
hbar(y=feature, x=mean_positive, color=support); hbar(y=feature, x=−mean_negative, color=oppose)
axvline(0); xlabel "mean signed normalized relevance"; legend; save PDF+PNG
```

### Figure 4 — Permutation rank-stability (Phase 2 scaling-step validation)
- **Type:** scatter. **X:** Phase-1 (fixed-order) feature rank 1–5. **Y:** Phase-2 (all-120-permutation)
  feature rank 1–5. **Points:** the five features, labeled. **Reference:** dashed $y=x$. **Annotation
  (from data):** Spearman $\rho$ and its $p$-value, written by the routine. Integer ticks, equal aspect.
- **Reads from:** `permutation_stability.csv`.
- **Why:** the explicit test that the ranking reflects **feature identity, not prompt position** —
  on-diagonal ⇒ position-robust; off-diagonal ⇒ position-inflated.
- **Pseudocode:**
```
df ← permutation_stability   # cols: feature, rank_fixed, rank_permuted
scatter(rank_fixed, rank_permuted); label each point with feature
plot diagonal y=x dashed; annotate f"Spearman ρ={rho:.2f}, p={p:.3g}" (computed, not typed)
xlabel "rank (fixed order)"; ylabel "rank (all 120 permutations)"; equal aspect; save PDF+PNG
```

*(A fifth figure — the deletion/insertion faithfulness curve — is produced as a **validation
diagnostic** in §15, not a headline result.)*

---

## 14. Mathematical & statistical validation matrix

Every claim is checkable. "Validated by hand" = re-derived above; "checked at runtime" = asserted in the
pipeline against the stated criterion.

| Claim | How validated | Pass criterion |
|---|---|---|
| ε-rule conserves relevance for linear layers (§3.2) | Algebra: $\sum_i R_{i\leftarrow j}=R_j\frac{z_j-b_j}{z_j+\varepsilon}$ | →$R_j$ as $\varepsilon\to0$, bias-free; **validated by hand** |
| Bilinear matmul rule conserves (§4.1) | Sum $R(A)=\sum_{jp}\frac{R_{jp}O_{jp}}{2O_{jp}+\varepsilon}\approx\tfrac12\sum R_{jp}$; same for $R(V)$ | total $\approx\sum R_{jp}$; **validated by hand** |
| Q/4·K/4·V/2 budget (§4.2) | Compose bilinear rule twice through softmax | ½(V)+¼(Q)+¼(K)=1; **validated by hand** |
| Softmax rule (§4.3) | Re-derive from Jacobian $s_k(\delta_{ki}-s_i)$ in input×grad form | matches Prop. 3.1 exactly; **validated by hand** |
| Input relevance = `input ⊙ grad` (§3.3, §6.2) | DTD identity under modified backward | matches LXT efficient mode; **validated by reference** |
| End-to-end conservation on real prompts | Runtime: compare $\sum_t R^0_t$ (sum form) vs injected $R^L$ | median relative residual < 1–2% (bf16 in use; flag/escalate to float32 if exceeded); **checked at runtime** |
| Per-sample normalization yields a simplex (§7.4) | $\sum_f \hat r_f$ over 5 features | = 1 ± 1e-6 per sample; **checked at runtime** |
| MEAN aggregation neutralizes token-count bias (§7.2) | Correlate per-feature token-count vs relevance | no significant correlation; **checked at runtime** |
| Feature means have honest uncertainty (§12.2) | Bootstrap B=10,000 over 500 samples | 95% CI reported; ranks asserted only for non-overlapping CIs; **statistical** |
| Features differ in relevance (§12.3) | Friedman omnibus → Wilcoxon post-hoc (10 pairs) | statistic + FDR-adjusted $p$; **statistical** |
| Cross-method agreement (§12.4) | Spearman $\rho$ vs published IG/SHAP/Occlusion ranks | $\rho, p$ reported; **statistical** |
| Identity dilution doesn't hide a decisive feature (§12.5) | All-pairs vs differs-only means | both reported; divergence flagged; **statistical** |
| Permutation removes position bias (§11, Plot 4) | Full 5!=120 set; coverage = 24/(feature,pos); Spearman fixed-vs-permuted | coverage exact; $\rho$ reported; top feature stable; **statistical** |
| Attribution is faithful (not an artifact) | Deletion/insertion AUC vs IG and vs random ordering (§15) | LRP deletion AUC beats random; reported; **checked at runtime** |
| Method is sensitive to the model | Model-randomization sanity check (Adebayo 2018) | attribution changes materially; **checked at runtime** |

---

## 15. Runtime sanity checks

Run *before* trusting any ranking; each maps to a row in §14.

1. **Conservation check.** Per sample verify $\sum_t R^0_t \approx R^{L}$ (sum form); report median
   relative residual. bfloat16 is in use by decision, so a small leak is expected — flag if it exceeds
   the §14 tolerance; float32 is the escape hatch only if conservation actually fails.
2. **Token-count neutrality check.** Confirm per-feature relevance is not correlated with the feature's
   token count (validates the MEAN aggregation choice).
3. **Model-randomization sanity check** (Adebayo et al., 2018). Re-initialize top layers; attribution
   must change materially, else the method is model-insensitive and meaningless.
4. **Deletion / insertion faithfulness curve (diagnostic figure).** Mask feature tokens in decreasing
   AttnLRP-relevance order; track the drop in the chosen logit / margin; compute AUC; compare across the
   two targets, against IG, and against a random-order baseline. A faithful attribution drops fast.
5. **Positional-coverage check (Phase 2).** Trivially satisfied by the full 120-permutation set
   (24 per feature-position); assert it anyway.
6. **Cross-method reality check.** If AttnLRP contradicts *all three* existing methods *and* fails the
   faithfulness curve, treat it as a pipeline bug, not a discovery.

---

## 16. Expected results

Predictions, framed to be wrong informatively, and tied to the published housing ranks
(lot 1.00; year built 3.00; bedrooms 3.67; bathrooms 3.67; property type 3.67).

- **Lot (hypothesis, both directions in play):** *if* AttnLRP also ranks `lot` first, that corroborates
  the IG/SHAP/Occlusion finding; but AttnLRP could equally **demote** it — in funds it *inverted* the
  perturbation ranking, sending the perturbation-top feature to rank 10. And because AttnLRP is
  `input×grad` and magnitude-sensitive, a high `lot` score must be read against the magnitude-bias
  caveat (§17 row 1), not taken at face value. Confirm, demote, or invert — each is informative.
- **Year built may rank second**, as it did under KernelSHAP and Occlusion (its distinctive 4-digit
  token pattern tends to attract relevance — compare Inception/Tenure in funds).
- **Bedrooms vs bathrooms parity.** The existing methods place them at parity (bed/bath ratio 0.87–1.02).
  If AttnLRP agrees, it reinforces that the bedroom-sensitivity asymmetry lives *downstream* of input
  reading (consistent with the macro-mapping vs input-attribution split in `main.tex`). The
  **differs-only** view (§12.5) is where any real bathroom signal should surface.
- **Single vs logit-difference correlation:** moderate-to-high (funds AttnLRP $\rho\approx0.745$).
- **Signed relevance:** mostly positive; conflict features likely bedrooms/bathrooms and property type.
- **Layer structure:** late-layer consolidation peak, possibly bimodal as in funds.
- **Attribution fraction:** higher than the funds' 4–9% because the simplified prompt strips instructions.
- **Phase 2:** the Phase-1 ranking should **survive** all 120 permutations if it is identity-driven;
  any feature whose rank collapses under permutation was position-inflated. We report this for every
  feature, not just the top one.

---

## 17. Limitations, known issues, and mitigations

| # | Issue | Why it matters | Mitigation |
|---|-------|----------------|------------|
| 1 | **Magnitude bias** of AttnLRP (`input×grad`) | Big numbers (lot sqft) inflate relevance | Truncate decimals (funds: 4-dp); report logit-diff alongside single; cross-check vs rank-based KernelSHAP |
| 2 | **Low feature-attribution fraction** | Rankings from a thin slice of total relevance | Simplified prompt lifts it; report the fraction (§7.6); rankings are relative-within-features |
| 3 | **Identity dilution** (40% identical baths/beds) | Decisive features look unimportant in aggregate | Differs-only conditioning (§12.5) beside all-pairs |
| 4 | **Multi-token numbers / tokenization** | Bad span mapping corrupts everything | Tokenizer offset mappings; PART_SENTENCE + MEAN; assert span coverage; `zpid` zero-control |
| 5 | **bfloat16 leaks relevance** | Conservation residual grows | bf16 chosen by the team; verify conservation (§15.1) and report the residual; escalate to float32 only if conservation fails |
| 6 | **AttnLRP unfaithful on Qwen3** (confirmed 🧪 "first-token skew"; QK-Norm not in the rule set) | Would corrupt the five-feature ranking and the position analysis | **Resolved — Qwen3 dropped.** Run on Llama-3.2-3B (✅, no QK-Norm); §9.5 |
| 7 | **Five features only** | Heating/cooling/parking excluded (as in the other methods) | Stated and intentional; matches the housing attribution scope exactly |
| 8 | **Single model (Llama-3.2-3B)** | No cross-model generality | Frame as model-specific; same model as Alan's funds LRP and the housing probing/amnesic runs |
| 9 | **LRP ≠ ground truth** | A pretty heatmap can be unfaithful | Deletion/insertion curve + sanity checks (§15) |
| 10 | **ε breaks strict conservation** | Small leakage into ε | Keep ε = $10^{-6}$; report conservation residual |
| 11 | **Property type rarely differs** | Small differs-only subset → low power for that feature | Flag reduced power for property type specifically (§12.6) |
| 12 | **Property-order (P1/P2) bias** | First-listing advantage masquerades as importance | Balanced P1/P2 swap in Phase 2 (§11.2) |
| 13 | **Single method (no CP-LRP cross-check)** | Lose the value-path vs attention-pattern contrast | Triangulate via the two AttnLRP targets + existing IG/SHAP/Occlusion; CP-LRP is a one-line add-on if needed |
| 14 | **Cross-model comparison** (AttnLRP on Llama vs IG/SHAP/Occ on Qwen3) | Disagreement could be a method *or* a model difference | Flag in §12.4; treat agreement (e.g., lot rank-1) as model-robust evidence, disagreement as ambiguous; never claim a same-model contrast |
| 15 | **12 GB VRAM / Blackwell GPU** | OOM or "no kernel image" can block the run | bs=1 + checkpointing + bf16 (§6.4); verify cu128 PyTorch build (§6.5); Phase 2 as a background job |

---

## 18. Deliverables and folder layout

Everything lives under `Attn-LRP/` (this folder — the single, canonical folder for this work).

```
Attn-LRP/
├── PLAN.md                                  # this validated technical plan
├── (later) results/
│   ├── attnlrp_single.csv                   # per-feature mean/std/top1/top3 + CI (single-logit), 5 features
│   ├── attnlrp_logitdiff.csv                # per-feature mean/std/top1/top3 + CI (logit-difference)
│   ├── signed_relevance.csv                 # per-feature mean positive / negative
│   ├── layer_feature_relevance.csv          # feature × layer mean normalized relevance
│   ├── cross_method_correlation.csv         # AttnLRP vs IG/SHAP/Occlusion + single-vs-logitdiff
│   ├── permutation_stability.csv            # Phase 2: rank_fixed vs rank_permuted, ρ, coverage
│   └── conservation_and_sanity.csv          # §14/§15 validation log
├── (later) figures/                         # PNG only (no PDF, per request)
│   ├── fig1_feature_importance_ci.png        # Figure 1 (headline)
│   ├── fig2_layer_feature_heatmap.png        # Figure 2 (mechanism)
│   ├── fig3_signed_relevance.png             # Figure 3 (conflict)
│   ├── fig4_permutation_rank_stability.png   # Figure 4 (Phase 2 validation)
│   └── diag_deletion_insertion_curve.png     # §15 faithfulness diagnostic
└── (later) writeup/
    └── housing_lrp_section.tex              # mirrors the AttnLRP parts of the funds LRP section
```

Reporting mirrors the AttnLRP parts of the funds LRP subsection (two ranking tables, signed table,
layer figure, cross-method correlation, interpretation) so the housing report can slot it beside the
existing IG/SHAP/Occlusion results and retire the `main.tex:757` "not run on housing" note.

---

## 19. Open clarifying questions for the team

Nearly everything is now locked: **AttnLRP only** · two targets (single-logit + logit-difference) ·
**Llama-3.2-3B-Instruct** (Qwen3 dropped) · five features (lot, bathrooms, bedrooms, year built,
property type) · simplified attribution prompt · MEAN-over-segment aggregation · **N = 500** ·
**bf16** · full **120-permutation** control · four figures · local **RTX 5070 Ti (12 GB)** + the
**`housing`** conda/venv env.

Only low-stakes defaults remain (override if you disagree — none block starting):

1. **Two-property aggregation for the headline:** sum of absolute relevance $|r_1|+|r_2|$ (default) vs
   signed sum vs max. Signed values are kept for Fig 3 regardless.
2. **Decimal truncation of feature values** to curb magnitude bias (funds used 4-dp): apply (default)
   vs leave raw.
3. **Background execution:** run the Phase-2 120-permutation sweep as a background job given the 12 GB
   card and checkpoint partials (default: yes).

*(Out of scope by decision: CP-LRP, and Qwen3 entirely — the LXT first-token-skew defect makes AttnLRP
unfaithful there.)*

---

## 20. References

- Achtibat, Hatefi, Dreyer, Jain, Wiegand, Lapuschkin, Samek. **AttnLRP: Attention-Aware Layer-Wise
  Relevance Propagation for Transformers.** ICML 2024. (`achtibat2024attnlrp` in `refs.bib`.)
  - Paper: https://arxiv.org/abs/2402.05602 · HTML: https://arxiv.org/html/2402.05602v1
  - PMLR: https://proceedings.mlr.press/v235/achtibat24a.html
- **LXT — LRP-eXplains-Transformers** (official implementation):
  https://github.com/rachtibat/LRP-eXplains-Transformers
- Ali, Schnake, Eberle, Montavon, Müller, Wolf. **XAI for Transformers: Better Explanations through
  Conservative Propagation (CP-LRP).** ICML 2022. (`ali2022xai`; cited for context — out of scope.)
- Montavon, Binder, Lapuschkin, Samek, Müller. **Layer-Wise Relevance Propagation: An Overview** (2019):
  https://iphome.hhi.de/samek/pdf/MonXAI19.pdf
- Adebayo et al. **Sanity Checks for Saliency Maps.** NeurIPS 2018 (model-randomization test, §15.4).
- Housing attribution baseline (the protocol we match): `main.tex` §"Real Estate Property"
  (`sec:housing-internal-single`) — IG / KernelSHAP / Occlusion on Qwen3-4B, five features, simplified
  prompt, 120-permutation control; "LRP not run" at `main.tex:757`. Funds template (Alan):
  `mutual_fund/mutual_fund.tex` §"Feature Attribution: LRP" (~600–896).

---

*End of validated technical plan. Next action after the §19 calls: implement Phase 1 (§§6–10) — AttnLRP
single-logit and logit-difference on Llama-3.2-3B over the five features, with the simplified attribution
prompt — run the validation suite (§§14–15), then add Phase 2 with the full 120-permutation set (§11)
and generate the four figures (§13).*
