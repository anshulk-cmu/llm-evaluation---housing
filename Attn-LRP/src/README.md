# Attn-LRP — implementation of `../PLAN.md`

AttnLRP feature attribution for the housing "which listing is more valuable" task.
**Open discovery**: we measure which of the five features actually drive the model's
choice — no feature (including `lot`) is assumed to be the answer.

- **Method:** AttnLRP only (no CP-LRP), via the LXT library, two targets
  (single-logit, logit-difference).
- **Model:** Llama-3.2-3B-Instruct (sole model; Qwen3 dropped — LXT first-token-skew defect).
- **Features (5):** lot, bathrooms, bedrooms, year built, property type.
- **Data:** `../../llm-evaluation-housing-reference/data/pairs_20pct_price_diff.csv`, N=500.
- **Phase 1** fixed feature order; **Phase 2** all 5!=120 permutations + P1/P2 swap.

See `../PLAN.md` for the math (§3–§5), statistics (§12), validation matrix (§14),
and figure specs (§13). Section references in the code point back to it.

## Files (`src/`)

| File | Role |
|---|---|
| `config.py` | All locked constants, paths, published housing ranks |
| `log.py` | Logging (console + timestamped file in `results/logs/`) |
| `common.py` | `clean_str`, `parse_choice` (mirror the housing pipeline) |
| `prompts.py` | Simplified attribution prompt + value spans + 120 permutations |
| `data.py` | Load / seeded sample of pairs |
| `attribution.py` | **GPU**: load Llama, apply LXT AttnLRP, per-token + per-layer relevance |
| `aggregate.py` | token→feature (MEAN over segments), simplex normalise, fraction |
| `stats.py` | bootstrap CIs, Friedman, Wilcoxon+FDR, Spearman, ranks |
| `sanity.py` | conservation, zpid control, token-count neutrality, deletion curve, randomisation |
| `plots.py` | 4 figures + deletion diagnostic (PDF+PNG, no hand-editing) |
| `run_phase1.py` | Phase 1 orchestrator |
| `run_phase2.py` | Phase 2 (permutation) orchestrator |
| `smoke_test.py` | Non-GPU end-to-end plumbing test (run this first) |

## Setup

```bash
# 1) Blackwell (RTX 5070 Ti) PyTorch FIRST — needs CUDA 12.8 wheels:
pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2) Everything else into the existing `housing` env:
pip install -r requirements.txt
```

`HF_TOKEN` must be set (Llama-3.2 is gated), or point `ATTNLRP_MODEL_DIR` at a
local snapshot of `Llama-3.2-3B-Instruct`.

## Run

```bash
cd src
python smoke_test.py                # validate plumbing (no GPU) - do this first
python preflight.py --full          # verify env + GPU + model + LXT before launch
python run_phase1.py --n 500        # AttnLRP, fixed order (batched on the GPU)
python run_phase2.py --n 500        # 120-permutation control (run in background)
python plots.py                     # build the four PNG figures from results/
```

Performance: the GPU runs are batched (`ATTNLRP_BATCH`, default 8 prompts per
forward+backward); CPU ops use `ATTNLRP_THREADS` (default min(24, cores)). Lower
`--batch-size` if VRAM is tight, raise it if there is headroom.

Useful flags: `run_phase1.py --batch-size 8`, `--no-layers` (skip the layer
heatmap to save VRAM), `--deletion-subset 0`, `--randomization-subset 0`.
Verbosity via `ATTNLRP_LOG_LEVEL=DEBUG`. Debug-only plumbing without LXT:
`ATTNLRP_DISABLE_LXT=1` (plain grad x input - NOT AttnLRP; never report it).

## Outputs (`../results/`, `../figures/`)

CSVs: `attnlrp_single.csv`, `attnlrp_logitdiff.csv`, `signed_relevance.csv`,
`layer_feature_relevance.csv`, `cross_method_correlation.csv`, `significance.csv`,
`permutation_stability.csv`, `position_sensitivity.csv`, `conservation_and_sanity.csv`,
`per_sample_relevance.csv`, `deletion_curve.csv`.
Figures (PNG only): `fig1_feature_importance_ci.png`, `fig2_layer_feature_heatmap.png`,
`fig3_signed_relevance.png`, `fig4_permutation_rank_stability.png`,
`diag_deletion_insertion_curve.png`.

## Validate before trusting (PLAN §15)

Check `conservation_and_sanity.csv`: conservation median residual within tolerance,
`zpid` relevance ≈ 0, token-count neutrality ρ ≈ 0, randomisation Δ clearly > 0;
and `deletion_curve.csv` (AttnLRP order should drop faster than random).
