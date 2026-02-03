# Mechanistic Interpretability of LLMs on Housing Price Comparison

**Understanding why language models fail at numerical reasoning through linear probing and causal analysis**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Models Evaluated](#models-evaluated)
4. [Experimental Pipeline](#experimental-pipeline)
5. [Stage 1: Model Accuracy Evaluation](#stage-1-model-accuracy-evaluation)
6. [Stage 2: Activation Extraction](#stage-2-activation-extraction)
7. [Stage 3: Linear Probing Analysis](#stage-3-linear-probing-analysis)
8. [Stage 3b: Logistic Regression Price Predictor](#stage-3b-logistic-regression-price-predictor)
9. [Key Findings](#key-findings)
10. [Generated Visualizations](#generated-visualizations)
11. [Next Steps: Causal Tracing](#next-steps-causal-tracing)
12. [Repository Structure](#repository-structure)

---

## Project Overview

This project investigates **why large language models (LLMs) struggle with numerical comparison tasks**, specifically comparing housing prices. We use mechanistic interpretability techniques to understand:

1. **What information do models encode?** (via linear probing)
2. **Where does the reasoning fail?** (via causal tracing - upcoming)
3. **Can we identify specific failure modes?** (attention head analysis - upcoming)

### Research Question

> If a model knows Property A has more bedrooms, more bathrooms, larger square footage, and was built more recently than Property B, why does it still fail to correctly predict which property costs more?

### Key Insight

**The "Last Mile" Problem**: Our probing analysis reveals that LLMs strongly encode all input features (87-96% linear probe accuracy) but fail catastrophically when predicting the target label (58-61% accuracy — barely above chance). The information exists in the model's representations, but something in the final computation fails to use it correctly.

---

## Dataset

### Source
- **Original**: Real estate listings data
- **Processed**: `data/pairs_20pct_price_diff.csv`

### Statistics
| Metric | Value |
|--------|-------|
| Total property pairs | 5,130 |
| Price difference threshold | ≥20% |
| Features per property | 5 (bedrooms, bathrooms, sqft, lot size, year built) |
| Binary labels | 6 (5 feature comparisons + 1 price comparison) |

### Binary Labels Derived
For each property pair, we compute:

| Label | Description |
|-------|-------------|
| `bathrooms_p1_more` | Property 1 has more bathrooms |
| `bedrooms_p1_more` | Property 1 has more bedrooms |
| `sqft_p1_larger` | Property 1 has larger square footage |
| `lot_p1_larger` | Property 1 has larger lot size |
| `year_p1_newer` | Property 1 was built more recently |
| `price_p1_higher` | Property 1 has higher price (**ground truth target**) |

---

## Models Evaluated

| Model | Parameters | Layers | Hidden Dim | Source |
|-------|-----------|--------|------------|--------|
| **Llama-3.2-3B-Instruct** | 3B | 28 | 3,072 | Meta |
| **Qwen3-4B-Instruct** | 4B | 36 | 2,560 | Alibaba |

Both models are instruction-tuned and accessed locally via HuggingFace Transformers.

---

## Experimental Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENTAL PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Stage 1: Model Evaluation                                               │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                     │
│  │ Property   │───▶│ Prompt     │───▶│ Model      │───▶ Accuracy        │
│  │ Pairs      │    │ Strategies │    │ Inference  │     by Strategy     │
│  └────────────┘    └────────────┘    └────────────┘                     │
│                                                                          │
│  Stage 2: Activation Extraction                                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                     │
│  │ Winning    │───▶│ Forward    │───▶│ Residual   │───▶ (N, L, D)       │
│  │ Prompts    │    │ Pass       │    │ Stream     │     Activations     │
│  └────────────┘    └────────────┘    └────────────┘                     │
│                                                                          │
│  Stage 3: Linear Probing                                                 │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                     │
│  │ Activations│───▶│ Logistic   │───▶│ Accuracy   │───▶ What info       │
│  │ + Labels   │    │ Regression │    │ per Layer  │     is encoded?     │
│  └────────────┘    └────────────┘    └────────────┘                     │
│                                                                          │
│  Stage 4: Causal Tracing (Next)                                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                     │
│  │ Activation │───▶│ Patch &    │───▶│ Causal     │───▶ Where does      │
│  │ Patching   │    │ Measure    │    │ Effects    │     reasoning fail? │
│  └────────────┘    └────────────┘    └────────────┘                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Model Accuracy Evaluation

### Prompt Strategies Tested

We evaluated multiple prompting strategies to find the best-performing approach:

| Strategy | Description | Llama Acc | Qwen Acc |
|----------|-------------|-----------|----------|
| `0_zeroshot_temp0` | Direct question, no examples | ~52% | ~54% |
| `1_zeroshot_cot_temp0` | Zero-shot chain-of-thought | ~55% | ~57% |
| `2_fewshot_cot_temp0` | **2-shot CoT (Winner)** | **~58%** | **~60%** |
| `3_fewshot_nocot_temp0` | 2-shot without reasoning | ~54% | ~56% |

### Key Observation

Even with the best prompting strategy (few-shot chain-of-thought), both models perform only slightly better than random chance (50%). This motivated the mechanistic analysis: **why do models fail despite seemingly understanding the task?**

---

## Stage 2: Activation Extraction

### Methodology

For each of the 5,130 property pairs:
1. Format the prompt using the winning strategy (`2_fewshot_cot_temp0`)
2. Run a forward pass through the model
3. Extract residual stream activations at the **last token position** for all layers
4. Store as numpy arrays: `(N_samples, N_layers, D_model)`

### Activation Files

| Model | Shape | File Size | Location |
|-------|-------|-----------|----------|
| Llama-3.2-3B | (5130, 28, 3072) | ~1.6 GB | `data/activations/llama-3.2-3b_2_fewshot_cot_temp0_activations.npz` |
| Qwen3-4B | (5130, 36, 2560) | ~1.8 GB | `data/activations/qwen3-4b_2_fewshot_cot_temp0_activations.npz` |

### Technical Details

- **Extraction method**:
  - Llama: TransformerLens (`hook_resid_post`)
  - Qwen: HuggingFace `output_hidden_states=True` (TransformerLens doesn't support Qwen3)
- **Token position**: Last token (-1), where the model makes its prediction
- **Batch size**: 50 (optimized for A6000 48GB GPU)

---

## Stage 3: Linear Probing Analysis

### What is Linear Probing?

Linear probing tests whether information is **linearly encoded** in a model's internal representations. We train a simple logistic regression classifier on the activations to predict binary labels. High accuracy indicates the information is accessible; low accuracy suggests the information is either absent or encoded non-linearly.

### Methodology

```
For each layer L in [0, ..., N_layers-1]:
    For each feature F in [bathrooms, bedrooms, sqft, lot, year, price]:
        1. Extract activations at layer L: X = activations[:, L, :]  # (5130, D_model)
        2. Get binary labels: y = labels[F]  # (5130,)
        3. Split: 70% train, 10% validation, 20% test
        4. Train logistic regression with GPU acceleration (cuML)
        5. Select best regularization C via validation set
        6. Evaluate on held-out test set
        7. Compute confidence intervals and p-values
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Train/Val/Test Split | 70% / 10% / 20% |
| Regularization values (C) | [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] |
| Max iterations | 2,000 (4,000 for final model) |
| Solver | cuML QN (GPU) or sklearn SAGA (CPU) |
| Significance threshold | p < 0.05 |
| Confidence intervals | Wilson score (95%) |

### Results: Llama-3.2-3B (28 layers)

| Feature | Best Layer | Test Accuracy | 95% CI | AUC | p-value |
|---------|-----------|---------------|--------|-----|---------|
| Year Built | **L15** | **95.9%** | [94.5%, 97.0%] | 0.990 | <0.0001 |
| Bedrooms | L15 | 91.0% | [89.1%, 92.6%] | 0.961 | <0.0001 |
| Bathrooms | L15 | 90.2% | [88.2%, 91.8%] | 0.942 | <0.0001 |
| Square Feet | L15 | 86.9% | [84.7%, 88.9%] | 0.927 | <0.0001 |
| Lot Size | L15 | 86.9% | [84.7%, 88.9%] | 0.927 | <0.0001 |
| **Price (Target)** | L19 | **60.7%** | [57.7%, 63.7%] | 0.628 | <0.0001 |

### Results: Qwen3-4B (36 layers)

| Feature | Best Layer | Test Accuracy | 95% CI | AUC | p-value |
|---------|-----------|---------------|--------|-----|---------|
| Year Built | **L23** | **94.7%** | [93.2%, 95.9%] | 0.969 | <0.0001 |
| Bedrooms | L22 | 92.5% | [90.7%, 94.0%] | 0.959 | <0.0001 |
| Bathrooms | L19 | 89.7% | [87.7%, 91.4%] | 0.944 | <0.0001 |
| Square Feet | L19 | 88.8% | [86.7%, 90.6%] | 0.943 | <0.0001 |
| Lot Size | L23 | 88.8% | [86.7%, 90.6%] | 0.927 | <0.0001 |
| **Price (Target)** | L34 | **58.6%** | [55.5%, 61.6%] | 0.604 | <0.0001 |

### Layer-wise Accuracy Progression

The accuracy follows a characteristic curve across layers:

```
Layer 0-5:   ~65-70%  │ Early layers: Basic feature extraction
Layer 6-11:  ~75-85%  │ Middle-early: Feature encoding builds
Layer 12-17: ~85-96%  │ **Peak encoding** (mid-network)
Layer 18-23: ~85-90%  │ Slight decline, information maintained
Layer 24-28: ~85-90%  │ Late layers: Output preparation
```

For **price_p1_higher**, accuracy stays flat at ~55-61% across ALL layers — the information is never encoded.

### The "Last Mile" Gap

```
┌─────────────────────────────────────────────────────────────┐
│                    THE LAST MILE PROBLEM                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input Features (avg)          Price (Target)               │
│   ┌───────────────────┐         ┌───────────────────┐       │
│   │                   │         │                   │       │
│   │      ~90%         │         │      ~60%         │       │
│   │                   │         │                   │       │
│   │   ████████████    │         │   ████            │       │
│   │   ████████████    │         │   ████            │       │
│   │   ████████████    │         │   ████            │       │
│   │   ████████████    │         │   ████            │       │
│   │   ████████████    │         │   ████            │       │
│   └───────────────────┘         └───────────────────┘       │
│                                                              │
│   GAP: ~30 percentage points                                 │
│                                                              │
│   The model KNOWS the features but FAILS to combine them     │
│   into a correct price prediction.                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Statistical Validation

- **Total probes**: 384 (168 Llama + 216 Qwen)
- **Significant probes**: 378/384 (98.4%) at p < 0.05
- **Confidence intervals**: Wilson score with 95% confidence
- **Bootstrap validation**: 1,000 iterations for robust CI estimation
- **All input feature probes**: p < 0.0001 (highly significant)
- **Price probes**: p < 0.0001 but accuracy barely above chance

### Interpretation

| Finding | Interpretation |
|---------|---------------|
| Input features at 87-96% | Information is **linearly accessible** in residual stream |
| Peak at Layer 15 (54% depth) | Mid-network layers encode semantic features |
| Price at 58-61% | Price comparison is **NOT linearly encoded** |
| Convergence warnings for price | Optimizer can't find stable weights (no signal) |
| Both models show same pattern | This is a **fundamental limitation**, not model-specific |

---

## Stage 3b: Logistic Regression Price Predictor

### Motivation

Stage 3 probed for multiple features (bedrooms, bathrooms, etc.) to understand what information models encode. This follow-up experiment asks a different question: **Can a simple classifier trained on activations outperform the LLM's own price prediction?**

If a logistic regression model trained on the LLM's internal representations achieves higher accuracy than the LLM's direct output, it would prove the information exists but the model's final computation layers are broken.

### Methodology

We train a logistic regression classifier on activations from **each layer** to predict the target task directly (which property costs more):

```
For each layer L in [0, ..., N_layers-1]:
    1. Extract activations: X = activations[:, L, :]  # (5130, D_model)
    2. Get price labels: y = price_p1_higher  # (5130,)
    3. Split: 70% train, 10% validation, 20% test (stratified)
    4. Hyperparameter search: C ∈ [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    5. Select best C via validation accuracy
    6. Train final model, evaluate on held-out test set
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Target | `price_p1_higher` (binary) |
| Train/Val/Test Split | 70% / 10% / 20% (stratified) |
| Regularization values (C) | [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] |
| Max iterations | 2,000 (4,000 for final model) |
| Solver | cuML QN (GPU-accelerated) |
| Runtime | ~4 minutes total (both models) |

### Results

| Model | Best Layer | Test Accuracy | 95% CI | AUC | Mean Acc (all layers) |
|-------|-----------|---------------|--------|-----|----------------------|
| **Llama-3.2-3B** | L27 (96% depth) | **61.3%** | [58.3%, 64.2%] | 0.629 | 57.5% |
| **Qwen3-4B** | L34 (94% depth) | **58.7%** | [55.6%, 61.6%] | 0.604 | 55.4% |

### Layer-wise Accuracy Progression

```
Llama-3.2-3B (28 layers):
Layer 0-10:  ~52-57%  │ Early layers: Weak price signal
Layer 11-20: ~57-60%  │ Mid layers: Signal builds gradually
Layer 21-27: ~59-61%  │ Late layers: Peak at L27 (61.3%)

Qwen3-4B (36 layers):
Layer 0-15:  ~51-57%  │ Early layers: Near chance
Layer 16-30: ~55-58%  │ Mid layers: Gradual improvement
Layer 31-35: ~57-59%  │ Late layers: Peak at L34 (58.7%)
```

### Comparison with Baselines

```
┌─────────────────────────────────────────────────────────────┐
│                    METHOD COMPARISON                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Chance        LLM Output      Best Probe      Zestimate   │
│   ┌─────┐       ┌─────┐        ┌─────┐         ┌─────┐     │
│   │     │       │     │        │     │         │     │     │
│   │ 50% │       │ 60% │        │ 61% │         │ 78% │     │
│   │     │       │     │        │     │         │     │     │
│   │ ██  │       │ ███ │        │ ███ │         │████ │     │
│   │ ██  │       │ ███ │        │ ███ │         │████ │     │
│   │ ██  │       │ ███ │        │ ███ │         │████ │     │
│   └─────┘       └─────┘        └─────┘         └─────┘     │
│                                                              │
│   The probe matches but doesn't beat the LLM output.        │
│   Price information is NOT linearly accessible.             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight

**The probe achieves ~61% accuracy — matching the LLM's own output (~60%).**

This confirms that price comparison information is **not linearly encoded** in the model's representations. Unlike input features (bedrooms, bathrooms, etc.) which are linearly accessible at 87-96%, the price comparison result simply doesn't exist in a form that a linear classifier can extract.

### Interpretation

| Finding | Interpretation |
|---------|---------------|
| Probe ≈ LLM output (~60%) | No hidden linear signal for price comparison |
| Best layer at 94-96% depth | Late layers have slightly more price-relevant info |
| Still 17pp below Zestimate | Fundamental gap in numerical reasoning capability |
| Accuracy flat across layers | Price signal never builds (unlike input features) |

### What This Means

1. **Not a "reading" problem**: The models encode input features excellently
2. **Not a "hidden knowledge" problem**: There's no linear price signal we're missing
3. **A computation problem**: The failure is in combining features → price estimate
4. **Confirms Last Mile hypothesis**: Information exists, but computation fails

### Output Files

| File | Location |
|------|----------|
| Detailed results (Llama) | `data/logit_results/logit_results_llama-3.2-3b_2_fewshot_cot_temp0.csv` |
| Detailed results (Qwen) | `data/logit_results/logit_results_qwen3-4b_2_fewshot_cot_temp0.csv` |
| Summary (Llama) | `data/logit_results/logit_summary_llama-3.2-3b_2_fewshot_cot_temp0.json` |
| Summary (Qwen) | `data/logit_results/logit_summary_qwen3-4b_2_fewshot_cot_temp0.json` |
| Accuracy vs Layer plot | `data/logit_results/plots/accuracy_vs_layer_2_fewshot_cot_temp0.png` |
| Method comparison plot | `data/logit_results/plots/method_comparison_2_fewshot_cot_temp0.png` |

---

## Key Findings

### 1. Models Encode All Input Features
Both Llama and Qwen strongly encode which property has:
- More bedrooms (91-92%)
- More bathrooms (90%)
- Larger square footage (87-89%)
- Larger lot size (87-89%)
- Newer year built (95-96%)

### 2. Models Fail at the Target Task
Despite encoding all relevant features, both models achieve only ~60% accuracy on predicting which property costs more — barely better than a coin flip.

### 3. The Bottleneck is in Late-Stage Processing
The information exists in the representations. The failure occurs when the model attempts to:
- Combine features into a price estimate
- Compare the implicit price estimates
- Output the correct token ("Property 1" vs "Property 2")

### 4. This is Not a Prompt Engineering Problem
We tested multiple prompt strategies. The best (few-shot CoT) only marginally improves accuracy. The failure is at the representation/computation level, not the input level.

### 5. Linear Probing Limitations
Linear probes test for **linear accessibility**, not causal use. The model might:
- Encode price non-linearly
- Have the information but not use it
- Use different circuits for different examples

This motivates **causal tracing** to identify which components are responsible.

---

## Generated Visualizations

All plots are saved to `data/probe_results/plots/`:

| File | Description |
|------|-------------|
| `heatmap_accuracy_llama-3.2-3b.png` | Layer × Feature accuracy heatmap |
| `heatmap_accuracy_qwen3-4b.png` | Layer × Feature accuracy heatmap |
| `accuracy_curves_llama-3.2-3b.png` | Accuracy across layers (all features) |
| `accuracy_curves_qwen3-4b.png` | Accuracy across layers (all features) |
| `best_layer_comparison.png` | Cross-model comparison + Last Mile visualization |
| `layer_depth_analysis.png` | Peak layer as % of model depth |
| `model_comparison_curves.png` | 2×3 grid comparing both models |
| `significance_heatmaps.png` | Statistical significance (-log₁₀ p-value) |
| `summary_dashboard.png` | **Comprehensive summary dashboard** |
| `results_table.tex` | LaTeX table for publications |

---

## Next Steps: Causal Tracing

Linear probing reveals **what** information is encoded, but not **whether** the model uses it. The next phase uses **activation patching** (causal tracing) to identify:

### Planned Experiments

1. **Layer-wise Patching**
   - Patch activations from correct → incorrect examples
   - Measure which layers are causally important for the output

2. **Attention Head Analysis**
   - Identify which attention heads attend to numerical tokens
   - Find "comparison" heads that might be responsible for failures

3. **MLP Patching**
   - Test whether MLP layers perform numerical computation
   - Identify failure points in the feedforward network

4. **Token-wise Analysis**
   - Which token positions are most important?
   - Does the model focus on the right numbers?

### Hypothesis

Based on probing results, we hypothesize that:
- **Early/middle layers**: Correctly encode numerical features
- **Late attention heads**: Fail to properly compare encoded values
- **Output MLP/unembedding**: May not map comparison results to correct tokens

---

## Repository Structure

```
Housing/
├── README.md                          # This file
├── data/
│   ├── pairs_20pct_price_diff.csv    # Processed dataset
│   ├── activations/                   # Extracted activations (.npz)
│   ├── probe_results/                 # Linear probing results
│   │   ├── plots/                     # Generated visualizations
│   │   ├── probe_results_*.csv        # Detailed results
│   │   ├── probe_matrix_*.csv         # Accuracy/AUC matrices
│   │   └── probe_best_layers_*.csv    # Peak layer per feature
│   ├── logit_results/                 # Price predictor results
│   │   ├── plots/                     # Accuracy curves, comparisons
│   │   ├── logit_results_*.csv        # Per-layer accuracy
│   │   └── logit_summary_*.json       # Best layer summary
│   └── results/                       # Model evaluation results
├── linear_probing/
│   ├── lp_config.py                   # Configuration
│   ├── lp_utils.py                    # Utility functions
│   ├── extract_activations.py        # Activation extraction
│   ├── probe.py                       # Main probing script
│   └── analyze_lp_results.py         # Visualization generation
├── logit_model/
│   ├── logit_config.py               # Configuration
│   ├── train_price_probe.py          # Train price predictor (cuML GPU)
│   └── analyze_results.py            # Generate plots
├── models/                            # Local model weights
├── logs/                              # SLURM job logs
├── run_linear_probing.sh             # Linear probing SLURM script
├── run_logit_model.sh                # Price predictor SLURM script
└── prompts.py                         # Prompt strategies
```

---

## Reproducibility

### Requirements
- Python 3.10+
- PyTorch 2.0+
- HuggingFace Transformers
- TransformerLens (for Llama)
- cuML/RAPIDS (for GPU-accelerated probing)
- scikit-learn, numpy, pandas, matplotlib, seaborn

### Running the Pipeline

```bash
# 1. Extract activations (requires GPU, ~2 hours)
sbatch run_extract_activations.sh

# 2. Run linear probing (requires GPU, ~35 minutes with cuML)
sbatch run_linear_probing.sh

# 3. Run price predictor (requires GPU, ~4 minutes with cuML)
sbatch run_logit_model.sh

# 4. Generate visualizations
python linear_probing/analyze_lp_results.py
python logit_model/analyze_results.py
```

---

## Acknowledgments

- TransformerLens library for activation extraction
- RAPIDS/cuML for GPU-accelerated machine learning
- HuggingFace for model access
- CMU computing resources

---

*Last updated: February 3, 2026*
