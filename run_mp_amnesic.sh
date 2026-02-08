#!/bin/bash
#SBATCH --job-name=mp_amnesic
#SBATCH --output=/home/anshulk/Housing/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/Housing/logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# Amnesic Probing with MP (Mean Projection) — ALL LAYERS
# ============================================================================
# 3-step causal framework (Dobrzeniecka et al., 2025):
#   Step 1: Erasure + Verification (rank-1 concept removal)
#   Step 2: Information Control (target vs random erasure comparison)
#   Step 3: Selectivity Control (gold label recovery test)
#
# MP removes exactly 1 direction per binary feature (vs INLP's 50-300).
# Closed-form computation — no iteration needed.
# Runs all layers for both models to find optimal layer.
# ============================================================================

echo "============================================================"
echo "Amnesic Probing with MP (Mean Projection) — ALL LAYERS"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================================"

cd /home/anshulk/Housing || { echo "Failed to cd"; exit 1; }
mkdir -p logs data/mp_results

# Activate environment
eval "$(conda shell.bash hook)"
conda activate housing || { echo "Failed to activate housing env"; exit 1; }

# Configuration
CONDITION="2_fewshot_cot_temp0"
LLAMA_N_LAYERS=28   # Llama-3.2-3B: (5130, 28, 3072)
QWEN_N_LAYERS=36    # Qwen3-4B: (5130, 36, 2560)

# ============================================================================
# DEPENDENCY CHECK
# ============================================================================

echo ""
echo "Checking dependencies..."

# Check cuML
if python -c "import cuml" 2>/dev/null; then
    echo "  cuML: INSTALLED (GPU probing enabled)"
else
    echo "  cuML: NOT FOUND (CPU sklearn fallback)"
fi

# Check activations
LLAMA_ACT="data/activations/llama-3.2-3b_${CONDITION}_activations.npz"
QWEN_ACT="data/activations/qwen3-4b_${CONDITION}_activations.npz"

LLAMA_READY=false
QWEN_READY=false

[ -f "$LLAMA_ACT" ] && { echo "  Llama activations: FOUND"; LLAMA_READY=true; } || echo "  Llama activations: NOT FOUND"
[ -f "$QWEN_ACT" ] && { echo "  Qwen activations: FOUND"; QWEN_READY=true; } || echo "  Qwen activations: NOT FOUND"

# Check samples (run selection if needed)
LLAMA_SAMPLES="data/amnesic_samples/selected_samples_llama-3.2-3b_${CONDITION}.csv"
QWEN_SAMPLES="data/amnesic_samples/selected_samples_qwen3-4b_${CONDITION}.csv"

[ -f "$LLAMA_SAMPLES" ] || { echo "  Running Llama sample selection..."; python amnesic_probing/select_samples.py --model llama-3.2-3b; }
[ -f "$QWEN_SAMPLES" ] || { echo "  Running Qwen sample selection..."; python amnesic_probing/select_samples.py --model qwen3-4b; }

echo ""

# ============================================================================
# RUN EXPERIMENTS — ALL LAYERS
# ============================================================================

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

LLAMA_EXIT=0
QWEN_EXIT=0

# --- Llama: all 28 layers ---
if [ "$LLAMA_READY" = true ]; then
    echo "============================================================"
    echo "MP Amnesic Probing: Llama-3.2-3B (ALL $LLAMA_N_LAYERS LAYERS)"
    echo "  Method: MP | Features: all | Random controls: 10"
    echo "============================================================"

    for LAYER in $(seq 0 $((LLAMA_N_LAYERS - 1))); do
        echo ""
        echo ">>> Llama layer $LAYER / $((LLAMA_N_LAYERS - 1))"
        python amnesic_probing/run_mp_experiment.py \
            --model llama-3.2-3b \
            --layer $LAYER \
            --gpu 0 \
            --method mp \
            --features all \
            --n-random-controls 10

        if [ $? -ne 0 ]; then
            echo "ERROR: Llama layer $LAYER failed"
            LLAMA_EXIT=1
        fi
    done
    echo ""
    echo "Llama complete (exit: $LLAMA_EXIT)"
fi

# --- Qwen: all 36 layers ---
if [ "$QWEN_READY" = true ]; then
    echo ""
    echo "============================================================"
    echo "MP Amnesic Probing: Qwen3-4B (ALL $QWEN_N_LAYERS LAYERS)"
    echo "  Method: MP | Features: all | Random controls: 10"
    echo "============================================================"

    for LAYER in $(seq 0 $((QWEN_N_LAYERS - 1))); do
        echo ""
        echo ">>> Qwen layer $LAYER / $((QWEN_N_LAYERS - 1))"
        python amnesic_probing/run_mp_experiment.py \
            --model qwen3-4b \
            --layer $LAYER \
            --gpu 0 \
            --method mp \
            --features all \
            --n-random-controls 10

        if [ $? -ne 0 ]; then
            echo "ERROR: Qwen layer $LAYER failed"
            QWEN_EXIT=1
        fi
    done
    echo ""
    echo "Qwen complete (exit: $QWEN_EXIT)"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo "Runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo ""
echo "Results:"
ls -lh data/mp_results/*.csv 2>/dev/null | wc -l
echo " CSV files generated"
echo ""
echo "Summaries:"
ls -lh data/mp_results/*.json 2>/dev/null | wc -l
echo " JSON files generated"
echo "============================================================"

# ============================================================================
# GENERATE PLOTS
# ============================================================================

echo ""
echo "Generating plots..."
python amnesic_probing/plot_mp_results.py \
    --results-dir data/mp_results \
    --output-dir data/mp_results/plots \
    && echo "Plots generated successfully" \
    || echo "WARNING: Plot generation failed (non-fatal)"

# Exit status
if [ "$LLAMA_READY" = false ] && [ "$QWEN_READY" = false ]; then
    echo "No activations found — nothing to process"
    exit 1
elif [ $LLAMA_EXIT -eq 0 ] && [ $QWEN_EXIT -eq 0 ]; then
    exit 0
else
    exit 1
fi
