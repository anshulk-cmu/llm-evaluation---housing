#!/bin/bash
#SBATCH --job-name=logit_price
#SBATCH --output=/home/anshulk/Housing/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/Housing/logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# SLURM Job Script for Logistic Regression Price Predictor
# ============================================================================
# Trains logistic regression on LLM activations to predict which property
# costs more. Tests if a simple classifier can outperform the LLM's output.
#
# Probes all layers for price prediction only:
#   - Llama-3.2-3B: 28 layers
#   - Qwen3-4B: 36 layers
#
# Uses 70/10/20 train/val/test split with statistical guarantees
# GPU acceleration via cuML (RAPIDS)
# ============================================================================

echo "============================================================"
echo "Logistic Regression Price Predictor on LLM Activations"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "============================================================"

# ============================================================================
# SETUP
# ============================================================================

cd /home/anshulk/Housing || { echo "Failed to cd to /home/anshulk/Housing"; exit 1; }

# Create directories
mkdir -p logs data/logit_results data/logit_results/logs data/logit_results/plots

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate housing || { echo "Failed to activate housing environment"; exit 1; }

# Configuration
CONDITION="2_fewshot_cot_temp0"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo ""
echo "Running pre-flight checks..."

# Check for activations
echo "  Checking for activation files..."
LLAMA_ACT="data/activations/llama-3.2-3b_${CONDITION}_activations.npz"
QWEN_ACT="data/activations/qwen3-4b_${CONDITION}_activations.npz"

LLAMA_READY=false
QWEN_READY=false

if [ -f "$LLAMA_ACT" ]; then
    LLAMA_SIZE=$(du -h "$LLAMA_ACT" | cut -f1)
    echo "    Llama activations found: $LLAMA_SIZE"
    LLAMA_READY=true
else
    echo "    WARNING: Llama activations not found at $LLAMA_ACT"
fi

if [ -f "$QWEN_ACT" ]; then
    QWEN_SIZE=$(du -h "$QWEN_ACT" | cut -f1)
    echo "    Qwen activations found: $QWEN_SIZE"
    QWEN_READY=true
else
    echo "    WARNING: Qwen activations not found at $QWEN_ACT"
fi

# Check Python packages
echo "  Checking Python packages..."
REQUIRED_PACKAGES="pandas numpy scikit-learn scipy tqdm matplotlib"
for pkg in $REQUIRED_PACKAGES; do
    if ! python -c "import $pkg" 2>/dev/null; then
        echo "  Installing missing package: $pkg"
        pip install $pkg --quiet
    fi
done
echo "  All required packages available"

# Check cuML for GPU acceleration
echo "  Checking GPU acceleration (cuML)..."
if python -c "import cuml" 2>/dev/null; then
    echo "    cuML already installed - GPU acceleration ENABLED"
else
    echo "    Installing cuML (RAPIDS) for GPU acceleration..."
    conda install -c rapidsai -c conda-forge -c nvidia \
        cuml=24.12 cupy cuda-version=12.0 \
        --yes --quiet 2>&1 | tail -5

    if python -c "import cuml" 2>/dev/null; then
        echo "    cuML installed successfully - GPU acceleration ENABLED"
    else
        echo "    WARNING: cuML installation failed - falling back to CPU"
    fi
fi

echo ""
echo "Pre-flight checks passed!"
echo ""

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ============================================================================
# RUN LOGIT MODEL - LLAMA
# ============================================================================

cd /home/anshulk/Housing

LLAMA_EXIT=0
QWEN_EXIT=0

# Check GPU status
if python -c "import cuml" 2>/dev/null; then
    GPU_STATUS="ENABLED (cuML)"
else
    GPU_STATUS="DISABLED (CPU sklearn)"
fi

if [ "$LLAMA_READY" = true ]; then
    echo "============================================================"
    echo "Training Price Predictor on Llama-3.2-3B Activations"
    echo "Condition: $CONDITION"
    echo "Split: 70/10/20 (train/val/test)"
    echo "Target: price_p1_higher (binary)"
    echo "Layers: 28"
    echo "GPU acceleration: $GPU_STATUS"
    echo "============================================================"

    python logit_model/train_price_probe.py \
        --model llama-3.2-3b \
        --condition $CONDITION

    LLAMA_EXIT=$?

    if [ $LLAMA_EXIT -eq 0 ]; then
        echo "SUCCESS: Llama price predictor completed"
    else
        echo "ERROR: Llama price predictor failed with exit code $LLAMA_EXIT"
    fi
else
    echo ""
    echo "SKIPPING Llama - activations not found"
fi

# ============================================================================
# RUN LOGIT MODEL - QWEN
# ============================================================================

if [ "$QWEN_READY" = true ]; then
    echo ""
    echo "============================================================"
    echo "Training Price Predictor on Qwen3-4B Activations"
    echo "Condition: $CONDITION"
    echo "Split: 70/10/20 (train/val/test)"
    echo "Target: price_p1_higher (binary)"
    echo "Layers: 36"
    echo "GPU acceleration: $GPU_STATUS"
    echo "============================================================"

    python logit_model/train_price_probe.py \
        --model qwen3-4b \
        --condition $CONDITION

    QWEN_EXIT=$?

    if [ $QWEN_EXIT -eq 0 ]; then
        echo "SUCCESS: Qwen price predictor completed"
    else
        echo "ERROR: Qwen price predictor failed with exit code $QWEN_EXIT"
    fi
else
    echo ""
    echo "SKIPPING Qwen - activations not found"
fi

# ============================================================================
# GENERATE PLOTS
# ============================================================================

echo ""
echo "============================================================"
echo "Generating Plots"
echo "============================================================"

python logit_model/analyze_results.py --condition $CONDITION

# ============================================================================
# VALIDATION
# ============================================================================

echo ""
echo "============================================================"
echo "Validating Results"
echo "============================================================"

python -c "
import pandas as pd
import json
from pathlib import Path

results_dir = Path('data/logit_results')

# Check Llama results
llama_csv = results_dir / 'logit_results_llama-3.2-3b_${CONDITION}.csv'
if llama_csv.exists():
    df = pd.read_csv(llama_csv)
    best = df.loc[df['test_accuracy'].idxmax()]
    print(f'✓ Llama results: {len(df)} layers probed')
    print(f'  Best layer: {int(best[\"layer\"])}')
    print(f'  Best accuracy: {best[\"test_accuracy\"]:.4f}')
    print(f'  95% CI: [{best[\"ci_lower\"]:.4f}, {best[\"ci_upper\"]:.4f}]')
else:
    print(f'✗ Llama results NOT found')

print()

# Check Qwen results
qwen_csv = results_dir / 'logit_results_qwen3-4b_${CONDITION}.csv'
if qwen_csv.exists():
    df = pd.read_csv(qwen_csv)
    best = df.loc[df['test_accuracy'].idxmax()]
    print(f'✓ Qwen results: {len(df)} layers probed')
    print(f'  Best layer: {int(best[\"layer\"])}')
    print(f'  Best accuracy: {best[\"test_accuracy\"]:.4f}')
    print(f'  95% CI: [{best[\"ci_lower\"]:.4f}, {best[\"ci_upper\"]:.4f}]')
else:
    print(f'✗ Qwen results NOT found')

print()
print('Baselines for comparison:')
print('  Chance: 50.0%')
print('  LLM Output: ~60%')
print('  Zestimate: 77.6%')
" 2>/dev/null || echo "Validation script failed"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "Logit Model Training Complete"
echo "============================================================"
echo "End time: $(date)"
echo "Llama exit code: $LLAMA_EXIT"
echo "Qwen exit code: $QWEN_EXIT"
echo ""
echo "Output files saved to: data/logit_results/"
echo ""
echo "CSV files:"
ls -lh data/logit_results/*.csv 2>/dev/null || echo "  No CSV files found"
echo ""
echo "Plots:"
ls -lh data/logit_results/plots/*.png 2>/dev/null || echo "  No plots found"
echo ""
echo "Logs:"
ls -lh data/logit_results/logs/*.log 2>/dev/null | head -5 || echo "  No log files found"
echo ""
echo "============================================================"
echo "Job Ended: $(date)"
echo "Total Runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo "============================================================"

# Return success if both succeeded (or were skipped)
if [ "$LLAMA_READY" = false ] && [ "$QWEN_READY" = false ]; then
    echo "✗ No activations found - nothing to train"
    exit 1
elif [ $LLAMA_EXIT -eq 0 ] && [ $QWEN_EXIT -eq 0 ]; then
    echo "✓ All training completed successfully!"
    exit 0
elif [ $LLAMA_EXIT -eq 0 ] || [ $QWEN_EXIT -eq 0 ]; then
    echo "⚠ Partial success (one model failed)"
    exit 1
else
    echo "✗ Both models failed"
    exit 1
fi
