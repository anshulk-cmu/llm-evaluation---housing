#!/bin/bash
#SBATCH --job-name=linear_probe
#SBATCH --output=/home/anshulk/Housing/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/Housing/logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# SLURM Job Script for Linear Probing
# ============================================================================
# This script runs linear probing on extracted activations to analyze
# what features are encoded in each layer of the model.
#
# Probes for 6 features across all layers:
#   - bathrooms_p1_more, bedrooms_p1_more, sqft_p1_larger
#   - lot_p1_larger, year_p1_newer, price_p1_higher (ground truth)
#
# Uses 70/10/20 train/val/test split with statistical guarantees
# GPU acceleration via cuML (RAPIDS) - utilizes ~70% of A6000 48GB
# ============================================================================

echo "============================================================"
echo "Linear Probing for Mechanistic Interpretability"
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
mkdir -p logs data/probe_results data/probe_results/logs

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
REQUIRED_PACKAGES="pandas numpy scikit-learn scipy tqdm"
for pkg in $REQUIRED_PACKAGES; do
    if ! python -c "import $pkg" 2>/dev/null; then
        echo "  Installing missing package: $pkg"
        pip install $pkg --quiet
    fi
done
echo "  All required packages available"

# Check and install cuML for GPU acceleration
echo "  Checking GPU acceleration (cuML)..."
if python -c "import cuml" 2>/dev/null; then
    echo "    cuML already installed - GPU acceleration ENABLED"
else
    echo "    Installing cuML (RAPIDS) for GPU acceleration..."
    echo "    This may take a few minutes..."
    conda install -c rapidsai -c conda-forge -c nvidia \
        cuml=24.12 cupy cuda-version=12.0 \
        --yes --quiet 2>&1 | tail -5

    if python -c "import cuml" 2>/dev/null; then
        echo "    cuML installed successfully - GPU acceleration ENABLED"
    else
        echo "    WARNING: cuML installation failed - falling back to CPU"
        echo "    Probing will still work but will be slower"
    fi
fi

echo ""
echo "Pre-flight checks passed!"
echo ""

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ============================================================================
# RUN PROBING - LLAMA
# ============================================================================

cd /home/anshulk/Housing/linear_probing

LLAMA_EXIT=0
QWEN_EXIT=0

# Check GPU status for logging
if python -c "import cuml" 2>/dev/null; then
    GPU_STATUS="ENABLED (cuML)"
else
    GPU_STATUS="DISABLED (CPU sklearn)"
fi

if [ "$LLAMA_READY" = true ]; then
    echo "============================================================"
    echo "Probing Llama-3.2-3B-Instruct activations"
    echo "Condition: $CONDITION"
    echo "Split: 70/10/20 (train/val/test)"
    echo "Features: 6 binary features"
    echo "Layers: 28"
    echo "GPU acceleration: $GPU_STATUS"
    echo "============================================================"

    python probe.py \
        --model llama-3.2-3b \
        --condition $CONDITION

    LLAMA_EXIT=$?

    if [ $LLAMA_EXIT -eq 0 ]; then
        echo "SUCCESS: Llama probing completed"
    else
        echo "ERROR: Llama probing failed with exit code $LLAMA_EXIT"
    fi
else
    echo ""
    echo "SKIPPING Llama probing - activations not found"
fi

# ============================================================================
# RUN PROBING - QWEN
# ============================================================================

if [ "$QWEN_READY" = true ]; then
    echo ""
    echo "============================================================"
    echo "Probing Qwen3-4B-Instruct activations"
    echo "Condition: $CONDITION"
    echo "Split: 70/10/20 (train/val/test)"
    echo "Features: 6 binary features"
    echo "Layers: 36"
    echo "GPU acceleration: $GPU_STATUS"
    echo "============================================================"

    python probe.py \
        --model qwen3-4b \
        --condition $CONDITION

    QWEN_EXIT=$?

    if [ $QWEN_EXIT -eq 0 ]; then
        echo "SUCCESS: Qwen probing completed"
    else
        echo "ERROR: Qwen probing failed with exit code $QWEN_EXIT"
    fi
else
    echo ""
    echo "SKIPPING Qwen probing - activations not found"
fi

# ============================================================================
# VALIDATION
# ============================================================================

echo ""
echo "============================================================"
echo "Validating Probe Results"
echo "============================================================"

cd /home/anshulk/Housing

python -c "
import pandas as pd
from pathlib import Path

results_dir = Path('data/probe_results')

# Check Llama results
llama_csv = results_dir / 'probe_results_llama-3.2-3b_${CONDITION}.csv'
if llama_csv.exists():
    df = pd.read_csv(llama_csv)
    print(f'✓ Llama results: {len(df)} probes')
    print(f'  Best test accuracy: {df[\"test_accuracy\"].max():.3f}')
    print(f'  Significant probes: {df[\"is_significant\"].sum()}/{len(df)}')
    best = df.loc[df['test_accuracy'].idxmax()]
    print(f'  Best probe: {best[\"feature\"]} @ Layer {int(best[\"layer\"])}')
else:
    print(f'✗ Llama results NOT found')

print()

# Check Qwen results
qwen_csv = results_dir / 'probe_results_qwen3-4b_${CONDITION}.csv'
if qwen_csv.exists():
    df = pd.read_csv(qwen_csv)
    print(f'✓ Qwen results: {len(df)} probes')
    print(f'  Best test accuracy: {df[\"test_accuracy\"].max():.3f}')
    print(f'  Significant probes: {df[\"is_significant\"].sum()}/{len(df)}')
    best = df.loc[df['test_accuracy'].idxmax()]
    print(f'  Best probe: {best[\"feature\"]} @ Layer {int(best[\"layer\"])}')
else:
    print(f'✗ Qwen results NOT found')
" 2>/dev/null || echo "Validation script failed"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "Linear Probing Complete"
echo "============================================================"
echo "End time: $(date)"
echo "Llama exit code: $LLAMA_EXIT"
echo "Qwen exit code: $QWEN_EXIT"
echo ""
echo "Output files saved to: data/probe_results/"
echo ""
echo "CSV files for analysis:"
ls -lh data/probe_results/*.csv 2>/dev/null || echo "  No CSV files found"
echo ""
echo "Logs saved to: data/probe_results/logs/"
ls -lh data/probe_results/logs/*.log 2>/dev/null | head -5 || echo "  No log files found"
echo ""
echo "============================================================"
echo "Job Ended: $(date)"
echo "Total Runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo "============================================================"

# Return success if both succeeded (or were skipped)
if [ "$LLAMA_READY" = false ] && [ "$QWEN_READY" = false ]; then
    echo "✗ No activations found - nothing to probe"
    exit 1
elif [ $LLAMA_EXIT -eq 0 ] && [ $QWEN_EXIT -eq 0 ]; then
    echo "✓ All probing completed successfully!"
    exit 0
elif [ $LLAMA_EXIT -eq 0 ] || [ $QWEN_EXIT -eq 0 ]; then
    echo "⚠ Partial success (one probing failed)"
    exit 1
else
    echo "✗ Both probing jobs failed"
    exit 1
fi
