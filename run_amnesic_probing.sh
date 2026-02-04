#!/bin/bash
#SBATCH --job-name=amnesic_probe
#SBATCH --output=/home/anshulk/Housing/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/Housing/logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# SLURM Job Script for Amnesic Probing
# ============================================================================
# This script runs amnesic probing to test whether the model USES the 
# features it encodes for price prediction (causal vs correlational).
#
# Based on:
#   - Elazar et al. (2021) "Amnesic Probing" (TACL)
#   - Ravfogel et al. (2020) "Null It Out: INLP" (ACL)
#
# Process:
#   1. INLP to erase each feature (bedrooms, bathrooms, year, lot, sqft)
#   2. Price prediction evaluation under each erasure condition
#   3. Analysis by prediction correctness (correct vs incorrect)
#   4. Random control to validate methodology
#
# GPU acceleration via cuML (RAPIDS) - uses 2x A6000 for parallel INLP
# Expected runtime: 2-4 hours with cuML
# ============================================================================

echo "============================================================"
echo "Amnesic Probing for Causal Feature Analysis"
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
mkdir -p logs data/amnesic_results data/amnesic_samples

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate housing || { echo "Failed to activate housing environment"; exit 1; }

# Configuration
CONDITION="2_fewshot_cot_temp0"
LLAMA_LAYER=15  # Best probe layer from linear probing results
QWEN_LAYER=19   # Best probe layer from linear probing results

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

# Check for selected samples
echo "  Checking for sample selection..."
LLAMA_SAMPLES="data/amnesic_samples/selected_samples_llama-3.2-3b_${CONDITION}.csv"
QWEN_SAMPLES="data/amnesic_samples/selected_samples_qwen3-4b_${CONDITION}.csv"

if [ -f "$LLAMA_SAMPLES" ]; then
    SAMPLE_COUNT=$(wc -l < "$LLAMA_SAMPLES")
    echo "    Llama sample selection found: $((SAMPLE_COUNT - 1)) samples"
else
    echo "    Running Llama sample selection..."
    python amnesic_probing/select_samples.py --model llama-3.2-3b
fi

if [ -f "$QWEN_SAMPLES" ]; then
    SAMPLE_COUNT=$(wc -l < "$QWEN_SAMPLES")
    echo "    Qwen sample selection found: $((SAMPLE_COUNT - 1)) samples"
else
    echo "    Running Qwen sample selection..."
    python amnesic_probing/select_samples.py --model qwen3-4b
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

# Check and verify cuML for GPU acceleration
echo "  Checking GPU acceleration (cuML)..."
if python -c "import cuml" 2>/dev/null; then
    echo "    cuML already installed - GPU acceleration ENABLED"
    GPU_STATUS="ENABLED (cuML)"
else
    echo "    WARNING: cuML not available - falling back to CPU"
    echo "    INLP will still work but will be slower (~10x)"
    GPU_STATUS="DISABLED (CPU sklearn)"
fi

# Check GPU availability
echo "  Checking GPU devices..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "    nvidia-smi not available"

echo ""
echo "Pre-flight checks passed!"
echo ""

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1

# ============================================================================
# RUN AMNESIC PROBING - LLAMA
# ============================================================================

cd /home/anshulk/Housing

LLAMA_EXIT=0
QWEN_EXIT=0

if [ "$LLAMA_READY" = true ]; then
    echo "============================================================"
    echo "Amnesic Probing: Llama-3.2-3B-Instruct"
    echo "============================================================"
    echo "Condition: $CONDITION"
    echo "Layer: $LLAMA_LAYER (best probe layer)"
    echo "Samples: 5,130 (all)"
    echo "Features: bedrooms, bathrooms, year_built, lot_size, sqft"
    echo "INLP iterations: 300 per feature"
    echo "GPU acceleration: $GPU_STATUS"
    echo "============================================================"
    echo ""

    python amnesic_probing/run_amnesic_experiment.py \
        --model llama-3.2-3b \
        --layer $LLAMA_LAYER \
        --gpu 0 \
        --features all

    LLAMA_EXIT=$?

    if [ $LLAMA_EXIT -eq 0 ]; then
        echo ""
        echo "SUCCESS: Llama amnesic probing completed"
    else
        echo ""
        echo "ERROR: Llama amnesic probing failed with exit code $LLAMA_EXIT"
    fi
else
    echo ""
    echo "SKIPPING Llama amnesic probing - activations not found"
fi

# ============================================================================
# RUN AMNESIC PROBING - QWEN (optional, if time permits)
# ============================================================================

if [ "$QWEN_READY" = true ]; then
    echo ""
    echo "============================================================"
    echo "Amnesic Probing: Qwen3-4B-Instruct"
    echo "============================================================"
    echo "Condition: $CONDITION"
    echo "Layer: $QWEN_LAYER (best probe layer)"
    echo "Samples: 5,130 (all)"
    echo "Features: bedrooms, bathrooms, year_built, lot_size, sqft"
    echo "INLP iterations: 300 per feature"
    echo "GPU acceleration: $GPU_STATUS"
    echo "============================================================"
    echo ""

    python amnesic_probing/run_amnesic_experiment.py \
        --model qwen3-4b \
        --layer $QWEN_LAYER \
        --gpu 1 \
        --features all

    QWEN_EXIT=$?

    if [ $QWEN_EXIT -eq 0 ]; then
        echo ""
        echo "SUCCESS: Qwen amnesic probing completed"
    else
        echo ""
        echo "ERROR: Qwen amnesic probing failed with exit code $QWEN_EXIT"
    fi
else
    echo ""
    echo "SKIPPING Qwen amnesic probing - activations not found"
fi

# ============================================================================
# VALIDATION
# ============================================================================

echo ""
echo "============================================================"
echo "Validating Amnesic Probing Results"
echo "============================================================"

python -c "
import pandas as pd
import json
from pathlib import Path

results_dir = Path('data/amnesic_results')

# Check Llama results
llama_csv = results_dir / 'amnesic_results_llama-3.2-3b_${CONDITION}.csv'
llama_summary = results_dir / 'amnesic_summary_llama-3.2-3b_${CONDITION}.json'

if llama_csv.exists():
    df = pd.read_csv(llama_csv)
    print(f'✓ Llama results: {len(df)} conditions tested')
    
    baseline = df[df['condition'] == 'baseline']['accuracy'].values[0]
    print(f'  Baseline accuracy: {baseline:.4f}')
    
    # Feature importance
    print(f'  Feature importance (by accuracy drop):')
    for _, row in df.iterrows():
        if row['condition'].startswith('erase_') and row['condition'] != 'erase_all_features':
            feat = row['condition'].replace('erase_', '')
            print(f'    {feat}: {row[\"accuracy_drop\"]:.4f} drop')
    
    # Control check
    random_row = df[df['condition'] == 'random_control']
    if len(random_row) > 0:
        print(f'  Random control drop: {random_row[\"accuracy_drop\"].values[0]:.4f}')
else:
    print(f'✗ Llama results NOT found')

print()

# Check Qwen results
qwen_csv = results_dir / 'amnesic_results_qwen3-4b_${CONDITION}.csv'
if qwen_csv.exists():
    df = pd.read_csv(qwen_csv)
    print(f'✓ Qwen results: {len(df)} conditions tested')
    baseline = df[df['condition'] == 'baseline']['accuracy'].values[0]
    print(f'  Baseline accuracy: {baseline:.4f}')
else:
    print(f'✗ Qwen results NOT found')
" 2>/dev/null || echo "Validation script failed"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "Amnesic Probing Complete"
echo "============================================================"
echo "End time: $(date)"
echo "Llama exit code: $LLAMA_EXIT"
echo "Qwen exit code: $QWEN_EXIT"
echo ""
echo "Output files saved to: data/amnesic_results/"
echo ""
echo "Key output files:"
ls -lh data/amnesic_results/*.csv 2>/dev/null || echo "  No CSV files found"
echo ""
echo "Projection matrices (for future use):"
ls -lh data/amnesic_results/P_*.npy 2>/dev/null || echo "  No projection matrices found"
echo ""
echo "Summary files:"
ls -lh data/amnesic_results/*summary*.json 2>/dev/null || echo "  No summary files found"
echo ""
echo "============================================================"
echo "Job Ended: $(date)"
echo "Total Runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo "============================================================"

# Return success if both succeeded (or were skipped)
if [ "$LLAMA_READY" = false ] && [ "$QWEN_READY" = false ]; then
    echo "✗ No activations found - nothing to process"
    exit 1
elif [ $LLAMA_EXIT -eq 0 ] && [ $QWEN_EXIT -eq 0 ]; then
    echo "✓ All amnesic probing completed successfully!"
    exit 0
elif [ $LLAMA_EXIT -eq 0 ] || [ $QWEN_EXIT -eq 0 ]; then
    echo "⚠ Partial success (one model failed)"
    exit 1
else
    echo "✗ Both amnesic probing jobs failed"
    exit 1
fi
