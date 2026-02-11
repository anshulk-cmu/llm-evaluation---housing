#!/bin/bash
#SBATCH --job-name=macro_mapping
#SBATCH --output=/home/anshulk/Housing/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/Housing/logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# SLURM Job Script for Behavioral Macro Mapping (Dual-GPU)
# ============================================================================
# Uses 2x A6000 GPUs with 50 samples per GPU (same pattern as original experiment).
#
# Steps 1, 3, 5-10: CPU-only (fast)
# Steps 2, 4: GPU inference on both GPUs simultaneously
#
# Usage:
#   sbatch run_macro_mapping.sh                        # run both models
#   sbatch run_macro_mapping.sh llama-3.2-3b           # run one model
#   sbatch run_macro_mapping.sh qwen3-4b               # run one model
# ============================================================================

echo "============================================================"
echo "Behavioral Macro Mapping Pipeline (Dual-GPU)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "GPU Mode: Dual-GPU (2x A6000, 50 samples/GPU)"
echo "============================================================"

# ============================================================================
# SETUP
# ============================================================================

cd /home/anshulk/Housing || { echo "Failed to cd to /home/anshulk/Housing"; exit 1; }

mkdir -p logs macro_mapping/data macro_mapping/outputs/graphs macro_mapping/outputs/tables

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate housing || { echo "Failed to activate housing environment"; exit 1; }

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo ""
echo "Running pre-flight checks..."

# Check GPUs
echo "  Checking GPUs..."
if ! nvidia-smi &> /dev/null; then
    echo "  ERROR: nvidia-smi not available"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  Found $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "  WARNING: Only $GPU_COUNT GPU(s) found. Will use single-GPU fallback."
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Check Python packages
echo "  Checking Python packages..."
REQUIRED_PACKAGES="pandas numpy torch transformers tqdm scipy"
for pkg in $REQUIRED_PACKAGES; do
    if ! python -c "import $pkg" 2>/dev/null; then
        echo "  ERROR: Python package '$pkg' not found"
        exit 1
    fi
done
echo "  All required packages found"

# Check data
echo "  Checking data file..."
if [ ! -f "data/pairs_20pct_price_diff.csv" ]; then
    echo "  ERROR: data/pairs_20pct_price_diff.csv not found"
    exit 1
fi
echo "  Data file found"

# Check models
echo "  Checking models..."
if [ ! -d "models/Llama-3.2-3B-Instruct" ]; then
    echo "  ERROR: models/Llama-3.2-3B-Instruct not found"
    exit 1
fi
if [ ! -d "models/Qwen3-4B-Instruct-2507" ]; then
    echo "  ERROR: models/Qwen3-4B-Instruct-2507 not found"
    exit 1
fi
echo "  All models found"

echo ""
echo "Pre-flight checks passed!"
echo ""

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ============================================================================
# PIPELINE FUNCTION
# ============================================================================

run_pipeline() {
    local model=$1
    echo ""
    echo "============================================================"
    echo "Running macro mapping pipeline for: $model"
    echo "============================================================"

    # Step 1: Select 100 samples (CPU)
    echo ""
    echo "[Step 1/10] Selecting 100 stratified samples..."
    python macro_mapping/select_samples.py --model $model \
        || { echo "FAILED at Step 1 for $model"; return 1; }

    # Step 2: Extract CoT reasoning (DUAL-GPU, 2048 tokens)
    echo ""
    echo "[Step 2/10] Extracting CoT reasoning (dual-GPU, 2048 tokens)..."
    python macro_mapping/extract_cot.py --model $model \
        || { echo "FAILED at Step 2 for $model"; return 1; }

    # Step 3: Setup perturbation plan (CPU)
    echo ""
    echo "[Step 3/10] Setting up perturbation plan..."
    python macro_mapping/setup_perturbations.py --model $model \
        || { echo "FAILED at Step 3 for $model"; return 1; }

    # Step 4: Run perturbation queries (DUAL-GPU, ~1.5 hrs)
    echo ""
    echo "[Step 4/10] Running perturbation queries (dual-GPU, ~1.5 hrs)..."
    python macro_mapping/run_perturbations.py --model $model \
        || { echo "FAILED at Step 4 for $model"; return 1; }

    # Step 5: Map reasoning stages (CPU)
    echo ""
    echo "[Step 5/10] Mapping reasoning stages..."
    python macro_mapping/map_reasoning_stages.py --model $model \
        || { echo "FAILED at Step 5 for $model"; return 1; }

    # Step 6: Compute exchange rates (CPU)
    echo ""
    echo "[Step 6/10] Computing feature exchange rates..."
    python macro_mapping/compute_exchange_rates.py --model $model \
        || { echo "FAILED at Step 6 for $model"; return 1; }

    # Step 7: Build reasoning graph (CPU)
    echo ""
    echo "[Step 7/10] Building reasoning graph..."
    python macro_mapping/build_reasoning_graph.py --model $model \
        || { echo "FAILED at Step 7 for $model"; return 1; }

    # Step 8: Decision boundary map (CPU)
    echo ""
    echo "[Step 8/10] Building decision boundary map..."
    python macro_mapping/decision_boundary.py --model $model \
        || { echo "FAILED at Step 8 for $model"; return 1; }

    # Step 9: Validate with regression (CPU)
    echo ""
    echo "[Step 9/10] Validating against logistic regression..."
    python macro_mapping/validate_regression.py --model $model \
        || { echo "FAILED at Step 9 for $model"; return 1; }

    # Step 10: Final graph (CPU)
    echo ""
    echo "[Step 10/10] Producing final reasoning graph..."
    python macro_mapping/produce_final_graph.py --model $model \
        || { echo "FAILED at Step 10 for $model"; return 1; }

    echo ""
    echo "Pipeline complete for $model"
}

# ============================================================================
# RUN
# ============================================================================

MODEL=${1:-"both"}
EXIT_CODE=0

if [ "$MODEL" = "both" ]; then
    run_pipeline "llama-3.2-3b" || EXIT_CODE=1
    run_pipeline "qwen3-4b" || EXIT_CODE=1
else
    run_pipeline "$MODEL" || EXIT_CODE=1
fi

# ============================================================================
# CLEANUP & SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "Job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "============================================================"

echo ""
echo "Final GPU memory state:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Macro mapping pipeline complete!"
    echo ""
    echo "Outputs:"
    echo "  - macro_mapping/data/           (intermediate CSVs)"
    echo "  - macro_mapping/outputs/graphs/ (reasoning graphs)"
    echo "  - macro_mapping/outputs/tables/ (summary tables)"
    if [ -d "macro_mapping/outputs" ]; then
        echo ""
        echo "Generated files:"
        find macro_mapping/outputs -type f | sort
    fi
else
    echo ""
    echo "FAILURE: Pipeline failed with exit code $EXIT_CODE"
    echo "Check logs: logs/slurm-${SLURM_JOB_ID}.err"
fi

echo "============================================================"
echo "Job Ended: $(date)"
echo "============================================================"

exit $EXIT_CODE
