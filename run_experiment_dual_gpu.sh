#!/bin/bash
#SBATCH --job-name=housing_dual_gpu
#SBATCH --output=/home/anshulk/Housing/logs/slurm-%j.out
#SBATCH --error=/home/anshulk/Housing/logs/slurm-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# SLURM Job Script for Dual-GPU LLM Real Estate Valuation Experiment
# ============================================================================
# This script runs the experiment using BOTH GPUs simultaneously for each model.
# Each GPU processes separate batches of 50 samples in parallel.
# Effective throughput: 100 samples per iteration (vs 50 in single-GPU mode)
# 
# ============================================================================

echo "============================================================"
echo "DUAL-GPU LLM Real Estate Valuation Experiment"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "GPU Mode: Dual-GPU (parallel batches)"
echo "============================================================"

# ============================================================================
# SETUP
# ============================================================================

# Navigate to project directory
cd /home/anshulk/Housing || { echo "Failed to cd to /home/anshulk/Housing"; exit 1; }

# Create logs directory if it doesn't exist
mkdir -p logs data/results data/checkpoints

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate housing || { echo "Failed to activate housing environment"; exit 1; }

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo ""
echo "Running pre-flight checks..."

# Check GPU availability
echo "  Checking GPUs..."
if ! nvidia-smi &> /dev/null; then
    echo "  ERROR: nvidia-smi not available"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  Found $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "  ERROR: Dual-GPU mode requires 2 GPUs, found $GPU_COUNT"
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Check conda environment
echo "  Checking conda environment..."
if ! conda list | grep -q "pandas"; then
    echo "  ERROR: pandas not found in environment"
    exit 1
fi

# Check required Python packages
echo "  Checking Python packages..."
REQUIRED_PACKAGES="pandas numpy torch transformers tqdm"
for pkg in $REQUIRED_PACKAGES; do
    if ! python -c "import $pkg" 2>/dev/null; then
        echo "  ERROR: Python package '$pkg' not found"
        exit 1
    fi
done
echo "  All required packages found"

# Check data file
echo "  Checking data file..."
if [ ! -f "data/pairs_20pct_price_diff.csv" ]; then
    echo "  ERROR: data/pairs_20pct_price_diff.csv not found"
    exit 1
fi
echo "  Data file found"

# Check models directory
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

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ============================================================================
# RUN EXPERIMENT
# ============================================================================

echo "Starting dual-GPU experiment..."
echo ""

# Run with --dual-gpu flag
python main.py --dual-gpu

EXIT_CODE=$?

# ============================================================================
# CLEANUP & SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "Job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

# Show GPU memory usage at end
echo ""
echo "Final GPU memory state:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# Print summary of results
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Experiment completed successfully!"
    echo ""
    echo "Results saved to:"
    echo "  - data/results/results_*.csv"
    echo "  - data/results/experiment_summary_*.csv"
    echo ""
    echo "Log file:"
    echo "  - logs/experiment_*.log"
    echo ""
    
    # Show result files
    if [ -d "data/results" ]; then
        echo "Generated files:"
        ls -lh data/results/
    fi
else
    echo ""
    echo "FAILURE: Experiment failed with exit code $EXIT_CODE"
    echo ""
    echo "Check logs for details:"
    echo "  - logs/slurm-${SLURM_JOB_ID}.err"
    echo "  - logs/experiment_*.log"
    echo ""
    
    # Show checkpoint files if they exist
    if [ -d "data/checkpoints" ] && [ "$(ls -A data/checkpoints 2>/dev/null)" ]; then
        echo "Checkpoints saved (you can resume):"
        ls -lh data/checkpoints/
    fi
fi

echo "============================================================"
echo "Job Ended: $(date)"
echo "Total Runtime: $SECONDS seconds"
echo "============================================================"

exit $EXIT_CODE
