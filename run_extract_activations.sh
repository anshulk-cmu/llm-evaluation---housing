#!/bin/bash
#SBATCH --job-name=extract_activations
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
# SLURM Job Script for Activation Extraction (Linear Probing Pipeline)
# ============================================================================
# This script extracts residual stream activations from LLMs for linear probing.
#
# Extraction methods:
#   - Llama-3.2-3B-Instruct: TransformerLens (supported)
#   - Qwen3-4B-Instruct-2507: HuggingFace manual hooks (not in TransformerLens)
#
# Both use FEW-SHOT COT with GREEDY DECODING (temperature=0, deterministic)
#   - Llama accuracy: 59.3%
#   - Qwen accuracy: 59.2%
# ============================================================================

echo "============================================================"
echo "Activation Extraction for Linear Probing"
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
mkdir -p logs data/activations data/checkpoints

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate housing || { echo "Failed to activate housing environment"; exit 1; }

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo ""
echo "Running pre-flight checks..."

# Check GPU
echo "  Checking GPU..."
if ! nvidia-smi &> /dev/null; then
    echo "  ERROR: nvidia-smi not available"
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Check TransformerLens
echo "  Checking TransformerLens..."
if ! python -c "import transformer_lens" 2>/dev/null; then
    echo "  Installing TransformerLens..."
    pip install transformer-lens
fi
python -c "import transformer_lens; print(f'  TransformerLens version: {transformer_lens.__version__}')"

# Check and install missing packages
echo "  Checking Python packages..."
REQUIRED_PACKAGES="pandas numpy torch transformers tqdm scikit-learn"
for pkg in $REQUIRED_PACKAGES; do
    if ! python -c "import $pkg" 2>/dev/null; then
        echo "  Installing missing package: $pkg"
        pip install $pkg --quiet
    fi
done
echo "  All required packages available"

# Check data and models
echo "  Checking data file..."
if [ ! -f "data/pairs_20pct_price_diff.csv" ]; then
    echo "  ERROR: data/pairs_20pct_price_diff.csv not found"
    exit 1
fi
NUM_SAMPLES=$(tail -n +2 data/pairs_20pct_price_diff.csv | wc -l)
echo "  Found $NUM_SAMPLES samples"

echo "  Checking models..."
if [ ! -d "models/Llama-3.2-3B-Instruct" ]; then
    echo "  ERROR: models/Llama-3.2-3B-Instruct not found"
    exit 1
fi
if [ ! -d "models/Qwen3-4B-Instruct-2507" ]; then
    echo "  ERROR: models/Qwen3-4B-Instruct-2507 not found"
    exit 1
fi
echo "  Both models found"

echo ""
echo "Pre-flight checks passed!"
echo ""

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ============================================================================
# EXTRACT ACTIVATIONS
# ============================================================================

cd /home/anshulk/Housing/linear_probing

# CORRECTED: Both models use 2_fewshot_cot_temp0 (deterministic, greedy decoding)
# This ensures we extract from the SAME prompt strategy with deterministic inference

BATCH_SIZE=50  # Optimized for A6000 (48GB)
CONDITION="2_fewshot_cot_temp0"

echo "============================================================"
echo "Extracting Llama-3.2-3B-Instruct activations"
echo "Method: TransformerLens"
echo "Condition: $CONDITION (few-shot CoT, greedy, deterministic)"
echo "Accuracy: 59.3%"
echo "============================================================"

python extract_activations.py \
    --model llama-3.2-3b \
    --condition $CONDITION \
    --batch-size $BATCH_SIZE \
    --device cuda \
    --no-resume

LLAMA_EXIT=$?

if [ $LLAMA_EXIT -ne 0 ]; then
    echo "ERROR: Llama extraction failed with exit code $LLAMA_EXIT"
else
    echo "SUCCESS: Llama extraction completed"
fi

echo ""
echo "============================================================"
echo "Extracting Qwen3-4B-Instruct activations"
echo "Method: HuggingFace manual hooks (not supported by TransformerLens)"
echo "Condition: $CONDITION (few-shot CoT, greedy, deterministic)"
echo "Accuracy: 59.2%"
echo "============================================================"

python extract_activations.py \
    --model qwen3-4b \
    --condition $CONDITION \
    --batch-size $BATCH_SIZE \
    --device cuda \
    --no-resume

QWEN_EXIT=$?

if [ $QWEN_EXIT -ne 0 ]; then
    echo "ERROR: Qwen extraction failed with exit code $QWEN_EXIT"
else
    echo "SUCCESS: Qwen extraction completed"
fi

# ============================================================================
# VALIDATION
# ============================================================================

echo ""
echo "============================================================"
echo "Validating Extracted Activations"
echo "============================================================"

cd /home/anshulk/Housing

# Run validation checks
python -c "
import numpy as np
from pathlib import Path

activations_dir = Path('data/activations')

# Check Llama
llama_file = activations_dir / 'llama-3.2-3b_${CONDITION}_activations.npz'
if llama_file.exists():
    data = np.load(llama_file, allow_pickle=True)
    print(f'✓ Llama file exists: {llama_file.name}')
    print(f'  Size: {llama_file.stat().st_size / (1024**2):.1f} MB')
    print(f'  Shape: {data[\"activations\"].shape}')
    print(f'  No NaNs: {not np.isnan(data[\"activations\"]).any()}')
else:
    print(f'✗ Llama file NOT found: {llama_file}')

print()

# Check Qwen
qwen_file = activations_dir / 'qwen3-4b_${CONDITION}_activations.npz'
if qwen_file.exists():
    data = np.load(qwen_file, allow_pickle=True)
    print(f'✓ Qwen file exists: {qwen_file.name}')
    print(f'  Size: {qwen_file.stat().st_size / (1024**2):.1f} MB')
    print(f'  Shape: {data[\"activations\"].shape}')
    print(f'  No NaNs: {not np.isnan(data[\"activations\"]).any()}')
else:
    print(f'✗ Qwen file NOT found: {qwen_file}')
" 2>/dev/null || echo "Validation script failed (may need pandas installed)"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================"
echo "Extraction Complete"
echo "============================================================"
echo "End time: $(date)"
echo "Llama exit code: $LLAMA_EXIT"
echo "Qwen exit code: $QWEN_EXIT"
echo ""

# Show GPU memory at end
echo "Final GPU memory state:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# List generated files
echo ""
echo "Generated activation files:"
ls -lh ../data/activations/*.npz 2>/dev/null || echo "  No files found"

# Clean up old checkpoints if successful
if [ $LLAMA_EXIT -eq 0 ] && [ $QWEN_EXIT -eq 0 ]; then
    echo ""
    echo "Cleaning up checkpoints..."
    rm -f ../data/checkpoints/extraction_*_${CONDITION}_*.pkl 2>/dev/null
    echo "Old checkpoints removed"
fi

echo ""
echo "============================================================"
echo "Job Ended: $(date)"
echo "Total Runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo "============================================================"

# Return success if both succeeded
if [ $LLAMA_EXIT -eq 0 ] && [ $QWEN_EXIT -eq 0 ]; then
    echo "✓ Both extractions successful!"
    exit 0
elif [ $LLAMA_EXIT -eq 0 ] || [ $QWEN_EXIT -eq 0 ]; then
    echo "⚠ Partial success (one extraction failed)"
    exit 1
else
    echo "✗ Both extractions failed"
    exit 1
fi
