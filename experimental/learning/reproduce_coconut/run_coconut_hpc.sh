#!/usr/bin/env bash
set -e

# HPC script for COCONUT Stage 0 training
# Requires: Python 3.11 with PyTorch 2.5.1 + CUDA
# Usage:
#   Quick test (3 epochs):  bash run_coconut_hpc.sh test
#   Full training (25 epochs): bash run_coconut_hpc.sh full

# Determine config based on argument
MODE="${1:-test}"
if [ "$MODE" = "full" ]; then
    CONFIG="args/gsm_cot.yaml"
    RUN_NAME="stage0_full"
    EPOCHS=25
else
    CONFIG="args/gsm_cot_test.yaml"
    RUN_NAME="stage0_test"
    EPOCHS=3
fi

# Activate conda environment (Python 3.11)
module load conda/24.3.0-0
conda activate 3_11

# Set up environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1,2,3  # Use GPUs 1,2,3 for COCONUT (GPU 0 for cross-model)

# Navigate to coconut directory
cd coconut

# Create output directory and log file
OUTPUT_DIR="../runs/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/coconut_${RUN_NAME}_${TIMESTAMP}.log"

echo "=========================================="
echo "COCONUT Stage 0 Training - HPC Run"
echo "=========================================="
echo "Mode: $MODE ($EPOCHS epochs)"
echo "Config: $CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES (3 GPUs for COCONUT)"
echo "GPU type: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="
echo ""

# Number of GPUs to use (defaults to 3 for shared node with cross-model ablation)
# Can be overridden with NPROC env var
NPROC="${NPROC:-3}"
echo "Using $NPROC GPUs for training"
echo ""

# Run training with tee to capture ALL output
{
    torchrun --nproc_per_node=$NPROC run.py $CONFIG
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Complete! Results saved to:"
echo "  - Checkpoints: $OUTPUT_DIR/checkpoint_*/"
echo "  - Log: $LOG_FILE"
echo "=========================================="
