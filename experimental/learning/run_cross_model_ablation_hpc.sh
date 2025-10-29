#!/usr/bin/env bash
set -e

# HPC script for cross-model ablation experiment
# Requires: Python 3.11 with PyTorch 2.5.1 + CUDA

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/cross_model_ablation}"

# Activate conda environment (Python 3.11)
module load conda/24.3.0-0
conda activate 3_11

# Set up environment
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 for cross-model (COCONUT uses GPUs 1-3)

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/cross_model_ablation_hpc_${TIMESTAMP}.log"

echo "=========================================="
echo "Cross-Model Ablation - HPC Run"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="
echo ""

# Run experiment with tee to capture ALL output
{
    python cross_model_ablation.py
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Complete! Results saved to:"
echo "  - $LOG_FILE"
echo "=========================================="
