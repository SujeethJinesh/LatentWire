#!/usr/bin/env bash
set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/cross_model_ablation}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1  # Force unbuffered output for real-time logging

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/cross_model_ablation_${TIMESTAMP}.log"

echo "Starting cross-model alignment ablation experiment..."
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Activate venv
source ../../venv_arm64/bin/activate

# Run experiment with tee to capture ALL output
{
    python cross_model_ablation.py
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Complete! Results saved to:"
echo "  - $LOG_FILE"
