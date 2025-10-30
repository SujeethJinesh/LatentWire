#!/usr/bin/env bash
set -e

# Procrustes Fixed Ablation - Wrapper Script with Logging
# Tests Procrustes alignment with all critical fixes across different layers

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/procrustes_fixed_ablation}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/procrustes_ablation_${TIMESTAMP}.log"

echo "Starting Procrustes Fixed Ablation Experiment"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Run experiment with tee to capture ALL output
{
    python3 procrustes_fixed_ablation.py
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "Results saved to:"
echo "  - Log: $LOG_FILE"
echo "  - JSON: $OUTPUT_DIR/procrustes_layer_ablation_*.json"
echo "=============================================="
