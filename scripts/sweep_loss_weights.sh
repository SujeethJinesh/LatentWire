#!/usr/bin/env bash
# Sweep different loss weight configurations to find optimal diversity
#
# Usage:
#   bash scripts/sweep_loss_weights.sh                    # Default: 300 steps per config
#   STEPS=200 bash scripts/sweep_loss_weights.sh          # Quick sweep
#   STEPS=500 bash scripts/sweep_loss_weights.sh          # Thorough sweep

set -euo pipefail

# Configuration from environment
SAMPLES="${SAMPLES:-1000}"
STEPS="${STEPS:-300}"
D_INTER="${D_INTER:-512}"
NUM_SLOTS="${NUM_SLOTS:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/loss_weight_sweep}"

# Force unbuffered output
export PYTHONUNBUFFERED=1

echo "========================================================================"
echo "Loss Weight Sweep: Finding Optimal Configuration"
echo "========================================================================"
echo "Configuration:"
echo "  SAMPLES=$SAMPLES"
echo "  STEPS=$STEPS (per configuration)"
echo "  D_INTER=$D_INTER"
echo "  NUM_SLOTS=$NUM_SLOTS"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo ""
echo "Testing 7 configurations:"
echo "  1. No semantic loss (0.0, K=4) - the 'buggy' version"
echo "  2. Very weak semantic (0.01, K=4)"
echo "  3. Weak semantic (0.05, K=4)"
echo "  4. Medium semantic (0.1, K=4)"
echo "  5. Strong semantic (0.5, K=4) - current"
echo "  6. Increased K-token (0.05, K=8)"
echo "  7. Increased K-token (0.05, K=12)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run sweep
echo "Running sweep..."
python scripts/sweep_loss_weights.py \
    --samples "$SAMPLES" \
    --steps "$STEPS" \
    --d_inter "$D_INTER" \
    --num_slots "$NUM_SLOTS" \
    2>&1 | tee "$OUTPUT_DIR/sweep.log"

EXIT_CODE=${PIPESTATUS[0]}

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "========================================================================"
    echo "SWEEP COMPLETE"
    echo "========================================================================"
    echo "Results saved to: $OUTPUT_DIR/"
    echo "  - sweep.log (full output)"
    echo "  - results.json (structured data)"
    echo "  - summary.txt (readable summary)"
    echo ""
    echo "Check summary:"
    echo "  cat $OUTPUT_DIR/summary.txt"
else
    echo ""
    echo "========================================================================"
    echo "SWEEP FAILED (exit code: $EXIT_CODE)"
    echo "========================================================================"
    echo "Check $OUTPUT_DIR/sweep.log for errors"
    exit $EXIT_CODE
fi
