#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Architecture sweep wrapper
# Tests 3 architectural variants to diagnose mode collapse:
# 1. Direct Sequence Compression (no mean pooling) - PROPOSED FIX
# 2. Mean Pool Bottleneck (single vector) - BROKEN
# 3. Mean Pool + Expand (full pipeline) - CURRENT BASELINE

SAMPLES="${SAMPLES:-1000}"
STEPS="${STEPS:-300}"
M="${M:-32}"
D_BOTTLENECK="${D_BOTTLENECK:-512}"

echo "Architecture Sweep Configuration:"
echo "  Samples: $SAMPLES"
echo "  Steps: $STEPS"
echo "  M (compressed length): $M"
echo "  d_bottleneck: $D_BOTTLENECK"
echo ""

# Ensure output directory exists
OUTPUT_DIR="runs/architecture_sweep"
mkdir -p "$OUTPUT_DIR"

# Run sweep
python scripts/sweep_architectures.py \
    --samples "$SAMPLES" \
    --steps "$STEPS" \
    --M "$M" \
    --d_bottleneck "$D_BOTTLENECK" \
    2>&1 | tee "$OUTPUT_DIR/sweep.log"

echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo "  - results.json (detailed data)"
echo "  - summary.txt (readable summary)"
echo "  - sweep.log (full training logs)"
