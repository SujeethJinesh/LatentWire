#!/usr/bin/env bash
# Test new Anchor-Guided Cross-Model Interlingua architecture
#
# Usage:
#   SAMPLES=100 STEPS=50 bash scripts/test_new_interlingua.sh       # Quick test
#   SAMPLES=1000 STEPS=500 bash scripts/test_new_interlingua.sh     # Full test
#   SAMPLES=1000 STEPS=500 TEST_QWEN=yes bash scripts/test_new_interlingua.sh  # With Qwen

set -euo pipefail

# Configuration from environment
SAMPLES="${SAMPLES:-100}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
D_INTER="${D_INTER:-512}"
NUM_SLOTS="${NUM_SLOTS:-32}"
TEST_QWEN="${TEST_QWEN:-no}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/test_new_interlingua}"

# Force unbuffered output
export PYTHONUNBUFFERED=1

echo "========================================================================"
echo "Testing: Anchor-Guided Cross-Model Interlingua"
echo "========================================================================"
echo "Configuration:"
echo "  SAMPLES=$SAMPLES"
echo "  STEPS=$STEPS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  LR=$LR"
echo "  D_INTER=$D_INTER"
echo "  NUM_SLOTS=$NUM_SLOTS"
echo "  TEST_QWEN=$TEST_QWEN"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build arguments
ARGS=(
    --samples "$SAMPLES"
    --steps "$STEPS"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --d_inter "$D_INTER"
    --num_slots "$NUM_SLOTS"
)

if [[ "$TEST_QWEN" == "yes" ]]; then
    ARGS+=(--test_qwen)
fi

# Run test
echo "Running test..."
python scripts/test_new_interlingua.py "${ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/test.log"

EXIT_CODE=${PIPESTATUS[0]}

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "========================================================================"
    echo "TEST PASSED"
    echo "========================================================================"
    echo "Log saved to: $OUTPUT_DIR/test.log"
    echo ""
    echo "Key checks:"
    echo "  ✓ All components instantiate correctly"
    echo "  ✓ Forward pass works"
    echo "  ✓ Loss computation succeeds"
    if [[ $STEPS -gt 0 ]]; then
        echo "  ✓ Training loop completes"
        echo "  Check diversity in log (should be >1 unique prediction)"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Check $OUTPUT_DIR/test.log for diversity results"
    echo "  2. If diverse predictions: Run longer test (STEPS=500)"
    echo "  3. If collapsed (all same): Debug architecture"
else
    echo ""
    echo "========================================================================"
    echo "TEST FAILED (exit code: $EXIT_CODE)"
    echo "========================================================================"
    echo "Check $OUTPUT_DIR/test.log for errors"
    exit $EXIT_CODE
fi
