#!/usr/bin/env bash
# Run end-to-end test for LatentWire pipeline
# This validates training, checkpoint saving/loading, and evaluation

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/e2e_test}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/e2e_test_${TIMESTAMP}.log"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "Running LatentWire End-to-End Test"
echo "================================================"
echo "Start time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Run the test with full output capture
{
    python test_end_to_end.py
    EXIT_CODE=$?

    echo ""
    echo "================================================"
    echo "Test completed at: $(date)"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ SUCCESS: All tests passed!"
    else
        echo "❌ FAILURE: Some tests failed. Check the log above."
    fi
    echo "================================================"

    exit $EXIT_CODE
} 2>&1 | tee "$LOG_FILE"