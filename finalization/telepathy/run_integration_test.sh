#!/usr/bin/env bash
# Run minimal integration test for LatentWire/Telepathy pipeline
#
# This test validates all major components work together:
# - Bridge training
# - Linear probe baseline
# - Statistical testing
# - Result aggregation
#
# Should complete in <5 minutes

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/integration_test}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "==========================================="
echo "LatentWire/Telepathy Integration Test"
echo "==========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Started: $(date)"
echo ""

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/integration_test_${TIMESTAMP}.log"

echo "Running integration tests..."
echo "Log file: $LOG_FILE"
echo ""

# Run integration test with output capture
{
    python telepathy/test_integration.py \
        --temp-dir "$OUTPUT_DIR/test_artifacts" \
        --keep-temp

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "==========================================="
        echo "✓ ALL INTEGRATION TESTS PASSED!"
        echo "==========================================="
    else
        echo ""
        echo "==========================================="
        echo "✗ INTEGRATION TESTS FAILED"
        echo "==========================================="
        echo "Check the log file for details: $LOG_FILE"
        echo "Test artifacts preserved in: $OUTPUT_DIR/test_artifacts"
    fi

    echo "Completed: $(date)"
    exit $EXIT_CODE

} 2>&1 | tee "$LOG_FILE"