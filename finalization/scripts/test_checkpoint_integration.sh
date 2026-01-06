#!/usr/bin/env bash
# Integration test script for checkpoint saving and resuming
# Tests the full checkpoint lifecycle on HPC with PyTorch available

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/checkpoint_integration_test}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/integration_test_${TIMESTAMP}.log"

echo "Starting checkpoint integration tests..."
echo "Log file: $LOG_FILE"
echo ""

# Run the integration test with tee to capture ALL output
{
    echo "=============================================================="
    echo "CHECKPOINT INTEGRATION TESTS"
    echo "=============================================================="
    echo "Start time: $(date)"
    echo "Python version: $(python3 --version)"
    echo "Working directory: $(pwd)"
    echo ""

    # Check PyTorch availability
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
        echo "ERROR: PyTorch not available!"
        exit 1
    }

    echo ""
    echo "Running integration test suite..."
    echo "=============================================================="

    # Run both test scripts
    echo ""
    echo "Test 1: Unit tests for checkpoint manager"
    echo "----------------------------------------------------------"
    python3 test_checkpoint_resume.py

    echo ""
    echo "Test 2: Integration test with mock training"
    echo "----------------------------------------------------------"
    python3 test_checkpoint_integration.py

    echo ""
    echo "=============================================================="
    echo "Tests completed at $(date)"
    echo "=============================================================="

} 2>&1 | tee "$LOG_FILE"

# Check if tests passed
if grep -q "ALL TESTS PASSED" "$LOG_FILE" && grep -q "ALL INTEGRATION TESTS PASSED" "$LOG_FILE"; then
    echo ""
    echo "✅ All checkpoint tests passed successfully!"
    echo "Results saved to: $LOG_FILE"
    exit 0
else
    echo ""
    echo "⚠️ Some tests failed. Check the log for details:"
    echo "  $LOG_FILE"
    exit 1
fi