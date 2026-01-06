#!/usr/bin/env bash
# Test script for checkpoint saving and resuming functionality
# This script should be run on the HPC where PyTorch is available

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/checkpoint_test}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/checkpoint_test_${TIMESTAMP}.log"

echo "Starting checkpoint save/resume tests..."
echo "Log file: $LOG_FILE"
echo ""

# Run the test with tee to capture ALL output
{
    echo "=============================================================="
    echo "CHECKPOINT SAVE/RESUME TESTS"
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
    echo "Running test suite..."
    echo "=============================================================="

    # Run the test
    python3 test_checkpoint_resume.py

    echo ""
    echo "=============================================================="
    echo "Test completed at $(date)"
    echo "=============================================================="

} 2>&1 | tee "$LOG_FILE"

# Check if tests passed
if grep -q "ALL TESTS PASSED" "$LOG_FILE"; then
    echo ""
    echo "✅ All checkpoint tests passed!"
    echo "Results saved to: $LOG_FILE"
    exit 0
else
    echo ""
    echo "⚠️ Some tests failed. Check the log for details:"
    echo "  $LOG_FILE"
    exit 1
fi