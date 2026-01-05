#!/usr/bin/env bash
# Quick test runner for system validation
# Run this before submitting expensive HPC jobs

set -e

echo "=========================================="
echo "LatentWire System Test Suite"
echo "=========================================="
echo "This will test all critical components:"
echo "  1. Checkpoint save/load"
echo "  2. Preemption handling"
echo "  3. Resume from checkpoint"
echo "  4. GPU elasticity (1-4 GPUs)"
echo "  5. Data loading performance"
echo "  6. Logging capture"
echo "  7. State management"
echo "  8. Git operations"
echo "  + Bonus: Minimal training loop"
echo ""
echo "Expected runtime: <5 minutes"
echo "=========================================="
echo ""

# Set up environment
export PYTHONUNBUFFERED=1  # Critical: Immediate output flushing
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create test output directory
TEST_DIR="runs/system_tests_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

# Run tests with output capture
LOG_FILE="$TEST_DIR/test_results.log"

echo "Running tests..."
echo "Log file: $LOG_FILE"
echo ""

# Run test suite
{
    python3 finalization/test_all.py
} 2>&1 | tee "$LOG_FILE"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "[SUCCESS] ALL TESTS PASSED!"
    echo "=========================================="
    echo "System is ready for HPC experiments."
    echo "Test results saved to: $LOG_FILE"
else
    echo "=========================================="
    echo "[FAILURE] SOME TESTS FAILED!"
    echo "=========================================="
    echo "Please review failures in: $LOG_FILE"
    echo "Fix issues before running expensive experiments."
fi

exit $EXIT_CODE