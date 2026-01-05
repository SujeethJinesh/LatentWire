#!/usr/bin/env bash
set -e

# =============================================================================
# Final Integration Test for Telepathy System
# =============================================================================
# This script validates all components work together before HPC submission:
# 1. Linear probe baseline training
# 2. Bridge training with 2 models
# 3. ROUGE evaluation on XSUM
# 4. Statistical testing
# 5. Result aggregation
#
# Should complete in <5 minutes on 1 GPU
# =============================================================================

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/integration_test}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/integration_test_${TIMESTAMP}.log"

echo "=============================================================="
echo "TELEPATHY INTEGRATION TEST"
echo "=============================================================="
echo "Start time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Run integration test with tee to capture all output
{
    echo "Running comprehensive integration test..."
    echo "This will test all components end-to-end"
    echo ""

    python telepathy/test_final_integration.py

    echo ""
    echo "=============================================================="
    echo "Integration test complete!"
    echo "=============================================================="

} 2>&1 | tee "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ ALL INTEGRATION TESTS PASSED"
    echo ""
    echo "Results saved to:"
    echo "  - $OUTPUT_DIR/"
    echo "  - $LOG_FILE"
    echo "  - telepathy/integration_test_results/"
    echo ""
    echo "System is ready for HPC submission!"
else
    echo ""
    echo "❌ INTEGRATION TESTS FAILED"
    echo ""
    echo "Please check the log file: $LOG_FILE"
    echo "Fix any issues before submitting to HPC"
    exit 1
fi