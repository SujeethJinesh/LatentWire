#!/usr/bin/env bash
# Full pipeline integration test for LatentWire
#
# This script runs a comprehensive end-to-end test that validates:
# 1. Training with checkpoint saving
# 2. Checkpoint resumption
# 3. Evaluation on checkpoints
# 4. All components working together
#
# Usage:
#   bash run_integration_test.sh

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/integration_test}"
LOG_FILE="${OUTPUT_DIR}/integration_test_$(date +"%Y%m%d_%H%M%S").log"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "LatentWire Full Pipeline Integration Test"
echo "========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Run the integration test with logging
{
    echo "Starting integration test at $(date)"
    echo ""

    # Run the Python integration test
    python test_integration.py \
        --verbose \
        --output "$OUTPUT_DIR/test_results.json"

    echo ""
    echo "Integration test completed at $(date)"

} 2>&1 | tee "$LOG_FILE"

# Check results
if [ -f "$OUTPUT_DIR/test_results.json" ]; then
    echo ""
    echo "Test results saved to: $OUTPUT_DIR/test_results.json"

    # Parse and display summary
    python -c "
import json
import sys

with open('$OUTPUT_DIR/test_results.json') as f:
    results = json.load(f)

print('\n' + '=' * 50)
print('FINAL TEST SUMMARY')
print('=' * 50)
print(f'Tests Passed: {results.get(\"tests_passed\", 0)}')
print(f'Tests Failed: {results.get(\"tests_failed\", 0)}')

if results.get('failures'):
    print(f'Failed Tests: {\", \".join(results[\"failures\"])}')

total_time = results.get('total_time', 0)
print(f'\\nTotal Time: {total_time:.2f}s')

# Exit with appropriate code
sys.exit(0 if results.get('tests_failed', 0) == 0 else 1)
"

    EXIT_CODE=$?
else
    echo "ERROR: Test results not found!"
    EXIT_CODE=1
fi

echo ""
echo "Full log saved to: $LOG_FILE"

exit $EXIT_CODE