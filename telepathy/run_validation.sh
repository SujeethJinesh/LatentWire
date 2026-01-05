#!/usr/bin/env bash
# Run validation checks before experiments
set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/validation}"
VALIDATION_MODE="${VALIDATION_MODE:-quick}"  # quick or full

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/validation_${TIMESTAMP}.log"
REPORT_FILE="$OUTPUT_DIR/validation_report_${TIMESTAMP}.json"

echo "=============================================================="
echo "TELEPATHY EXPERIMENT VALIDATION"
echo "=============================================================="
echo "Mode: $VALIDATION_MODE"
echo "Log file: $LOG_FILE"
echo "Report file: $REPORT_FILE"
echo ""

# Determine validation flags
if [ "$VALIDATION_MODE" = "full" ]; then
    FLAGS="--full"
else
    FLAGS=""
fi

# Run validation with tee to capture output
{
    echo "Starting validation at $(date)"
    echo "Working directory: $(pwd)"
    echo "Python path: $PYTHONPATH"
    echo "Python version: $(python3 --version)"
    echo ""

    python3 telepathy/validate_experiments.py $FLAGS --save-report "$REPORT_FILE"

    echo ""
    echo "Validation completed at $(date)"
} 2>&1 | tee "$LOG_FILE"

# Check exit code
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ VALIDATION SUCCESSFUL"
    echo ""
    echo "You can now run experiments with confidence!"
    echo "Reports saved to:"
    echo "  - Log: $LOG_FILE"
    echo "  - JSON: $REPORT_FILE"
else
    echo ""
    echo "❌ VALIDATION FAILED"
    echo ""
    echo "Please review the errors in:"
    echo "  - $LOG_FILE"
    echo "  - $REPORT_FILE"
    echo ""
    echo "Fix the issues before running experiments."
    exit 1
fi