#!/bin/bash
# run_passkey.sh
#
# Phase 20: Passkey Retrieval Sanity Check
#
# GOAL: Test if the bridge can transmit a 5-digit code exactly.
# This determines whether GSM8K failure was due to:
#   - BANDWIDTH (bridge can't transmit precise data)
#   - REASONING (bridge can transmit, but model can't reason)
#
# Interpretation:
#   < 10% accuracy: Bridge is "vibe only" - need architecture changes
#   10-80% accuracy: Bridge is lossy - may need more tokens
#   > 80% accuracy: Bridge works! GSM8K needs COCONUT curriculum

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/passkey_check}"
STEPS="${STEPS:-1000}"
SOFT_TOKENS="${SOFT_TOKENS:-16}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/passkey_${TIMESTAMP}.log"

echo "============================================================"
echo "PASSKEY RETRIEVAL EXPERIMENT"
echo "============================================================"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "Soft tokens: $SOFT_TOKENS"
echo "Steps: $STEPS"
echo ""
echo "This test answers: Is the bridge a DATA CHANNEL or just VIBE TRANSFER?"
echo "============================================================"

{
    python telepathy/train_telepathy_passkey.py \
        --output_dir "$OUTPUT_DIR" \
        --steps "$STEPS" \
        --soft_tokens "$SOFT_TOKENS" \
        --batch_size "$BATCH_SIZE" \
        --eval_every 200 \
        --diversity_weight 0.1
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo "Results: $OUTPUT_DIR/passkey_results.json"
echo "Log: $LOG_FILE"
echo ""
echo "Next steps based on results:"
echo "  If < 10%: Rethink architecture (precision bottleneck)"
echo "  If 10-80%: Try --soft_tokens 32 or 64"
echo "  If > 80%: Implement COCONUT curriculum for GSM8K"
echo "============================================================"
