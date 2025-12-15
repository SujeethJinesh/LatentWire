#!/bin/bash
# run_unified_comparison.sh
#
# Runs ALL baselines in ONE script for paper comparison:
# 1. Bridge (Llama→Mistral) - Main method
# 2. Prompt-Tuning (Mistral only) - Proves sender is essential
# 3. Text-Relay - Latency comparison
# 4. Few-shot prompting (5-shot)
# 5. Zero-shot baselines
#
# Output: Single JSON with all results
#
# Usage:
#   bash run_unified_comparison.sh
#   OUTPUT_DIR=runs/my_comparison bash run_unified_comparison.sh

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/unified_comparison}"
DATASETS="${DATASETS:-sst2 agnews trec}"
SOFT_TOKENS="${SOFT_TOKENS:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
FEWSHOT_SHOTS="${FEWSHOT_SHOTS:-5}"
SEED="${SEED:-42}"
SKIP_TEXT_RELAY="${SKIP_TEXT_RELAY:-}"

# Environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/unified_comparison_${TIMESTAMP}.log"

echo "=============================================="
echo "UNIFIED COMPARISON EXPERIMENT"
echo "=============================================="
echo "Timestamp: $TIMESTAMP"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "Datasets: $DATASETS"
echo "Soft tokens: $SOFT_TOKENS"
echo "Train steps: $TRAIN_STEPS"
echo "Eval samples: $EVAL_SAMPLES"
echo "=============================================="
echo ""

# Build args
EXTRA_ARGS=""
if [ -n "$SKIP_TEXT_RELAY" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip_text_relay"
fi

# Run unified comparison with tee to capture all output
{
    python telepathy/run_unified_comparison.py \
        --datasets $DATASETS \
        --output_dir "$OUTPUT_DIR" \
        --soft_tokens $SOFT_TOKENS \
        --train_steps $TRAIN_STEPS \
        --eval_samples $EVAL_SAMPLES \
        --fewshot_shots $FEWSHOT_SHOTS \
        --seed $SEED \
        $EXTRA_ARGS
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================="
echo "COMPLETE!"
echo "=============================================="
echo "Results: $OUTPUT_DIR/unified_results_*.json"
echo "Log: $LOG_FILE"
echo ""
echo "To analyze results:"
echo "  cat $OUTPUT_DIR/unified_results_*.json | python -m json.tool"
