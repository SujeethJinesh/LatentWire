#!/usr/bin/env bash
#
# Text Baseline - Upper Bound Performance
#
# Evaluates both LLMs with FULL text prompts to establish best possible performance.
# This is the target we're trying to match with compressed representations.
#
# Usage: PYTHONPATH=. bash scripts/baselines/text_baseline.sh
#

set -e

echo "========================================================================"
echo "TEXT BASELINE - Upper Bound"
echo "========================================================================"
echo ""
echo "This establishes the best possible performance with full text prompts."
echo "The compressed interlingua should aim to match this quality."
echo ""
echo "========================================================================"
echo ""

# Configuration
LLAMA_ID="${LLAMA_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-7B-Instruct}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-10000}"  # Use full validation set
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/baselines/text}"

echo "Configuration:"
echo "  LLAMA_ID: $LLAMA_ID"
echo "  QWEN_ID: $QWEN_ID"
echo "  DATASET: $DATASET"
echo "  SAMPLES: $SAMPLES"
echo "  MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/text_baseline_${TIMESTAMP}.log"

echo "========================================================================"
echo "Running text baseline evaluation..."
echo "Log file: $LOG_FILE"
echo "========================================================================"
echo ""

# Run evaluation with both models on full text
{
echo "Evaluating Llama with full text prompts..."
python scripts/baselines/evaluate_text_baseline.py \
    --model_id "$LLAMA_ID" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --save_dir "$OUTPUT_DIR/llama"

echo ""
echo "Evaluating Qwen with full text prompts..."
python scripts/baselines/evaluate_text_baseline.py \
    --model_id "$QWEN_ID" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --save_dir "$OUTPUT_DIR/qwen"

echo ""
echo "========================================================================"
echo "Text baseline complete!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - $OUTPUT_DIR/llama/"
echo "  - $OUTPUT_DIR/qwen/"
echo "  - $LOG_FILE"
echo ""
} 2>&1 | tee "$LOG_FILE"

echo "To view results:"
echo "  cat $OUTPUT_DIR/llama/results.json | python -m json.tool | less"
echo "  cat $OUTPUT_DIR/qwen/results.json | python -m json.tool | less"
echo ""
