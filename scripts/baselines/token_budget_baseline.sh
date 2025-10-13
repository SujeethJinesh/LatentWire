#!/usr/bin/env bash
#
# Token Budget Baseline - Fair Comparison
#
# Evaluates LLMs with TEXT TRUNCATED to M tokens (same budget as latent representation).
# This is the critical fairness baseline - shows what you get with simple truncation
# vs learned compression.
#
# Usage: PYTHONPATH=. bash scripts/baselines/token_budget_baseline.sh
#

set -e

echo "========================================================================"
echo "TOKEN BUDGET BASELINE - Fair Comparison"
echo "========================================================================"
echo ""
echo "Truncates text prompts to M tokens (same budget as compressed latent)."
echo "This establishes what simple truncation achieves with the same token budget."
echo ""
echo "KEY QUESTION: Does learned compression beat simple truncation?"
echo ""
echo "========================================================================"
echo ""

# Configuration
LLAMA_ID="${LLAMA_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-7B-Instruct}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-10000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"
LATENT_LEN="${LATENT_LEN:-32}"  # Match this to LatentWire M
OUTPUT_DIR="${OUTPUT_DIR:-runs/baselines/token_budget}"

echo "Configuration:"
echo "  LLAMA_ID: $LLAMA_ID"
echo "  QWEN_ID: $QWEN_ID"
echo "  DATASET: $DATASET"
echo "  SAMPLES: $SAMPLES"
echo "  TOKEN_BUDGET: $LATENT_LEN tokens"
echo "  MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/token_budget_${TIMESTAMP}.log"

echo "========================================================================"
echo "Running token budget baseline..."
echo "Log file: $LOG_FILE"
echo "========================================================================"
echo ""

# Run evaluation with truncated text
{
echo "Evaluating Llama with $LATENT_LEN-token truncated prompts..."
python latentwire/eval.py \
    --llama_id "$LLAMA_ID" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --mode token_budget \
    --token_budget "$LATENT_LEN" \
    --save_dir "$OUTPUT_DIR/llama_m${LATENT_LEN}" \
    --models llama

echo ""
echo "Evaluating Qwen with $LATENT_LEN-token truncated prompts..."
python latentwire/eval.py \
    --qwen_id "$QWEN_ID" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --mode token_budget \
    --token_budget "$LATENT_LEN" \
    --save_dir "$OUTPUT_DIR/qwen_m${LATENT_LEN}" \
    --models qwen

echo ""
echo "========================================================================"
echo "Token budget baseline complete!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - $OUTPUT_DIR/llama_m${LATENT_LEN}/"
echo "  - $OUTPUT_DIR/qwen_m${LATENT_LEN}/"
echo "  - $LOG_FILE"
echo ""
echo "INTERPRETATION:"
echo "  If LatentWire beats this baseline, learned compression is working!"
echo "  If LatentWire is below this, the model isn't learning to compress."
echo ""
} 2>&1 | tee "$LOG_FILE"

echo "To view results:"
echo "  cat $OUTPUT_DIR/llama_m${LATENT_LEN}/results.json | python -m json.tool | less"
echo "  cat $OUTPUT_DIR/qwen_m${LATENT_LEN}/results.json | python -m json.tool | less"
echo ""
