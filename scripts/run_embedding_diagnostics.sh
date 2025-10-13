#!/usr/bin/env bash
#
# Comprehensive Embedding Diagnostics
# Answers critical questions about embedding distribution mismatch
#
# Usage: PYTHONPATH=. bash scripts/run_embedding_diagnostics.sh
#

set -e

echo "========================================================================"
echo "EMBEDDING DIAGNOSTICS"
echo "========================================================================"
echo ""
echo "This script performs comprehensive analysis of embedding statistics"
echo "to understand why learned embeddings fail."
echo ""
echo "Analysis includes:"
echo "  - Per-token RMS distribution (min, max, mean, std)"
echo "  - Per-dimension statistics"
echo "  - Nearest vocab token alignment (cosine similarity)"
echo "  - Covariance structure"
echo "  - Effect of RMS scaling and batch normalization"
echo ""
echo "Questions answered:"
echo "  1. Where did '115-120× magnitude mismatch' come from?"
echo "  2. What's different between text and learned embeddings?"
echo "  3. Why does RMS scaling destroy everything?"
echo "  4. What properties does the LLM need?"
echo ""
echo "========================================================================"
echo ""

# Configuration
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/embed_diagnostics}"
CHECKPOINT="${CHECKPOINT:-}"

echo "Configuration:"
echo "  MODEL_ID: $MODEL_ID"
echo "  DATASET: $DATASET"
echo "  SAMPLES: $SAMPLES"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
if [ -n "$CHECKPOINT" ]; then
    echo "  CHECKPOINT: $CHECKPOINT"
else
    echo "  CHECKPOINT: None (will use synthetic embeddings)"
fi
echo ""

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/diagnostics_${TIMESTAMP}.log"

echo "========================================================================"
echo "Starting diagnostics..."
echo "Log file: $LOG_FILE"
echo "========================================================================"
echo ""

# Run diagnostics with tee to capture output
{
if [ -n "$CHECKPOINT" ]; then
    python scripts/run_embedding_diagnostics.py \
        --model_id "$MODEL_ID" \
        --dataset "$DATASET" \
        --samples "$SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR" \
        --checkpoint "$CHECKPOINT"
else
    python scripts/run_embedding_diagnostics.py \
        --model_id "$MODEL_ID" \
        --dataset "$DATASET" \
        --samples "$SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR" \
        --no-checkpoint
fi
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================================================"
echo "Diagnostics complete!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - $OUTPUT_DIR/diagnostics.json"
echo "  - $LOG_FILE"
echo ""
echo "To view logs:"
echo "  cat $LOG_FILE | less"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/diagnostics.json | python -m json.tool | less"
echo ""
