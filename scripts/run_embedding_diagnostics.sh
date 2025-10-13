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
SAMPLES="${SAMPLES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/embed_diagnostics}"
CHECKPOINT="${CHECKPOINT:-}"

echo "Configuration:"
echo "  MODEL_ID: $MODEL_ID"
echo "  DATASET: $DATASET"
echo "  SAMPLES: $SAMPLES"
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

echo "========================================================================"
echo "Starting diagnostics..."
echo "========================================================================"
echo ""

# Run diagnostics
if [ -n "$CHECKPOINT" ]; then
    python scripts/run_embedding_diagnostics.py \
        --model_id "$MODEL_ID" \
        --dataset "$DATASET" \
        --samples "$SAMPLES" \
        --output_dir "$OUTPUT_DIR" \
        --checkpoint "$CHECKPOINT"
else
    python scripts/run_embedding_diagnostics.py \
        --model_id "$MODEL_ID" \
        --dataset "$DATASET" \
        --samples "$SAMPLES" \
        --output_dir "$OUTPUT_DIR" \
        --no-checkpoint
fi

echo ""
echo "========================================================================"
echo "Diagnostics complete!"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/diagnostics.json"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/diagnostics.json | python -m json.tool | less"
echo ""
