#!/usr/bin/env bash
#
# Comprehensive Embedding Diagnostics
# Analyzes REAL learned embeddings from trained checkpoints
#
# Usage: CHECKPOINT=path/to/checkpoint bash scripts/run_embedding_diagnostics.sh
#
# IMPORTANT: CHECKPOINT environment variable is REQUIRED
#            Synthetic testing is prohibited - only real data is analyzed
#

set -e

echo "========================================================================"
echo "EMBEDDING DIAGNOSTICS (Real Data Only)"
echo "========================================================================"
echo ""
echo "This script analyzes REAL learned embeddings from trained checkpoints"
echo "and compares them to text embeddings to understand failure modes."
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

# Validate checkpoint
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: CHECKPOINT environment variable is required."
    echo ""
    echo "Usage: CHECKPOINT=path/to/checkpoint bash scripts/run_embedding_diagnostics.sh"
    echo ""
    echo "Synthetic testing is prohibited - you must provide a real checkpoint."
    exit 1
fi

if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint directory does not exist: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$CHECKPOINT/encoder.pt" ]; then
    echo "ERROR: No encoder.pt found in checkpoint: $CHECKPOINT"
    exit 1
fi

echo "Configuration:"
echo "  MODEL_ID: $MODEL_ID"
echo "  DATASET: $DATASET"
echo "  SAMPLES: $SAMPLES"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  CHECKPOINT: $CHECKPOINT"
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
python scripts/run_embedding_diagnostics.py \
    --model_id "$MODEL_ID" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT"
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
