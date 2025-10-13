#!/usr/bin/env bash
#
# Comprehensive Embedding Diagnostics
# Trains a quick checkpoint then analyzes REAL learned embeddings
#
# Usage: PYTHONPATH=. bash scripts/run_embedding_diagnostics.sh
#
# This script works END-TO-END from scratch:
#   1. Runs quick training (500 steps) to generate real checkpoint
#   2. Analyzes learned embeddings from that checkpoint
#   3. Compares to text embeddings
#
# NO synthetic data - only real trained embeddings
#

set -e

echo "========================================================================"
echo "EMBEDDING DIAGNOSTICS (Real Data - End-to-End)"
echo "========================================================================"
echo ""
echo "This script:"
echo "  1. TRAINS a quick checkpoint (500 steps, real data)"
echo "  2. ANALYZES learned embeddings from that checkpoint"
echo "  3. COMPARES to text embeddings"
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
QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-7B-Instruct}"
DATASET="${DATASET:-squad}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-1000}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/embed_diagnostics}"
LATENT_LEN="${LATENT_LEN:-32}"
D_Z="${D_Z:-256}"

echo "Configuration:"
echo "  LLAMA_ID: $MODEL_ID"
echo "  QWEN_ID: $QWEN_ID"
echo "  DATASET: $DATASET"
echo "  TRAIN_SAMPLES: $TRAIN_SAMPLES"
echo "  TRAIN_EPOCHS: $TRAIN_EPOCHS"
echo "  EVAL_SAMPLES: $EVAL_SAMPLES"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  LATENT_LEN: $LATENT_LEN"
echo "  D_Z: $D_Z"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/diagnostics_${TIMESTAMP}.log"

echo "========================================================================"
echo "PHASE 1: TRAINING (Quick checkpoint generation)"
echo "========================================================================"
echo ""

CHECKPOINT_DIR="$OUTPUT_DIR/checkpoint"

# Run quick training to generate real checkpoint
{
echo "Training quick checkpoint with real data..."
echo "  Epochs: $TRAIN_EPOCHS"
echo "  Samples: $TRAIN_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Latent: M=$LATENT_LEN, d_z=$D_Z"
echo ""

python latentwire/train.py \
    --llama_id "$MODEL_ID" \
    --qwen_id "$QWEN_ID" \
    --samples "$TRAIN_SAMPLES" \
    --epochs "$TRAIN_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --latent_len "$LATENT_LEN" \
    --d_z "$D_Z" \
    --encoder_type byte \
    --dataset "$DATASET" \
    --sequential_models \
    --warm_anchor_text "Answer: " \
    --first_token_ce_weight 0.5 \
    --output_dir "$CHECKPOINT_DIR"

echo ""
echo "Training complete!"
echo ""
} 2>&1 | tee -a "$LOG_FILE"

# Find the checkpoint (should be latest)
if [ ! -f "$CHECKPOINT_DIR/encoder.pt" ]; then
    echo "ERROR: Training did not produce checkpoint at $CHECKPOINT_DIR/encoder.pt"
    exit 1
fi

echo "========================================================================"
echo "PHASE 2: ANALYSIS (Embedding diagnostics)"
echo "========================================================================"
echo ""

# Run diagnostics with tee to capture output
{
python scripts/run_embedding_diagnostics.py \
    --model_id "$MODEL_ID" \
    --dataset "$DATASET" \
    --samples "$EVAL_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT_DIR"
} 2>&1 | tee -a "$LOG_FILE"

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
