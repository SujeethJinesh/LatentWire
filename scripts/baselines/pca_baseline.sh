#!/usr/bin/env bash
#
# PCA Baseline - Linear Compression
#
# Tests if simple PCA (linear projection) of text embeddings achieves
# comparable performance to learned non-linear encoding.
#
# KEY QUESTION: Do we need the learned encoder, or is PCA sufficient?
#
# Usage: PYTHONPATH=. bash scripts/baselines/pca_baseline.sh
#

set -e

echo "========================================================================"
echo "PCA BASELINE - Linear Compression"
echo "========================================================================"
echo ""
echo "Compresses text embeddings with PCA to M dimensions, then reconstructs."
echo "This tests if linear compression is sufficient or if we need"
echo "the learned non-linear encoder."
echo ""
echo "========================================================================"
echo ""

# Configuration
LLAMA_ID="${LLAMA_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-1000}"
LATENT_LEN="${LATENT_LEN:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/baselines/pca}"

echo "Configuration:"
echo "  LLAMA_ID: $LLAMA_ID"
echo "  DATASET: $DATASET"
echo "  SAMPLES: $SAMPLES"
echo "  LATENT_DIM (M): $LATENT_LEN"
echo "  MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/pca_baseline_${TIMESTAMP}.log"

echo "========================================================================"
echo "Running PCA baseline..."
echo "Log file: $LOG_FILE"
echo "========================================================================"
echo ""

# Run PCA baseline
{
python scripts/baselines/pca_baseline.py \
    --llama_id "$LLAMA_ID" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --latent_len "$LATENT_LEN" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --save_dir "$OUTPUT_DIR/M${LATENT_LEN}"

echo ""
echo "========================================================================"
echo "PCA baseline complete!"
echo "========================================================================"
echo ""
} 2>&1 | tee "$LOG_FILE"

echo "Results saved to:"
echo "  - $OUTPUT_DIR/M${LATENT_LEN}/results.json"
echo "  - $LOG_FILE"
echo ""
echo "To view:"
echo "  cat $OUTPUT_DIR/M${LATENT_LEN}/results.json | python -m json.tool"
echo ""
