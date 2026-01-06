#!/usr/bin/env bash
#
# Simple Embedding Experiment Sweep
# Tests transformations on real text embeddings (no training needed)
#
# Usage: PYTHONPATH=. bash scripts/run_embed_sweep_simple.sh
#

set -e

echo "========================================================================"
echo "SIMPLE EMBEDDING EXPERIMENT SWEEP (LIGHTWEIGHT ONLY)"
echo "========================================================================"
echo ""
echo "This script tests LIGHTWEIGHT embedding transformations WITHOUT training."
echo "It uses text embeddings as input and tests if transformations help."
echo ""
echo "Important:"
echo "  - Sequential generation (no batching)"
echo "  - Runtime: ~30-60 minutes on GPU, ~2-4 hours on CPU"
echo "  - 100 samples × 10 configs = 1000 generation calls"
echo ""
echo "Experiments: 10 LIGHTWEIGHT configs"
echo ""
echo "Categories:"
echo "  1. Baseline (1 config) - upper bound"
echo "  2. RMS matching (8 configs) - fixes magnitude mismatch"
echo "  3. Batch distribution (1 config) - normalizes statistics"
echo ""
echo "Heavy transforms (K-nearest, anchor+offset, soft codebook) are REMOVED"
echo "because they require full vocabulary searches (too expensive)."
echo ""
echo "Results will be ranked by F1 score."
echo "========================================================================"
echo ""

# Configuration
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/embed_sweep_simple}"

echo "Configuration:"
echo "  MODEL_ID: $MODEL_ID"
echo "  DATASET: $DATASET"
echo "  SAMPLES: $SAMPLES"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/sweep_${TIMESTAMP}.log"

echo "========================================================================"
echo "Starting experiments..."
echo "Log file: $LOG_FILE"
echo "========================================================================"
echo ""

# Run experiments with tee to capture output
{
python scripts/run_embed_sweep_simple.py \
    --model_id "$MODEL_ID" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "Complete! Generating analysis..."
echo "========================================================================"
echo ""

# Run analysis
python scripts/analyze_sweep_results.py --summary "$OUTPUT_DIR/summary.json"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================================================"
echo "Results saved to:"
echo "  - $OUTPUT_DIR/summary.json"
echo "  - $LOG_FILE"
echo "========================================================================"
echo ""
echo "To view logs:"
echo "  cat $LOG_FILE | less"
echo ""
echo "To view raw summary:"
echo "  cat $OUTPUT_DIR/summary.json | python -m json.tool"
echo ""
echo "To re-run analysis:"
echo "  python scripts/analyze_sweep_results.py --summary $OUTPUT_DIR/summary.json"
echo ""
