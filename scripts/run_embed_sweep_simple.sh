#!/usr/bin/env bash
#
# Simple Embedding Experiment Sweep
# Tests transformations on real text embeddings (no training needed)
#
# Usage: PYTHONPATH=. bash scripts/run_embed_sweep_simple.sh
#

set -e

echo "========================================================================"
echo "SIMPLE EMBEDDING EXPERIMENT SWEEP WITH HYPERPARAMETER SWEEPS"
echo "========================================================================"
echo ""
echo "This script tests embedding transformations WITHOUT training."
echo "It uses text embeddings as input and tests if transformations help."
echo ""
echo "Advantages:"
echo "  - Fast: ~15-30 minutes total"
echo "  - No training needed"
echo "  - Isolates the transformation effect"
echo "  - Sweeps hyperparameters to find optimal settings"
echo ""
echo "Total Experiments: ~35"
echo ""
echo "Categories:"
echo "  1. Baseline (1 config)"
echo "  2. RMS matching (3 configs: scale=0.8,1.0,1.2)"
echo "  3. Batch distribution (1 config)"
echo "  4. K-nearest projection (10 configs: k∈{3,5,7,10}, α∈{0.2,0.3,0.5,0.7,0.9})"
echo "  5. Anchor + offset (7 configs: ε∈{0.01,0.02,0.05,0.1,0.15,0.2,0.3})"
echo "  6. Soft codebook (6 configs: size∈{128,256,512,1024}, τ∈{0.7,1.0})"
echo ""
echo "Results will be ranked by F1 score to identify winners."
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

echo "========================================================================"
echo "Starting experiments..."
echo "========================================================================"
echo ""

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

echo ""
echo "========================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================================================"
echo ""
echo "To view raw summary:"
echo "  cat $OUTPUT_DIR/summary.json | python -m json.tool"
echo ""
echo "To re-run analysis:"
echo "  python scripts/analyze_sweep_results.py --summary $OUTPUT_DIR/summary.json"
echo ""
