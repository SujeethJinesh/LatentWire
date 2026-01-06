#!/usr/bin/env bash
#
# Embedding Experiment Sweep Script
# Runs all embedding distribution experiments quickly (1 epoch each)
#
# Usage: PYTHONPATH=. bash scripts/run_embed_sweep.sh
#

set -e  # Exit on error

echo "========================================================================"
echo "EMBEDDING EXPERIMENT SWEEP"
echo "========================================================================"
echo ""
echo "This script will run ~11 experiments to test different approaches"
echo "for bridging the embedding distribution mismatch between learned"
echo "encoders and frozen LLM expectations."
echo ""
echo "Each experiment trains for 1 epoch (~1000 samples) and evaluates"
echo "on 200 samples for quick feedback."
echo ""
echo "Estimated time: ~30-60 minutes total (depends on GPU)"
echo "========================================================================"
echo ""

# Configuration
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-1000}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LATENT_LEN="${LATENT_LEN:-32}"
D_Z="${D_Z:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/embed_sweep}"

echo "Configuration:"
echo "  MODEL_ID: $MODEL_ID"
echo "  DATASET: $DATASET"
echo "  SAMPLES: $SAMPLES"
echo "  EPOCHS: $EPOCHS"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  LATENT_LEN: $LATENT_LEN"
echo "  D_Z: $D_Z"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# Set up Python path
export PYTHONPATH=.

# Enable MPS fallback for Mac
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Check if we should use fp16
USE_FP16=""
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, using fp16"
    USE_FP16="--fp16"
else
    echo "No CUDA detected, using fp32"
fi

echo ""
echo "========================================================================"
echo "Starting experiment sweep..."
echo "========================================================================"
echo ""

# Run all experiments
python scripts/run_embed_sweep_train.py \
    --experiment all \
    --model_id "$MODEL_ID" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --latent_len "$LATENT_LEN" \
    --d_z "$D_Z" \
    --output_dir "$OUTPUT_DIR" \
    $USE_FP16

echo ""
echo "========================================================================"
echo "Sweep complete!"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary:"
cat "$OUTPUT_DIR/experiment_summary.json" | python -m json.tool | head -50 || echo "(Summary file not found)"
echo ""
echo "To analyze results:"
echo "  python -m json.tool $OUTPUT_DIR/experiment_summary.json"
echo ""
echo "To view individual experiment logs:"
echo "  tail $OUTPUT_DIR/*/diagnostics.jsonl"
echo ""
