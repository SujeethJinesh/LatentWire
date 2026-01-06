#!/usr/bin/env bash
set -e

# Linear Probe Layer Sweep
# Evaluates linear probes across multiple layers to find optimal layer

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/linear_probe_sweep}"
DATASET="${DATASET:-sst2}"
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-33}"  # Llama 3.1 8B has 32 layers + embeddings
LAYER_STEP="${LAYER_STEP:-4}"
NUM_SEEDS="${NUM_SEEDS:-3}"   # Fewer seeds for sweep (faster)

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/layer_sweep_${DATASET}_${TIMESTAMP}.log"

echo "Starting Linear Probe Layer Sweep"
echo "Dataset: $DATASET"
echo "Model: $MODEL_ID"
echo "Layers: ${LAYER_START}:${LAYER_STEP}:${LAYER_END}"
echo "Seeds per layer: $NUM_SEEDS"
echo "Log file: $LOG_FILE"
echo ""

# Run layer sweep with tee to capture ALL output
{
    python telepathy/linear_probe_hidden_states.py \
        --dataset "$DATASET" \
        --model_id "$MODEL_ID" \
        --layer_sweep \
        --layer_start "$LAYER_START" \
        --layer_end "$LAYER_END" \
        --layer_step "$LAYER_STEP" \
        --num_seeds "$NUM_SEEDS" \
        --normalize l2 \
        --pooling last_token \
        --test_size 0.2 \
        --batch_size 8 \
        --device cuda \
        --output_dir "$OUTPUT_DIR"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Complete! Results saved to:"
echo "  - $OUTPUT_DIR/results.json"
echo "  - $LOG_FILE"
