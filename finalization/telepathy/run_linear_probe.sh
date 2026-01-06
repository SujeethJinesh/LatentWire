#!/usr/bin/env bash
set -e

# Linear Probe Hidden States Baseline
# Runs standard linear probe evaluation on LLM hidden states

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/linear_probe}"
DATASET="${DATASET:-sst2}"
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
LAYER_IDX="${LAYER_IDX:-16}"
NUM_SEEDS="${NUM_SEEDS:-5}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/linear_probe_${DATASET}_layer${LAYER_IDX}_${TIMESTAMP}.log"

echo "Starting Linear Probe Baseline"
echo "Dataset: $DATASET"
echo "Model: $MODEL_ID"
echo "Layer: $LAYER_IDX"
echo "Seeds: $NUM_SEEDS"
echo "Log file: $LOG_FILE"
echo ""

# Run linear probe evaluation with tee to capture ALL output
{
    python telepathy/linear_probe_hidden_states.py \
        --dataset "$DATASET" \
        --model_id "$MODEL_ID" \
        --layer_idx "$LAYER_IDX" \
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
