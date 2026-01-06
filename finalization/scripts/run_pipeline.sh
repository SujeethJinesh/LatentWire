#!/usr/bin/env bash
# Main pipeline script for LatentWire training and evaluation.
#
# This script runs the complete training and evaluation pipeline.

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/pipeline}"
SAMPLES="${SAMPLES:-87599}"
EPOCHS="${EPOCHS:-24}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LATENT_LEN="${LATENT_LEN:-32}"
D_Z="${D_Z:-256}"
DATASET="${DATASET:-squad}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/pipeline_${TIMESTAMP}.log"

echo "Starting LatentWire pipeline..."
echo "Log file: $LOG_FILE"
echo "Configuration:"
echo "  Samples: $SAMPLES"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Latent length: $LATENT_LEN"
echo "  Latent dimension: $D_Z"
echo "  Dataset: $DATASET"
echo ""

# Run training with tee to capture output
{
    echo "============================================================"
    echo "Phase 1: Training"
    echo "============================================================"

    python latentwire/train.py \
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
        --samples "$SAMPLES" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --latent_len "$LATENT_LEN" \
        --d_z "$D_Z" \
        --encoder_type byte \
        --dataset "$DATASET" \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --output_dir "$OUTPUT_DIR/checkpoint"

    echo ""
    echo "============================================================"
    echo "Phase 2: Evaluation"
    echo "============================================================"

    # Find the latest checkpoint
    LATEST_CHECKPOINT=$(ls -d "$OUTPUT_DIR"/checkpoint/epoch* 2>/dev/null | sort -V | tail -1)

    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "Error: No checkpoint found in $OUTPUT_DIR/checkpoint"
        exit 1
    fi

    echo "Evaluating checkpoint: $LATEST_CHECKPOINT"

    python latentwire/eval.py \
        --ckpt "$LATEST_CHECKPOINT" \
        --samples 200 \
        --max_new_tokens 12 \
        --dataset "$DATASET" \
        --sequential_eval \
        --fresh_eval \
        --calibration embed_rms \
        --latent_anchor_mode text \
        --latent_anchor_text "Answer: " \
        --append_bos_after_prefix yes \
        --output_dir "$OUTPUT_DIR/eval"

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Pipeline complete! Results saved to:"
echo "  - Checkpoint: $OUTPUT_DIR/checkpoint"
echo "  - Evaluation: $OUTPUT_DIR/eval"
echo "  - Log: $LOG_FILE"