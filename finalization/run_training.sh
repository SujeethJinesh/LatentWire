#!/bin/bash
# Example training script for LatentWire

# Set environment variables
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/experiment}"
SAMPLES="${SAMPLES:-1000}"
EPOCHS="${EPOCHS:-1}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting LatentWire training..."
echo "Output directory: $OUTPUT_DIR"
echo "Samples: $SAMPLES"
echo "Epochs: $EPOCHS"
echo ""

# Run training
python3 latentwire/train.py \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
    --samples "$SAMPLES" \
    --epochs "$EPOCHS" \
    --batch_size 64 \
    --latent_len 32 \
    --d_z 256 \
    --encoder_type byte \
    --dataset squad \
    --sequential_models \
    --warm_anchor_text "Answer: " \
    --first_token_ce_weight 0.5 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Training complete! Results saved to: $OUTPUT_DIR"
