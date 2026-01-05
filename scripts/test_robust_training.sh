#!/usr/bin/env bash
set -e

# Test the robust training wrapper locally with small settings
# This is for quick validation before submitting to HPC

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/test_robust}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Testing robust training wrapper..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run with minimal settings for quick test
python telepathy/robust_training.py \
    --checkpoint_dir "$OUTPUT_DIR" \
    --max_retries 2 \
    --max_oom_retries 3 \
    --batch_size 8 \
    --min_batch_size 2 \
    --memory_threshold_gb 10.0 \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
    --samples 100 \
    --epochs 1 \
    --latent_len 8 \
    --d_z 64 \
    --encoder_type byte \
    --dataset squad \
    --sequential_models \
    --warm_anchor_text "Answer: " \
    --first_token_ce_weight 0.5

echo ""
echo "Test complete! Check logs in: $OUTPUT_DIR/robust_training.log"