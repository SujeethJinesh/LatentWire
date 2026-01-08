#!/usr/bin/env bash
# Wrapper script to handle SQuAD loading issues on HPC with Python 3.11

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/experiment}"
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Set datasets cache to avoid fresh download issues
export HF_HOME="${HF_HOME:-/projects/m000066/sujinesh/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/training_${TIMESTAMP}.log"

echo "Starting training with SQuAD dataset fix..."
echo "Log file: $LOG_FILE"
echo "Python version: $(python3 --version)"
echo "Datasets cache: $HF_DATASETS_CACHE"
echo ""

# Run training with proper error handling
{
    python latentwire/train.py \
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
        --samples 1000 \
        --epochs 1 \
        --batch_size 64 \
        --latent_len 32 \
        --d_z 256 \
        --encoder_type byte \
        --dataset squad \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --output_dir "$OUTPUT_DIR"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete! Check results in $OUTPUT_DIR"