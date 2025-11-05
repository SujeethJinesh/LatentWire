#!/usr/bin/env bash
set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/cross_attention}"
SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
TRANSLATOR_TYPE="${TRANSLATOR_TYPE:-cross_attn}"
SOFT_TOKENS="${SOFT_TOKENS:-32}"
DEPTH="${DEPTH:-2}"
HEADS="${HEADS:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
EVAL_EVERY="${EVAL_EVERY:-200}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-1234}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/cross_attention_${TIMESTAMP}.log"

echo "Starting cross-attention interlingua experiment..."
echo "Log file: $LOG_FILE"
echo ""
echo "Configuration:"
echo "  Source model: $SOURCE_MODEL"
echo "  Target model: $TARGET_MODEL"
echo "  Translator type: $TRANSLATOR_TYPE"
echo "  Soft tokens: $SOFT_TOKENS"
echo "  Training steps: $TRAIN_STEPS"
echo "  Per-device batch: $PER_DEVICE_BATCH"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Run the cross-attention training with torchrun for 4 GPUs
{
    torchrun --nproc_per_node=4 cross_model/experiments/cross_attention.py \
        --source_model "$SOURCE_MODEL" \
        --target_model "$TARGET_MODEL" \
        --translator_type "$TRANSLATOR_TYPE" \
        --soft_tokens "$SOFT_TOKENS" \
        --depth "$DEPTH" \
        --heads "$HEADS" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --train_steps "$TRAIN_STEPS" \
        --warmup_steps "$WARMUP_STEPS" \
        --per_device_batch "$PER_DEVICE_BATCH" \
        --eval_every "$EVAL_EVERY" \
        --eval_samples "$EVAL_SAMPLES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --seed "$SEED" \
        --bf16 \
        --save_path "$OUTPUT_DIR/translator_checkpoint.pt"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Complete! Results saved to:"
echo "  - $OUTPUT_DIR/translator_checkpoint.pt"
echo "  - $LOG_FILE"