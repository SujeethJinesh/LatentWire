#!/usr/bin/env bash
set -e

# Configuration for improved cross-attention with bottlenecked gated architecture
OUTPUT_DIR="${OUTPUT_DIR:-runs/cross_attention_v2}"
SOURCE_MODEL="${SOURCE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
TRANSLATOR_TYPE="${TRANSLATOR_TYPE:-bottleneck_gated}"
BOTTLENECK_DIM="${BOTTLENECK_DIM:-1024}"  # Bottleneck for efficiency
SOFT_TOKENS="${SOFT_TOKENS:-48}"          # More soft tokens
DEPTH="${DEPTH:-6}"                        # Deeper but narrower
HEADS="${HEADS:-16}"                       # More heads in bottleneck space
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"
EVAL_EVERY="${EVAL_EVERY:-200}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
ATTENTION_REG_STEPS="${ATTENTION_REG_STEPS:-500}"
SEED="${SEED:-1234}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/cross_attention_v2_${TIMESTAMP}.log"

echo "Starting improved cross-attention interlingua experiment..."
echo "Log file: $LOG_FILE"
echo ""
echo "Configuration:"
echo "  Source model: $SOURCE_MODEL"
echo "  Target model: $TARGET_MODEL"
echo "  Translator type: $TRANSLATOR_TYPE"
echo "  Bottleneck dim: $BOTTLENECK_DIM"
echo "  Soft tokens: $SOFT_TOKENS"
echo "  Depth: $DEPTH layers"
echo "  Training steps: $TRAIN_STEPS"
echo "  Per-device batch: $PER_DEVICE_BATCH"
echo "  Effective batch size: $((PER_DEVICE_BATCH * 4)) (4 H100 GPUs)"
echo "  Attention reg steps: $ATTENTION_REG_STEPS"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "Key improvements:"
echo "  - Bottlenecked architecture (${BOTTLENECK_DIM}d internal)"
echo "  - Gated cross-attention (Flamingo-style)"
echo "  - Left-padding for decoder-only models"
echo "  - Question-only inputs, answer-only loss"
echo "  - Attention regularization for first ${ATTENTION_REG_STEPS} steps"
echo ""

# Run the improved cross-attention training
{
    torchrun --nproc_per_node=4 cross_model/experiments/cross_attention_v2.py \
        --source_model "$SOURCE_MODEL" \
        --target_model "$TARGET_MODEL" \
        --translator_type "$TRANSLATOR_TYPE" \
        --bottleneck_dim "$BOTTLENECK_DIM" \
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
        --attention_reg_steps "$ATTENTION_REG_STEPS" \
        --seed "$SEED" \
        --bf16 \
        --save_path "$OUTPUT_DIR/translator_v2_checkpoint.pt"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Complete! Results saved to:"
echo "  - $OUTPUT_DIR/translator_v2_checkpoint.pt"
echo "  - $LOG_FILE"