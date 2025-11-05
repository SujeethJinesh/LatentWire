#!/usr/bin/env bash
set -e

OUTPUT_DIR="${OUTPUT_DIR:-runs/dit_interlingua}"
SOURCE_MODEL="${SOURCE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"

MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
EVAL_EVERY="${EVAL_EVERY:-200}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-1234}"

# Translator sizing
B="${B:-1024}"           # bottleneck width
COND_TOKENS="${COND_TOKENS:-64}"
DIT_DEPTH="${DIT_DEPTH:-6}"
DIT_HEADS="${DIT_HEADS:-16}"
LM_AUX_WEIGHT="${LM_AUX_WEIGHT:-0.5}"

mkdir -p "$OUTPUT_DIR"
TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/dit_${TS}.log"

echo "Starting DiT interlingua experiment on 4×H100..."
echo "Log file: $LOG_FILE"
echo "Effective batch size: $((PER_DEVICE_BATCH * 4))"
{
  torchrun --nproc_per_node=4 dit_interlingua.py \
    --source_model "$SOURCE_MODEL" \
    --target_model "$TARGET_MODEL" \
    --max_prompt_len "$MAX_PROMPT_LEN" \
    --per_device_batch "$PER_DEVICE_BATCH" \
    --train_steps "$TRAIN_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --eval_every "$EVAL_EVERY" \
    --eval_samples "$EVAL_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --seed "$SEED" \
    --bf16 \
    --bottleneck_dim "$B" \
    --cond_tokens "$COND_TOKENS" \
    --dit_depth "$DIT_DEPTH" \
    --dit_heads "$DIT_HEADS" \
    --lm_aux_weight "$LM_AUX_WEIGHT" \
    --save_path "$OUTPUT_DIR/translator.pt"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Complete. Artifacts:"
echo "  - $OUTPUT_DIR/translator.pt"
echo "  - $LOG_FILE"
