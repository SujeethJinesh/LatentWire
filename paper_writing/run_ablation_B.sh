#!/usr/bin/env bash
set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load gcc/13.1.0
    module load conda/24.3.0-0
    module load stockcuda/12.6.2
    module load cudnn/cuda12/9.3.0.75
fi

echo "=========================================="
echo "ABLATION B – KL ONLY"
echo "=========================================="
echo "Start time: $(date)"
echo ""

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

detect_nproc() {
    if [[ -n "${NUM_GPUS:-}" ]]; then
        echo "$NUM_GPUS"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
        if [[ "$count" -gt 0 ]]; then
            echo "$count"
            return
        fi
    fi
    python - <<'PY' 2>/dev/null || echo 1
import torch
print(torch.cuda.device_count() or 1)
PY
}

NPROC=$(detect_nproc)
echo "Using $NPROC GPU(s) (set NUM_GPUS to override)."

SOURCE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
TARGET_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
PER_DEVICE_BATCH=2
EVAL_EVERY=250
EVAL_SAMPLES=200
MAX_NEW_TOKENS=256
PROMPT_MODE="soft_plus_text"
KL_MAX_LEN=512
KL_TOKENS=20

OUTPUT_DIR="paper_writing/runs/ablB_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$OUTPUT_DIR"
SUMMARY_LOG="$OUTPUT_DIR/summary.log"
echo "Run directory: $OUTPUT_DIR" | tee "$SUMMARY_LOG"

CONFIG_NAME="ablB_kl_only"
EXP_DIR="$OUTPUT_DIR/$CONFIG_NAME"
mkdir -p "$EXP_DIR"
LOG_FILE="$EXP_DIR/train.log"

RANDOM_PORT=$((29500 + RANDOM % 1000))

{
torchrun --standalone --nproc_per_node="$NPROC" --master_port "$RANDOM_PORT" paper_writing/cross_attention.py \
    --source_model "$SOURCE_MODEL" \
    --target_model "$TARGET_MODEL" \
    --per_device_batch "$PER_DEVICE_BATCH" \
    --eval_every "$EVAL_EVERY" \
    --eval_samples "$EVAL_SAMPLES" \
    --eval_batch_size 36 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --eval_prompt_mode "$PROMPT_MODE" \
    --decode_loss_weight 0.0 \
    --bridge dit \
    --soft_tokens -1 \
    --dit_dim 512 \
    --dit_depth 6 \
    --dit_heads 8 \
    --dit_steps_train 4 \
    --dit_steps_eval 8 \
    --dit_dropout 0.1 \
    --dit_pool mean \
    --dit_loss_weight 0.1 \
    --info_nce_weight 0.05 \
    --train_steps 2000 \
    --warmup_steps 200 \
    --early_stop_patience 3 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --seed 1234 \
    --bf16 \
    --no_compile \
    --save_path "$EXP_DIR/checkpoint.pt" \
    --log_dir "$EXP_DIR"
} 2>&1 | tee "$LOG_FILE"

echo "End time: $(date)" | tee -a "$SUMMARY_LOG"
echo "Logs: $LOG_FILE" | tee -a "$SUMMARY_LOG"
