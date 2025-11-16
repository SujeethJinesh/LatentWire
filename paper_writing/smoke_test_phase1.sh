#!/usr/bin/env bash
set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load gcc/13.1.0 || true
    module load conda/24.3.0-0 || true
    module load stockcuda/12.6.2 || true
    module load cudnn/cuda12/9.3.0.75 || true
fi

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SOURCE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
TARGET_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

OUTPUT_DIR="paper_writing/runs/smoke_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train.log"

RANDOM_PORT=$((29500 + RANDOM % 1000))

torchrun --standalone --nproc_per_node=1 --master_port "$RANDOM_PORT" paper_writing/cross_attention.py \
    --source_model "$SOURCE_MODEL" \
    --target_model "$TARGET_MODEL" \
    --bridge dit \
    --dit_dim 512 \
    --dit_depth 2 \
    --dit_heads 4 \
    --dit_steps_train 1 \
    --dit_steps_eval 1 \
    --soft_tokens 64 \
    --per_device_batch 1 \
    --train_steps 20 \
    --warmup_steps 5 \
    --eval_every 10 \
    --eval_samples 8 \
    --eval_batch_size 4 \
    --max_new_tokens 64 \
    --decode_loss_weight 0.0 \
    --info_nce_weight 0.01 \
    --lr 1e-4 \
    --weight_decay 0.0 \
    --early_stop_patience 2 \
    --seed 1234 \
    --dataset gsm8k \
    --eval_prompt_mode soft_plus_text \
    --bf16 \
    --no_compile \
    --save_path "$OUTPUT_DIR/checkpoint.pt" \
    --log_dir "$OUTPUT_DIR" \
    | tee "$LOG_FILE"

