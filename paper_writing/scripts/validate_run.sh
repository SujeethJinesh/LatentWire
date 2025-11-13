#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

EXP_DIR="paper_writing/runs/validation_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$EXP_DIR"

TORCHRUN="torchrun --standalone --nproc_per_node=1"

$TORCHRUN paper_writing/cross_attention.py \
  --source_model mistralai/Mistral-7B-Instruct-v0.3 \
  --target_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --per_device_batch 1 \
  --eval_every 50 \
  --eval_samples 20 \
  --eval_batch_size 4 \
  --max_new_tokens 128 \
  --train_steps 50 \
  --warmup_steps 10 \
  --soft_tokens 64 \
  --bridge dit \
  --lr 1e-4 \
  --seed 1234 \
  --bf16 \
  --no_compile \
  --save_path "$EXP_DIR/translator.pt" \
  --log_dir "$EXP_DIR"
