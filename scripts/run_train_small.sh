#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

python latentwire/train.py \
  --llama_id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --qwen_id  "Qwen/Qwen2-0.5B-Instruct" \
  --samples  512 \
  --epochs   1 \
  --batch_size 1 \
  --latent_len 8 \
  --d_z 256 \
  --max_bytes 512 \
  --max_answer_tokens 32 \
  --save_dir ./ckpt
