#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

python latentwire/train.py \
  --llama_id "sshleifer/tiny-gpt2" \
  --qwen_id  "hf-internal-testing/tiny-random-GPTNeoXForCausalLM" \
  --samples  64 \
  --epochs   1 \
  --batch_size 1 \
  --latent_len 4 \
  --d_z 64 \
  --max_bytes 256 \
  --max_answer_tokens 16 \
  --hotpot_config distractor \
  --save_dir ./ckpt_toy
