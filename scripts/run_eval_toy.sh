#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

python latentwire/eval.py \
  --ckpt ./ckpt_toy \
  --samples 32 \
  --hotpot_config distractor \
  --max_new_tokens 24
