#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

python latentwire/eval.py \
  --ckpt ./ckpt \
  --samples 200 \
  --max_new_tokens 24
