#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export RUN="mps_m8_simple_$(date +%Y%m%d_%H%M%S)"
export OUT="runs/$RUN"; mkdir -p "$OUT"

# 1) Train (Preset A: safe)
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -u latentwire/train.py \
  --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --qwen_id  "Qwen/Qwen1.5-7B-Chat" \
  --samples  1024 \
  --epochs   1 \
  --batch_size 16 \
  --latent_len 8 \
  --d_z 256 \
  --encoder_type simple-st \
  --hotpot_config distractor \
  --grad_ckpt \
  --fp16_mps \
  --auto_resume \
  --save_every 100 \
  --save_dir "$OUT/ckpt" \
  2>&1 | tee -a "$OUT/train.log"

PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -u latentwire/eval.py \
  --ckpt "$OUT/ckpt" \
  --samples 50 \
  --max_new_tokens 20 \
  --hotpot_config distractor \
  --out_dir "$OUT" \
  2>&1 | tee "$OUT/eval.log"

# Quick extract (what to send to me)
echo "==== STEP_TIMES ===="
grep -E "sec/step" "$OUT/train.log" | tail -n 5
echo "==== EVAL_JSON ===="
cat "$OUT/metrics.json"