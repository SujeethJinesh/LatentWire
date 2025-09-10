#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export RUN="squad_m16_small_20250909_225821"
export OUT="runs/$RUN"; mkdir -p "$OUT"

trap 'rm -f "$OUT/eval.pid"' EXIT
echo $$ > "$OUT/eval.pid"

echo "Starting fast-iterate train (TinyLlama + Qwen2-0.5B, M=16, anchor='Answer: ')..."
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -u latentwire/train.py \
  --dataset squad \
  --llama_id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --qwen_id  "Qwen/Qwen2-0.5B-Instruct" \
  --samples  16000 \
  --epochs   5 \
  --batch_size 32 \
  --latent_len 16 \
  --d_z 256 \
  --encoder_type simple-st \
  --encoder_use_chat_template \
  --warm_anchor_text "Answer: " \
  --fp16_mps \
  --sequential_models \
  --grad_ckpt \
  --auto_resume \
  --save_every 500 \
  --save_dir "$OUT/ckpt" \
  2>&1 | tee -a "$OUT/train.log"

echo "Evaluating on SQuAD..."
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -u latentwire/eval.py \
  --ckpt "$OUT/ckpt" \
  --dataset squad \
  --samples 200 \
  --max_new_tokens 8 \
  --token_budget_mode content_only \
  --token_budget_k 32 \
  --latent_anchor_text "Answer: " \
  --out_dir "$OUT/squad_eval" \
  --sequential_eval \
  --debug \
  2>&1 | tee "$OUT/squad_eval/eval.log"

echo "========================================="
echo "EVAL COMPLETE at $(date)"
echo "========================================="
cat "$OUT/squad_eval/metrics.json" || true
