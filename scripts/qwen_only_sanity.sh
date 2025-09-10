#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export RUN="squad_qwen_only_m16_$(date +%Y%m%d_%H%M%S)"
export OUT="runs/$RUN"; mkdir -p "$OUT"

echo "Starting Qwen-only fast-iterate train (Qwen2-0.5B, M=16, anchor='Answer: ')..."
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -u latentwire/train.py \
  --dataset squad \
  --qwen_id  "Qwen/Qwen2-0.5B-Instruct" \
  --llama_id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --samples  16384 \
  --epochs   5 \
  --batch_size 32 \
  --latent_len 16 \
  --d_z 256 \
  --encoder_type simple-st \
  --encoder_use_chat_template \
  --warm_anchor_text "Answer: " \
  --fp16_mps \
  --sequential_models \
  --lambda_llama 0.0 \
  --lambda_qwen  1.0 \
  --acceptance_reg 0.05 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --auto_resume \
  --save_every 500 \
  --save_dir "$OUT/ckpt" \
  --debug \
  2>&1 | tee -a "$OUT/train.log"

echo "Evaluating on SQuAD (robust decode: min_new_tokens=2, eos_ban_steps=6)..."
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -u latentwire/eval.py \
  --ckpt "$OUT/ckpt" \
  --dataset squad \
  --samples 200 \
  --max_new_tokens 8 \
  --min_new_tokens 2 \
  --eos_ban_steps 6 \
  --token_budget_mode content_only \
  --token_budget_k 32 \
  --latent_anchor_text "Answer: " \
  --out_dir "$OUT/squad_eval" \
  --sequential_eval \
  --debug \
  2>&1 | tee "$OUT/squad_eval/eval.log"

echo "DONE. Results in $OUT"
