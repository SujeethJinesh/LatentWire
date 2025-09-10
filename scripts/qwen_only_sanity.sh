#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

# ========= TRAIN =========
export RUN="squad_qwen_only_m16_20250910_032844"
export OUT="runs/$RUN"; mkdir -p "$OUT"

# echo "Starting Qwen-only train (SQuAD, M=16, anchor='Answer: ')..."
# PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
# python -u latentwire/train.py \
#   --dataset squad \
#   --llama_id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
#   --qwen_id  "Qwen/Qwen2-0.5B-Instruct" \
#   --samples  16384 \
#   --epochs   5 \
#   --batch_size 32 \
#   --latent_len 16 \
#   --d_z 256 \
#   --encoder_type simple-st \
#   --encoder_use_chat_template \
#   --warm_anchor_text "Answer: " \
#   --fp16_mps \
#   --sequential_models \
#   --auto_resume \
#   --save_every 500 \
#   --save_dir "$OUT/ckpt" \
#   --debug \
#   2>&1 | tee -a "$OUT/train.log"

# ========= EVAL =========
EOUT="$OUT/squad_eval"; mkdir -p "$EOUT"

echo "Evaluating on SQuAD (robust decode: min_new_tokens=3, eos_ban_steps=6; first-token nucleus)..."
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -u latentwire/eval.py \
  --ckpt "$OUT/ckpt" \
  --dataset squad \
  --samples 200 \
  --max_new_tokens 8 \
  --token_budget_mode content_only \
  --token_budget_k 32 \
  --latent_anchor_text "Answer: " \
  --out_dir "$EOUT" \
  --sequential_eval \
  --min_new_tokens 3 \
  --eos_ban_steps 6 \
  --first_token_top_p 0.9 \
  --first_token_temperature 0.7 \
  --prefix_gain 1.0 \
  --debug \
  2>&1 | tee "$EOUT/eval.log"

# Optional: quick prefix gain sweep (no retrain) to confirm amplitude sensitivity
for G in 1.25 1.5 2.0; do
  EOUT_SWEEP="$OUT/squad_eval_gain_${G}"; mkdir -p "$EOUT_SWEEP"
  echo "Re-evaluating with prefix_gain=${G}..."
  PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python -u latentwire/eval.py \
    --ckpt "$OUT/ckpt" \
    --dataset squad \
    --samples 200 \
    --max_new_tokens 8 \
    --token_budget_mode content_only \
    --token_budget_k 32 \
    --latent_anchor_text "Answer: " \
    --out_dir "$EOUT_SWEEP" \
    --sequential_eval \
    --min_new_tokens 3 \
    --eos_ban_steps 6 \
    --first_token_top_p 0.9 \
    --first_token_temperature 0.7 \
    --prefix_gain $G \
    --debug \
    2>&1 | tee "$EOUT_SWEEP/eval.log"
done

echo "========================================="
echo "EVAL COMPLETE at $(date)"
echo "Results in: $OUT"
echo "========================================="
