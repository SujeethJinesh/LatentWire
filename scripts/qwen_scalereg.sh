#!/usr/bin/env bash
set -euo pipefail

# Activate env
source .venv/bin/activate

export RUN="squad_m16_scalereg_$(date +%Y%m%d_%H%M%S)"
export OUT="runs/$RUN"
mkdir -p "$OUT" "$OUT/squad_eval"

echo "Starting train (TinyLlama + Qwen2-0.5B, M=16, scale_L2=0.10, anchor='Answer: ')..."
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -u latentwire/train.py \
  --dataset squad \
  --llama_id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --qwen_id  "Qwen/Qwen2-0.5B-Instruct" \
  --samples  16384 \
  --epochs   5 \
  --batch_size 32 \
  --latent_len 16 \
  --d_z 256 \
  --encoder_type simple-st \
  --encoder_use_chat_template \
  --warm_anchor_text "Answer: " \
  --scale_l2 0.10 \
  --fp16_mps \
  --sequential_models \
  --auto_resume \
  --save_every 500 \
  --save_dir "$OUT/ckpt" \
  --debug 2>&1 | tee -a "$OUT/train.log"

echo "Evaluating on SQuAD (min_new_tokens=2, eos_ban_steps=6)..."
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -u latentwire/eval.py \
  --ckpt "$OUT/ckpt" \
  --dataset squad \
  --samples 200 \
  --max_new_tokens 8 \
  --latent_anchor_text "Answer: " \
  --out_dir "$OUT/squad_eval" \
  --sequential_eval \
  --device mps \
  --min_new_tokens 2 \
  --eos_ban_steps 6 \
  --first_token_top_p 0.9 \
  --first_token_temperature 0.7 \
  --prefix_gain 1.0 \
  --debug 2>&1 | tee "$OUT/squad_eval/eval.log"

# If you want to quickly check that evaluation is not starved by a quiet prefix,
# uncomment this sweep to see sensitivity to amplitude.
for PG in 0.5 1.0 2.0; do
  echo "Evaluating with prefix_gain=${PG} ..."
  PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python -u latentwire/eval.py \
    --ckpt "$OUT/ckpt" \
    --dataset squad \
    --samples 200 \
    --max_new_tokens 8 \
    --latent_anchor_text "Answer: " \
    --out_dir "$OUT/squad_eval_pg${PG//./}" \
    --sequential_eval \
    --device mps \
    --min_new_tokens 2 \
    --eos_ban_steps 6 \
    --first_token_top_p 0.9 \
    --first_token_temperature 0.7 \
    --prefix_gain "$PG" \
    --debug 2>&1 | tee "$OUT/squad_eval_pg${PG//./}/eval.log"
done

echo "==== METRICS (primary eval) ===="
cat "$OUT/squad_eval/metrics.json" || true
