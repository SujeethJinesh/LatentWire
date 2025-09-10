#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export RUN="squad_m16_big_$(date +%Y%m%d_%H%M%S)"
export OUT="runs/$RUN"; mkdir -p "$OUT"

echo $$ > "$OUT/eval.pid"
echo "Process PID: $$ (saved to $OUT/eval.pid)"
echo "To emergency stop: kill \$(cat $OUT/eval.pid)"

cleanup() {
    echo "Restoring power settings..."
    sudo pmset -b sleep 1
    sudo pmset -b disablesleep 0
    sudo pmset -b lidwake 1
    sudo pmset -b standby 1
    sudo pmset -b autopoweroff 1
    rm -f "$OUT/eval.pid"
}
trap cleanup EXIT

battery_level=$(pmset -g batt | grep -Eo "[0-9]+%" | cut -d% -f1)
if [ "${battery_level:-100}" -lt 40 ]; then
    echo "Warning: Battery at ${battery_level}%. Consider charging first."
    read -p "Continue anyway? (y/n) " -n 1 -r; echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

echo "========================================="
echo "Starting LatentWire training + evaluation (BIG MODELS, SQuAD)"
echo "Started at: $(date)"
echo "Output directory: $OUT"
echo "========================================="

echo "Configuring power settings for travel..."
sudo pmset -b sleep 0
sudo pmset -b disablesleep 1
sudo pmset -b lidwake 0
sudo pmset -b standby 0
sudo pmset -b autopoweroff 0

echo "Starting train (M=16, anchor='Answer:') on SQuAD..."
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -u latentwire/train.py \
  --dataset squad \
  --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --qwen_id  "Qwen/Qwen1.5-7B-Chat" \
  --samples  12000 \
  --epochs   3 \
  --batch_size 16 \
  --latent_len 16 \
  --d_z 256 \
  --encoder_type simple-st \
  --encoder_use_chat_template \
  --warm_anchor_text "Answer:" \
  --grad_ckpt \
  --fp16_mps \
  --sequential_models \
  --auto_resume \
  --save_every 500 \
  --save_dir "$OUT/ckpt" \
  --debug \
  2>&1 | tee -a "$OUT/train.log"

echo "Evaluating on SQuAD (anchor='Answer:', K=32 content_only)..."
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
caffeinate -dims python -u latentwire/eval.py \
  --ckpt "$OUT/ckpt" \
  --dataset squad \
  --samples 100 \
  --max_new_tokens 5 \
  --token_budget_mode content_only \
  --token_budget_k 32 \
  --latent_anchor_text "Answer:" \
  --out_dir "$OUT/squad_eval" \
  --sequential_eval \
  --debug \
  2>&1 | tee "$OUT/squad_eval/eval.log"

echo "========================================="
echo "EVAL COMPLETE at $(date)"
echo "========================================="

echo "==== STEP_TIMES ===="
grep -E "sec/step" "$OUT/train.log" | tail -n 5 || true

echo "==== SQUAD EVAL_JSON ===="
cat "$OUT/squad_eval/metrics.json" || true

echo "========================================="
echo "All results saved to: $OUT"
echo "========================================="
