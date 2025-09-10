#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export RUN="mps_m16_simple_$(date +%Y%m%d_%H%M%S)"
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
if [ "$battery_level" -lt 40 ]; then
    echo "Warning: Battery at ${battery_level}%. Consider charging first."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if [ -f "$OUT/eval.log" ] && [ $(stat -f%z "$OUT/eval.log" 2>/dev/null || echo 0) -gt 100000000 ]; then
    echo "Rotating large log file (>100MB)..."
    mv "$OUT/eval.log" "$OUT/eval.log.$(date +%Y%m%d_%H%M%S)"
fi

echo "========================================="
echo "Starting LatentWire training + evaluation"
echo "Started at: $(date)"
echo "Output directory: $OUT"
echo "========================================="

echo "Configuring power settings for travel..."
sudo pmset -b sleep 0
sudo pmset -b disablesleep 1
sudo pmset -b lidwake 0
sudo pmset -b standby 0
sudo pmset -b autopoweroff 0

echo "Starting train (M=16 for learnability)..."
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
caffeinate -dims python -u latentwire/train.py \
  --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --qwen_id  "Qwen/Qwen1.5-7B-Chat" \
  --samples  4096 \
  --epochs   5 \
  --batch_size 16 \
  --latent_len 16 \
  --d_z 256 \
  --encoder_type simple-st \
  --hotpot_config distractor \
  --grad_ckpt \
  --fp16_mps \
  --auto_resume \
  --save_every 200 \
  --save_dir "$OUT/ckpt" \
  2>&1 | tee -a "$OUT/train.log"

# Eval: Hotpot (distractor), fair token budget with larger K
echo "Evaluating on Hotpot (distractor)..."
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
caffeinate -dims python -u latentwire/eval.py \
  --ckpt "$OUT/ckpt" \
  --dataset hotpot \
  --hotpot_config distractor \
  --samples 50 \
  --max_new_tokens 6 \
  --token_budget_mode content_only \
  --token_budget_k 32 \
  --out_dir "$OUT/hotpot_eval" \
  --sequential_eval \
  2>&1 | tee "$OUT/hotpot_eval/eval.log"

# Eval: SQuAD
echo "Evaluating on SQuAD..."
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
caffeinate -dims python -u latentwire/eval.py \
  --ckpt "$OUT/ckpt" \
  --dataset squad \
  --samples 100 \
  --max_new_tokens 5 \
  --token_budget_mode content_only \
  --token_budget_k 32 \
  --out_dir "$OUT/squad_eval" \
  --sequential_eval \
  2>&1 | tee "$OUT/squad_eval/eval.log"

echo "========================================="
echo "EVAL COMPLETE at $(date)"
echo "========================================="

echo "==== STEP_TIMES ===="
if [ -f "$OUT/train.log" ]; then
    grep -E "sec/step" "$OUT/train.log" | tail -n 5 || true
else
    echo "No train.log found"
fi

echo "==== HOTPOT EVAL_JSON ===="
if [ -f "$OUT/hotpot_eval/metrics.json" ]; then
    cat "$OUT/hotpot_eval/metrics.json"
else
    echo "No hotpot metrics.json found"
fi

echo "==== SQUAD EVAL_JSON ===="
if [ -f "$OUT/squad_eval/metrics.json" ]; then
    cat "$OUT/squad_eval/metrics.json"
else
    echo "No squad metrics.json found"
fi

echo "========================================="
echo "All results saved to: $OUT"
echo "========================================="
