#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export RUN="mps_m8_simple_$(date +%Y%m%d_%H%M%S)"
export OUT="runs/$RUN"; mkdir -p "$OUT"

# Save PID for emergency stop
echo $$ > "$OUT/eval.pid"
echo "Process PID: $$ (saved to $OUT/eval.pid)"
echo "To emergency stop: kill \$(cat $OUT/eval.pid)"

# Ensure cleanup happens even if script fails
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

# Check battery level before starting
battery_level=$(pmset -g batt | grep -Eo "[0-9]+%" | cut -d% -f1)
if [ "$battery_level" -lt 40 ]; then
    echo "Warning: Battery at ${battery_level}%. Consider charging first."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Log rotation to prevent disk fill
if [ -f "$OUT/eval.log" ] && [ $(stat -f%z "$OUT/eval.log" 2>/dev/null || echo 0) -gt 100000000 ]; then
    echo "Rotating large log file (>100MB)..."
    mv "$OUT/eval.log" "$OUT/eval.log.$(date +%Y%m%d_%H%M%S)"
fi

# Time estimates
echo "========================================="
echo "Starting LatentWire evaluation"
echo "Started at: $(date)"
echo "Estimated completion: ~30-45 minutes for 50 samples"
echo "Output directory: $OUT"
echo "========================================="

# Prevent sleep
echo "Configuring power settings for travel..."
sudo pmset -b sleep 0
sudo pmset -b disablesleep 1
sudo pmset -b lidwake 0
sudo pmset -b standby 0
sudo pmset -b autopoweroff 0

echo "Starting train..."
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -u latentwire/train.py \
  --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --qwen_id  "Qwen/Qwen1.5-7B-Chat" \
  --samples  2048 \
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

# Run eval
echo "Starting evaluation with fp16 precision and aggressive memory management..."
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 \
caffeinate -dims python -u latentwire/eval.py \
  --ckpt "$OUT/ckpt" \
  --samples 50 \
  --max_new_tokens 20 \
  --hotpot_config distractor \
  --out_dir "$OUT" \
  --sequential_eval \
  2>&1 | tee "$OUT/eval.log"

echo "========================================="
echo "EVAL COMPLETE at $(date)"
echo "========================================="

# Quick extract (what to send to me)
echo "==== STEP_TIMES ===="
if [ -f "$OUT/train.log" ]; then
    grep -E "sec/step" "$OUT/train.log" | tail -n 5
else
    echo "No train.log found"
fi

echo "==== EVAL_JSON ===="
if [ -f "$OUT/metrics.json" ]; then
    cat "$OUT/metrics.json"
else
    echo "No metrics.json found yet"
fi

echo "========================================="
echo "Evaluation completed successfully!"
echo "Results saved to: $OUT"
echo "========================================="