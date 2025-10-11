#!/bin/bash
# Minimal viable training for quick experiments - 15 minute runs
# Use this to quickly test hyperparameter changes
set -euo pipefail

# H100 Optimizations
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_COMPILE=1
export USE_FLASH_ATTENTION_2=1
export TF32_MODE=1

# === MINIMAL CONFIGURATION ===
# Absolute minimum for meaningful signal
SAMPLES=5000       # 6% of full dataset
EPOCHS=10         # Minimal epochs
BATCH_SIZE=80     # Slightly larger than smoke test
LR=3e-4          # Higher LR for fast convergence

CHECKPOINT_DIR="runs/minimal/test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CHECKPOINT_DIR/logs"

echo "=== LatentWire Minimal Training (15 min) ==="
echo "Configuration:"
echo "  - Samples: $SAMPLES"
echo "  - Epochs: $EPOCHS"
echo "  - Total steps: $((SAMPLES * EPOCHS / BATCH_SIZE)) = ~625"
echo "  - Estimated time: 15 minutes"
echo "  - Expected first-token acc: 15-20%"
echo "  - Expected F1: 0.05-0.10"
echo ""

python latentwire/train.py \
  --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --models llama \
  --dataset squad \
  --samples $SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --latent_len 32 \
  --d_z 256 \
  --encoder_type byte \
  --first_token_ce_weight 5.0 \
  --first_token_entropy_weight 1.0 \
  --k_ce_weight 2.0 \
  --K 4 \
  --use_lora \
  --lora_r 64 \
  --lora_alpha 128 \
  --warm_anchor_text "Answer: " \
  --save_dir "$CHECKPOINT_DIR" \
  --save_every 50 \
  --grad_diag_interval 10 \
  --diagnostic_log "$CHECKPOINT_DIR/logs/diagnostics.jsonl" \
  --llama_device_map "auto" \
  --grad_ckpt \
  --seed 42 \
  2>&1 | tee "$CHECKPOINT_DIR/logs/training.log"

# Quick eval
python latentwire/eval.py \
  --ckpt "$CHECKPOINT_DIR" \
  --samples 100 \
  --dataset squad \
  --max_new_tokens 8 \
  --out_dir "$CHECKPOINT_DIR/eval" \
  2>&1 | tee "$CHECKPOINT_DIR/logs/eval.log"

# Results
echo ""
tail -5 "$CHECKPOINT_DIR/logs/diagnostics.jsonl" | jq -r '.models.llama | "Step \(.global_step): first_acc=\(.first_acc*100)%, entropy=\(.first_entropy)"' 2>/dev/null || true