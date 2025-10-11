#!/bin/bash
# Optimized training script for 4x H100 GPUs - Minimum viable configuration
# Balances training time, checkpoint storage, and expected performance
set -euo pipefail

# H100 Optimizations
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_COMPILE=1
export USE_FLASH_ATTENTION_2=1
export TF32_MODE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# === OPTIMIZED CONFIGURATION ===
# Reduced but still viable for meaningful results
SAMPLES=20000       # 25% of full dataset (was 80000)
EPOCHS=20          # Reduced from 50 (still 10x more than smoke test)
BATCH_SIZE=256     # Larger batch for faster training (was 128)
GRAD_ACCUM=1       # Effective batch = 256
LR=2e-4           # Slightly higher LR for faster convergence

# Checkpoint Configuration
CHECKPOINT_DIR="runs/optimized/lora_20ep"
SAVE_EVERY=500    # More frequent saves for early stopping
MAX_CHECKPOINTS=5  # Keep only 5 best checkpoints to save disk space

echo "=== LatentWire Optimized Training on 4x H100s ==="
echo "Configuration:"
echo "  - Samples: $SAMPLES (25% of full dataset)"
echo "  - Epochs: $EPOCHS (10x smoke test)"
echo "  - Batch Size: $BATCH_SIZE (2x hero config)"
echo "  - Steps per epoch: $((SAMPLES / BATCH_SIZE)) = 78"
echo "  - Total steps: $((SAMPLES * EPOCHS / BATCH_SIZE)) = 1,560"
echo "  - Checkpoints: Every $SAVE_EVERY steps (~3 per training)"
echo "  - Estimated training time: ~45 minutes"
echo ""
echo "Expected Outcomes:"
echo "  - First-token accuracy: 25-35%"
echo "  - F1 Score: 0.15-0.25"
echo "  - Good enough to validate approach"
echo ""

mkdir -p "$CHECKPOINT_DIR/logs"

python latentwire/train.py \
  --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --models llama \
  --dataset squad \
  --samples $SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --grad_accum_steps $GRAD_ACCUM \
  --lr $LR \
  --lr_scheduler "cosine" \
  --warmup_steps 100 \
  --weight_decay 0.01 \
  --latent_len 32 \
  --d_z 256 \
  --encoder_type byte \
  --sequential_models \
  --first_token_ce_weight 3.0 \
  --k_token_ce_weight 1.5 \
  --k_token_ce_k 8 \
  --kd_weight 0.5 \
  --kd_tau 2.0 \
  --entropy_weight 0.5 \
  --use_lora 1 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --warm_anchor_text "Answer: " \
  --save_dir "$CHECKPOINT_DIR" \
  --save_every $SAVE_EVERY \
  --max_checkpoints_to_keep $MAX_CHECKPOINTS \
  --save_best \
  --eval_every 200 \
  --eval_samples 200 \
  --early_stopping_patience 5 \
  --early_stopping_metric "first_token_top1" \
  --diagnostic_log "$CHECKPOINT_DIR/logs/diagnostics.jsonl" \
  --diagnostic_interval 20 \
  --llama_device_map "auto" \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --max_grad_norm 0.5 \
  --seed 42 \
  2>&1 | tee "$CHECKPOINT_DIR/logs/training.log"

echo ""
echo "=== Training Complete ==="
echo "Running quick evaluation..."

python latentwire/eval.py \
  --ckpt "$CHECKPOINT_DIR/ckpt_best" \
  --samples 500 \
  --dataset squad \
  --max_new_tokens 12 \
  --out_dir "$CHECKPOINT_DIR/eval" \
  2>&1 | tee "$CHECKPOINT_DIR/logs/eval.log"

# Display results
if [ -f "$CHECKPOINT_DIR/eval/metrics.json" ]; then
    echo "=== Results ==="
    python -c "
import json
with open('$CHECKPOINT_DIR/eval/metrics.json') as f:
    m = json.load(f)
    print(f'Text F1: {m.get(\"text\",{}).get(\"llama\",{}).get(\"f1\",0):.3f}')
    print(f'Latent F1: {m.get(\"latent\",{}).get(\"llama\",{}).get(\"f1\",0):.3f}')
    print(f'First-token Acc: {m.get(\"latent\",{}).get(\"llama\",{}).get(\"first_token_top1\",0):.1%}')
"
fi