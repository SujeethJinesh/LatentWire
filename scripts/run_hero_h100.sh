#!/bin/bash
# Hero training script optimized for 4x H100 GPUs
# Scales up from smoke test (640 samples, 2 epochs) to full training (80K samples, 50 epochs)
set -euo pipefail

# H100 Optimizations
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_COMPILE=1
export USE_FLASH_ATTENTION_2=1
export TF32_MODE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0

# Training Configuration
SAMPLES=80000        # Full dataset (was 640 in smoke test)
EPOCHS=50           # Extended training (was 2 in smoke test)
BATCH_SIZE=128      # Doubled from smoke test (was 64)
GRAD_ACCUM=2        # Effective batch = 256
LR=1e-4            # Higher learning rate for faster convergence
CHECKPOINT_DIR="runs/hero/lora_50ep"
LOG_DIR="$CHECKPOINT_DIR/logs"

# Create directories
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

echo "=== LatentWire Hero Training on 4x H100s ==="
echo "Configuration:"
echo "  - Hardware: 4x H100 GPUs (320GB total VRAM)"
echo "  - Samples: $SAMPLES (full SQuAD training set)"
echo "  - Epochs: $EPOCHS"
echo "  - Batch Size: $BATCH_SIZE per GPU"
echo "  - Gradient Accumulation: $GRAD_ACCUM steps"
echo "  - Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  - Learning Rate: $LR"
echo "  - Checkpoint Dir: $CHECKPOINT_DIR"
echo ""
echo "Expected Outcomes:"
echo "  - First-token accuracy: 40-50% by epoch 25"
echo "  - F1 Score: 0.30-0.40 by epoch 50"
echo "  - GPU Utilization: 85-90% memory usage"
echo "  - Training Speed: 1.5-2.0 sec/step"
echo ""

python latentwire/train.py \
  --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --models llama \
  --dataset squad \
  --samples $SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --grad_accum_steps $GRAD_ACCUM \
  --lr $LR \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --latent_len 32 \
  --d_z 256 \
  --encoder_type byte \
  --max_enc_tokens 2048 \
  --sequential_models \
  --first_token_ce_weight 2.0 \
  --k_token_ce_weight 1.0 \
  --k_token_ce_k 8 \
  --kd_weight 0.5 \
  --kd_tau 2.0 \
  --kd_first_k 8 \
  --latent_align_weight 0.5 \
  --entropy_weight 0.5 \
  --label_smoothing 0.1 \
  --use_lora 1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
  --warm_anchor_text "Answer: " \
  --save_dir "$CHECKPOINT_DIR" \
  --save_every 1000 \
  --save_best \
  --eval_every 500 \
  --eval_samples 500 \
  --diagnostic_log "$LOG_DIR/diagnostics.jsonl" \
  --diagnostic_interval 50 \
  --llama_device_map "auto" \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --max_grad_norm 1.0 \
  --seed 42 \
  --deterministic \
  --log_level INFO \
  --require_cuda "yes" \
  2>&1 | tee "$LOG_DIR/training.log"

echo ""
echo "=== Training Complete ==="
echo "Checkpoint saved to: $CHECKPOINT_DIR"
echo ""

# Run evaluation on the best checkpoint
echo "Running evaluation on best checkpoint..."
python latentwire/eval.py \
  --ckpt "$CHECKPOINT_DIR/ckpt_best" \
  --samples 1000 \
  --dataset squad \
  --max_new_tokens 24 \
  --chunk_size 64 \
  --out_dir "$CHECKPOINT_DIR/eval" \
  --embedding_replay true \
  --embedding_baseline_modes '["raw","anchor","adapter"]' \
  --llama_device_map "auto" \
  2>&1 | tee "$LOG_DIR/evaluation.log"

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: $CHECKPOINT_DIR/eval/metrics.json"
echo ""

# Display summary
if [ -f "$CHECKPOINT_DIR/eval/metrics.json" ]; then
    echo "=== Results Summary ==="
    python -c "
import json
with open('$CHECKPOINT_DIR/eval/metrics.json') as f:
    metrics = json.load(f)
    print(f'Text Baseline F1: {metrics.get(\"text\", {}).get(\"llama\", {}).get(\"f1\", 0):.3f}')
    print(f'Latent F1: {metrics.get(\"latent\", {}).get(\"llama\", {}).get(\"f1\", 0):.3f}')
    print(f'First-token Acc: {metrics.get(\"latent\", {}).get(\"llama\", {}).get(\"first_token_top1\", 0):.1%}')
    print(f'Compression: {metrics.get(\"compression\", {}).get(\"llama\", 0):.1f}x')

    # Embedding baselines
    if 'embedding_baselines' in metrics:
        print('\\nEmbedding Baselines:')
        for mode in ['raw', 'anchor', 'adapter']:
            if mode in metrics['embedding_baselines']:
                f1 = metrics['embedding_baselines'][mode].get('metrics', {}).get('llama', {}).get('f1', 0)
                print(f'  - {mode:8s}: F1 = {f1:.3f}')
"
fi