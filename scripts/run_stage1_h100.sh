#!/bin/bash
# Stage 1: Adapter-only training with compression
# Tests if adapter can reconstruct from compressed embeddings
set -euo pipefail

# H100 Optimizations
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_COMPILE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "==================================="
echo "STAGE 1: ADAPTER-ONLY TRAINING"
echo "==================================="
echo ""
echo "Purpose: Test if adapter can reconstruct embeddings from compression"
echo "Expected: ~70% F1 (from 82% baseline)"
echo ""

# We CAN use larger batch sizes since no KD/encoder
BATCH_SIZE=128  # Much larger than before!

python train_adapter_only.py \
  --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --compress_dim 512 \
  --compress_method pca \
  --adapter_hidden_mult 4 \
  --adapter_dropout 0.1 \
  --adapter_lr 5e-4 \
  --samples 10000 \
  --epochs 3 \
  --batch_size $BATCH_SIZE \
  --recon_weight 1.0 \
  --ce_weight 1.0 \
  --eval_every 1 \
  --eval_samples 500 \
  --save_dir "runs/stage1_adapter_only"

echo ""
echo "Stage 1 complete! Check runs/stage1_adapter_only for results"