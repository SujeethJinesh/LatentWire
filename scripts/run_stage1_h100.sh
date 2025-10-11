#!/bin/bash
# Stage 1: Adapter-only training with compression
# Tests if adapter can reconstruct from compressed embeddings
set -euo pipefail

# Set PYTHONPATH if not already set
export PYTHONPATH="${PYTHONPATH:-.}"

# H100 Optimizations
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_COMPILE=1

# Disable NVIDIA MPS to avoid daemon connection errors (Error 805)
# MPS can cause CUDA initialization failures on some HPC configurations
export CUDA_MPS_PIPE_DIRECTORY=/dev/null
export CUDA_MPS_LOG_DIRECTORY=/dev/null

# Verify GPU detection before training
echo "Checking GPU availability..."
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU count: {torch.cuda.device_count()}'); [print(f\"  GPU {i}: {torch.cuda.get_device_name(i)}\") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('  ERROR: No GPUs detected!')"
echo ""

if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: No CUDA GPUs detected! Cannot proceed with training."
    echo "Please check:"
    echo "  - nvidia-smi shows GPUs"
    echo "  - CUDA drivers are loaded"
    echo "  - PyTorch CUDA installation: pip list | grep torch"
    exit 1
fi

# Configuration
CHECKPOINT_DIR="runs/stage1_adapter_only"
BATCH_SIZE=32  # Reduced to avoid OOM on multi-GPU setup
SAMPLES=10000
EPOCHS=3

echo "==================================="
echo "STAGE 1: ADAPTER-ONLY TRAINING"
echo "==================================="
echo ""
echo "Configuration:"
echo "  - Model: meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "  - Compression: 4096 → 512 (8x compression via PCA)"
echo "  - Batch Size: $BATCH_SIZE (larger due to no KD/encoder)"
echo "  - Samples: $SAMPLES"
echo "  - Epochs: $EPOCHS"
echo "  - Steps: $((SAMPLES * EPOCHS / BATCH_SIZE)) total"
echo "  - Output: $CHECKPOINT_DIR"
echo ""
echo "Expected Performance:"
echo "  - Baseline (no compression): 82% F1"
echo "  - Target (with compression): 70% F1"
echo "  - If <50% F1: Adapter architecture issue"
echo ""

# Create log directory
mkdir -p "$CHECKPOINT_DIR/logs"

# Log start time
echo "Starting training at $(date)" | tee "$CHECKPOINT_DIR/logs/training.log"
echo "" | tee -a "$CHECKPOINT_DIR/logs/training.log"

# Run training with output logged
python train_adapter_only.py \
  --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --compress_dim 512 \
  --compress_method pca \
  --adapter_hidden_mult 4 \
  --adapter_dropout 0.1 \
  --adapter_lr 5e-4 \
  --samples $SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --recon_weight 1.0 \
  --ce_weight 1.0 \
  --eval_every 1 \
  --eval_samples 500 \
  --save_dir "$CHECKPOINT_DIR" \
  --diagnostic_log "$CHECKPOINT_DIR/logs/diagnostics.jsonl" \
  2>&1 | tee -a "$CHECKPOINT_DIR/logs/training.log"

echo "" | tee -a "$CHECKPOINT_DIR/logs/training.log"
echo "=== Training Complete at $(date) ===" | tee -a "$CHECKPOINT_DIR/logs/training.log"

# Run evaluation if checkpoint exists
if [ -f "$CHECKPOINT_DIR/adapter_only_best.pt" ]; then
    echo "Running evaluation..." | tee -a "$CHECKPOINT_DIR/logs/training.log"

    python -c "
import torch
import json
from pathlib import Path

checkpoint = torch.load('$CHECKPOINT_DIR/adapter_only_best.pt', map_location='cpu')
config = checkpoint.get('config', {})
best_f1 = checkpoint.get('best_f1', 0)
epoch = checkpoint.get('epoch', 0)

print(f'Best checkpoint from epoch {epoch}')
print(f'Best F1: {best_f1:.3f}')

# Save summary
summary = {
    'best_f1': best_f1,
    'epoch': epoch,
    'compression': f\"{config.get('input_dim', 4096)} → {config.get('compress_dim', 512)}\",
    'method': config.get('compress_method', 'pca'),
    'samples': config.get('samples', 0)
}

with open('$CHECKPOINT_DIR/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
" 2>&1 | tee -a "$CHECKPOINT_DIR/logs/training.log"
fi

# Display results summary
echo "" | tee -a "$CHECKPOINT_DIR/logs/training.log"
echo "==================================" | tee -a "$CHECKPOINT_DIR/logs/training.log"
echo "STAGE 1 RESULTS SUMMARY" | tee -a "$CHECKPOINT_DIR/logs/training.log"
echo "==================================" | tee -a "$CHECKPOINT_DIR/logs/training.log"

if [ -f "$CHECKPOINT_DIR/summary.json" ]; then
    python -c "
import json
with open('$CHECKPOINT_DIR/summary.json') as f:
    s = json.load(f)
    print(f\"  Compression: {s['compression']}\")
    print(f\"  Method: {s['method']}\")
    print(f\"  Best F1: {s['best_f1']:.1%}\")
    print(f\"  Achieved in epoch: {s['epoch']}\")
    print('')
    if s['best_f1'] > 0.65:
        print('  ✅ SUCCESS: Adapter works! Proceed to Stage 2.')
    elif s['best_f1'] > 0.50:
        print('  ⚠️ PARTIAL: Adapter partially works. May need tuning.')
    else:
        print('  ❌ FAILURE: Adapter concept may be flawed.')
" 2>&1 | tee -a "$CHECKPOINT_DIR/logs/training.log"
else
    echo "  No results found. Check logs for errors." | tee -a "$CHECKPOINT_DIR/logs/training.log"
fi

echo "" | tee -a "$CHECKPOINT_DIR/logs/training.log"
echo "Logs saved to:" | tee -a "$CHECKPOINT_DIR/logs/training.log"
echo "  - $CHECKPOINT_DIR/logs/training.log" | tee -a "$CHECKPOINT_DIR/logs/training.log"
echo "  - $CHECKPOINT_DIR/logs/diagnostics.jsonl" | tee -a "$CHECKPOINT_DIR/logs/training.log"