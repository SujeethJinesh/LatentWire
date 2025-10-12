#!/bin/bash
# Stage 1 Phase 1b: Reconstruction + generation-aware training
# Adds K-token CE and Prefix KD to teach QA format
# Goal: Improve F1 from 24% (Phase 1a) to 50-70% by preserving task pragmatics
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
CHECKPOINT_DIR="runs/stage1b_phase1b"
BATCH_SIZE=8   # Reduced to 8 for memory (K-token CE and KD are memory-intensive)
SAMPLES=10000
EPOCHS=5      # Increased from 3 to allow generation objectives to converge
K_TOKENS=2    # Reduced to 2 tokens (less memory, still effective)
LAMBDA_KCE=0.5  # Weight for K-token CE
LAMBDA_KD=0.5   # Weight for Prefix KD
KD_TAU=1.0    # Temperature for KD

# Memory optimization: Set chunking for KD teacher forward passes
export KD_TEACHER_CHUNK=2  # Process 2 examples at a time in teacher forward pass

echo "================================================"
echo "STAGE 1 PHASE 1B: RECONSTRUCTION + GENERATION"
echo "================================================"
echo ""
echo "Configuration:"
echo "  - Model: meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "  - Compression: 4096 → 1024 (4x compression via GPU-accelerated PCA)"
echo "  - PCA: Randomized SVD (memory-efficient, GPU-only) on 5k samples"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Samples: $SAMPLES"
echo "  - Epochs: $EPOCHS"
echo "  - Steps: $((SAMPLES * EPOCHS / BATCH_SIZE)) total"
echo "  - Eval: 100 samples, batched (batch_size=32)"
echo "  - Output: $CHECKPOINT_DIR"
echo ""
echo "Loss Components:"
echo "  1. Reconstruction: Cosine (1.0×) + MSE (0.1×) - semantic preservation"
echo "  2. K-token CE: K=$K_TOKENS, λ=$LAMBDA_KCE - supervise first K answer tokens"
echo "  3. Prefix KD: τ=$KD_TAU, λ=$LAMBDA_KD - distill QA behavior from text teacher"
echo ""
echo "Memory Optimizations:"
echo "  - Batch size: $BATCH_SIZE (reduced for generation objectives)"
echo "  - K tokens: $K_TOKENS (reduced to save memory)"
echo "  - KD teacher chunking: $KD_TEACHER_CHUNK examples at a time"
echo "  - Model loaded once (reused from LMWrapper)"
echo "  - If OOM persists: Set LAMBDA_KD=0 to disable KD (CE only)"
echo ""
echo "Phase 1b Goals:"
echo "  - Improve over Phase 1a baseline (F1 24%)"
echo "  - Target F1: 50-70% (learn to stop after answer, preserve QA format)"
echo "  - Target FirstTok@1: 12-20% (PLAN.md milestone)"
echo "  - If successful: Proceed to full system (learned encoder + dual-LLM)"
echo "  - If partial (30-50%): Tune loss weights, increase K or epochs"
echo ""

# Create log directory
mkdir -p "$CHECKPOINT_DIR/logs"

# Log start time
echo "Starting training at $(date)" | tee "$CHECKPOINT_DIR/logs/training.log"
echo "" | tee -a "$CHECKPOINT_DIR/logs/training.log"

# Run Phase 1b training with generation objectives
python train_adapter_only_phase1b.py \
  --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --compress_dim 1024 \
  --compress_method pca \
  --pca_samples 5000 \
  --adapter_hidden_mult 4 \
  --adapter_dropout 0.1 \
  --adapter_lr 5e-4 \
  --samples $SAMPLES \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --k_tokens $K_TOKENS \
  --lambda_kce $LAMBDA_KCE \
  --lambda_kd $LAMBDA_KD \
  --kd_tau $KD_TAU \
  --eval_every 1 \
  --eval_samples 100 \
  --save_dir "$CHECKPOINT_DIR" \
  --diagnostic_log "$CHECKPOINT_DIR/logs/diagnostics.jsonl" \
  2>&1 | tee -a "$CHECKPOINT_DIR/logs/training.log"

echo "" | tee -a "$CHECKPOINT_DIR/logs/training.log"
echo "=== Training Complete at $(date) ===" | tee -a "$CHECKPOINT_DIR/logs/training.log"

# Run evaluation if checkpoint exists
if [ -f "$CHECKPOINT_DIR/adapter_phase1_best.pt" ]; then
    echo "Running evaluation..." | tee -a "$CHECKPOINT_DIR/logs/training.log"

    python -c "
import torch
import json
from pathlib import Path

checkpoint = torch.load('$CHECKPOINT_DIR/adapter_phase1_best.pt', map_location='cpu')
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
    'samples': config.get('samples', 0),
    'k_tokens': config.get('k_tokens', 4),
    'lambda_kce': config.get('lambda_kce', 0.5),
    'lambda_kd': config.get('lambda_kd', 0.5)
}

with open('$CHECKPOINT_DIR/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
" 2>&1 | tee -a "$CHECKPOINT_DIR/logs/training.log"
fi

# Display results summary
echo "" | tee -a "$CHECKPOINT_DIR/logs/training.log"
echo "=====================================" | tee -a "$CHECKPOINT_DIR/logs/training.log"
echo "STAGE 1 PHASE 1B RESULTS SUMMARY" | tee -a "$CHECKPOINT_DIR/logs/training.log"
echo "=====================================" | tee -a "$CHECKPOINT_DIR/logs/training.log"

if [ -f "$CHECKPOINT_DIR/summary.json" ]; then
    python -c "
import json
with open('$CHECKPOINT_DIR/summary.json') as f:
    s = json.load(f)
    print(f\"  Compression: {s['compression']}\")
    print(f\"  Method: {s['method']}\")
    print(f\"  Best F1: {s['best_f1']:.1%}\")
    print(f\"  Achieved in epoch: {s['epoch']}\")
    print(f\"  K-token CE: K={s.get('k_tokens', 4)}, λ={s.get('lambda_kce', 0.5)}\")
    print(f\"  Prefix KD: λ={s.get('lambda_kd', 0.5)}\")
    print('')
    if s['best_f1'] >= 0.70:
        print('  ✅ SUCCESS: Phase 1b achieves target!')
        print('  Ready to proceed to full system (learned encoder + dual-LLM)')
    elif s['best_f1'] >= 0.50:
        print('  ⚠️ PARTIAL: Improvement over Phase 1a but below target')
        print('  Consider: longer training, higher K, different loss weights')
    elif s['best_f1'] >= 0.30:
        print('  ⚠️ MODEST: Generation objectives helping but more work needed')
    else:
        print('  ❌ BELOW PHASE 1A: Check loss function bugs or anchor alignment')
" 2>&1 | tee -a "$CHECKPOINT_DIR/logs/training.log"
else
    echo "  No results found. Check logs for errors." | tee -a "$CHECKPOINT_DIR/logs/training.log"
fi

echo "" | tee -a "$CHECKPOINT_DIR/logs/training.log"
echo "Logs saved to:" | tee -a "$CHECKPOINT_DIR/logs/training.log"
echo "  - $CHECKPOINT_DIR/logs/training.log" | tee -a "$CHECKPOINT_DIR/logs/training.log"
echo "  - $CHECKPOINT_DIR/logs/diagnostics.jsonl" | tee -a "$CHECKPOINT_DIR/logs/training.log"