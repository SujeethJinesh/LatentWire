#!/bin/bash
# Fast Phase 1b Weight Sweep: 2 minutes per lambda
# Reduced from 10k samples to 1k samples for 10× speedup
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_COMPILE=1

# Disable MPS
export CUDA_MPS_PIPE_DIRECTORY=/dev/null
export CUDA_MPS_LOG_DIRECTORY=/dev/null

echo "================================================"
echo "FAST PHASE 1B WEIGHT SWEEP"
echo "================================================"
echo ""
echo "Fast sweep: 1000 samples, ~125 steps per lambda"
echo "Goal: Quickly identify mode collapse threshold"
echo ""

# Test different lambda values
LAMBDAS=(0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5)

for LAMBDA in "${LAMBDAS[@]}"; do
    echo ""
    echo "========================================"
    echo "TESTING λ = $LAMBDA"
    echo "========================================"
    echo ""

    CHECKPOINT_DIR="runs/phase1b_fast_sweep/lambda_${LAMBDA}"
    mkdir -p "$CHECKPOINT_DIR/logs"

    echo "Starting training at $(date)" | tee "$CHECKPOINT_DIR/logs/training.log"

    # Run fast training (1k samples, 1 epoch, batch_size=8)
    # ~125 steps, should take ~2 minutes
    python train_adapter_only_phase1b.py \
      --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
      --compress_dim 1024 \
      --compress_method pca \
      --pca_samples 5000 \
      --adapter_hidden_mult 4 \
      --adapter_dropout 0.1 \
      --adapter_lr 5e-4 \
      --samples 1000 \
      --epochs 1 \
      --batch_size 8 \
      --k_tokens 2 \
      --lambda_kce $LAMBDA \
      --lambda_kd $LAMBDA \
      --kd_tau 1.0 \
      --eval_every 1 \
      --eval_samples 100 \
      --save_dir "$CHECKPOINT_DIR" \
      --diagnostic_log "$CHECKPOINT_DIR/logs/diagnostics.jsonl" \
      2>&1 | tee -a "$CHECKPOINT_DIR/logs/training.log"

    echo "Completed λ=$LAMBDA at $(date)" | tee -a "$CHECKPOINT_DIR/logs/training.log"

    # Extract final metrics
    if [ -f "$CHECKPOINT_DIR/logs/diagnostics.jsonl" ]; then
        echo ""
        echo "Final metrics for λ=$LAMBDA:"
        tail -1 "$CHECKPOINT_DIR/logs/diagnostics.jsonl" | python -c "
import sys, json
d = json.load(sys.stdin)
eval_f1 = d.get('f1', d.get('eval_f1', 0))
print(f\"  F1: {eval_f1:.1%}\")
print(f\"  Cosine sim: {d.get('recon_cosine_sim', 0):.3f}\")
print(f\"  loss_recon: {d.get('loss_recon', 0):.3f}\")
print(f\"  loss_kce: {d.get('loss_kce', 0):.3f}\")
print(f\"  loss_kd: {d.get('loss_kd', 0):.3f}\")
"
    fi
done

echo ""
echo "========================================"
echo "FAST WEIGHT SWEEP COMPLETE"
echo "========================================"
echo ""
echo "Analyzing results across all λ values..."
echo ""

# Compare all results
echo "λ value | F1    | Cosine | Status"
echo "--------|-------|--------|------------------"

for LAMBDA in "${LAMBDAS[@]}"; do
    CHECKPOINT_DIR="runs/phase1b_fast_sweep/lambda_${LAMBDA}"
    if [ -f "$CHECKPOINT_DIR/logs/diagnostics.jsonl" ]; then
        tail -1 "$CHECKPOINT_DIR/logs/diagnostics.jsonl" | python -c "
import sys, json
d = json.load(sys.stdin)
eval_f1 = d.get('f1', d.get('eval_f1', 0))
cosine = d.get('recon_cosine_sim', 0)

# Check for mode collapse
collapsed = cosine < 0.3 or eval_f1 < 0.05

if collapsed:
    status = '❌ COLLAPSED'
elif eval_f1 > 0.25:
    status = '✅ IMPROVED'
elif eval_f1 > 0.20:
    status = '⚠️  STABLE'
else:
    status = '⚠️  DEGRADED'

print(f\"$LAMBDA      | {eval_f1:.1%} | {cosine:.3f}  | {status}\")
"
    fi
done

echo ""
echo "Legend:"
echo "  ✅ IMPROVED: F1 > 25% (better than Phase 1a baseline)"
echo "  ⚠️  STABLE: F1 ≥ 20% (similar to baseline)"
echo "  ⚠️  DEGRADED: F1 < 20% (worse than baseline)"
echo "  ❌ COLLAPSED: Cosine < 0.3 or F1 < 5% (mode collapse)"
echo ""
echo "Recommendation: Use highest λ that maintains ✅ or ⚠️  STABLE"
echo ""
