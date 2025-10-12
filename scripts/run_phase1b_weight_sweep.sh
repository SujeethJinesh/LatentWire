#!/bin/bash
# Phase 1b Weight Sweep: Find λ values that work
# Tests: λ ∈ {0.001, 0.005, 0.01, 0.05, 0.1}
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_COMPILE=1

# Disable MPS
export CUDA_MPS_PIPE_DIRECTORY=/dev/null
export CUDA_MPS_LOG_DIRECTORY=/dev/null

echo "================================================"
echo "PHASE 1B WEIGHT SWEEP"
echo "================================================"
echo ""
echo "Testing λ_kce and λ_kd values to find stable configuration"
echo "Goal: Find weights where generation helps without mode collapse"
echo ""

# Test different lambda values
LAMBDAS=(0.001 0.005 0.01 0.05 0.1)

for LAMBDA in "${LAMBDAS[@]}"; do
    echo ""
    echo "========================================"
    echo "TESTING λ = $LAMBDA"
    echo "========================================"
    echo ""

    CHECKPOINT_DIR="runs/phase1b_sweep/lambda_${LAMBDA}"
    mkdir -p "$CHECKPOINT_DIR/logs"

    echo "Starting training at $(date)" | tee "$CHECKPOINT_DIR/logs/training.log"

    # Run short training (1 epoch to see if it collapses)
    python train_adapter_only_phase1b.py \
      --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
      --compress_dim 1024 \
      --compress_method pca \
      --pca_samples 5000 \
      --adapter_hidden_mult 4 \
      --adapter_dropout 0.1 \
      --adapter_lr 5e-4 \
      --samples 10000 \
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
print(f\"  F1: {d.get('eval_f1', 0):.1%}\")
print(f\"  Cosine sim: {d.get('recon_cosine_sim', 0):.3f}\")
print(f\"  loss_recon: {d.get('loss_recon', 0):.3f}\")
print(f\"  loss_kce: {d.get('loss_kce', 0):.3f}\")
print(f\"  loss_kd: {d.get('loss_kd', 0):.3f}\")
"
    fi
done

echo ""
echo "========================================"
echo "WEIGHT SWEEP COMPLETE"
echo "========================================"
echo ""
echo "Analyzing results across all λ values..."
echo ""

# Compare all results
for LAMBDA in "${LAMBDAS[@]}"; do
    CHECKPOINT_DIR="runs/phase1b_sweep/lambda_${LAMBDA}"
    if [ -f "$CHECKPOINT_DIR/logs/diagnostics.jsonl" ]; then
        echo "λ = $LAMBDA:"
        tail -1 "$CHECKPOINT_DIR/logs/diagnostics.jsonl" | python -c "
import sys, json
d = json.load(sys.stdin)
f1 = d.get('eval_f1', 0)
cosine = d.get('recon_cosine_sim', 0)
loss_r = d.get('loss_recon', 0)
loss_k = d.get('loss_kce', 0)
loss_d = d.get('loss_kd', 0)

# Check for mode collapse
collapsed = cosine < 0.3 or f1 < 0.05

status = '❌ COLLAPSED' if collapsed else ('✅ STABLE' if f1 > 0.20 else '⚠️  PARTIAL')

print(f\"  {status}: F1={f1:.1%}, Cosine={cosine:.3f}, recon={loss_r:.2f}, kce={loss_k:.2f}, kd={loss_d:.2f}\")
"
    fi
done

echo ""
echo "Recommendations:"
echo "  - ✅ STABLE: Use this λ for full LatentWire"
echo "  - ⚠️  PARTIAL: Consider even weaker λ or annealing"
echo "  - ❌ COLLAPSED: λ too strong, mode collapse occurred"
echo ""
