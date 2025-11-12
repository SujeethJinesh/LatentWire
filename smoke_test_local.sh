#!/usr/bin/env bash
set -e

echo "=========================================="
echo "MacBook Smoke Test for DiT Training"
echo "=========================================="
echo ""
echo "This will:"
echo "  - Load Mistral-7B + Llama-8B (~30 GB RAM)"
echo "  - Run 2 training steps (CPU/MPS)"
echo "  - Run minimal evaluation (4 samples)"
echo "  - Should complete in ~10-15 minutes"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run minimal smoke test
python paper_writing/cross_attention.py \
  --source_model "mistralai/Mistral-7B-Instruct-v0.3" \
  --target_model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --dataset gsm8k \
  --bridge dit \
  --dit_dim 256 \
  --soft_tokens 16 \
  --dit_depth 2 \
  --dit_heads 4 \
  --dit_steps_train 2 \
  --train_steps 2 \
  --per_device_batch 1 \
  --eval_every 2 \
  --eval_samples 4 \
  --eval_batch_size 2 \
  --max_new_tokens 20 \
  --lr 1e-4 \
  --warmup_steps 1 \
  --bf16 \
  --no_compile \
  --save_path /tmp/smoke_test_dit.pt

echo ""
echo "=========================================="
echo "Smoke Test Complete!"
echo "=========================================="
echo ""
echo "If you got here without errors, the code is working!"
echo "The HPC cluster should work once CUDA is fixed."
echo ""
echo "Next steps:"
echo "  1. Wait for HPC cluster to be fixed"
echo "  2. Request node with: --exclude=n26"
echo "  3. Run: bash paper_writing/run_ablations.sh"
