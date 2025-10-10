#!/usr/bin/env bash
# Smoke test for embedding baselines - creates checkpoint then tests embeddings
# Optimized for 4x H100 GPUs on HPC cluster
set -euo pipefail

# Always use all 4 H100s
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Configuration optimized for 4x H100
SAMPLES_TRAIN=80  # Can handle more with H100 memory (80GB per GPU)
SAMPLES_EVAL=50   # More samples for better statistics
BATCH_SIZE=4      # Leverage H100 memory (4 samples across 4 GPUs = 1 per GPU)
CHECKPOINT_DIR="runs/smoke/embedding_test/ckpt"
EVAL_DIR="runs/smoke/embedding_test/eval"
LOG_DIR="runs/smoke/embedding_test/logs"

echo "=== LatentWire Embedding Baseline Smoke Test (4x H100 Config) ==="
echo "Hardware: 4x H100 GPUs (80GB each)"
echo "Creating checkpoint with $SAMPLES_TRAIN training samples (batch_size=$BATCH_SIZE)"
echo "Then evaluating on $SAMPLES_EVAL samples with 3 embedding modes"
echo

# Step 1: Create a minimal checkpoint with quick training
echo "Step 1/2: Training to create checkpoint..."
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

python latentwire/train.py \
  --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --models llama \
  --dataset squad \
  --samples $SAMPLES_TRAIN \
  --epochs 2 \
  --batch_size $BATCH_SIZE \
  --grad_accum_steps 1 \
  --latent_len 32 \
  --d_z 256 \
  --encoder_type byte \
  --sequential_models \
  --first_token_ce_weight 0.5 \
  --warm_anchor_text "Answer: " \
  --save_dir "$CHECKPOINT_DIR" \
  --save_every 40 \
  --diagnostic_log "$LOG_DIR/train_diagnostics.jsonl" \
  --llama_device_map "auto" \
  --require_cuda "yes" \
  2>&1 | tee "$LOG_DIR/training.log"

echo
echo "Step 2/2: Running embedding baseline tests..."

# Step 2: Run embedding baseline evaluation
# H100s can handle larger chunk sizes for faster evaluation
python -m latentwire.cli.eval \
  --config configs/baseline/embedding_baselines.json \
  --override "ckpt=$CHECKPOINT_DIR" \
  --override "samples=$SAMPLES_EVAL" \
  --override "out_dir=$EVAL_DIR" \
  --override "llama_id=meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --override "dataset=squad" \
  --override "max_new_tokens=24" \
  --override "chunk_size=16" \
  --override "embedding_replay=true" \
  --override 'embedding_baseline_modes=["raw","anchor","adapter"]' \
  --override "llama_device_map=auto" \
  --tag "embedding-smoke" \
  2>&1 | tee "$LOG_DIR/embedding_baseline.log"

echo
echo "=== Smoke Test Complete ==="
echo "Results saved to:"
echo "  - Metrics: $EVAL_DIR/metrics.json"
echo "  - Predictions: $EVAL_DIR/predictions.jsonl"
echo "  - Logs: $LOG_DIR/"
echo

# Display summary if metrics exist
if [ -f "$EVAL_DIR/metrics.json" ]; then
    echo "=== Quick Results Summary ==="
    python -c "
import json
with open('$EVAL_DIR/metrics.json') as f:
    metrics = json.load(f)
    print('Text Baseline F1: {:.3f}'.format(metrics.get('text', {}).get('metrics', {}).get('f1', 0)))
    print('Embedding Baselines:')
    for mode in ['raw', 'anchor', 'adapter']:
        key = f'embed_{mode}'
        if key in metrics:
            f1 = metrics[key].get('metrics', {}).get('f1', 0)
            print(f'  - {mode:8s}: F1 = {f1:.3f}')
    print()
    print('Key Insight: Embedding baselines should be close to text baseline')
    print('If they are significantly lower, inputs_embeds interface may have issues')
"
fi