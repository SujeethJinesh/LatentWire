#!/usr/bin/env bash
# Smoke test for embedding baselines - creates checkpoint then tests embeddings
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Configuration
SAMPLES_TRAIN=40  # 2 epochs x 20 steps
SAMPLES_EVAL=30   # Enough for meaningful statistics
CHECKPOINT_DIR="runs/smoke/embedding_test/ckpt"
EVAL_DIR="runs/smoke/embedding_test/eval"
LOG_DIR="runs/smoke/embedding_test/logs"

echo "=== LatentWire Embedding Baseline Smoke Test ==="
echo "Creating checkpoint with $SAMPLES_TRAIN training samples, then evaluating on $SAMPLES_EVAL samples"
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
  --batch_size 1 \
  --latent_len 32 \
  --d_z 256 \
  --encoder_type byte \
  --sequential_models \
  --first_token_ce_weight 0.5 \
  --warm_anchor_text "Answer: " \
  --save_dir "$CHECKPOINT_DIR" \
  --save_every 20 \
  --diagnostic_log "$LOG_DIR/train_diagnostics.jsonl" \
  2>&1 | tee "$LOG_DIR/training.log"

echo
echo "Step 2/2: Running embedding baseline tests..."

# Step 2: Run embedding baseline evaluation
python -m latentwire.cli.eval \
  --config configs/baseline/embedding_baselines.json \
  --override "ckpt=$CHECKPOINT_DIR" \
  --override "samples=$SAMPLES_EVAL" \
  --override "out_dir=$EVAL_DIR" \
  --override "llama_id=meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --override "dataset=squad" \
  --override "max_new_tokens=24" \
  --override "embedding_replay=true" \
  --override 'embedding_baseline_modes=["raw","anchor","adapter"]' \
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