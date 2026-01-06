#!/bin/bash
# Monitor Stage 1 training progress
set -euo pipefail

CHECKPOINT_DIR="${1:-runs/stage1_adapter_only}"

echo "=================================="
echo "MONITORING STAGE 1 TRAINING"
echo "=================================="
echo ""
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo ""

# Check if training is running
if [ -f "$CHECKPOINT_DIR/logs/training.log" ]; then
    echo "Training log found. Showing last 20 lines:"
    echo "---"
    tail -20 "$CHECKPOINT_DIR/logs/training.log"
    echo ""
fi

# Check diagnostics
if [ -f "$CHECKPOINT_DIR/logs/diagnostics.jsonl" ]; then
    echo "Latest metrics:"
    echo "---"

    # Get last few diagnostic entries
    tail -5 "$CHECKPOINT_DIR/logs/diagnostics.jsonl" | python -c "
import json
import sys

for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        if d.get('type') == 'full_eval':
            print(f\"Step {d['step']}: F1={d['f1']:.3f}, EM={d['em']:.3f}\")
        elif d.get('type') == 'quick_eval':
            print(f\"Step {d['step']}: Quick Acc={d['quick_eval_acc']:.1%}\")
        else:
            print(f\"Step {d.get('step', '?')}: Loss={d.get('loss', 0):.4f}, Recon={d.get('recon_loss', 0):.4f}, CE={d.get('ce_loss', 0):.4f}\")
    except:
        pass
"
    echo ""
fi

# Check GPU usage
if [ -f "$CHECKPOINT_DIR/logs/diagnostics.jsonl" ]; then
    echo "GPU Memory Usage:"
    echo "---"
    tail -1 "$CHECKPOINT_DIR/logs/diagnostics.jsonl" | python -c "
import json
import sys
try:
    d = json.loads(sys.stdin.read().strip())
    print(f\"Memory: {d.get('gpu_memory_gb', 0):.1f} GB\")
    print(f\"Utilization: {d.get('gpu_utilization', 0):.1%}\")
except:
    print('No GPU data available')
"
    echo ""
fi

# Check if checkpoint exists
if [ -f "$CHECKPOINT_DIR/adapter_only_best.pt" ]; then
    echo "Best checkpoint info:"
    echo "---"
    python -c "
import torch
ckpt = torch.load('$CHECKPOINT_DIR/adapter_only_best.pt', map_location='cpu')
print(f\"Best F1: {ckpt.get('best_f1', 0):.3f}\")
print(f\"From epoch: {ckpt.get('epoch', 0)}\")
config = ckpt.get('config', {})
print(f\"Compression: {config.get('input_dim', 4096)} → {config.get('compress_dim', 512)}\")
"
    echo ""
fi

# Show summary if exists
if [ -f "$CHECKPOINT_DIR/summary.json" ]; then
    echo "Training Summary:"
    echo "---"
    cat "$CHECKPOINT_DIR/summary.json"
    echo ""
fi

echo "=================================="
echo "To watch live updates, run:"
echo "  tail -f $CHECKPOINT_DIR/logs/training.log"
echo ""
echo "To see diagnostics stream:"
echo "  tail -f $CHECKPOINT_DIR/logs/diagnostics.jsonl | jq '.'"
echo "=================================="