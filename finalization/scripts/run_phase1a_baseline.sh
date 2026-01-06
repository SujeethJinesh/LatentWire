#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Phase 1a baseline reproduction (pure reconstruction, no LoRA)
# Usage (recommended):
#   git pull && rm -rf runs && PYTHONPATH=. bash scripts/run_phase1a_baseline.sh
#
# Environment overrides:
#   SAMPLES=10000 EPOCHS=3 BATCH_SIZE=32 MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct bash ...

SAMPLES="${SAMPLES:-10000}"
PCA_SAMPLES="${PCA_SAMPLES:-5000}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/phase1a_baseline}"
PCA_CACHE_PATH="${PCA_CACHE_PATH:-cache/phase1a_pca.pt}"
PCA_BATCH_SIZE="${PCA_BATCH_SIZE:-512}"

echo "==============================================="
echo "Phase 1a Baseline (Pure Reconstruction)"
echo "==============================================="
echo "Model:        $MODEL"
echo "Samples:      $SAMPLES (PCA samples: $PCA_SAMPLES)"
echo "Epochs:       $EPOCHS"
echo "Batch size:   $BATCH_SIZE"
echo "Max length:   $MAX_LENGTH"
echo "Output dir:   $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$PCA_CACHE_PATH")"

LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).log"
DIAG_FILE="$OUTPUT_DIR/diagnostics.jsonl"

CMD="python train_adapter_only_phase1.py \
    --model_id \"$MODEL\" \
    --samples $SAMPLES \
    --pca_samples $PCA_SAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --pca_batch_size $PCA_BATCH_SIZE \
    --save_dir \"$OUTPUT_DIR\" \
    --diagnostic_log \"$DIAG_FILE\" \
    --pca_cache_path \"$PCA_CACHE_PATH\""

echo "Running:"
echo "  $CMD"
echo ""

eval "$CMD" 2>&1 | tee "$LOG_FILE"

echo ""
echo "-----------------------------------------------"
echo "Baseline Summary"
echo "-----------------------------------------------"

python - <<'PY'
import json
from pathlib import Path
import sys

diag_path = Path(sys.argv[1])
if not diag_path.exists():
    print("Diagnostics not found.")
    sys.exit(0)

best = {"f1": 0.0, "em": 0.0}
with diag_path.open() as f:
    for line in f:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("type") == "full_eval":
            f1 = data.get("f1", 0.0)
            em = data.get("em", 0.0)
            if f1 > best["f1"]:
                best = {"f1": f1, "em": em}

if best["f1"] == 0.0:
    print("No evaluation records yet.")
else:
    print(f"Best F1: {best['f1']*100:.2f}%")
    print(f"Best EM: {best['em']*100:.2f}%")
PY "$DIAG_FILE"

echo ""
echo "Artifacts:"
echo "  - Training log:      $LOG_FILE"
echo "  - Diagnostics (JSONL): $DIAG_FILE"
echo "  - Best checkpoint:   $OUTPUT_DIR/adapter_phase1_best.pt"
echo ""
