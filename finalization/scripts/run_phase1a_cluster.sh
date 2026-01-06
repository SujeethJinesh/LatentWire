#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Comprehensive Phase 1a run (baseline + LoRA sweep)
#
# Recommended invocation on the cluster:
#   git pull && rm -rf runs && PYTHONPATH=. bash scripts/run_phase1a_cluster.sh
#
# Environment overrides (optional):
#   SAMPLES=8000 EPOCHS=2 GEN_WEIGHT=0.05 PYTHONPATH=. bash scripts/run_phase1a_cluster.sh

ROOT_DIR="${ROOT_DIR:-runs/phase1a_cluster}"
BASELINE_DIR="$ROOT_DIR/baseline"
SWEEP_DIR="$ROOT_DIR/lora_sweep"
SUMMARY_FILE="$ROOT_DIR/summary.txt"

# Shared knobs (override via env as needed)
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-10000}"
PCA_SAMPLES="${PCA_SAMPLES:-5000}"
PCA_TOKEN_CAP="${PCA_TOKEN_CAP:-850000}"  # Use all tokens with randomized SVD (~13GB input, efficient workspace)
EPOCHS="${EPOCHS:-3}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-64}"
LORA_BATCH_SIZE="${LORA_BATCH_SIZE:-48}"
BATCH_SIZE="$BASE_BATCH_SIZE"
MAX_LENGTH="${MAX_LENGTH:-256}"
COMPRESS_DIM="${COMPRESS_DIM:-1024}"
ADAPTER_LR="${ADAPTER_LR:-5e-4}"
COSINE_WEIGHT="${COSINE_WEIGHT:-1.0}"
MSE_WEIGHT="${MSE_WEIGHT:-0.1}"
GEN_WEIGHT="${GEN_WEIGHT:-0.0}"          # 0.0 = strict Phase 1a; bump (e.g. 0.05) if LoRA needs gradients
PCA_CACHE_PATH="${PCA_CACHE_PATH:-cache/phase1a_pca.pt}"
PCA_BATCH_SIZE="${PCA_BATCH_SIZE:-512}"

echo "=================================================="
echo "Phase 1a Cluster Run"
echo "Model:        $MODEL"
echo "Dataset:      $DATASET"
echo "Samples:      $SAMPLES  (PCA: $PCA_SAMPLES)"
echo "Epochs:       $EPOCHS"
echo "Batch size (baseline): $BASE_BATCH_SIZE"
echo "Batch size (LoRA):     $LORA_BATCH_SIZE"
echo "Compression:  4096 -> $COMPRESS_DIM"
echo "Gen loss wt:  $GEN_WEIGHT"
echo "Root dir:     $ROOT_DIR"
echo "=================================================="
echo ""

mkdir -p "$BASELINE_DIR" "$SWEEP_DIR"
mkdir -p "$(dirname "$PCA_CACHE_PATH")"
rm -f "$SUMMARY_FILE"

###############################################
# Phase 1a baseline (pure reconstruction)
###############################################

BASE_LOG="$BASELINE_DIR/train_$(date +%Y%m%d_%H%M%S).log"
BASE_DIAG="$BASELINE_DIR/diagnostics.jsonl"

echo "[Baseline] Starting pure reconstruction run..."
BASE_CMD=(python train_adapter_only_phase1.py
    --model_id "$MODEL"
    --dataset "$DATASET"
    --samples "$SAMPLES"
    --pca_samples "$PCA_SAMPLES"
    --pca_token_cap "$PCA_TOKEN_CAP"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --max_length "$MAX_LENGTH"
    --compress_dim "$COMPRESS_DIM"
    --compress_method pca
    --adapter_lr "$ADAPTER_LR"
    --cosine_weight "$COSINE_WEIGHT"
    --mse_weight "$MSE_WEIGHT"
    --save_dir "$BASELINE_DIR"
    --diagnostic_log "$BASE_DIAG"
    --pca_cache_path "$PCA_CACHE_PATH"
    --pca_batch_size "$PCA_BATCH_SIZE"
    --gen_loss_weight 0.0
)

echo "Command: ${BASE_CMD[*]}"
("${BASE_CMD[@]}") 2>&1 | tee "$BASE_LOG"

BASE_REPORT=$(python - "$BASE_DIAG" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
best_f1 = 0.0
best_em = 0.0
if path.exists():
    with path.open() as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "full_eval":
                f1 = data.get("f1", 0.0)
                em = data.get("em", 0.0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_em = em
if best_f1 == 0.0:
    print("No evaluation entries yet.")
else:
    print(f"Best F1: {best_f1*100:.2f}%")
    print(f"Best EM: {best_em*100:.2f}%")
PY)

{
  echo "Baseline results:"
  echo "  Log:  $BASE_LOG"
  echo "  Diag: $BASE_DIAG"
  echo "$BASE_REPORT"
  echo ""
} | tee -a "$SUMMARY_FILE"

###############################################
# Phase 1a + LoRA sweep (optional gradients)
###############################################

echo "[LoRA] Launching sweep ..."
GEN_ENV="GEN_WEIGHT_DEFAULT=$GEN_WEIGHT"
OUTPUT_ENV="OUTPUT_BASE=$SWEEP_DIR"

(
  export GEN_WEIGHT_DEFAULT="$GEN_WEIGHT"
  export PCA_CACHE_PATH="$PCA_CACHE_PATH"
  export PCA_BATCH_SIZE="$PCA_BATCH_SIZE"
  export PCA_TOKEN_CAP="$PCA_TOKEN_CAP"
  export BATCH_SIZE="$LORA_BATCH_SIZE"
  export OUTPUT_BASE="$SWEEP_DIR"
  export MODEL DATASET SAMPLES PCA_SAMPLES EPOCHS MAX_LENGTH COMPRESS_DIM ADAPTER_LR COSINE_WEIGHT MSE_WEIGHT
  bash scripts/sweep_phase1a_lora.sh
) | tee "$SWEEP_DIR/sweep_$(date +%Y%m%d_%H%M%S).log"

if [[ -f "$SWEEP_DIR/sweep_summary.txt" ]]; then
  echo "" >> "$SUMMARY_FILE"
  echo "LoRA sweep summary:" >> "$SUMMARY_FILE"
  cat "$SWEEP_DIR/sweep_summary.txt" >> "$SUMMARY_FILE"
fi

###############################################
# Final summary
###############################################

echo ""
echo "=================================================="
echo "Experiment complete. Summary:"
echo "=================================================="
cat "$SUMMARY_FILE"
echo ""
echo "Artifacts stored under: $ROOT_DIR"
echo "  - Baseline logs/checkpoints: $BASELINE_DIR"
echo "  - LoRA sweep logs/checkpoints: $SWEEP_DIR"
echo ""
