#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Phase 1a + LoRA sweep launcher.
# Suggested usage:
#   git pull && rm -rf runs && PYTHONPATH=. bash scripts/sweep_phase1a_lora.sh

SAMPLES="${SAMPLES:-5000}"
PCA_SAMPLES="${PCA_SAMPLES:-4000}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-24}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DATASET="${DATASET:-squad}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
COSINE_WEIGHT="${COSINE_WEIGHT:-1.0}"
MSE_WEIGHT="${MSE_WEIGHT:-0.1}"
ADAPTER_LR="${ADAPTER_LR:-5e-4}"
COMPRESS_DIM="${COMPRESS_DIM:-1024}"
GEN_WEIGHT_DEFAULT="${GEN_WEIGHT_DEFAULT:-0.05}"

OUTPUT_BASE="${OUTPUT_BASE:-runs/phase1a_lora_sweep}"
mkdir -p "$OUTPUT_BASE"
SUMMARY_FILE="$OUTPUT_BASE/sweep_summary.txt"

# name:rank:alpha:layers:gen_loss_weight
CONFIGS=(
  "baseline:0:0:none:0.0"
  "r4_a8_l8:4:8:8:${GEN_WEIGHT_DEFAULT}"
  "r8_a16_l12:8:16:12:${GEN_WEIGHT_DEFAULT}"
  "r16_a32_full:16:32:all:${GEN_WEIGHT_DEFAULT}"
)

cat > "$SUMMARY_FILE" <<EOF_SUMMARY
Phase 1a + LoRA Sweep
Started: $(date)
================================================
Model: $MODEL
Dataset: $DATASET
Samples: $SAMPLES (PCA: $PCA_SAMPLES)
Epochs:  $EPOCHS
Batch:   $BATCH_SIZE
Compression: 4096 -> $COMPRESS_DIM
Adapter LR: $ADAPTER_LR
Loss: Cosine ($COSINE_WEIGHT) + MSE ($MSE_WEIGHT)
EOF_SUMMARY

echo "=================================================="
echo "PHASE 1A + LORA SWEEP"
echo "=================================================="
echo "Model:        $MODEL"
echo "Dataset:      $DATASET"
echo "Samples:      $SAMPLES (PCA: $PCA_SAMPLES)"
echo "Epochs:       $EPOCHS"
echo "Batch size:   $BATCH_SIZE"
echo "Compression:  4096 -> $COMPRESS_DIM"
echo "Configs:      ${#CONFIGS[@]}"
echo "Output dir:   $OUTPUT_BASE"
echo ""

declare -A RESULTS

is_number() {
  [[ $1 =~ ^[0-9]+([.][0-9]+)?$ ]]
}

for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r NAME R ALPHA LAYERS GEN_WEIGHT <<< "$cfg"
  echo "--------------------------------------------------"
  echo "Configuration: $NAME"
  echo "  LoRA rank:   $R"
  echo "  LoRA alpha:  $ALPHA"
  echo "  Layers:      $LAYERS"
  echo "  Gen weight:  $GEN_WEIGHT"
  echo ""

  RUN_DIR="$OUTPUT_BASE/$NAME"
  mkdir -p "$RUN_DIR"
  LOG_FILE="$RUN_DIR/train_$(date +%Y%m%d_%H%M%S).log"
  DIAG_FILE="$RUN_DIR/diagnostics.jsonl"

  CMD=(python train_adapter_only_phase1.py
        --model_id "$MODEL"
        --dataset "$DATASET"
        --samples "$SAMPLES"
        --pca_samples "$PCA_SAMPLES"
        --epochs "$EPOCHS"
        --batch_size "$BATCH_SIZE"
        --max_length "$MAX_LENGTH"
        --compress_dim "$COMPRESS_DIM"
        --compress_method pca
        --adapter_lr "$ADAPTER_LR"
        --cosine_weight "$COSINE_WEIGHT"
        --mse_weight "$MSE_WEIGHT"
        --save_dir "$RUN_DIR"
        --diagnostic_log "$DIAG_FILE"
        --gen_loss_weight "$GEN_WEIGHT"
  )

  if [[ "$R" != "0" ]]; then
    CMD+=(--use_lora --lora_r "$R" --lora_alpha "$ALPHA")
    if [[ "$LAYERS" != "all" ]]; then
      CMD+=(--lora_layers "$LAYERS")
    fi
  fi

  echo "Running: ${CMD[*]}" | tee "$LOG_FILE"
  ( "${CMD[@]}" ) 2>&1 | tee -a "$LOG_FILE"

  BEST_F1="N/A"
  BEST_EM="N/A"
  if [[ -f "$DIAG_FILE" ]]; then
    METRICS=$(python - "$DIAG_FILE" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
best_f1 = 0.0
best_em = 0.0
seen = False
if path.exists():
    with path.open() as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "full_eval":
                seen = True
                f1 = data.get("f1", 0.0)
                em = data.get("em", 0.0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_em = em
if seen:
    print(f"{best_f1} {best_em}")
PY)
    if [[ -n "$METRICS" ]]; then
      read -r BEST_F1 BEST_EM <<< "$METRICS"
    fi
  fi

  RESULTS[$NAME]="$BEST_F1 $BEST_EM"
  {
    echo ""
    echo "Configuration: $NAME"
    echo "  LoRA: r=$R, alpha=$ALPHA, layers=$LAYERS"
    echo "  gen_weight: $GEN_WEIGHT"
    echo "  F1: $BEST_F1"
    echo "  EM: $BEST_EM"
    echo "  Log: $LOG_FILE"
  } >> "$SUMMARY_FILE"

done

echo ""
echo "=================================================="
echo "SWEEP COMPLETE"
echo "=================================================="

echo "Configuration           F1        EM"
for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r NAME _ <<< "$cfg"
  read -r F1 EM <<< "${RESULTS[$NAME]}"
  if is_number "$F1" && is_number "$EM"; then
    FORMATTED=$(python - "$F1" "$EM" <<'PY'
import sys
f1 = float(sys.argv[1]) * 100
em = float(sys.argv[2]) * 100
print(f"{f1:6.2f}% {em:6.2f}%")
PY)
    printf "%-22s %s\n" "$NAME" "$FORMATTED"
  else
    printf "%-22s %-7s %-7s\n" "$NAME" "N/A" "N/A"
  fi
  echo "$NAME $F1 $EM" >> "$SUMMARY_FILE"
done

echo ""
echo "Summary file: $SUMMARY_FILE"
echo "Artifacts saved under: $OUTPUT_BASE"
