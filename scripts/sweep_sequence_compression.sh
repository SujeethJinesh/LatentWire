#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Fix CUDA/MPS initialization issues on clusters
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Kill MPS daemon if it's causing issues
echo quit | nvidia-cuda-mps-control 2>/dev/null || true

# Verify CUDA is accessible
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA OK: {torch.cuda.device_count()} GPUs')" 2>/dev/null; then
    echo "ERROR: CUDA not accessible despite GPUs being visible"
    echo "Attempting to fix..."

    # Try to reset CUDA state
    nvidia-smi --gpu-reset 2>/dev/null || true

    # Test again
    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "FATAL: Cannot initialize CUDA. Please run manually:"
        echo "  export CUDA_VISIBLE_DEVICES=0,1,2,3"
        echo "  echo quit | nvidia-cuda-mps-control"
        echo "  python3 -c 'import torch; print(torch.cuda.is_available())'"
        exit 1
    fi
fi

echo "CUDA initialized successfully"
echo ""

# Experiment 1: Comprehensive Sequence Compression + LoRA Sweep
#
# Tests moderate sequence compression (2-5×) with LoRA adaptation
# to find optimal configuration for preserving QA performance.
#
# Suggested usage:
#   git pull && rm -rf runs/seq_compression_sweep && PYTHONPATH=. bash scripts/sweep_sequence_compression.sh

MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-10000}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
EVAL_SAMPLES="${EVAL_SAMPLES:-100}"
LR="${LR:-5e-4}"

OUTPUT_BASE="${OUTPUT_BASE:-runs/seq_compression_sweep}"
mkdir -p "$OUTPUT_BASE"
SUMMARY_FILE="$OUTPUT_BASE/sweep_summary.txt"

cat > "$SUMMARY_FILE" <<EOF_SUMMARY
Sequence Compression + LoRA Sweep
Started: $(date)
================================================
Model: $MODEL
Dataset: $DATASET
Training samples: $SAMPLES
Eval samples: $EVAL_SAMPLES
Epochs: $EPOCHS
Batch size: $BATCH_SIZE
Learning rate: $LR
================================================

EOF_SUMMARY

echo "=================================================="
echo "SEQUENCE COMPRESSION + LORA SWEEP"
echo "=================================================="
echo "Model:        $MODEL"
echo "Dataset:      $DATASET"
echo "Samples:      $SAMPLES train, $EVAL_SAMPLES eval"
echo "Epochs:       $EPOCHS"
echo "Batch size:   $BATCH_SIZE"
echo "Output dir:   $OUTPUT_BASE"
echo ""
echo "This sweep will test:"
echo "  - Sequence lengths: 256, 192, 128, 96, 64"
echo "  - LoRA ranks: 8, 16, 32"
echo "  - LoRA layers: 8, 16, 32, all"
echo "  - Pooling methods: learned_attention, convolutional"
echo ""
echo "Expected runtime: 12-24 hours on 4×H100"
echo "=================================================="
echo ""

declare -A RESULTS

is_number() {
  [[ $1 =~ ^[0-9]+([.][0-9]+)?$ ]]
}

# Configuration matrix:
# Format: name:seq_len:pooling:use_lora:lora_r:lora_alpha:lora_layers

CONFIGS=(
  # Baseline: No LoRA
  "seq256_noLoRA:256:learned_attention:0:0:0:0"
  "seq192_noLoRA:192:learned_attention:0:0:0:0"
  "seq128_noLoRA:128:learned_attention:0:0:0:0"
  "seq96_noLoRA:96:learned_attention:0:0:0:0"
  "seq64_noLoRA:64:learned_attention:0:0:0:0"

  # LoRA rank sweep at seq=128
  "seq128_r8_l8:128:learned_attention:1:8:16:8"
  "seq128_r16_l8:128:learned_attention:1:16:32:8"
  "seq128_r32_l8:128:learned_attention:1:32:64:8"

  # LoRA layer sweep at seq=128, r=16
  "seq128_r16_l16:128:learned_attention:1:16:32:16"
  "seq128_r16_l32:128:learned_attention:1:16:32:32"
  "seq128_r16_all:128:learned_attention:1:16:32:0"

  # Best config at different sequence lengths
  "seq256_r16_l8:256:learned_attention:1:16:32:8"
  "seq192_r16_l8:192:learned_attention:1:16:32:8"
  "seq96_r16_l8:96:learned_attention:1:16:32:8"
  "seq64_r16_l8:64:learned_attention:1:16:32:8"

  # Pooling method comparison at seq=128, r=16, l=8
  "seq128_r16_l8_conv:128:convolutional:1:16:32:8"

  # Aggressive compression with strong LoRA
  "seq64_r32_l16:64:learned_attention:1:32:64:16"
  "seq48_r32_l16:48:learned_attention:1:32:64:16"
  "seq32_r32_all:32:learned_attention:1:32:64:0"
)

echo "Total configurations: ${#CONFIGS[@]}"
echo ""

for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r NAME SEQ_LEN POOLING USE_LORA LORA_R LORA_ALPHA LORA_LAYERS <<< "$cfg"

  echo "--------------------------------------------------"
  echo "Configuration: $NAME"
  echo "  Sequence length: $SEQ_LEN (compression: ~$(python3 -c "print(f'{300/$SEQ_LEN:.2f}')"))×"
  echo "  Pooling: $POOLING"
  if [[ "$USE_LORA" == "1" ]]; then
    echo "  LoRA: r=$LORA_R, alpha=$LORA_ALPHA, layers=${LORA_LAYERS:-all}"
  else
    echo "  LoRA: disabled"
  fi
  echo ""

  RUN_DIR="$OUTPUT_BASE/$NAME"
  mkdir -p "$RUN_DIR"
  LOG_FILE="$RUN_DIR/train_$(date +%Y%m%d_%H%M%S).log"
  DIAG_FILE="$RUN_DIR/diagnostics.jsonl"

  CMD=(python train_sequence_compression.py
       --model_id "$MODEL"
       --dataset "$DATASET"
       --samples "$SAMPLES"
       --epochs "$EPOCHS"
       --batch_size "$BATCH_SIZE"
       --max_length "$MAX_LENGTH"
       --eval_samples "$EVAL_SAMPLES"
       --lr "$LR"
       --target_sequence_length "$SEQ_LEN"
       --pooling_method "$POOLING"
       --save_dir "$RUN_DIR"
       --diagnostic_log "$DIAG_FILE"
  )

  if [[ "$USE_LORA" == "1" ]]; then
    CMD+=(--use_lora --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA")
    if [[ "$LORA_LAYERS" != "0" ]]; then
      CMD+=(--lora_layers "$LORA_LAYERS")
    fi
  fi

  echo "Running: ${CMD[*]}" | tee "$LOG_FILE"
  ( "${CMD[@]}" ) 2>&1 | tee -a "$LOG_FILE"

  # Extract best F1 from diagnostics
  BEST_F1="N/A"
  BEST_EM="N/A"
  if [[ -f "$DIAG_FILE" ]]; then
    METRICS=$(python3 - "$DIAG_FILE" <<'PY'
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
PY
    )
    if [[ -n "$METRICS" ]]; then
      read -r BEST_F1 BEST_EM <<< "$METRICS"
    fi
  fi

  RESULTS[$NAME]="$BEST_F1 $BEST_EM"

  # Write to summary
  {
    echo ""
    echo "Configuration: $NAME"
    echo "  Sequence: $SEQ_LEN (compression: $(python3 -c "print(f'{300/$SEQ_LEN:.2f}x')"))"
    echo "  Pooling: $POOLING"
    if [[ "$USE_LORA" == "1" ]]; then
      echo "  LoRA: r=$LORA_R, alpha=$LORA_ALPHA, layers=${LORA_LAYERS:-all}"
    else
      echo "  LoRA: disabled"
    fi
    echo "  F1: $BEST_F1"
    echo "  EM: $BEST_EM"
    echo "  Log: $LOG_FILE"
  } >> "$SUMMARY_FILE"

done

echo ""
echo "=================================================="
echo "SWEEP COMPLETE"
echo "=================================================="
echo ""

# Print results table
echo "Results Summary:"
echo "----------------"
echo ""
printf "%-25s %-12s %-8s %-8s %s\n" "Configuration" "Compression" "F1" "EM" "Notes"
echo "$(printf '%.0s-' {1..80})"

for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r NAME SEQ_LEN POOLING USE_LORA LORA_R LORA_ALPHA LORA_LAYERS <<< "$cfg"
  read -r F1 EM <<< "${RESULTS[$NAME]}"

  COMP_RATIO=$(python3 -c "print(f'{300/$SEQ_LEN:.2f}x')")

  if is_number "$F1" && is_number "$EM"; then
    F1_PCT=$(python3 -c "print(f'{float("'$F1'") * 100:.2f}%')")
    EM_PCT=$(python3 -c "print(f'{float("'$EM'") * 100:.2f}%')")

    # Add notes based on config
    NOTES=""
    if [[ "$USE_LORA" == "0" ]]; then
      NOTES="(no LoRA)"
    elif [[ "$LORA_LAYERS" == "0" ]]; then
      NOTES="(all layers)"
    else
      NOTES="(first $LORA_LAYERS layers)"
    fi

    printf "%-25s %-12s %-8s %-8s %s\n" "$NAME" "$COMP_RATIO" "$F1_PCT" "$EM_PCT" "$NOTES"
  else
    printf "%-25s %-12s %-8s %-8s %s\n" "$NAME" "$COMP_RATIO" "N/A" "N/A" "(failed)"
  fi

  echo "$NAME $SEQ_LEN $F1 $EM" >> "$SUMMARY_FILE"
done

echo ""
echo "Summary file: $SUMMARY_FILE"
echo "Artifacts saved under: $OUTPUT_BASE"
echo ""

# Find best configuration
echo "Finding best configuration..."
echo ""

BEST_CONFIG=""
BEST_F1_VAL=0.0

for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r NAME _ <<< "$cfg"
  read -r F1 _ <<< "${RESULTS[$NAME]}"

  if is_number "$F1"; then
    if (( $(python3 -c "print($F1 > $BEST_F1_VAL)") )); then
      BEST_F1_VAL=$F1
      BEST_CONFIG=$NAME
    fi
  fi
done

if [[ -n "$BEST_CONFIG" ]]; then
  echo "Best configuration: $BEST_CONFIG"
  echo "Best F1: $(python3 -c "print(f'{$BEST_F1_VAL * 100:.2f}%')")"
  echo ""
  echo "Recommended next steps:"
  echo "1. Analyze $OUTPUT_BASE/$BEST_CONFIG/diagnostics.jsonl for training dynamics"
  echo "2. Run longer training (more epochs) with best config"
  echo "3. Test on larger evaluation set"
fi

echo ""
echo "Sweep complete! $(date)"
