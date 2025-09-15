#!/usr/bin/env bash
set -euo pipefail

# ------------- USER TOGGLES (edit these) -----------------

# Phase toggles
DO_TRAIN=1             # set to 0 to skip training
DO_FINAL_EVAL=1        # full eval after all training

# Eval knobs
DATASET="squad"        # "squad", "squad_v2", or "hotpot"
SAMPLES=200            # full eval sample count
SMOKE_SAMPLES=200      # per-epoch eval sample count (kept high for stronger signal)
MAX_NEW_TOKENS=12
SEQUENTIAL_EVAL=1      # per-model auto encoder-text alignment
FRESH_EVAL=1           # recompute Z for eval outputs
LOAD_4BIT=0            # for constrained GPUs
CHUNK_SIZE=8
TOKEN_BUDGET_MODE="content_only"   # "content_only" or "chat_full"
TOKEN_BUDGET_K=32                  # match LATENT_LEN for a fairer budget
FIRST_TOKEN_TOP_P=1.0              # hardened: deterministic first step
FIRST_TOKEN_TEMPERATURE=0.0        # hardened: deterministic first step

# Anchor & decode controls
LATENT_ANCHOR_MODE="text"
LATENT_ANCHOR_TEXT="Answer: "      # note the trailing space
APPEND_BOS_AFTER_PREFIX="yes"      # **important**: align eval first-step with training
CALIBRATION="embed_rms"
PREFIX_GAIN=1.0

# Debug printing (safe to leave on)
DEBUG=1
DEBUG_PRINT_FIRST=5
DEBUG_TOPK=10
DEBUG_TOPK_EXAMPLES=2

# Training knobs
EPOCHS=16
BATCH_SIZE=128
TRAIN_SAMPLES=87599
ENCODER_TYPE="byte"                 # stronger, token-level input
ENCODER_USE_CHAT_TEMPLATE=0         # training wants raw sources
LATENT_LEN=32
D_Z=256
BYTE_MAX=2048                       # byte budget for the encoder
LR=5e-5
SCALE_L2=0.05
ADAPTER_RMS_L2=0.0
MAX_GRAD_NORM=1.0
WARM_ANCHOR_TEXT="Answer: "         # Matches eval anchor
FIRST_TOKEN_CE=0.5                  # λ_first (first-token CE weight)
TRAIN_APPEND_BOS="yes"              # BOS after prefix+anchor during first-token CE
SAVE_EVERY=1                        # save only at the end of each epoch (we copy ckpt dir)
SEQUENTIAL_MODELS=1

# Model IDs
LLAMA_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN_ID="Qwen/Qwen2.5-7B-Instruct"

# GPU selection
CUDA_VISIBLE_DEVICES="0,1"

# ------------- PATHS & RUNTIME -----------------

source .venv/bin/activate
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

# Run folder name
RUN="8B_clean_answer_ftce"  # new run with first-token CE + BOS alignment
RUN_DIR="runs/${RUN}"
CKPT_DIR="${RUN_DIR}/ckpt"
mkdir -p "$RUN_DIR" "$CKPT_DIR"

# Log file
ts="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RUN_DIR}/pipeline_${ts}.log"

# Ensure helper is present
if [[ ! -f "latentwire/common.py" ]]; then
  echo "FATAL: missing latentwire/common.py" >&2
  exit 1
fi

# Utility functions
print_header() {
  echo ""
  echo "========================================="
  echo "$1"
  echo "========================================="
  echo ""
}

print_metrics() {
  local dir="$1"
  if [[ -f "${dir}/metrics.json" ]]; then
    echo "✓ Metrics from: ${dir}/metrics.json"
    EV_DIR="${dir}" python - <<'PY'
import os, json, sys
p = os.environ.get("EV_DIR")
if not p:
    print("EV_DIR env var missing", file=sys.stderr); sys.exit(1)
with open(os.path.join(p,"metrics.json")) as f:
    m=json.load(f)

def get(*ks, d=None, default=None):
    cur=d or m
    for k in ks:
        if cur is None: return default
        cur = cur.get(k)
    return cur if cur is not None else default

def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"

print(f"  Text F1:     Llama {fmt(get('text','llama','f1'))} | Qwen {fmt(get('text','qwen','f1'))}")
print(f"  Latent F1:   Llama {fmt(get('latent','llama','f1'))} | Qwen {fmt(get('latent','qwen','f1'))}")
print(f"  FirstTok@1:  Llama {fmt(get('latent','llama','first_token_top1'))} | Qwen {fmt(get('latent','qwen','first_token_top1'))}")
print(f"  FirstTok@5:  Llama {fmt(get('latent','llama','first_token_top5'))} | Qwen {fmt(get('latent','qwen','first_token_top5'))}")
PY
  else
    echo "✗ Metrics missing in ${dir}"
  fi
}

run_eval() {
  local ckpt_path="$1"
  local out_dir="$2"
  local n_samples="$3"
  local quiet="${4:-0}"

  mkdir -p "$out_dir"

  EVAL_ARGS=(
    --ckpt "$ckpt_path"
    --llama_id "$LLAMA_ID" --qwen_id "$QWEN_ID"
    --dataset "$DATASET" --samples "$n_samples"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --latent_anchor_mode "$LATENT_ANCHOR_MODE"
    --latent_anchor_text "$LATENT_ANCHOR_TEXT"
    --append_bos_after_prefix "$APPEND_BOS_AFTER_PREFIX"
    --calibration "$CALIBRATION" --prefix_gain "$PREFIX_GAIN"
    --token_budget_mode "$TOKEN_BUDGET_MODE" --token_budget_k "$TOKEN_BUDGET_K"
    --first_token_top_p "$FIRST_TOKEN_TOP_P"
    --first_token_temperature "$FIRST_TOKEN_TEMPERATURE"
    --min_new_tokens 3 --eos_ban_steps 6
    --chunk_size "$CHUNK_SIZE"
    --out_dir "$out_dir"
    --sequential_eval
  )

  if [[ $FRESH_EVAL -eq 1 ]]; then
    EVAL_ARGS+=(--fresh_eval)
  fi
  if [[ $LOAD_4BIT -eq 1 ]]; then
    EVAL_ARGS+=(--load_4bit)
  fi
  if [[ $DEBUG -eq 1 ]] && [[ $quiet -eq 0 ]]; then
    EVAL_ARGS+=(--debug --debug_print_first "$DEBUG_PRINT_FIRST"
                --debug_topk "$DEBUG_TOPK" --debug_topk_examples "$DEBUG_TOPK_EXAMPLES")
  fi

  if [[ $quiet -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    python -u latentwire/eval.py "${EVAL_ARGS[@]}" > /dev/null 2>&1
  else
    echo "Evaluating: ${ckpt_path} -> ${out_dir}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    python -u latentwire/eval.py "${EVAL_ARGS[@]}"
  fi
}

# ------------- START PIPELINE -----------------

{
  print_header "Starting pipeline at $(date)"

  # ======================================================
  # PHASE 1: TRAINING WITH EPOCH EVALUATIONS
  # ======================================================
  if [[ $DO_TRAIN -eq 1 ]]; then
    print_header "PHASE 1: TRAINING WITH EPOCH EVALUATIONS"
    echo "Training for $EPOCHS epochs with evaluation after each"
    echo "Checkpoint will be saved to: ${CKPT_DIR}"
    echo ""

    for epoch in $(seq 1 $EPOCHS); do
      print_header "EPOCH $epoch/$EPOCHS"

      TRAIN_ARGS=(
        --dataset "$DATASET" --samples "$TRAIN_SAMPLES"
        --epochs 1 --batch_size "$BATCH_SIZE"
        --encoder_type "$ENCODER_TYPE"
        --latent_len "$LATENT_LEN" --d_z "$D_Z"
        --qwen_id "$QWEN_ID" --llama_id "$LLAMA_ID"
        --lr "$LR" --scale_l2 "$SCALE_L2"
        --adapter_rms_l2 "$ADAPTER_RMS_L2"
        --max_grad_norm "$MAX_GRAD_NORM"
        --save_dir "$CKPT_DIR"
        --save_every "$SAVE_EVERY"
        --save_training_stats --debug --auto_resume
        --max_bytes "$BYTE_MAX"
        --first_token_ce_weight "$FIRST_TOKEN_CE"
        --train_append_bos_after_prefix "$TRAIN_APPEND_BOS"
      )

      if [[ "${ENCODER_USE_CHAT_TEMPLATE}" == "1" ]]; then
        TRAIN_ARGS+=(--encoder_use_chat_template)
      fi
      if [[ -n "${WARM_ANCHOR_TEXT}" ]]; then
        TRAIN_ARGS+=(--warm_anchor_text "$WARM_ANCHOR_TEXT")
      fi
      if [[ $SEQUENTIAL_MODELS -eq 1 ]]; then
        TRAIN_ARGS+=(--sequential_models)
      fi
      if [[ $LOAD_4BIT -eq 1 ]]; then
        TRAIN_ARGS+=(--load_4bit)
      fi

      echo "Training epoch $epoch..."
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      python -u latentwire/train.py "${TRAIN_ARGS[@]}"

      # Save checkpoint for this epoch (canonical snapshot)
      epoch_ckpt="${RUN_DIR}/epoch${epoch}"
      rm -rf "$epoch_ckpt"
      cp -r "$CKPT_DIR" "$epoch_ckpt"

      # Evaluate this epoch
      echo ""
      echo "Evaluating epoch $epoch checkpoint..."
      epoch_eval="${RUN_DIR}/eval_epoch${epoch}"
      run_eval "$epoch_ckpt" "$epoch_eval" "$SMOKE_SAMPLES" 0

      echo ""
      print_metrics "$epoch_eval"
      echo ""
    done

    print_header "TRAINING COMPLETE - EPOCH SUMMARY"
    for epoch in $(seq 1 $EPOCHS); do
      echo "Epoch $epoch results:"
      print_metrics "${RUN_DIR}/eval_epoch${epoch}"
      echo ""
    done
  fi

  # ======================================================
  # PHASE 2: FINAL FULL EVALUATION
  # ======================================================
  if [[ $DO_FINAL_EVAL -eq 1 ]]; then
    print_header "PHASE 2: FINAL FULL EVALUATION"

    # Find best epoch based on latent F1 (average of the two models)
    best_epoch=0
    best_f1=0
    for epoch in $(seq 1 $EPOCHS); do
      eval_dir="${RUN_DIR}/eval_epoch${epoch}"
      if [[ -f "${eval_dir}/metrics.json" ]]; then
        f1=$(python - <<PY 2>/dev/null || echo "0"
import json
with open('${eval_dir}/metrics.json') as f:
    m = json.load(f)
ll = m.get('latent',{}).get('llama',{}).get('f1',0) or 0
qw = m.get('latent',{}).get('qwen',{}).get('f1',0) or 0
print((float(ll) + float(qw)) / 2.0)
PY
)
        if (( $(echo "$f1 > $best_f1" | bc -l) )); then
          best_f1=$f1
          best_epoch=$epoch
        fi
      fi
    done

    if [[ $best_epoch -gt 0 ]]; then
      echo "Best epoch: $best_epoch (avg latent F1: $best_f1)"
      echo "Running full evaluation on best checkpoint..."

      best_ckpt="${RUN_DIR}/epoch${best_epoch}"
      final_eval="${RUN_DIR}/eval_final_best"
      run_eval "$best_ckpt" "$final_eval" "$SAMPLES" 0

      echo ""
      print_metrics "$final_eval"
    else
      echo "WARNING: No valid epoch evaluations found"
    fi
  fi

  print_header "PIPELINE SUMMARY"
  echo "Run ID: $RUN"
  echo "Completed: $(date)"
  echo ""
  echo "Outputs:"
  for epoch in $(seq 1 $EPOCHS); do
    echo "  Epoch $epoch: ${RUN_DIR}/epoch${epoch}/"
    echo "           Eval: ${RUN_DIR}/eval_epoch${epoch}/"
  done
  if [[ $DO_FINAL_EVAL -eq 1 ]] && [[ ${best_epoch:-0} -gt 0 ]]; then
    echo "  Best checkpoint: ${RUN_DIR}/epoch${best_epoch}/"
    echo "  Final eval: ${RUN_DIR}/eval_final_best/"
  fi

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "All output saved to: $LOG_FILE"
echo "Check ${RUN_DIR}/eval_epoch*/ for per-epoch metrics"
