#!/usr/bin/env bash
set -euo pipefail

# ------------- USER TOGGLES (edit these) -----------------

# Phase toggles
DO_SMOKE_EVAL=0        # quick small eval to sanity-check the run
DO_FULL_EVAL=1         # full eval (uses SAMPLES below)
DO_TRAIN=1             # set to 1 when you're ready to retrain

# Eval knobs
DATASET="squad"        # "squad", "squad_v2", or "hotpot"
SAMPLES=200            # full eval sample count
SMOKE_SAMPLES=16       # smoke eval sample count
MAX_NEW_TOKENS=12
SEQUENTIAL_EVAL=1      # 1 = per-model auto encoder-text alignment
FRESH_EVAL=1           # 1 = recompute Z for eval outputs
LOAD_4BIT=0            # for constrained GPUs
CHUNK_SIZE=8
TOKEN_BUDGET_MODE="content_only"   # "content_only" or "chat_full"
TOKEN_BUDGET_K=16
FIRST_TOKEN_TOP_P=1.0
FIRST_TOKEN_TEMPERATURE=0.0

# Anchor & decode controls
LATENT_ANCHOR_MODE="text"
LATENT_ANCHOR_TEXT="Answer: "    # note the trailing space
APPEND_BOS_AFTER_PREFIX="no"    # force a BOS after prefix+anchor to reset state
CALIBRATION="embed_rms"
PREFIX_GAIN=1.0                 # small boost to prefix amplitude after calibration

# Debug printing (safe to leave on)
DEBUG=1
DEBUG_PRINT_FIRST=5
DEBUG_TOPK=10
DEBUG_TOPK_EXAMPLES=2

# Training knobs (enable DO_TRAIN=1 to use)
EPOCHS=8
BATCH_SIZE=128
TRAIN_SAMPLES=87599
ENCODER_TYPE="simple-st"     # "byte" or "simple-st"
ENCODER_USE_CHAT_TEMPLATE=1  # only for SimpleEncoder
LATENT_LEN=16
D_Z=256
LR=5e-5
SCALE_L2=0.05
ADAPTER_RMS_L2=0.0
MAX_GRAD_NORM=1.0
WARM_ANCHOR_TEXT="Answer: "  # <-- keep consistent with eval; used during training
SAVE_EVERY=2000              # checkpoint pruning cadence
SEQUENTIAL_MODELS=0          # dual loss by default; can set to 1

# Model IDs
LLAMA_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN_ID="Qwen/Qwen2.5-7B-Instruct"

# GPU selection (comma-separated IDs)
CUDA_VISIBLE_DEVICES="0,1"

# ------------- PATHS & RUNTIME -----------------

source .venv/bin/activate
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

# Run folder name
RUN="8B_runs"
RUN_DIR="runs/${RUN}"
CKPT_DIR="${RUN_DIR}/ckpt"
mkdir -p "$RUN_DIR" "$CKPT_DIR"

# Eval output directories (smoke & full)
EV_TAG="${DATASET}_answer_${LATENT_ANCHOR_MODE}"
EV_DIR="${RUN_DIR}/eval_${EV_TAG}"
EV_SMOKE_DIR="${RUN_DIR}/eval_${EV_TAG}_smoke"

# Log file
ts="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RUN_DIR}/pipeline_${ts}.log"

# Ensure new refactored helper is present
if [[ ! -f "latentwire/common.py" ]]; then
  echo "FATAL: missing latentwire/common.py. Please add the helper module I provided." >&2
  exit 1
fi

# Utility to print a section header
print_header() {
  echo ""
  echo "========================================="
  echo "$1"
  echo "========================================="
  echo ""
}

# Utility to render key metrics from a finished eval directory
print_metrics() {
  local dir="$1"
  if [[ -f "${dir}/metrics.json" ]]; then
    echo "✓ Evaluation metrics saved at: ${dir}/metrics.json"
    echo ""
    echo "Key metrics:"
    # Pass EV_DIR to python via environment to avoid ${...} expansion pitfalls
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

compL = get("compression","llama", default="-")
compQ = get("compression","qwen", default="-")
print(f"  Compression: Llama {fmt(compL,1)}x | Qwen {fmt(compQ,1)}x")

print(f"  Text F1:     Llama {fmt(get('text','llama','f1'))} | Qwen {fmt(get('text','qwen','f1'))}")
print(f"  Latent F1:   Llama {fmt(get('latent','llama','f1'))} | Qwen {fmt(get('latent','qwen','f1'))}")

je = get('joint','em'); jf = get('joint','f1')
if je is not None or jf is not None:
    print(f"  Joint:       EM {fmt(je)} | F1 {fmt(jf)}")
PY
  else
    echo "✗ Evaluation metrics missing in ${dir}"
  fi
}

# ------------- START PIPELINE -----------------

{
  print_header "Starting pipeline at $(date)"

  # --- Quick sanity: verify checkpoint presence (skip if training will create it)
  if [[ $DO_TRAIN -eq 0 ]]; then
    for f in "${CKPT_DIR}/config.json" "${CKPT_DIR}/encoder.pt" "${CKPT_DIR}/adapter_llama.pt" "${CKPT_DIR}/adapter_qwen.pt"; do
      if [[ ! -f "$f" ]]; then
        echo "FATAL: expected checkpoint artifact missing: $f"
        echo "Hint: enable DO_TRAIN=1 or point CKPT_DIR to an existing run."
        exit 1
      fi
    done
  fi

  # ======================================================
  # PHASE 1: TRAINING (optional)
  # ======================================================
  print_header "PHASE 1: TRAINING (optional)"
  echo "Checkpoint will be saved to: ${CKPT_DIR}"
  echo ""

  if [[ $DO_TRAIN -eq 1 ]]; then
    TRAIN_ARGS=(
      --dataset "$DATASET" --samples "$TRAIN_SAMPLES"
      --epochs "$EPOCHS" --batch_size "$BATCH_SIZE"
      --encoder_type "$ENCODER_TYPE"
      --latent_len "$LATENT_LEN" --d_z "$D_Z"
      --qwen_id "$QWEN_ID" --llama_id "$LLAMA_ID"
      --lr "$LR" --scale_l2 "$SCALE_L2" --adapter_rms_l2 "$ADAPTER_RMS_L2"
      --max_grad_norm "$MAX_GRAD_NORM"
      --save_dir "$CKPT_DIR" --save_every "$SAVE_EVERY"
      --save_training_stats --debug --auto_resume
    )

    # Optional flags
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

    echo "Running training..."
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    python -u latentwire/train.py "${TRAIN_ARGS[@]}"
    echo "Training complete."
    echo ""
  else
    echo "(training skipped)"
    echo ""
  fi

  # After training: assert ckpt artifacts exist
  for f in "${CKPT_DIR}/config.json" "${CKPT_DIR}/encoder.pt" "${CKPT_DIR}/adapter_llama.pt" "${CKPT_DIR}/adapter_qwen.pt"; do
    if [[ ! -f "$f" ]]; then
      echo "FATAL: expected checkpoint artifact missing after training: $f"
      exit 1
    fi
  done

  # ======================================================
  # PHASE 2: EVALUATION
  # ======================================================
  print_header "PHASE 2: EVALUATION"
  echo "Using checkpoint from: ${CKPT_DIR}"
  echo ""

  run_eval () {
    local out_dir="$1"
    local n_samples="$2"

    mkdir -p "$out_dir"

    # Compose eval command
    EVAL_ARGS=(
      --ckpt "$CKPT_DIR"
      --dataset "$DATASET" --samples "$n_samples"
      --max_new_tokens "$MAX_NEW_TOKENS"
      --latent_anchor_mode "$LATENT_ANCHOR_MODE" --latent_anchor_text "$LATENT_ANCHOR_TEXT"
      --append_bos_after_prefix "$APPEND_BOS_AFTER_PREFIX"
      --calibration "$CALIBRATION" --prefix_gain "$PREFIX_GAIN"
      --token_budget_mode "$TOKEN_BUDGET_MODE" --token_budget_k "$TOKEN_BUDGET_K"
      --first_token_top_p "$FIRST_TOKEN_TOP_P" --first_token_temperature "$FIRST_TOKEN_TEMPERATURE"
      --min_new_tokens 2 --eos_ban_steps 6
      --chunk_size "$CHUNK_SIZE"
      --out_dir "$out_dir"
    )

    # Optional eval flags
    if [[ $SEQUENTIAL_EVAL -eq 1 ]]; then
      EVAL_ARGS+=(--sequential_eval)
    fi
    if [[ $FRESH_EVAL -eq 1 ]]; then
      EVAL_ARGS+=(--fresh_eval)
    fi
    if [[ $LOAD_4BIT -eq 1 ]]; then
      EVAL_ARGS+=(--load_4bit)
    fi
    if [[ $DEBUG -eq 1 ]]; then
      EVAL_ARGS+=(--debug --debug_print_first "$DEBUG_PRINT_FIRST" --debug_topk "$DEBUG_TOPK" --debug_topk_examples "$DEBUG_TOPK_EXAMPLES")
    fi

    echo "Running eval (samples=${n_samples}) -> ${out_dir}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    python -u latentwire/eval.py "${EVAL_ARGS[@]}"

    echo ""
    print_metrics "$out_dir"
    echo ""
  }

  # (A) Smoke eval
  if [[ $DO_SMOKE_EVAL -eq 1 ]]; then
    run_eval "$EV_SMOKE_DIR" "$SMOKE_SAMPLES"
  else
    echo "(smoke eval skipped)"
  fi

  # (B) Full eval
  if [[ $DO_FULL_EVAL -eq 1 ]]; then
    run_eval "$EV_DIR" "$SAMPLES"
  else
    echo "(full eval skipped)"
  fi

  print_header "PIPELINE SUMMARY"
  echo "Run ID: $RUN"
  echo "Completed: $(date)"
  echo "Outputs:"
  echo "  Training checkpoint: ${CKPT_DIR}/"
  if [[ $DO_FULL_EVAL -eq 1 ]]; then
    echo "  Evaluation (full):   ${EV_DIR}/"
  fi
  if [[ $DO_SMOKE_EVAL -eq 1 ]]; then
    echo "  Evaluation (smoke):  ${EV_SMOKE_DIR}/"
  fi

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "All output has been saved to:"
echo "  Full log: $LOG_FILE"
[[ $DO_FULL_EVAL -eq 1 ]]  && echo "  Full evaluation: ${EV_DIR}/metrics.json"
[[ $DO_SMOKE_EVAL -eq 1 ]] && echo "  Smoke evaluation: ${EV_SMOKE_DIR}/metrics.json"
