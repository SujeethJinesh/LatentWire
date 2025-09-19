#!/usr/bin/env bash
set -euo pipefail

# Hero run v3 (Attempt A+B+C, compact verification)
# - Fixes t=0 alignment by using chat anchor (no "Answer: " text anchor)
# - Uses chat templates for encoder text (neutral chat) to match LLM distribution
# - Stronger first-token CE and short-horizon K-token CE
# - No eval-time quantization of Z (remove confounders); bf16 spot-check
# - Small sample & few epochs to finish in ~2 hours on 4x H100 (adjust if needed)
# - Saves full metrics JSON and prints compact summary

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$THIS_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# --------------- Config ---------------
RUN_NAME="hero_v3_ABCs"
OUT_ROOT="runs/${RUN_NAME}"
mkdir -p "$OUT_ROOT"

# Data & model choices
DATASET="squad"
LLAMA_ID="meta-llama/Llama-3.1-8B-Instruct"
QWEN_ID="Qwen/Qwen2.5-7B-Instruct"

# Encoder (Attempt A: STQuery; Attempt B: byte fallback off; Attempt C: early-K losses already in code)
ENCODER_TYPE="simple-stq"                # you're using MiniLM
ENC_BACKBONE="sentence-transformers/all-MiniLM-L6-v2"
ENC_USE_CHAT=1                            # neutral chat wrapper for encoder inputs

# Budget: keep short so we can iterate quickly
SAMPLES=${SAMPLES:-1600}
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-16}
LATENT_LEN=${LATENT_LEN:-48}             # ensures >=4x compression on SQuAD prompts
MAX_NEW=16

# Train/Eval deterministic knobs
SEED=123
APPEND_BOS="auto"                         # let code pick BOS usage per model
ANCHOR_MODE="auto"                        # CRITICAL: uses chat assistant header, not "Answer: "
ANCHOR_TEXT=""                            # force empty text-anchor
CALIB="embed_rms"
PREF_GAIN=1.0

# Loss weights (conservative but stronger than v2)
FIRST_W=6.0                               # first-token CE
KCE_W=0.3                                 # K-token CE on first ~K tokens
KD_W=0.2                                  # KD between latent and text scaffolds
K_TOKENS=8
TEMP=1.25                                  # for KD

# Other training
LR=3e-4
WARMUP=100
GRAD_ACC=1
GRAD_CLIP=1.0
WEIGHT_DECAY=0.01

# Devices
DEV="cuda"
MAX_BYTES=512
AMP_BF16=1
GRAD_CKPT=0

# Optional: sequential eval for stability across big models
SEQ_EVAL=1
FRESH_EVAL=1

RUN_DIR="${OUT_ROOT}"
LOG_FILE="${OUT_ROOT}/pipeline_$(date +%Y%m%d_%H%M%S).log"

# --------------- Helpers ---------------
print_header () { echo -e "\\n=========================================\\n$1\\n=========================================\\n"; }

run_train () {
  print_header "TRAINING (${EPOCHS} epochs, ${SAMPLES} samples, latent M=${LATENT_LEN})"
  python -u -m latentwire.train \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    --max_new_tokens "$MAX_NEW" \
    --device "$DEV" \
    --seed "$SEED" \
    --out_dir "$RUN_DIR" \
    --llama_id "$LLAMA_ID" \
    --qwen_id "$QWEN_ID" \
    --encoder_type "$ENCODER_TYPE" \
    --encoder_backbone "$ENC_BACKBONE" \
    $( [[ "$ENC_USE_CHAT" -eq 1 ]] && echo --encoder_use_chat_template ) \
    --latent_len "$LATENT_LEN" \
    --max_bytes "$MAX_BYTES" \
    --train_append_bos_after_prefix "$APPEND_BOS" \
    --warm_anchor_text "$ANCHOR_TEXT" \
    --first_token_ce_weight "$FIRST_W" \
    --k_token_ce_weight "$KCE_W" \
    --kd_first_k_weight "$KD_W" \
    --first_k "$K_TOKENS" \
    --kd_temperature "$TEMP" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_steps "$WARMUP" \
    --grad_accum "$GRAD_ACC" \
    --grad_clip_norm "$GRAD_CLIP" \
    $( [[ "$AMP_BF16" -eq 1 ]] && echo --amp_bf16 ) \
    $( [[ "$GRAD_CKPT" -eq 1 ]] && echo --grad_ckpt )
}

run_eval () {
  CKPT_DIR="$1"
  OUT_EVAL="$2"
  N="$3"
  BF16="$4"
  print_header "EVAL (N=$N) @ $CKPT_DIR -> $OUT_EVAL"
  mkdir -p "$OUT_EVAL"
  python -u -m latentwire.eval \
    --ckpt "$CKPT_DIR" \
    --out_dir "$OUT_EVAL" \
    --dataset "$DATASET" \
    --samples "$N" \
    --max_new_tokens "$MAX_NEW" \
    --device "$DEV" \
    --seed "$SEED" \
    --latent_len "$LATENT_LEN" \
    --append_bos_after_prefix "$APPEND_BOS" \
    --latent_anchor_mode "$ANCHOR_MODE" \
    --latent_anchor_text "$ANCHOR_TEXT" \
    --calibration "$CALIB" \
    --prefix_gain "$PREF_GAIN" \
    --sequential_eval "$SEQ_EVAL" \
    --fresh_eval "$FRESH_EVAL" \
    $( [[ "$BF16" -eq 1 ]] && echo --force_bf16 )
}

print_metrics () {
  DIR="$1"
  echo ""
  echo "==== METRICS SNAPSHOT ($DIR) ===="
  if [[ -f "$DIR/metrics.json" ]]; then
    python - <<'PY'
import json,sys,os,math
with open(os.path.join(sys.argv[1], "metrics.json")) as f:
    m=json.load(f)
def g(*ks, default=None):
    cur=m
    for k in ks:
        if cur is None: break
        cur=cur.get(k) if isinstance(cur, dict) else None
    return cur if cur is not None else default
def fmt(x):
    try: return f"{float(x):.3f}"
    except: return "-"
print(f"  Text  F1: Llama {fmt(g('text','llama','f1'))} | Qwen {fmt(g('text','qwen','f1'))}")
print(f"  Latent F1: Llama {fmt(g('latent','llama','f1'))} | Qwen {fmt(g('latent','qwen','f1'))}")
print(f"  FirstTok@1: Llama {fmt(g('latent','llama','first_token_top1'))} | Qwen {fmt(g('latent','qwen','first_token_top1'))}")
print(f"  FirstTok@5: Llama {fmt(g('latent','llama','first_token_top5'))} | Qwen {fmt(g('latent','qwen','first_token_top5'))}")
print(f"  NLL/token:  Llama {fmt(g('latent','llama','nll_token'))} | Qwen {fmt(g('latent','qwen','nll_token'))}")
print(f"  Compression: Llama {fmt(g('compression','llama'))}× | Qwen {fmt(g('compression','qwen'))}×")
PY
  else
    echo "✗ metrics.json missing in $DIR"
  fi
}

# --------------- Pipeline ---------------
{
  print_header "Starting pipeline at $(date)"
  run_train
  # quick smoke eval on the last checkpoint
  LAST="${RUN_DIR}/epoch${EPOCHS}"
  run_eval "$LAST" "${RUN_DIR}/eval_smoke_bf16" 200 1
  print_metrics "${RUN_DIR}/eval_smoke_bf16"

  # optional full eval on small set
  run_eval "$LAST" "${RUN_DIR}/eval_small" $SAMPLES 1
  print_metrics "${RUN_DIR}/eval_small"

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "All output saved to: $LOG_FILE"
