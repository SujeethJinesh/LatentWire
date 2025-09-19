#!/usr/bin/env bash
set -euo pipefail

# =========================================
#  LatentWire — Hail‑Mary All‑in‑One Run
#  One pipeline, A+B+C knobs enabled
#  (STQueryEncoder + asymmetric latent + KD + first‑token CE + calibration)
# =========================================

# ---- Quick preset (change to HERO for full run) ----
PRESET="${PRESET:-FAST_2H}"   # FAST_2H | HERO

# ---- Phases toggles (we keep one continuous run; these enable knobs) ----
DO_TRAIN=1
DO_FINAL_EVAL=1

# ---- Dataset & eval ----
DATASET="squad"
SAMPLES=1000             # full eval sample count
SMOKE_SAMPLES=400        # per-epoch eval sample count (keep high enough to see signal)
MAX_NEW_TOKENS=16
CHUNK_SIZE=96             # eval decode batching
TOKEN_BUDGET_MODE="content_only"
TOKEN_BUDGET_K=32

# Deterministic first token for fair latent/text comparison
FIRST_TOKEN_TOP_P=1.0
FIRST_TOKEN_TEMPERATURE=0.0

# ---- Anchor & calibration (important for frozen acceptance) ----
# Use chat-aware anchoring (per-model assistant header) + explicit BOS after prefix
LATENT_ANCHOR_MODE="auto"
LATENT_ANCHOR_TEXT=""
APPEND_BOS_AFTER_PREFIX="yes"
CALIBRATION="embed_rms"
PREFIX_GAIN=1.15
SEQ_EVAL=1
FRESH_EVAL=1

# ---- Training core knobs (A+B+C ON) ----
# Encoder: STQueryEncoder (MiniLM)
ENCODER_TYPE="stq"
HF_ENCODER_ID="sentence-transformers/all-MiniLM-L6-v2"
MAX_ENC_TOKENS=1024

# Latent shape (asymmetric within a fixed budget)
LATENT_LEN=48                   # 4×+ compression target (avg ~ 240–260 text tokens)
LATENT_SHARED_LEN=36
LATENT_PRIVATE_LEN=6            # per model; code splits remaining

# Capacity & stability
D_Z=256
LR=3e-5
SCALE_L2=0.05
ADAPTER_RMS_L2=0.0
MAX_GRAD_NORM=1.0
ADAPTER_HIDDEN_MULT=2
ADAPTER_COLORIZE=1
ADAPTER_METADATA=1

# First-token cross-entropy (stabilizes BOS acceptance)
WARM_ANCHOR_TEXT=""
FIRST_TOKEN_CE=3.0
TRAIN_APPEND_BOS="yes"          # keep BOS alignment with eval

# Regularizers & KD (B & C)
MANIFOLD_STAT_WEIGHT=0.001
STATE_KD_WEIGHT=0.10            # KD on early layers (A+B+C "on")
STATE_KD_LAYERS="0,1,2"
K=8                              # top‑K teacher guidance steps
K_CE_WEIGHT=1.2
KD_FIRST_K_WEIGHT=1.5
KD_TAU=1.25
ADAPTIVE_K_START=${ADAPTIVE_K_START:-4}
ADAPTIVE_K_END=${ADAPTIVE_K_END:-8}
LATENT_KEEP_START=${LATENT_KEEP_START:-1.0}
LATENT_KEEP_END=${LATENT_KEEP_END:-0.85}
LATENT_KEEP_POWER=${LATENT_KEEP_POWER:-2.0}

# Hardware & quant
LOAD_4BIT=${LOAD_4BIT:-0}       # disable by default to remove NF4 confounders
SEQUENTIAL_MODELS=1             # sequential backprop to avoid param/state on different devices
GRAD_CKPT=1

# Models
LLAMA_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN_ID="Qwen/Qwen2.5-7B-Instruct"

# GPU layout (4× H100 expected)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LLAMA_DEVICES="${LLAMA_DEVICES:-0,1}"
QWEN_DEVICES="${QWEN_DEVICES:-2,3}"
GPU_MEM_GIB="${GPU_MEM_GIB:-78}"
LLAMA_DEVICE_MAP="auto"
QWEN_DEVICE_MAP="auto"

# ---- Presets ----
if [[ "$PRESET" == "HERO" ]]; then
  EPOCHS=12
  BATCH_SIZE=24
  GRAD_ACCUM_STEPS=32
  TRAIN_SAMPLES=40000           # ~half SQuAD train; faster epochs than 87k
elif [[ "$PRESET" == "FAST_2H" ]]; then
  EPOCHS=13
  BATCH_SIZE=24
  GRAD_ACCUM_STEPS=32
  TRAIN_SAMPLES=7000            # quick verification; ~2–4 epochs in ~2h on 4×H100
  SAMPLES=400                   # full eval budget smaller in fast mode
  SMOKE_SAMPLES=200
else
  echo "Unknown PRESET='$PRESET' (use FAST_2H or HERO)"; exit 2
fi

# ---- Paths ----
RUN="8B_hailmary_allknobs_${PRESET,,}"
RUN_DIR="runs/${RUN}"
CKPT_DIR="${RUN_DIR}/ckpt"
mkdir -p "$RUN_DIR" "$CKPT_DIR"

ts="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RUN_DIR}/pipeline_${ts}.log"

# ---- Environment ----
source .venv/bin/activate
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES LLAMA_DEVICES QWEN_DEVICES

# ---- Helpers ----
print_header() { echo -e "\n=========================================\n$1\n=========================================\n"; }

resolve_ckpt_for_eval() {
  local path="$1"
  if [[ -f "${path}/state.pt" ]]; then
    echo "${path}/state.pt"
  elif [[ -f "${path}/encoder.pt" ]]; then
    echo "$path"
  else
    echo "$path"
  fi
}

TRAIN_ARGS_COMMON=(
  --dataset "$DATASET" --samples "$TRAIN_SAMPLES"
  --epochs 1 --batch_size "$BATCH_SIZE"
  --encoder_type "$ENCODER_TYPE"
  --hf_encoder_id "$HF_ENCODER_ID" --max_enc_tokens "$MAX_ENC_TOKENS"
  --latent_len "$LATENT_LEN" --latent_shared_len "$LATENT_SHARED_LEN" --latent_private_len "$LATENT_PRIVATE_LEN" --d_z "$D_Z"
  --qwen_id "$QWEN_ID" --llama_id "$LLAMA_ID"
  --lr "$LR" --scale_l2 "$SCALE_L2" --adapter_rms_l2 "$ADAPTER_RMS_L2" --max_grad_norm "$MAX_GRAD_NORM"
  --max_bytes 2048 --max_answer_tokens 24
  --first_token_ce_weight "$FIRST_TOKEN_CE"
  --train_append_bos_after_prefix "$TRAIN_APPEND_BOS"
  --encoder_use_chat_template
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --adapter_hidden_mult "$ADAPTER_HIDDEN_MULT"
  --manifold_stat_weight "$MANIFOLD_STAT_WEIGHT"
  --state_kd_weight "$STATE_KD_WEIGHT" --state_kd_layers "$STATE_KD_LAYERS"
  --K "$K" --k_ce_weight "$K_CE_WEIGHT" --kd_first_k_weight "$KD_FIRST_K_WEIGHT" --kd_tau "$KD_TAU"
  --adaptive_k_start "$ADAPTIVE_K_START" --adaptive_k_end "$ADAPTIVE_K_END"
  --latent_keep_start "$LATENT_KEEP_START" --latent_keep_end "$LATENT_KEEP_END" --latent_keep_power "$LATENT_KEEP_POWER"
  --llama_device_map "$LLAMA_DEVICE_MAP" --qwen_device_map "$QWEN_DEVICE_MAP"
  --llama_devices "$LLAMA_DEVICES" --qwen_devices "$QWEN_DEVICES" --gpu_mem_gib "$GPU_MEM_GIB"
)

EVAL_ARGS_COMMON=(
  --dataset "$DATASET" --max_new_tokens "$MAX_NEW_TOKENS"
  --latent_anchor_mode "$LATENT_ANCHOR_MODE" --latent_anchor_text "$LATENT_ANCHOR_TEXT"
  --append_bos_after_prefix "$APPEND_BOS_AFTER_PREFIX"
  --calibration "$CALIBRATION" --prefix_gain "$PREFIX_GAIN"
  --token_budget_mode "$TOKEN_BUDGET_MODE" --token_budget_k "$TOKEN_BUDGET_K"
  --first_token_top_p "$FIRST_TOKEN_TOP_P" --first_token_temperature "$FIRST_TOKEN_TEMPERATURE"
  # Disable eval-time Z quantization until acceptance improves
  # --latent_quant_bits 6 --latent_quant_group_size 32 --latent_quant_scale_bits 16
  --min_new_tokens 3 --eos_ban_steps 6 --chunk_size "$CHUNK_SIZE"
  --hf_encoder_id "$HF_ENCODER_ID" --max_enc_tokens "$MAX_ENC_TOKENS"
  --llama_device_map "$LLAMA_DEVICE_MAP" --qwen_device_map "$QWEN_DEVICE_MAP"
  --llama_devices "$LLAMA_DEVICES" --qwen_devices "$QWEN_DEVICES" --gpu_mem_gib "$GPU_MEM_GIB"
)

run_eval() {
  local ckpt_path="$1"; local out_dir="$2"; local n_samples="$3"
  mkdir -p "$out_dir"
  local resolved; resolved=$(resolve_ckpt_for_eval "$ckpt_path")
  local args=( --ckpt "$resolved" --llama_id "$LLAMA_ID" --qwen_id "$QWEN_ID" \
               --samples "$n_samples" --out_dir "$out_dir" "${EVAL_ARGS_COMMON[@]}" )
  [[ ${SEQ_EVAL:-0} -eq 1 ]] && args+=(--sequential_eval)
  [[ ${FRESH_EVAL:-0} -eq 1 ]] && args+=(--fresh_eval)
  [[ $LOAD_4BIT -eq 1 ]] && args+=(--load_4bit)
  set -x
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u -m latentwire.eval "${args[@]}"
  set +x
}

print_metrics() {
  local dir="$1"
  if [[ -f "${dir}/metrics.json" ]]; then
    echo "✓ Metrics from: ${dir}/metrics.json"
    EV_DIR="${dir}" python - <<'PY'
import os, json
p = os.environ['EV_DIR']; m=json.load(open(os.path.join(p,"metrics.json")))
def g(*ks, d=None, default=None):
    cur=d or m
    for k in ks: cur = cur.get(k, {})
    return cur if cur else default
def f(x): 
    try: return f"{float(x):.3f}"
    except: return "-"
print(f"  Text F1:    Llama {f(g('text','llama','f1'))} | Qwen {f(g('text','qwen','f1'))}")
print(f"  Latent F1:  Llama {f(g('latent','llama','f1'))} | Qwen {f(g('latent','qwen','f1'))}")
print(f"  FirstTok@1: Llama {f(g('latent','llama','first_token_top1'))} | Qwen {f(g('latent','qwen','first_token_top1'))}")
print(f"  NLL/token:  Llama {f(g('latent','llama','nll_token'))} | Qwen {f(g('latent','qwen','nll_token'))}")
PY
  else
    echo "✗ No metrics.json in ${dir}"
  fi
}

print_top_predictions() {
  local dir="$1"
  local limit="${2:-5}"
  local preds="${dir}/predictions.jsonl"
  if [[ ! -f "$preds" ]]; then
    echo "✗ No predictions.jsonl in ${dir}"
    return
  fi
  echo "Top ${limit} latent predictions from $preds"
  PRED_PATH="$preds" LIMIT="$limit" python - <<'PY'
import json, os

path = os.environ["PRED_PATH"]
limit = int(os.environ.get("LIMIT", "5"))

rows = []
with open(path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= limit:
            break
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

for i, row in enumerate(rows, 1):
    llama = row.get("latent_pred_llama") or row.get("text_pred_llama")
    qwen = row.get("latent_pred_qwen") or row.get("text_pred_qwen")
    gold = row.get("gold")
    print(f"  {i}. Llama: {llama!s} | Qwen: {qwen!s} | Gold: {gold!s}")
PY
}

# ---- Pipeline ----
{
  print_header "Starting pipeline at $(date)"
  echo "Preset: $PRESET  |  GPUs: $CUDA_VISIBLE_DEVICES  |  Llama devices: $LLAMA_DEVICES  |  Qwen devices: $QWEN_DEVICES"
  echo "Run dir: $RUN_DIR"
  echo ""

  if [[ $DO_TRAIN -eq 1 ]]; then
    print_header "TRAIN + PER-EPOCH EVAL (All knobs enabled)"
    for epoch in $(seq 1 $EPOCHS); do
      print_header "EPOCH ${epoch}/${EPOCHS}"

      TRAIN_ARGS=( "${TRAIN_ARGS_COMMON[@]}" --save_dir "$CKPT_DIR" --save_every 1000000 --save_training_stats --debug --auto_resume )
      [[ $GRAD_CKPT -eq 1 ]] && TRAIN_ARGS+=(--grad_ckpt)
      [[ -n "${WARM_ANCHOR_TEXT}" ]] && TRAIN_ARGS+=(--warm_anchor_text "$WARM_ANCHOR_TEXT")
      [[ $SEQUENTIAL_MODELS -eq 1 ]] && TRAIN_ARGS+=(--sequential_models)
      [[ $LOAD_4BIT -eq 1 ]] && TRAIN_ARGS+=(--load_4bit)
      [[ "$ADAPTER_COLORIZE" == "1" ]] && TRAIN_ARGS+=(--adapter_colorize)

      # Pre-train eval if resuming
      if [[ -f "${CKPT_DIR}/state.pt" || -f "${CKPT_DIR}/encoder.pt" ]]; then
        echo "Running pre-train eval on existing checkpoint..."; run_eval "$CKPT_DIR" "${RUN_DIR}/eval_epoch${epoch}_pre" "$SMOKE_SAMPLES"
      fi

      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/train.py "${TRAIN_ARGS[@]}"
      epoch_ckpt="${RUN_DIR}/epoch${epoch}"; rm -rf "$epoch_ckpt"; cp -r "$CKPT_DIR" "$epoch_ckpt"
      run_eval "$epoch_ckpt" "${RUN_DIR}/eval_epoch${epoch}" "$SMOKE_SAMPLES"
      print_metrics "${RUN_DIR}/eval_epoch${epoch}"
      print_top_predictions "${RUN_DIR}/eval_epoch${epoch}" 5
    done
  fi

  if [[ $DO_FINAL_EVAL -eq 1 ]]; then
    print_header "FINAL FULL EVAL (best epoch by avg latent F1)"
    best_info=$(RUN_DIR="$RUN_DIR" EPOCHS="$EPOCHS" python - <<'PY'
import os, json
run_dir = os.environ["RUN_DIR"]
epochs = int(os.environ["EPOCHS"])
best_epoch = 0
best = 0.0
for epoch in range(1, epochs + 1):
    path = os.path.join(run_dir, f"eval_epoch{epoch}", "metrics.json")
    try:
        with open(path) as f:
            m = json.load(f)
    except Exception:
        continue
    ll = float(m.get("latent", {}).get("llama", {}).get("f1", 0) or 0)
    qw = float(m.get("latent", {}).get("qwen", {}).get("f1", 0) or 0)
    cur = (ll + qw) / 2.0
    if cur > best:
        best = cur
        best_epoch = epoch
print(best_epoch)
print(best)
PY
)
    best_epoch=$(echo "$best_info" | sed -n '1p')
    best_f1=$(echo "$best_info" | sed -n '2p')
    if [[ ${best_epoch:-0} -gt 0 ]]; then
      echo "Best epoch: $best_epoch (avg latent F1: $best_f1)"
      best_ckpt="${RUN_DIR}/epoch${best_epoch}"
      final_eval="${RUN_DIR}/eval_final_best"
      run_eval "$best_ckpt" "$final_eval" "$SAMPLES"
      print_metrics "$final_eval"
      print_top_predictions "$final_eval" 5
    else
      echo "WARNING: No epoch metrics found; skipping final full eval."
    fi
  fi

  print_header "PIPELINE SUMMARY"
  echo "Run: $RUN  |  Completed: $(date)"
  echo "All outputs under: $RUN_DIR"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "All output saved to: $LOG_FILE"
