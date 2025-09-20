#!/usr/bin/env bash
set -euo pipefail

# Hero run v2: STQueryEncoder (MiniLM) + adapters
# Key changes vs v1:
#  - Append BOS after latent prefix for both train & eval (stabilizes first token)
#  - Stronger first-token CE (λ_first=8.0)
#  - Sequential eval explicitly enabled with a value (--sequential_eval 1)
#  - Latent length M=48 (≥4× compression on SQuAD, alleviates bottleneck from M=32)
#  - Per-epoch spot-check eval in bf16 to rule out 4-bit quantization artifacts
#  - Deterministic first token at eval + extra metrics printing

DO_TRAIN=1
DO_FINAL_EVAL=1

DATASET="squad"
SAMPLES=1000
SMOKE_SAMPLES=400
MAX_NEW_TOKENS=16
FRESH_EVAL=1
LOAD_4BIT=1
CHUNK_SIZE=8
TOKEN_BUDGET_MODE="content_only"
TOKEN_BUDGET_K=48    # match M for fair token-budget control

# Decoding knobs
FIRST_TOKEN_TOP_P=1.0
FIRST_TOKEN_TEMPERATURE=0.0

# Latent anchoring & calibration
LATENT_ANCHOR_MODE="text"
LATENT_ANCHOR_TEXT="Answer: "      # trailing space intentional
APPEND_BOS_AFTER_PREFIX="yes"      # **change** vs v1
CALIBRATION="embed_rms"
PREFIX_GAIN=1.15

# Training
EPOCHS=12
BATCH_SIZE=16
GRAD_ACCUM_STEPS=32
TRAIN_SAMPLES=10000
ENCODER_TYPE="stq"
HF_ENCODER_ID="sentence-transformers/all-MiniLM-L6-v2"
MAX_ENC_TOKENS=1024

# Latent / adapter sizes
LATENT_LEN=48                 # **change** vs v1 (32 -> 48) still ≥ 4× compression
LATENT_SHARED_LEN=36
LATENT_PRIVATE_LEN=6
D_Z=256
BYTE_MAX=2048
LR=3e-5
SCALE_L2=0.05
ADAPTER_RMS_L2=0.0
MAX_GRAD_NORM=1.0
WARM_ANCHOR_TEXT="Answer: "
FIRST_TOKEN_CE=8.0            # **stronger** first-token CE
TRAIN_APPEND_BOS="yes"        # **align** train & eval BOS behavior
MAX_ANSWER_TOKENS=24
ADAPTER_HIDDEN_MULT=2
ADAPTER_COLORIZE=1
ADAPTER_METADATA=1
MANIFOLD_STAT_WEIGHT=0.001
STATE_KD_WEIGHT=0.0
STATE_KD_LAYERS="0,1,2"
K=8
K_CE_WEIGHT=1.2
KD_FIRST_K_WEIGHT=1.5
KD_TAU=1.25

LLAMA_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN_ID="Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES="0,1,2,3"
LLAMA_DEVICES="0,1"
QWEN_DEVICES="2,3"
GPU_MEM_GIB=78
LLAMA_DEVICE_MAP="auto"
QWEN_DEVICE_MAP="auto"

source .venv/bin/activate
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

TRAIN_ARGS_COMMON=(
  --dataset "$DATASET" --samples "$TRAIN_SAMPLES"
  --epochs 1 --batch_size "$BATCH_SIZE"
  --encoder_type "$ENCODER_TYPE"
  --hf_encoder_id "$HF_ENCODER_ID" --max_enc_tokens "$MAX_ENC_TOKENS"
  --latent_len "$LATENT_LEN" --latent_shared_len "$LATENT_SHARED_LEN" --latent_private_len "$LATENT_PRIVATE_LEN" --d_z "$D_Z"
  --qwen_id "$QWEN_ID" --llama_id "$LLAMA_ID"
  --lr "$LR" --scale_l2 "$SCALE_L2" --adapter_rms_l2 "$ADAPTER_RMS_L2" --max_grad_norm "$MAX_GRAD_NORM"
  --max_bytes "$BYTE_MAX" --max_answer_tokens "$MAX_ANSWER_TOKENS"
  --first_token_ce_weight "$FIRST_TOKEN_CE"
  --train_append_bos_after_prefix "$TRAIN_APPEND_BOS"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --adapter_hidden_mult "$ADAPTER_HIDDEN_MULT"
  --manifold_stat_weight "$MANIFOLD_STAT_WEIGHT"
  --state_kd_weight "$STATE_KD_WEIGHT" --state_kd_layers "$STATE_KD_LAYERS"
  --K "$K" --k_ce_weight "$K_CE_WEIGHT" --kd_first_k_weight "$KD_FIRST_K_WEIGHT" --kd_tau "$KD_TAU"
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
  --latent_quant_bits 6 --latent_quant_group_size 32 --latent_quant_scale_bits 16
  --min_new_tokens 3 --eos_ban_steps 6 --chunk_size "$CHUNK_SIZE"
  --sequential_eval
  --hf_encoder_id "$HF_ENCODER_ID" --max_enc_tokens "$MAX_ENC_TOKENS"
  --llama_device_map "$LLAMA_DEVICE_MAP" --qwen_device_map "$QWEN_DEVICE_MAP"
  --llama_devices "$LLAMA_DEVICES" --qwen_devices "$QWEN_DEVICES" --gpu_mem_gib "$GPU_MEM_GIB"
)

RUN="8B_hero_stq_m48_bos"
RUN_DIR="runs/${RUN}"
CKPT_DIR="${RUN_DIR}/ckpt"
mkdir -p "$RUN_DIR" "$CKPT_DIR"

ts="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RUN_DIR}/pipeline_${ts}.log"

print_header() { echo ""; echo "========================================="; echo "$1"; echo "========================================="; echo ""; }

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

print_metrics() {
  local dir="$1"
  if [[ -f "${dir}/metrics.json" ]]; then
    echo "✓ Metrics from: ${dir}/metrics.json"
    EV_DIR="${dir}" python - <<'PY'
import os, json, numpy as np
p = os.environ.get("EV_DIR")
with open(os.path.join(p,"metrics.json")) as f: m=json.load(f)

def g(*ks, d=None, default=None):
    cur=d or m
    for k in ks: cur = cur.get(k, {}) if isinstance(cur, dict) else {}
    return cur if cur else default

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
    echo "✗ Metrics missing in ${dir}"
  fi
}

run_eval() {
  local ckpt_path="$1"; local out_dir="$2"; local n_samples="$3"; local use_bf16="${4:-0}"
  mkdir -p "$out_dir"
  local resolved; resolved=$(resolve_ckpt_for_eval "$ckpt_path")
  EVAL_ARGS=( --ckpt "$resolved" --llama_id "$LLAMA_ID" --qwen_id "$QWEN_ID" --samples "$n_samples" --out_dir "$out_dir" "${EVAL_ARGS_COMMON[@]}" )
  if [[ $FRESH_EVAL -eq 1 ]]; then EVAL_ARGS+=(--fresh_eval); fi
  if [[ $LOAD_4BIT -eq 1 ]] && [[ "$use_bf16" -eq 0 ]]; then EVAL_ARGS+=(--load_4bit); fi
  echo "Evaluating: ${resolved} -> ${out_dir} (bf16=${use_bf16})"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/eval.py "${EVAL_ARGS[@]}"
}

{
  print_header "Starting pipeline at $(date)"
  if [[ $DO_TRAIN -eq 1 ]]; then
    print_header "TRAIN + PER-EPOCH EVAL"
    for epoch in $(seq 1 $EPOCHS); do
      print_header "EPOCH $epoch/$EPOCHS"
      TRAIN_ARGS=( "${TRAIN_ARGS_COMMON[@]}" --save_dir "$CKPT_DIR" --save_every 1000000 --save_training_stats --debug --auto_resume )
      TRAIN_ARGS+=(--grad_ckpt)
      if [[ -n "${WARM_ANCHOR_TEXT}" ]]; then TRAIN_ARGS+=(--warm_anchor_text "$WARM_ANCHOR_TEXT"); fi
      if [[ $LOAD_4BIT -eq 1 ]]; then TRAIN_ARGS+=(--load_4bit); fi
      if [[ "$ADAPTER_COLORIZE" == "1" ]]; then TRAIN_ARGS+=(--adapter_colorize); fi

      # Pre-train eval if resuming
      if [[ -f "${CKPT_DIR}/state.pt" || -f "${CKPT_DIR}/encoder.pt" ]]; then
        echo "Running pre-train eval on existing checkpoint..."
        run_eval "$CKPT_DIR" "${RUN_DIR}/eval_epoch${epoch}_pre" "$SMOKE_SAMPLES" 1
        print_metrics "${RUN_DIR}/eval_epoch${epoch}_pre"
      fi

      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/train.py "${TRAIN_ARGS[@]}"

      epoch_ckpt="${RUN_DIR}/epoch${epoch}"; rm -rf "$epoch_ckpt"; cp -r "$CKPT_DIR" "$epoch_ckpt"

      # 4-bit eval (fast)
      run_eval "$epoch_ckpt" "${RUN_DIR}/eval_epoch${epoch}" "$SMOKE_SAMPLES" 0
      print_metrics "${RUN_DIR}/eval_epoch${epoch}"

      # bf16 spot-check to rule out 4bit artifacts
      run_eval "$epoch_ckpt" "${RUN_DIR}/eval_epoch${epoch}_bf16" "$SMOKE_SAMPLES" 1
      print_metrics "${RUN_DIR}/eval_epoch${epoch}_bf16"
    done
  fi

  if [[ $DO_FINAL_EVAL -eq 1 ]]; then
    print_header "FINAL FULL EVAL ON LAST CKPT"
    run_eval "${RUN_DIR}/epoch${EPOCHS}" "${RUN_DIR}/eval_final" "$SAMPLES" 1
    print_metrics "${RUN_DIR}/eval_final"
  fi
} 2>&1 | tee "$LOG_FILE"

echo ""; echo "All output saved to: $LOG_FILE"
