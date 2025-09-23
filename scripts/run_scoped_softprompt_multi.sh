#!/usr/bin/env bash
set -euo pipefail

# run_scoped_softprompt_multi.sh
# Streamlined pipeline: (A) train shared latent encoder + prefix adapters on clean
# hub checkpoints; (B) evaluate latent vs text baselines with deterministic chat
# templates. LoRA bring-up is intentionally omitted – focus is on latent-wire
# acceptance only.

: "${RUN_TAG:=scoped_softprompt_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="runs/${RUN_TAG}"
CKPT_DIR="${RUN_DIR}/ckpt"
CKPT_DIR_STAGEA="${CKPT_DIR}/stageA"
CKPT_DIR_STAGEB="${CKPT_DIR}/stageB"
mkdir -p "$RUN_DIR" "$CKPT_DIR_STAGEA" "$CKPT_DIR_STAGEB"
LOG="${RUN_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

# Base hub models
LLAMA_ID="${LLAMA_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-7B-Instruct}"

# Optional hero flag for extended runs
hero=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hero)
      hero=1
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

# Data / eval
DATASET="${DATASET:-squad}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"

if [[ $hero -eq 1 ]]; then
  TRAIN_SAMPLES=8000
  EPOCHS_B=8
  SAMPLES="${SAMPLES:-1000}"
else
  TRAIN_SAMPLES=640
  EPOCHS_B=4
  SAMPLES="${SAMPLES:-200}"
fi

# Latent budget and optimiser defaults
LATENT_LEN="${LATENT_LEN:-64}"
D_Z="${D_Z:-256}"
BATCH_SIZE_B="${BATCH_SIZE_B:-32}"
BATCH_SIZE_A="${BATCH_SIZE_A:-32}"

# Mandatory chat templating across the stack
export LW_APPLY_CHAT_TEMPLATE=1
export PYTHONPATH="${PYTHONPATH:-.}"

# GPU placement
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LLAMA_DEVICES="${LLAMA_DEVICES:-0,1}"
QWEN_DEVICES="${QWEN_DEVICES:-2,3}"
GPU_MEM_GIB="${GPU_MEM_GIB:-78}"

COMMON_DEVMAP=(
  --llama_device_map auto
  --qwen_device_map auto
  --llama_devices "$LLAMA_DEVICES"
  --qwen_devices "$QWEN_DEVICES"
  --gpu_mem_gib "$GPU_MEM_GIB"
)

# Ensure PEFT is available
python - <<'PY'
try:
    import peft, accelerate  # noqa
    print("✓ PEFT present")
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft>=0.12.0", "accelerate>=0.33.0"])
    print("✓ Installed PEFT + Accelerate")
print("✓ Environment ready")
PY

# CUDA preflight summary
echo -e "\n=== Preflight: CUDA / SLURM / bitsandbytes ===\n" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import os, torch, subprocess, sys
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "is_available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    subprocess.run(["nvidia-smi"], check=False)
except Exception as exc:
    print("nvidia-smi not runnable:", exc)
try:
    import bitsandbytes as bnb
    print("bitsandbytes:", getattr(bnb, "__version__", "?"))
except Exception as exc:
    print("bitsandbytes import failed:", exc)
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    print("FATAL: No CUDA devices visible. Aborting before training.")
    sys.exit(2)
PY

# --- Stage A: Latent encoder bring-up ---
echo -e "\n=== Stage A: Latent Fit ===\n" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/train.py \
  --dataset "$DATASET" --samples "$TRAIN_SAMPLES" --epochs "$EPOCHS_B" \
  --batch_size "$BATCH_SIZE_A" --grad_accum_steps 16 \
  --encoder_type stq --hf_encoder_id sentence-transformers/all-MiniLM-L6-v2 \
  --encoder_use_chat_template \
  --latent_len "$LATENT_LEN" --d_z "$D_Z" \
  --llama_id "$LLAMA_ID" --qwen_id "$QWEN_ID" \
  --save_dir "$CKPT_DIR_STAGEA" --auto_resume --save_training_stats \
  --train_append_bos_after_prefix yes \
  --use_chat_template \
  --warm_anchor_mode chat \
  --first_token_ce_weight 1.0 \
  --K 4 \
  --k_ce_weight 0.5 --kd_first_k_weight 0.0 --state_kd_weight 0.0 \
  --max_grad_norm 1.0 \
  --adapter_hidden_mult 2 \
  --manifold_stat_weight 0.0 \
  --max_answer_tokens 24 \
  "${COMMON_DEVMAP[@]}" 2>&1 | tee -a "$LOG"

# --- Stage B: Prefix-training only ---
echo -e "\n=== Stage B: Prefix Training ===\n" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/train.py \
  --dataset "$DATASET" --samples "$TRAIN_SAMPLES" --epochs "$EPOCHS_B" \
  --batch_size "$BATCH_SIZE_B" --grad_accum_steps 16 \
  --encoder_type stq --hf_encoder_id sentence-transformers/all-MiniLM-L6-v2 \
  --encoder_use_chat_template \
  --latent_len "$LATENT_LEN" --d_z "$D_Z" \
  --llama_id "$LLAMA_ID" --qwen_id "$QWEN_ID" \
  --use_prefix --prefix_tokens 24 --prefix_projection --peft_prefix_all_layers yes \
  --save_dir "$CKPT_DIR_STAGEB" --auto_resume --resume_from "$CKPT_DIR_STAGEA" --no_load_optimizer --save_training_stats \
  --train_append_bos_after_prefix yes \
  --freeze_encoder \
  --use_chat_template \
  --warm_anchor_mode chat \
  --first_token_ce_weight 2.0 \
  --K 4 \
  --k_ce_weight 0.5 --kd_first_k_weight 0.0 --state_kd_weight 0.0 \
  --max_grad_norm 1.0 \
  --adapter_hidden_mult 2 \
  --manifold_stat_weight 0.0 \
  --max_answer_tokens 24 \
  "${COMMON_DEVMAP[@]}" 2>&1 | tee -a "$LOG"

# --- Stage C: Evaluation on clean hubs + learned prefixes ---
echo -e "\n=== Stage C: Eval ===\n" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/eval.py \
  --ckpt "$CKPT_DIR_STAGEB" --llama_id "$LLAMA_ID" --qwen_id "$QWEN_ID" \
  --samples "$SAMPLES" --dataset "$DATASET" \
  --fresh_eval --max_new_tokens "$MAX_NEW_TOKENS" \
  --chunk_size "$CHUNK_SIZE" \
  --latent_anchor_mode chat --append_bos_after_prefix yes \
  --use_chat_template yes \
  --first_token_top_p 1.0 --first_token_temperature 0.0 \
  --token_budget_mode content_only --token_budget_k "$LATENT_LEN" \
  "${COMMON_DEVMAP[@]}" 2>&1 | tee -a "$LOG"

echo -e "\n✓ Completed. Logs at $LOG\n"
