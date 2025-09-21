#!/usr/bin/env bash
set -euo pipefail

# run_scoped_softprompt_multi.sh
# One‑button runner to stage (A) tiny LoRA, merge; (B) deep Prefix‑Tuning;
# (C) evaluate latent soft‑prompt + text baselines with proper chat templates.
#
# Assumes your package entry points still live at latentwire/train.py and latentwire/eval.py.
# This script *adds no new Python entry point*; it only toggles features via flags/env.

: "${RUN_TAG:=scoped_softprompt_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="runs/${RUN_TAG}"
CKPT_DIR="${RUN_DIR}/ckpt"
mkdir -p "$RUN_DIR" "$CKPT_DIR"
LOG="${RUN_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

# Models
LLAMA_ID="${LLAMA_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-7B-Instruct}"

# Data/Eval
DATASET="${DATASET:-squad}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-320}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-200}"
SAMPLES="${SAMPLES:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

# Latent
LATENT_LEN="${LATENT_LEN:-64}"           # Relax compression for acceptance
D_Z="${D_Z:-256}"

# Tuning Params
BATCH_SIZE_A="${BATCH_SIZE_A:-8}"
BATCH_SIZE_B="${BATCH_SIZE_B:-4}"

# Chat templating (non‑negotiable)
export LW_APPLY_CHAT_TEMPLATE=1

# Ensure the latentwire package is discoverable when running via python -m / latentwire/*.py
export PYTHONPATH="${PYTHONPATH:-.}"

# GPU maps
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LLAMA_DEVICES="${LLAMA_DEVICES:-0,1}"
QWEN_DEVICES="${QWEN_DEVICES:-2,3}"
GPU_MEM_GIB="${GPU_MEM_GIB:-78}"

# Common flags
COMMON_DEVMAP=(
  --llama_device_map auto
  --qwen_device_map auto
  --llama_devices "$LLAMA_DEVICES"
  --qwen_devices "$QWEN_DEVICES"
  --gpu_mem_gib "$GPU_MEM_GIB"
)

# Install peft if missing (allows Slurm nodes without preinstall)
python - <<'PY'
try:
    import peft, accelerate  # noqa
    print("✓ PEFT present")
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable,"-m","pip","install","-q","peft>=0.12.0","accelerate>=0.33.0"])
print("✓ Environment ready")
PY

# Hardened CUDA preflight
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

# Stage A: tiny LoRA (first ~12 layers, attn+mlp)
echo -e "\n=== Stage A: LoRA (tiny) ===\n" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/train.py \
  --dataset "$DATASET" --samples "$TRAIN_SAMPLES" --epochs 1 \
  --batch_size "$BATCH_SIZE_A" --grad_accum_steps 16 --grad_ckpt \
  --encoder_type stq --hf_encoder_id sentence-transformers/all-MiniLM-L6-v2 \
  --latent_len "$LATENT_LEN" --d_z "$D_Z" \
  --llama_id "$LLAMA_ID" --qwen_id "$QWEN_ID" \
  --use_lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules attn_mlp_firstN:12 \
  --save_dir "$CKPT_DIR" --auto_resume --save_training_stats \
  --train_append_bos_after_prefix yes \
  --first_token_ce_weight 4.0 \
  --adapter_hidden_mult 2 \
  --manifold_stat_weight 0.001 \
  --max_answer_tokens 24 \
  "${COMMON_DEVMAP[@]}" 2>&1 | tee -a "$LOG"

# Merge LoRA into base weights for stability of Stage B (Prefix)
echo -e "\n=== Stage A -> merge LoRA ===\n" | tee -a "$LOG"
RUN_TAG="$RUN_TAG" LLAMA_ID="$LLAMA_ID" QWEN_ID="$QWEN_ID" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python - <<'PY'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

run_tag = os.environ["RUN_TAG"]
ckpt = os.path.join("runs", run_tag, "ckpt")
llama_id = os.environ.get("LLAMA_ID")
qwen_id = os.environ.get("QWEN_ID")

def merge_adapter(base_id: str, adapter_dir: str, out_dir: str) -> None:
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Missing adapter directory: {adapter_dir}")
    base = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto", torch_dtype="auto")
    peft = PeftModel.from_pretrained(base, adapter_dir)
    peft.merge_and_unload(safe_merge=True)
    peft.save_pretrained(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    tokenizer.save_pretrained(out_dir)

merge_adapter(llama_id, os.path.join(ckpt, "lora_llama"), os.path.join(ckpt, "merged_llama"))
merge_adapter(qwen_id, os.path.join(ckpt, "lora_qwen"), os.path.join(ckpt, "merged_qwen"))
print(f"✓ Merged LoRA adapters into {ckpt}/merged_*")
PY

# Stage B: Deep Prefix (Prefix‑Tuning) with merged bases
echo -e "\n=== Stage B: Deep Prefix ===\n" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/train.py \
  --dataset "$DATASET" --samples "$TRAIN_SAMPLES" --epochs 1 \
  --batch_size "$BATCH_SIZE_B" --grad_accum_steps 16 \
  --encoder_type stq --hf_encoder_id sentence-transformers/all-MiniLM-L6-v2 \
  --latent_len "$LATENT_LEN" --d_z "$D_Z" \
  --llama_id "${CKPT_DIR}/merged_llama" \
  --qwen_id "${CKPT_DIR}/merged_qwen" \
  --use_prefix --prefix_tokens 24 --prefix_projection --peft_prefix_all_layers yes \
  --save_dir "$CKPT_DIR" --resume_from "$CKPT_DIR" --no_load_optimizer --save_training_stats \
  --train_append_bos_after_prefix yes \
  --first_token_ce_weight 2.0 \
  --k_ce_weight 0.0 --kd_first_k_weight 0.0 --state_kd_weight 0.0 \
  --adapter_hidden_mult 2 \
  --manifold_stat_weight 0.001 \
  --max_answer_tokens 24 \
  "${COMMON_DEVMAP[@]}" 2>&1 | tee -a "$LOG"

# Stage C: Eval (text vs latent, sequential_eval, proper chat templates)
echo -e "\n=== Stage C: Eval ===\n" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u latentwire/eval.py \
  --ckpt "$CKPT_DIR" --samples "$SAMPLES" --dataset "$DATASET" \
  --latent_quant_bits 6 --latent_quant_group_size 32 --latent_quant_scale_bits 16 \
  --sequential_eval --max_new_tokens "$MAX_NEW_TOKENS" \
  --latent_anchor_mode text --latent_anchor_text "Answer: " --append_bos_after_prefix yes \
  --use_chat_template yes \
  --first_token_top_p 1.0 --first_token_temperature 0.0 \
  --token_budget_mode content_only --token_budget_k "$LATENT_LEN" \
  "${COMMON_DEVMAP[@]}" 2>&1 | tee -a "$LOG"

echo -e "\n✓ Completed. Logs at $LOG\n"
