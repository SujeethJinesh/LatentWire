#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/prefetch_assets.sh <LLAMA_MODEL_ID> <QWEN_MODEL_ID>
# Example:
#   bash scripts/prefetch_assets.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 Qwen/Qwen2-0.5B-Instruct

LLM_A=${1:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
LLM_B=${2:-"Qwen/Qwen1.5-7B-Chat"}

export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

echo "Using caches:"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

source .venv/bin/activate 2>/dev/null || true

python - <<PY
import os
from huggingface_hub import snapshot_download
from datasets import load_dataset

llm_a = os.environ.get('LLM_A', '') or "${LLM_A}"
llm_b = os.environ.get('LLM_B', '') or "${LLM_B}"

print(f"Prefetching models: {llm_a} , {llm_b}")
for mid in [llm_a, llm_b]:
    try:
        snapshot_download(repo_id=mid, allow_patterns=["*.safetensors","*.json","*.model","*.txt","*.py","*.md"], local_files_only=False)
        print(f"  ✓ downloaded: {mid}")
    except Exception as e:
        print(f"  ! warning: failed to prefetch {mid}: {e}")

print("Prefetching HotpotQA (fullwiki, distractor) train/validation")
for cfg in ["fullwiki","distractor"]:
    for split in ["train","validation"]:
        try:
            ds = load_dataset("hotpot_qa", cfg, split=split)
            _ = ds.select(range(min(5, len(ds))))  # touch to force cache
            print(f"  ✓ cached hotpot_qa/{cfg}:{split} -> {len(ds)} rows")
        except Exception as e:
            print(f"  ! warning: failed hotpot_qa {cfg}:{split}: {e}")
print("Done.")
PY
