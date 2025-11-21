#!/usr/bin/env bash
set -euo pipefail

# Phase 2 hybrid conditioning diagnostic (Llama -> Mistral).
# Keeps literal prompts and injects DiT outputs as residual adapters instead of prepending soft tokens.
# Default: uses all visible GPUs (override with NUM_GPUS), Llama 3.1 8B as source, Mistral 7B as target.

if command -v module >/dev/null 2>&1; then
    module purge
    module load gcc/13.1.0
    module load conda/24.3.0-0
    module load stockcuda/12.6.2
    module load cudnn/cuda12/9.3.0.75
fi

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

detect_nproc() {
    if [[ -n "${NUM_GPUS:-}" ]]; then
        echo "$NUM_GPUS"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
        if [[ "$count" -gt 0 ]]; then
            echo "$count"
            return
        fi
    fi
    python - <<'PY' 2>/dev/null || echo 1
import torch
print(torch.cuda.device_count() or 1)
PY
}

NUM_GPUS=$(detect_nproc)
LABEL=${LABEL:-phase2_hybrid_adapter}
SOURCE_MODEL=${SOURCE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
TARGET_MODEL=${TARGET_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
DIT_TEACHER=${DIT_TEACHER:-prompt}
PROMPT_MODE=${PROMPT_MODE:-soft_plus_text}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
SOFT_TOKENS=${SOFT_TOKENS:-64}
ADAPTER_SCALE=${ADAPTER_SCALE:-1.0}
TOKEN_ALIGNMENT_WEIGHT=${TOKEN_ALIGNMENT_WEIGHT:-0.1}

EXTRA_ARGS=${EXTRA_ARGS:-"--soft_tokens ${SOFT_TOKENS} --soft_injection adapter --adapter_scale ${ADAPTER_SCALE} --prompt_alignment_weight 0.001 --dit_loss_weight 0.1 --token_alignment_weight ${TOKEN_ALIGNMENT_WEIGHT}"}

echo "=========================================="
echo "Hybrid Conditioning Diagnostic"
echo "------------------------------------------"
echo "Label:            $LABEL"
echo "GPUs:             $NUM_GPUS"
echo "Source model:     $SOURCE_MODEL"
echo "Target model:     $TARGET_MODEL"
echo "Teacher:          $DIT_TEACHER"
echo "Eval prompt mode: $PROMPT_MODE"
echo "Extra args:       $EXTRA_ARGS"
echo "=========================================="

PYTHONPATH=. NUM_GPUS="$NUM_GPUS" \
SOURCE_MODEL="$SOURCE_MODEL" TARGET_MODEL="$TARGET_MODEL" \
DIT_TEACHER="$DIT_TEACHER" PROMPT_MODE="$PROMPT_MODE" \
PER_DEVICE_BATCH="$PER_DEVICE_BATCH" EXTRA_ARGS="$EXTRA_ARGS" \
AUTO_SOFT_ONLY=0 \
bash paper_writing/run_phase2_swap.sh

latest=$(ls -td paper_writing/runs/phase2_swap_* 2>/dev/null | head -n 1 || true)
if [[ -z "$latest" ]]; then
    echo "ERROR: No phase2_swap_* directory found in paper_writing/runs/"
    exit 1
fi

run_name=$(basename "$latest")
labeled_dir="paper_writing/runs/${LABEL}_${run_name}"

if ! mv "$latest" "$labeled_dir"; then
    echo "ERROR: Failed to rename $latest to $labeled_dir"
    exit 1
fi

preserved="paper_writing/preserved_data/${LABEL}_${run_name}"
mkdir -p "paper_writing/preserved_data"

if ! cp -R "$labeled_dir" "$preserved"; then
    echo "ERROR: Failed to copy $labeled_dir to $preserved"
    exit 1
fi

echo "Saved run to: $labeled_dir"
echo "Copied artifacts to: $preserved"
