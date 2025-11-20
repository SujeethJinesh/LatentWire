#!/usr/bin/env bash
set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load gcc/13.1.0
    module load conda/24.3.0-0
    module load stockcuda/12.6.2
    module load cudnn/cuda12/9.3.0.75
fi

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_GPUS=${NUM_GPUS:-$(python - <<'PY'
import torch
print(torch.cuda.device_count() or 1)
PY
)}

declare -a CONFIGS=(
    "label=phase1_96tok_breakthrough \
     SOURCE_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
     TARGET_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct \
     DIT_TEACHER=answer \
     PROMPT_MODE=soft_plus_text \
     EXTRA_ARGS='--soft_tokens 96 --prompt_alignment_weight 0.001 --dit_loss_weight 0.1'"

    "label=phase2_prompt_aligned \
     SOURCE_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct \
     TARGET_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
     DIT_TEACHER=prompt \
     PROMPT_MODE=soft_plus_text \
     EXTRA_ARGS='--soft_tokens 64 --token_alignment_weight 0.1 --prompt_alignment_weight 0.001'"
)

for entry in "${CONFIGS[@]}"; do
    eval "$entry"
    echo "=========================================="
    echo "Launching Breakthrough config: $label"
    echo "  Source: $SOURCE_MODEL"
    echo "  Target: $TARGET_MODEL"
    echo "  Teacher: $DIT_TEACHER"
    echo "  Prompt mode: $PROMPT_MODE"
    echo "  Extra args: $EXTRA_ARGS"
    echo "=========================================="

    PYTHONPATH=. NUM_GPUS="$NUM_GPUS" \
    SOURCE_MODEL="$SOURCE_MODEL" TARGET_MODEL="$TARGET_MODEL" \
    DIT_TEACHER="$DIT_TEACHER" PROMPT_MODE="$PROMPT_MODE" \
    EXTRA_ARGS="$EXTRA_ARGS" AUTO_SOFT_ONLY=0 \
    bash paper_writing/run_phase2_swap.sh

    latest=$(ls -td paper_writing/runs/phase2_swap_* | head -n 1)
    run_name=$(basename "$latest")
    labeled_dir="paper_writing/runs/${label}_${run_name}"
    mv "$latest" "$labeled_dir"
    preserved="paper_writing/preserved_data/${label}_${run_name}"
    cp -R "$labeled_dir" "$preserved"
    echo "Saved run to $labeled_dir"
    echo "Copied artifacts to $preserved"
    echo ""
done
