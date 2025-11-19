#!/usr/bin/env bash
set -euo pipefail

# Sequential single-GPU sweep for Phase 2 configs.
# Runs both prompt-supervised (soft-only) and answer-supervised (soft+text) variants.

if command -v module >/dev/null 2>&1; then
    module purge
    module load gcc/13.1.0
    module load conda/24.3.0-0
    module load stockcuda/12.6.2
    module load cudnn/cuda12/9.3.0.75
fi

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=${NUM_GPUS:-1}

declare -a CONFIGS=(
    "label=prompt_softonly DIT_TEACHER=prompt PROMPT_MODE=soft_only"
    "label=answer_softplus DIT_TEACHER=answer PROMPT_MODE=soft_plus_text"
)

for entry in "${CONFIGS[@]}"; do
    eval "$entry"
    echo "=========================================="
    echo "Running Phase2 config: $label"
    echo "  DIT_TEACHER=$DIT_TEACHER"
    echo "  PROMPT_MODE=$PROMPT_MODE"
    echo "  NUM_GPUS=$NUM_GPUS"
    echo "=========================================="
    PYTHONPATH=. NUM_GPUS="$NUM_GPUS" DIT_TEACHER="$DIT_TEACHER" PROMPT_MODE="$PROMPT_MODE" \
        bash paper_writing/run_phase2_swap.sh

    latest=$(ls -td paper_writing/runs/phase2_swap_* | head -n 1)
    run_name=$(basename "$latest")
    relabeled="paper_writing/runs/${label}_${run_name}"
    mv "$latest" "$relabeled"
    preserved="paper_writing/preserved_data/${label}_${run_name}"
    cp -R "$relabeled" "$preserved"
    echo "Saved run to: $relabeled"
    echo "Copied artifacts to: $preserved"
done
