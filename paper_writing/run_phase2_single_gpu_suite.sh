#!/usr/bin/env bash
set -euo pipefail

# Sequential sweep for Phase 2 configs (supports arbitrary GPU count via NUM_GPUS env var).
# Runs both prompt-supervised (soft-only) and answer-supervised (soft+text) variants.
# Usage: NUM_GPUS=4 bash run_phase2_single_gpu_suite.sh

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
    "label=prompt_softonly DIT_TEACHER=prompt PROMPT_MODE=soft_only EXTRA_ARGS='--prompt_alignment_weight 1.0 --dit_loss_weight 1.0'"
    "label=answer_softplus DIT_TEACHER=answer PROMPT_MODE=soft_plus_text EXTRA_ARGS='--prompt_alignment_weight 0.001 --dit_loss_weight 0.1'"
)

for entry in "${CONFIGS[@]}"; do
    eval "$entry"
    echo "=========================================="
    echo "Running Phase2 config: $label"
    echo "  DIT_TEACHER=$DIT_TEACHER"
    echo "  PROMPT_MODE=$PROMPT_MODE"
    echo "  NUM_GPUS=$NUM_GPUS"
    echo "=========================================="
    PYTHONPATH=. NUM_GPUS="$NUM_GPUS" DIT_TEACHER="$DIT_TEACHER" PROMPT_MODE="$PROMPT_MODE" EXTRA_ARGS="$EXTRA_ARGS" \
        bash paper_writing/run_phase2_swap.sh

    # Find the most recently created phase2_swap directory
    latest=$(ls -td paper_writing/runs/phase2_swap_* 2>/dev/null | head -n 1 || true)
    if [[ -z "$latest" ]]; then
        echo "ERROR: No phase2_swap_* directory found in paper_writing/runs/"
        exit 1
    fi

    run_name=$(basename "$latest")
    relabeled="paper_writing/runs/${label}_${run_name}"

    if ! mv "$latest" "$relabeled"; then
        echo "ERROR: Failed to rename $latest to $relabeled"
        exit 1
    fi

    preserved="paper_writing/preserved_data/${label}_${run_name}"
    mkdir -p "paper_writing/preserved_data"

    if ! cp -R "$relabeled" "$preserved"; then
        echo "ERROR: Failed to copy $relabeled to $preserved"
        exit 1
    fi

    echo "Saved run to: $relabeled"
    echo "Copied artifacts to: $preserved"
done
