#!/bin/bash
#SBATCH --job-name=reasoning_hailmary
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=06:00:00
#SBATCH --mem=128GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/reasoning_hailmary_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/reasoning_hailmary_%j.err

# =============================================================================
# REASONING HAIL MARY EXPERIMENTS
# =============================================================================
# A consolidated suite of quick experiments to explore reasoning transfer.
#
# Experiments (estimated times):
#   1. Token Capacity Scaling (M=8,16,32,64)     ~45 min
#   2. Difficulty Transfer (easy->hard)          ~30 min
#   3. CoT Prompt Engineering                    ~45 min
#   4. Longer Training (500-5000 steps)          ~60 min
#   5. GSM8K Baseline                            ~45 min
#   6. Linear Probe Upper Bound                  ~15 min
#                                          Total: ~4 hours
#
# SUCCESS CRITERIA:
#   - Any experiment showing >50% accuracy on curriculum arithmetic
#   - Any experiment showing >10% on GSM8K
#   - Clear evidence that X helps reasoning (where X = tokens/CoT/training)
#
# Submit with: sbatch telepathy/experimental/submit_reasoning_hailmary.slurm
# Monitor with: squeue -u $USER && tail -f runs/reasoning_hailmary_*.log
# =============================================================================

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "REASONING HAIL MARY EXPERIMENTS"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo ""
echo "This experiment suite tests multiple approaches to reasoning:"
echo "  1. Token capacity scaling (M=8,16,32,64)"
echo "  2. Difficulty transfer (train easy, test hard)"
echo "  3. CoT prompt engineering (receiver-side)"
echo "  4. Longer training (500-5000 steps)"
echo "  5. GSM8K baseline (real math problems)"
echo "  6. Linear probe upper bound"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME="/projects/m000066/sujinesh/.cache/huggingface"
export TRANSFORMERS_CACHE="/projects/m000066/sujinesh/.cache/huggingface/transformers"

# Create directories
mkdir -p runs figures
mkdir -p /projects/m000066/sujinesh/.cache/huggingface

# Git setup
echo ""
echo "Git Configuration..."
git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job $SLURM_JOB_ID"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@hpc.cluster"

echo "Pulling latest code..."
if ! git pull; then
    echo "WARNING: git pull failed, continuing with existing code"
fi

# GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="runs/reasoning_hailmary_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# =============================================================================
# RUN EXPERIMENTS
# =============================================================================
echo "Starting reasoning hail mary experiments..."
echo ""

python -u telepathy/experimental/reasoning_hailmary.py \
    --output_dir "$OUTPUT_DIR" \
    --gpu 0 \
    --bf16 \
    --experiments 1 2 3 4 5 6 \
    2>&1 | tee "${OUTPUT_DIR}/experiment_${SLURM_JOB_ID}.log"

EXPERIMENT_STATUS=$?

echo ""
echo "=============================================================="
echo "Experiments completed with status: $EXPERIMENT_STATUS"
echo "=============================================================="

# =============================================================================
# RESULTS SUMMARY
# =============================================================================
echo ""
echo "Results Summary:"
echo "------------------------------------------------------------"

if [ -f "${OUTPUT_DIR}/all_results.json" ]; then
    python -c "
import json
import sys

try:
    with open('${OUTPUT_DIR}/all_results.json') as f:
        results = json.load(f)

    print()
    print('REASONING HAIL MARY RESULTS')
    print('=' * 60)
    print(f'Total runtime: {results.get(\"total_time_seconds\", 0)/3600:.1f} hours')
    print()

    for exp_name, exp_data in results.get('experiments', {}).items():
        print(f'{exp_name}:')

        if 'results' in exp_data:
            for key, val in exp_data['results'].items():
                if isinstance(val, dict):
                    if 'accuracy' in val:
                        print(f'  {key}: {val[\"accuracy\"]:.1f}%')
                    elif 'eval' in val and isinstance(val['eval'], dict):
                        print(f'  {key}: {val[\"eval\"].get(\"accuracy\", 0):.1f}%')
        elif 'eval' in exp_data and isinstance(exp_data['eval'], dict):
            print(f'  Accuracy: {exp_data[\"eval\"].get(\"accuracy\", 0):.1f}%')
        elif 'accuracy' in exp_data:
            print(f'  Accuracy: {exp_data[\"accuracy\"]:.1f}%')
        print()

    # Success criteria check
    print('SUCCESS CRITERIA CHECK:')
    print('-' * 40)

    any_success = False
    for exp_name, exp_data in results.get('experiments', {}).items():
        if 'results' in exp_data:
            for key, val in exp_data['results'].items():
                if isinstance(val, dict):
                    acc = val.get('accuracy', val.get('eval', {}).get('accuracy', 0) if isinstance(val.get('eval'), dict) else 0)
                    if 'gsm8k' in exp_name.lower() and acc > 10:
                        print(f'  SUCCESS: {exp_name} {key} = {acc:.1f}% (>10% on GSM8K)')
                        any_success = True
                    elif acc > 50:
                        print(f'  SUCCESS: {exp_name} {key} = {acc:.1f}% (>50% on arithmetic)')
                        any_success = True
        elif 'eval' in exp_data and isinstance(exp_data['eval'], dict):
            acc = exp_data['eval'].get('accuracy', 0)
            if 'gsm8k' in exp_name.lower() and acc > 10:
                print(f'  SUCCESS: {exp_name} = {acc:.1f}% (>10% on GSM8K)')
                any_success = True
            elif acc > 50:
                print(f'  SUCCESS: {exp_name} = {acc:.1f}% (>50% on arithmetic)')
                any_success = True

    if not any_success:
        print('  No experiments met success criteria.')
        print('  This is expected - reasoning transfer is hard!')

except Exception as e:
    print(f'Could not parse results: {e}')
    import traceback
    traceback.print_exc()
" 2>/dev/null || echo "Could not display results summary"
fi

# =============================================================================
# GIT COMMIT AND PUSH
# =============================================================================
echo ""
echo "Saving results to git..."

git add "$OUTPUT_DIR"/ 2>/dev/null || true
git add runs/reasoning_hailmary_*.log runs/reasoning_hailmary_*.err 2>/dev/null || true

COMMIT_MSG="results: reasoning hail mary experiments (SLURM job $SLURM_JOB_ID)

EXPERIMENT: Reasoning Hail Mary - Consolidated Suite
PURPOSE: Explore multiple approaches to enable reasoning via soft tokens

Experiments Run:
1. Token Capacity Scaling (M=8,16,32,64) on curriculum arithmetic
2. Difficulty Transfer (train easy, test hard)
3. CoT Prompt Engineering (receiver-side prefixes)
4. Longer Training (500-5000 steps)
5. GSM8K Baseline (real math reasoning)
6. Linear Probe Upper Bound

Exit status: $EXPERIMENT_STATUS
Output directory: $OUTPUT_DIR

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

if git commit -m "$COMMIT_MSG"; then
    echo "Created commit, pushing..."
    RETRY_COUNT=0
    while [ $RETRY_COUNT -lt 3 ]; do
        if git push; then
            echo "Successfully pushed to remote"
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            echo "Push attempt $RETRY_COUNT failed"
            [ $RETRY_COUNT -lt 3 ] && sleep 5 && git pull --rebase=false || true
        fi
    done
else
    echo "No changes to commit"
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "=============================================================="
echo "JOB COMPLETED"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds ($(echo "scale=1; $SECONDS/3600" | bc) hours)"
echo ""
echo "Key files:"
echo "  - ${OUTPUT_DIR}/all_results.json"
echo "  - ${OUTPUT_DIR}/experiment_${SLURM_JOB_ID}.log"
echo "  - ${OUTPUT_DIR}/exp*_bridge_*.pt (checkpoints)"
echo ""
echo "To view results locally:"
echo "  git pull"
echo "  cat ${OUTPUT_DIR}/all_results.json | python -m json.tool"
echo "=============================================================="
