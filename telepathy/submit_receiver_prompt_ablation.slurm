#!/bin/bash
#SBATCH --job-name=receiver_prompt_ablation
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=8:00:00
#SBATCH --mem=128GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/receiver_prompt_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/receiver_prompt_%j.err

# =============================================================================
# Receiver-Side Prompt Engineering Ablation Experiment
# =============================================================================
# This experiment tests whether prepending reasoning-inducing prompts AFTER
# the soft tokens improves performance. The hypothesis is that soft tokens
# provide compressed context, and the reasoning prompt triggers the model
# to use that context more effectively.
#
# Prompts tested:
#   - none: Just "Answer: " (baseline)
#   - cot: "Let's think step by step. Answer: "
#   - cot_concise: "Think carefully. Answer: "
#   - direct: "Answer directly: Answer: "
#   - careful: "Be precise. Answer: "
#   - json: "Respond in JSON format. Answer: "
#
# Hypothesis:
#   - CoT prompts may help with reasoning tasks (SQuAD, HotpotQA)
#   - "Be precise" may help with factual accuracy
#   - Baseline (none) serves as control
#
# Expected Runtime: ~6-7 hours on single H100 GPU
#
# Submit with: sbatch telepathy/submit_receiver_prompt_ablation.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "RECEIVER-SIDE PROMPT ENGINEERING ABLATION"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
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
git pull || echo "Pull failed - continuing with existing code"

# GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# =============================================================================
# Define experiment configuration
# =============================================================================
OUTPUT_DIR="runs/receiver_prompt_ablation"
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/ablation_${TIMESTAMP}.log"

# Find the most recent checkpoint to use
# Priority: look for best checkpoint, then latest epoch
CKPT_DIR=""
for candidate in runs/*/best runs/*/epoch* ; do
    if [ -f "$candidate/state.pt" ] || [ -f "$candidate/config.json" ]; then
        CKPT_DIR="$candidate"
    fi
done

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found. Please train a model first."
    echo "Looking in: runs/*/best or runs/*/epoch*"
    exit 1
fi

echo "Using checkpoint: $CKPT_DIR"
echo ""

# Receiver prompt presets to test
PRESETS=("none" "cot" "cot_concise" "direct" "careful")

# Dataset and samples configuration
DATASET="squad"
SAMPLES=500  # Use subset for faster iteration
MAX_NEW_TOKENS=12

echo "=============================================================="
echo "EXPERIMENT CONFIGURATION"
echo "=============================================================="
echo "Checkpoint: $CKPT_DIR"
echo "Dataset: $DATASET"
echo "Samples: $SAMPLES"
echo "Presets: ${PRESETS[*]}"
echo "=============================================================="
echo ""

# =============================================================================
# Run ablation
# =============================================================================
{
    echo "Starting receiver prompt ablation at $(date)"
    echo ""

    RESULTS_JSON="$OUTPUT_DIR/ablation_results.json"
    echo "{\"experiment\": \"receiver_prompt_ablation\", \"timestamp\": \"$TIMESTAMP\", \"checkpoint\": \"$CKPT_DIR\", \"results\": {}" > "$RESULTS_JSON"

    for preset in "${PRESETS[@]}"; do
        echo "=============================================================="
        echo "Testing preset: $preset"
        echo "=============================================================="

        PRESET_OUTPUT="$OUTPUT_DIR/preset_${preset}"
        mkdir -p "$PRESET_OUTPUT"

        python latentwire/eval.py \
            --ckpt "$CKPT_DIR" \
            --dataset "$DATASET" \
            --samples "$SAMPLES" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --receiver_prompt_preset "$preset" \
            --models llama \
            --sequential_eval \
            --fresh_eval \
            --calibration embed_rms \
            --latent_anchor_mode text \
            --latent_anchor_text "Answer: " \
            --append_bos_after_prefix yes \
            --out_dir "$PRESET_OUTPUT" \
            2>&1 | tee "$PRESET_OUTPUT/eval_${preset}.log"

        # Extract key metrics from the output
        echo ""
        echo "Completed preset: $preset"
        echo ""
    done

    echo ""
    echo "=============================================================="
    echo "ABLATION COMPLETE"
    echo "=============================================================="
    echo "End time: $(date)"
    echo ""

    # Generate summary
    echo "Generating summary..."
    python -c "
import json
import os
import glob

output_dir = '$OUTPUT_DIR'
results = {
    'experiment': 'receiver_prompt_ablation',
    'checkpoint': '$CKPT_DIR',
    'dataset': '$DATASET',
    'samples': $SAMPLES,
    'presets': {}
}

for preset in ['none', 'cot', 'cot_concise', 'direct', 'careful']:
    preset_dir = os.path.join(output_dir, f'preset_{preset}')
    summary_files = glob.glob(os.path.join(preset_dir, '*summary.json'))

    if summary_files:
        with open(summary_files[0]) as f:
            data = json.load(f)

        llama = data.get('latent', {}).get('llama', {})
        results['presets'][preset] = {
            'em': llama.get('em', 0),
            'f1': llama.get('f1', 0),
            'nll_token': llama.get('nll_token', 0),
            'first_tok_acc': llama.get('first_tok_acc', {}),
        }
        print(f'{preset:15} EM={llama.get(\"em\", 0):.3f}  F1={llama.get(\"f1\", 0):.3f}')

with open(os.path.join(output_dir, 'ablation_summary.json'), 'w') as f:
    json.dump(results, f, indent=2)

print()
print('Summary saved to:', os.path.join(output_dir, 'ablation_summary.json'))
" 2>&1

} 2>&1 | tee "$LOG_FILE"

# =============================================================================
# Git commit and push
# =============================================================================
echo ""
echo "Saving results to git..."

git add "$OUTPUT_DIR"/ 2>/dev/null || true
git add runs/receiver_prompt_*.log runs/receiver_prompt_*.err 2>/dev/null || true

COMMIT_MSG="results: receiver-side prompt engineering ablation (SLURM job $SLURM_JOB_ID)

Experiment: Test reasoning-inducing prompts after soft tokens
Dataset: $DATASET ($SAMPLES samples)
Checkpoint: $CKPT_DIR
Presets tested: ${PRESETS[*]}

Hypothesis: Prepending 'Let's think step by step' or similar prompts
after soft tokens may help the receiver model reason more effectively
with the compressed context provided by the soft tokens.

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

if git commit -m "$COMMIT_MSG"; then
    echo "Created commit, pushing..."
    git push || echo "Push failed"
else
    echo "No changes to commit"
fi

echo ""
echo "=============================================================="
echo "JOB COMPLETED"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "To view results locally:"
echo "  git pull"
echo "  cat $OUTPUT_DIR/ablation_summary.json"
echo "=============================================================="
