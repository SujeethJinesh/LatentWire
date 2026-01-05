#!/bin/bash
#SBATCH --job-name=memory_safe_bridge
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/memory_safe_bridge_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/memory_safe_bridge_%j.err

# =============================================================================
# Memory-Safe Bridge Training with Dynamic Configuration
#
# This script automatically configures batch size and gradient accumulation
# based on model sizes to prevent OOM errors on 80GB H100 GPUs.
# =============================================================================
# Submit with: sbatch telepathy/submit_memory_safe_bridge.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "SLURM Job Information"
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

# Create runs directory if needed
mkdir -p runs figures

# Pull latest code
echo "Pulling latest code..."
git pull

# ============================================================
# Configuration: Modify these to test different model pairs
# ============================================================
SOURCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
TARGET_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
SOFT_TOKENS=128
DEPTH=4
MAX_LENGTH=1536
LR=2e-4
STEPS=3000
WARMUP_STEPS=100

echo ""
echo "=============================================================="
echo "Getting Memory-Safe Configuration"
echo "=============================================================="

# Get memory-safe configuration using Python script
CONFIG_OUTPUT=$(python3 -c "
from telepathy.memory_configs import get_memory_safe_config
import json

config = get_memory_safe_config(
    source_model='$SOURCE_MODEL',
    target_model='$TARGET_MODEL',
    max_length=$MAX_LENGTH,
    soft_tokens=$SOFT_TOKENS,
    depth=$DEPTH,
)

if 'error' in config:
    print(f'ERROR: {config[\"error\"]}')
    exit(1)

# Output configuration as JSON for parsing
print(json.dumps({
    'batch_size': config['batch_size'],
    'grad_accum': config['gradient_accumulation_steps'],
    'memory_gb': config['estimated_memory_gb']
}))
")

# Parse the configuration
BATCH_SIZE=$(echo $CONFIG_OUTPUT | python3 -c "import sys, json; print(json.loads(sys.stdin.read())['batch_size'])")
GRAD_ACCUM=$(echo $CONFIG_OUTPUT | python3 -c "import sys, json; print(json.loads(sys.stdin.read())['grad_accum'])")
MEMORY_GB=$(echo $CONFIG_OUTPUT | python3 -c "import sys, json; print(json.loads(sys.stdin.read())['memory_gb'])")

echo "Model Configuration:"
echo "  Source: $SOURCE_MODEL"
echo "  Target: $TARGET_MODEL"
echo ""
echo "Memory-Safe Training Parameters:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Estimated Memory: ${MEMORY_GB} GB / 80 GB"
echo ""
echo "Bridge Configuration:"
echo "  Soft Tokens: $SOFT_TOKENS"
echo "  Perceiver Depth: $DEPTH"
echo "  Max Sequence Length: $MAX_LENGTH"
echo ""
echo "Training Configuration:"
echo "  Learning Rate: $LR"
echo "  Training Steps: $STEPS"
echo "  Warmup Steps: $WARMUP_STEPS"

# ============================================================
# Run Training with Memory-Safe Configuration
# ============================================================
echo ""
echo "=============================================================="
echo "Starting Training with Memory-Safe Configuration"
echo "=============================================================="

python3 telepathy/train_telepathy_v15.py \
    --source_model "$SOURCE_MODEL" \
    --target_model "$TARGET_MODEL" \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --soft_tokens $SOFT_TOKENS \
    --depth $DEPTH \
    --lr $LR \
    --steps $STEPS \
    --warmup_steps $WARMUP_STEPS \
    --save_path "runs/memory_safe_bridge_${SLURM_JOB_ID}.pt" \
    --save_every 500 \
    --eval_every 500 \
    --bf16

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"

    # Run evaluation if model was saved
    if [ -f "runs/memory_safe_bridge_${SLURM_JOB_ID}.pt" ]; then
        echo ""
        echo "=============================================================="
        echo "Running Evaluation"
        echo "=============================================================="

        python3 telepathy/eval_telepathy_v15.py \
            --bridge_path "runs/memory_safe_bridge_${SLURM_JOB_ID}.pt" \
            --dataset gsm8k \
            --max_samples 100 \
            --output_file "runs/eval_memory_safe_${SLURM_JOB_ID}.json"
    fi
else
    echo "Training failed with error code $?"
fi

# Push results back to git
echo ""
echo "=============================================================="
echo "Pushing Results to Git"
echo "=============================================================="
git add -A
git commit -m "results: memory-safe bridge training (SLURM job $SLURM_JOB_ID)

Models: $SOURCE_MODEL -> $TARGET_MODEL
Batch Size: $BATCH_SIZE (auto-configured)
Grad Accum: $GRAD_ACCUM (auto-configured)
Memory Usage: ${MEMORY_GB} GB / 80 GB

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>" || true
git push || true

echo ""
echo "=============================================================="
echo "Job completed at $(date)"
echo "=============================================================="