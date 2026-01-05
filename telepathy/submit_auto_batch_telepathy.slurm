#!/bin/bash
#SBATCH --job-name=telepathy_auto_batch
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/telepathy_auto_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/telepathy_auto_%j.err

# =============================================================================
# Telepathy training with automatic batch size optimization
# Prevents OOM by calculating safe batch sizes for model pairs
# =============================================================================
# Submit with: sbatch telepathy/submit_auto_batch_telepathy.slurm
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

# Create directories
mkdir -p runs/telepathy figures

# Pull latest code
echo "Pulling latest code..."
git pull

# Configuration
SOURCE_MODEL="${SOURCE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LATENT_LEN="${LATENT_LEN:-128}"
D_Z="${D_Z:-768}"
DATASET="${DATASET:-arxiv}"
SAMPLES="${SAMPLES:-10000}"
EPOCHS="${EPOCHS:-10}"
OUTPUT_DIR="runs/telepathy/auto_batch_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "=============================================================="
echo "Training Configuration"
echo "=============================================================="
echo "Source Model: $SOURCE_MODEL"
echo "Target Model: $TARGET_MODEL"
echo "Latent Length: $LATENT_LEN"
echo "Latent Dimension: $D_Z"
echo "Dataset: $DATASET"
echo "Samples: $SAMPLES"
echo "Epochs: $EPOCHS"
echo "Output: $OUTPUT_DIR"
echo "=============================================================="
echo ""

# Get optimal batch size
echo "Calculating optimal batch size for model pair..."
BATCH_CONFIG=$(python telepathy/optimize_batch_size.py \
    --source-model "$SOURCE_MODEL" \
    --target-model "$TARGET_MODEL" \
    --gpu-memory 80.0 \
    --safety-margin 0.2 \
    --latent-len "$LATENT_LEN" \
    --d-z "$D_Z" \
    --desired-batch-size 64 \
    --json)

# Extract batch size and gradient accumulation
BATCH_SIZE=$(echo "$BATCH_CONFIG" | python -c "import sys, json; print(json.load(sys.stdin)['max_batch_size'])")
GRAD_ACCUM=$(echo "$BATCH_CONFIG" | python -c "import sys, json; print(json.load(sys.stdin)['gradient_accumulation_steps'])")
EFFECTIVE_BS=$(echo "$BATCH_CONFIG" | python -c "import sys, json; print(json.load(sys.stdin)['effective_batch_size'])")
EST_MEMORY=$(echo "$BATCH_CONFIG" | python -c "import sys, json; print(json.load(sys.stdin)['estimated_memory_usage_gb'])")

echo "Batch Size Optimization Results:"
echo "  Max Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM steps"
echo "  Effective Batch Size: $EFFECTIVE_BS"
echo "  Estimated Memory: ${EST_MEMORY} GB"
echo ""
echo "=============================================================="
echo "Starting Training"
echo "=============================================================="

# Create log file
LOG_FILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTPUT_DIR"

# Run training with optimal batch size
{
    python telepathy/train_telepathy.py \
        --source_model "$SOURCE_MODEL" \
        --target_model "$TARGET_MODEL" \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --latent_len "$LATENT_LEN" \
        --d_z "$D_Z" \
        --epochs "$EPOCHS" \
        --dataset "$DATASET" \
        --samples "$SAMPLES" \
        --output_dir "$OUTPUT_DIR" \
        --learning_rate 1e-4 \
        --warmup_steps 500 \
        --eval_every 1000 \
        --save_every 5000 \
        --log_every 100
} 2>&1 | tee "$LOG_FILE"

# Save batch optimization config
echo "$BATCH_CONFIG" > "$OUTPUT_DIR/batch_config.json"

# Run evaluation on best checkpoint
echo ""
echo "=============================================================="
echo "Running Evaluation"
echo "=============================================================="

if [ -f "$OUTPUT_DIR/best_checkpoint.pt" ]; then
    python telepathy/eval_telepathy.py \
        --checkpoint "$OUTPUT_DIR/best_checkpoint.pt" \
        --source_model "$SOURCE_MODEL" \
        --target_model "$TARGET_MODEL" \
        --dataset "$DATASET" \
        --samples 500 \
        --batch_size "$BATCH_SIZE" \
        --output_file "$OUTPUT_DIR/evaluation_results.json" \
        2>&1 | tee "$OUTPUT_DIR/evaluation.log"
else
    echo "No best checkpoint found, skipping evaluation"
fi

# Generate plots if available
if [ -f "$OUTPUT_DIR/training_metrics.json" ]; then
    echo ""
    echo "Generating training plots..."
    python telepathy/plot_training_metrics.py \
        --metrics-file "$OUTPUT_DIR/training_metrics.json" \
        --output-dir "$OUTPUT_DIR/figures" \
        2>&1 | tee "$OUTPUT_DIR/plotting.log"
fi

# Push results back to git
echo ""
echo "=============================================================="
echo "Pushing Results to Git"
echo "=============================================================="

git add -A
git commit -m "results: telepathy auto-batch training $SOURCE_MODEL -> $TARGET_MODEL (SLURM job $SLURM_JOB_ID)

Configuration:
- Source: $SOURCE_MODEL
- Target: $TARGET_MODEL
- Batch Size: $BATCH_SIZE (effective: $EFFECTIVE_BS)
- Gradient Accumulation: $GRAD_ACCUM
- Latent: ${LATENT_LEN}x${D_Z}
- Dataset: $DATASET ($SAMPLES samples)
- Estimated Memory: ${EST_MEMORY} GB

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true

git push || true

echo ""
echo "=============================================================="
echo "Job Summary"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Models: $SOURCE_MODEL -> $TARGET_MODEL"
echo "Batch Size Used: $BATCH_SIZE (effective: $EFFECTIVE_BS)"
echo "Memory Usage: ${EST_MEMORY} GB"
echo "Results saved to: $OUTPUT_DIR"
echo "Completion time: $(date)"
echo "=============================================================="