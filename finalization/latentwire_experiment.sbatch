#!/bin/bash
#SBATCH --job-name=latentwire_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=08:00:00
#SBATCH --mem=32GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/experiment_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/experiment_%j.err
#SBATCH --signal=B:TERM@30

# =============================================================================
# LatentWire Production Experiment - Optimized Single GPU
# =============================================================================
# Runs complete experiment with 1 GPU in ~6 hours
# Fully automated - submit and disconnect
#
# Submit with: sbatch finalization/latentwire_experiment.sbatch
# Monitor with: squeue -u $USER
# Check logs: tail -f runs/experiment_*.log
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory (CRITICAL: must use /projects path)
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR" || exit 1

# Create output directory with job ID
OUTPUT_DIR="runs/job_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Master log file for all output
MASTER_LOG="$OUTPUT_DIR/master.log"

# Function to log with timestamp
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG"
}

# Cleanup function for graceful exit
cleanup() {
    log_msg "Cleanup triggered - saving state"

    # Try to push any results we have
    cd "$WORK_DIR"
    git add runs/"job_${SLURM_JOB_ID}"* 2>/dev/null || true
    git commit -m "partial: interrupted job $SLURM_JOB_ID" 2>/dev/null || true
    git push 2>/dev/null || true

    log_msg "Job terminated"
    exit 0
}

# Set trap for cleanup on exit
trap cleanup EXIT TERM INT

# =============================================================================
# JOB INFORMATION
# =============================================================================

log_msg "=============================================================="
log_msg "SLURM Job Information"
log_msg "=============================================================="
log_msg "Job ID: $SLURM_JOB_ID"
log_msg "Node: $SLURMD_NODENAME"
log_msg "GPUs: $CUDA_VISIBLE_DEVICES"
log_msg "Start time: $(date)"
log_msg "Working directory: $WORK_DIR"
log_msg "Output directory: $OUTPUT_DIR"
log_msg "=============================================================="

# =============================================================================
# GIT SETUP
# =============================================================================

log_msg "Setting up git..."

# Configure git if needed
git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job $SLURM_JOB_ID"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@hpc.cluster"

# Pull latest code
log_msg "Pulling latest code..."
if ! git pull; then
    log_msg "WARNING: git pull failed, attempting stash and retry..."
    git stash push -m "Job $SLURM_JOB_ID auto-stash"
    git pull || log_msg "Pull failed - continuing with existing code"
    git stash pop 2>/dev/null || true
fi

# =============================================================================
# ENVIRONMENT SETUP - Critical for avoiding dataclass errors
# =============================================================================

log_msg "Setting up Python environment..."

# CRITICAL: Ignore user site-packages to avoid conflicts
export PYTHONNOUSERSITE=1

# Set Python environment variables
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# HuggingFace cache location
export HF_HOME=/projects/m000066/sujinesh/.cache/huggingface
export TRANSFORMERS_CACHE=/projects/m000066/sujinesh/.cache/huggingface

# H100 optimizations
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOW_TF32=1

# Setup virtual environment
if [ -f ".venv/bin/activate" ]; then
    log_msg "Activating existing virtual environment..."
    source .venv/bin/activate
else
    log_msg "Creating new virtual environment..."
    python3 -m venv .venv --system-site-packages
    source .venv/bin/activate

    # Install critical packages
    log_msg "Installing packages..."
    pip install --upgrade pip --no-user
    pip install "datasets>=3.0.0,<4.0.0" --no-user  # Critical: use v3.x
    pip install -r requirements.txt --no-user
fi

# Verify critical imports work
log_msg "Verifying environment..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || exit 1
python3 -c "import datasets; print(f'Datasets: {datasets.__version__}')" || exit 1
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" || exit 1

# =============================================================================
# EXPERIMENT CONFIGURATION - Optimized for single GPU
# =============================================================================

# Training configuration (reduced for faster completion)
export DATASET="squad"
export SAMPLES="${SAMPLES:-2000}"      # Reduced from 5000
export EPOCHS="${EPOCHS:-6}"           # Reduced from 8
export BATCH_SIZE="${BATCH_SIZE:-2}"   # Reduced for 32GB memory
export LATENT_LEN="${LATENT_LEN:-32}"
export D_Z="${D_Z:-256}"

# Evaluation configuration
export EVAL_SAMPLES="${EVAL_SAMPLES:-1000}"
export SEEDS="${SEEDS:-42 123 456}"

# Use gradient accumulation for effective larger batch
GRAD_ACCUM=16  # Effective batch = 2 * 16 = 32

log_msg "Configuration:"
log_msg "  Dataset: $DATASET"
log_msg "  Samples: $SAMPLES"
log_msg "  Epochs: $EPOCHS"
log_msg "  Batch Size: $BATCH_SIZE (×$GRAD_ACCUM accumulation = $(($BATCH_SIZE * $GRAD_ACCUM)) effective)"
log_msg "  Latent: ${LATENT_LEN}×${D_Z}"
log_msg "  GPUs: 1 × H100"

# =============================================================================
# GPU MONITORING (background)
# =============================================================================

log_msg "Starting GPU monitoring..."
(
    while true; do
        nvidia-smi --query-gpu=timestamp,gpu_name,memory.used,memory.total,utilization.gpu \
            --format=csv >> "$OUTPUT_DIR/gpu_usage.csv" 2>/dev/null
        sleep 30
    done
) &
MONITOR_PID=$!

# =============================================================================
# TRAINING PHASE
# =============================================================================

log_msg ""
log_msg "=============================================================="
log_msg "PHASE 1: TRAINING"
log_msg "=============================================================="

CHECKPOINT_DIR="$OUTPUT_DIR/checkpoint"
TRAIN_LOG="$OUTPUT_DIR/training.log"

# Run training with optimizations for single GPU
{
    python3 latentwire/train.py \
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
        --dataset "$DATASET" \
        --samples "$SAMPLES" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum_steps "$GRAD_ACCUM" \
        --latent_len "$LATENT_LEN" \
        --d_z "$D_Z" \
        --save_dir "$CHECKPOINT_DIR" \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --mixed_precision bf16 \
        --grad_ckpt \
        --save_every 2 \
        --elastic_gpu
} 2>&1 | tee "$TRAIN_LOG"

TRAIN_STATUS=${PIPESTATUS[0]}

if [ $TRAIN_STATUS -eq 0 ]; then
    log_msg "Training completed successfully"
else
    log_msg "WARNING: Training failed with status $TRAIN_STATUS"
    log_msg "Continuing with evaluation if checkpoint exists..."
fi

# =============================================================================
# EVALUATION PHASE
# =============================================================================

log_msg ""
log_msg "=============================================================="
log_msg "PHASE 2: EVALUATION"
log_msg "=============================================================="

# Find the latest checkpoint
CHECKPOINT=$(find "$CHECKPOINT_DIR" -type d -name "epoch*" 2>/dev/null | sort -V | tail -1)

if [ -z "$CHECKPOINT" ]; then
    log_msg "ERROR: No checkpoint found, skipping evaluation"
else
    log_msg "Using checkpoint: $CHECKPOINT"

    RESULTS_DIR="$OUTPUT_DIR/results"
    mkdir -p "$RESULTS_DIR"

    # Run evaluation for each seed
    for SEED in $SEEDS; do
        log_msg "Evaluating with seed $SEED..."

        EVAL_LOG="$OUTPUT_DIR/eval_seed${SEED}.log"
        SEED_DIR="$RESULTS_DIR/seed${SEED}"
        mkdir -p "$SEED_DIR"

        {
            python3 latentwire/eval.py \
                --ckpt "$CHECKPOINT" \
                --dataset "$DATASET" \
                --samples "$EVAL_SAMPLES" \
                --seed "$SEED" \
                --sequential_eval \
                --fresh_eval \
                --calibration embed_rms \
                --latent_anchor_mode text \
                --latent_anchor_text "Answer: " \
                --append_bos_after_prefix yes \
                --max_new_tokens 12 \
                --out_dir "$SEED_DIR"
        } 2>&1 | tee "$EVAL_LOG"

        # Copy metrics to standard location
        if [ -f "$SEED_DIR/metrics.json" ]; then
            cp "$SEED_DIR/metrics.json" "$RESULTS_DIR/seed${SEED}.json"
            log_msg "Seed $SEED evaluation completed"
        else
            log_msg "WARNING: Seed $SEED evaluation failed"
        fi
    done
fi

# =============================================================================
# ANALYSIS PHASE
# =============================================================================

log_msg ""
log_msg "=============================================================="
log_msg "PHASE 3: ANALYSIS"
log_msg "=============================================================="

# Run any analysis scripts if they exist
if [ -f "scripts/analyze_all_results.py" ] && [ -d "$RESULTS_DIR" ]; then
    log_msg "Running results analysis..."
    python3 scripts/analyze_all_results.py \
        --results_dir "$RESULTS_DIR" \
        2>&1 | tee "$OUTPUT_DIR/analysis.log" || true
fi

# Kill GPU monitor
kill $MONITOR_PID 2>/dev/null || true

# =============================================================================
# SUMMARY
# =============================================================================

log_msg ""
log_msg "=============================================================="
log_msg "EXPERIMENT SUMMARY"
log_msg "=============================================================="
log_msg "Job ID: $SLURM_JOB_ID"
log_msg "Runtime: $((SECONDS / 3600))h $((SECONDS % 3600 / 60))m"
log_msg "Output directory: $OUTPUT_DIR"

if [ -f "$CHECKPOINT_DIR/diagnostics.jsonl" ]; then
    log_msg "Training diagnostics: $CHECKPOINT_DIR/diagnostics.jsonl"
fi

if [ -d "$RESULTS_DIR" ]; then
    log_msg "Evaluation results: $RESULTS_DIR/"

    # Show quick metrics if available
    for json in "$RESULTS_DIR"/seed*.json; do
        if [ -f "$json" ]; then
            SEED_NAME=$(basename "$json" .json)
            F1=$(python3 -c "import json; print(json.load(open('$json'))['metrics']['latent']['f1'])" 2>/dev/null || echo "N/A")
            log_msg "  $SEED_NAME F1: $F1"
        fi
    done
fi

# =============================================================================
# GIT COMMIT & PUSH
# =============================================================================

log_msg ""
log_msg "Saving results to git..."

cd "$WORK_DIR"

# Add all results (but not large checkpoints)
git add "$OUTPUT_DIR"/*.log "$OUTPUT_DIR"/*.csv "$OUTPUT_DIR"/*.json 2>/dev/null || true
git add "$OUTPUT_DIR"/results/*.json 2>/dev/null || true
git add "$OUTPUT_DIR"/checkpoint/diagnostics.jsonl 2>/dev/null || true

# Commit with proper message
COMMIT_MSG="results: latentwire experiment (SLURM job $SLURM_JOB_ID)

Configuration:
- Samples: $SAMPLES
- Epochs: $EPOCHS
- Batch Size: ${BATCH_SIZE}×${GRAD_ACCUM}=$((BATCH_SIZE * GRAD_ACCUM))
- GPUs: 1×H100
- Runtime: $((SECONDS / 3600))h $((SECONDS % 3600 / 60))m

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>"

git commit -m "$COMMIT_MSG" 2>/dev/null || log_msg "No changes to commit"

# Push with retries
for attempt in 1 2 3; do
    if git push; then
        log_msg "Results pushed to git successfully"
        break
    else
        log_msg "Push attempt $attempt failed, retrying..."
        sleep 5
    fi
done

log_msg ""
log_msg "=============================================================="
log_msg "Job completed successfully at $(date)"
log_msg "Results available at: $OUTPUT_DIR"
log_msg "=============================================================="

# Exit cleanly
exit 0