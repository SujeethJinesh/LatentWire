#!/bin/bash
#SBATCH --job-name=latentwire_experiment
#SBATCH --nodes=1
#SBATCH --gpus=2                                    # Using 2 GPUs for efficiency
#SBATCH --account=marlowe-m000066                   # CRITICAL: Correct account
#SBATCH --partition=preempt                         # CRITICAL: Correct partition
#SBATCH --time=11:30:00                             # 11.5 hours to allow for cleanup
#SBATCH --mem=80GB                                  # 80GB for 2 GPU setup
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/experiment_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/experiment_%j.err

# =============================================================================
# PRODUCTION LATENTWIRE EXPERIMENT - FULLY AUTOMATED SUBMISSION
# =============================================================================
# This script runs a complete LatentWire experiment with all environment fixes,
# proper logging, automatic git commits, and hands-off operation.
#
# Features:
# - Python environment conflict resolution (PYTHONNOUSERSITE=1)
# - Virtual environment setup with correct packages (datasets v3.x)
# - Comprehensive logging with tee capture
# - Automatic git commit/push of results
# - GPU utilization monitoring
# - Graceful error handling and recovery
# =============================================================================
# Submit with: sbatch finalization/run_experiment.sbatch
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory - MUST use /projects path
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

# =============================================================================
# JOB INFORMATION
# =============================================================================

echo "=============================================================="
echo "SLURM Job Information"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs allocated: $SLURM_GPUS"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "Memory limit: ${SLURM_MEM_PER_NODE}MB"
echo "Time limit: $SLURM_JOB_TIME_LIMIT"
echo "=============================================================="

# =============================================================================
# ENVIRONMENT SETUP - CRITICAL FOR PYTHON CONFLICTS
# =============================================================================

echo ""
echo "Setting up Python environment..."

# CRITICAL: Disable user site-packages to avoid dataclasses conflicts
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORK_DIR"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Optionally use MPI for multi-GPU (if needed)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available

# Create necessary directories
mkdir -p runs figures logs

# =============================================================================
# VIRTUAL ENVIRONMENT SETUP
# =============================================================================

echo ""
echo "Checking virtual environment..."

# Check if virtual environment exists and is healthy
VENV_EXISTS=false
if [ -d .venv ] && [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    # Test if we can import key packages
    if python3 -c "import torch, transformers, datasets" 2>/dev/null; then
        VENV_EXISTS=true
        echo "Using existing virtual environment"
    else
        echo "Virtual environment exists but is broken, recreating..."
        deactivate 2>/dev/null || true
        rm -rf .venv
    fi
fi

# Create virtual environment if needed
if [ "$VENV_EXISTS" = false ]; then
    echo "Creating fresh virtual environment..."
    python3 -m venv .venv --system-site-packages
    source .venv/bin/activate

    echo "Installing critical packages..."
    python3 -m pip install --upgrade pip --no-user

    # Install with specific versions to avoid conflicts
    python3 -m pip install torch --no-user 2>&1 | grep -v "already satisfied" || true
    python3 -m pip install transformers==4.45.2 --no-user 2>&1 | grep -v "already satisfied" || true
    python3 -m pip install "datasets>=3.0,<4.0" --no-user 2>&1 | grep -v "already satisfied" || true
    python3 -m pip install accelerate --no-user 2>&1 | grep -v "already satisfied" || true

    # Install other requirements
    if [ -f requirements.txt ]; then
        python3 -m pip install -r requirements.txt --no-user 2>&1 | grep -v "already satisfied" || true
    fi
fi

# =============================================================================
# VERIFY ENVIRONMENT
# =============================================================================

echo ""
echo "Verifying Python environment..."
python3 -c "
import sys
import os
print(f'Python: {sys.version}')
print(f'Executable: {sys.executable}')
print(f'PYTHONNOUSERSITE: {os.environ.get(\"PYTHONNOUSERSITE\", \"not set\")}')
print()

# Test critical imports
modules = ['dataclasses', 'torch', 'transformers', 'datasets', 'accelerate']
for module in modules:
    try:
        __import__(module)
        print(f'✓ {module}')
    except ImportError as e:
        print(f'✗ {module}: {e}')
        sys.exit(1)

# Check GPU availability
import torch
if torch.cuda.is_available():
    print(f'\\nGPUs available: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('WARNING: No GPUs detected!')
"

if [ $? -ne 0 ]; then
    echo "Environment verification failed! Exiting..."
    exit 1
fi

# =============================================================================
# GIT SETUP
# =============================================================================

echo ""
echo "Configuring Git..."

# Configure git identity if not set
git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job $SLURM_JOB_ID"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@marlowe.hpc"

# Pull latest code with conflict resolution
echo "Pulling latest code..."
if ! git pull; then
    echo "WARNING: git pull failed. Attempting to stash and retry..."
    git stash push -m "SLURM job $SLURM_JOB_ID auto-stash at $(date)"
    git pull --rebase=false || echo "Pull failed - continuing with existing code"
    git stash pop 2>/dev/null || true
fi

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Set experiment parameters (can be overridden via environment)
EXPERIMENT_NAME="${EXPERIMENT_NAME:-production_${SLURM_JOB_ID}}"
OUTPUT_DIR="runs/${EXPERIMENT_NAME}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/experiment_${TIMESTAMP}.log"

# Model configuration
SOURCE_MODEL="${SOURCE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# Training hyperparameters (optimized for 2 GPUs)
SAMPLES="${SAMPLES:-10000}"          # Moderate sample size
EPOCHS="${EPOCHS:-12}"               # Sufficient for convergence
BATCH_SIZE="${BATCH_SIZE:-8}"        # Batch per GPU (16 total)
LATENT_LEN="${LATENT_LEN:-32}"      # Standard compression
D_Z="${D_Z:-256}"                    # Latent dimension

# Evaluation parameters
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"

echo ""
echo "Experiment Configuration:"
echo "  Name: $EXPERIMENT_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  Models: $SOURCE_MODEL -> $TARGET_MODEL"
echo "  Training: ${SAMPLES} samples, ${EPOCHS} epochs, batch=${BATCH_SIZE}"
echo "  Latent: ${LATENT_LEN} tokens, dim=${D_Z}"
echo "  Evaluation: ${EVAL_SAMPLES} samples"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# GPU MONITORING (Background Process)
# =============================================================================

echo ""
echo "Starting GPU monitoring..."

# Start GPU monitoring in background
(
    while true; do
        nvidia-smi --query-gpu=timestamp,gpu_name,index,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader >> "${OUTPUT_DIR}/gpu_usage.csv" 2>/dev/null
        sleep 30
    done
) &
GPU_MONITOR_PID=$!

# Ensure we kill the monitor on exit
trap "kill $GPU_MONITOR_PID 2>/dev/null || true" EXIT

# =============================================================================
# MAIN EXPERIMENT EXECUTION
# =============================================================================

echo ""
echo "Starting main experiment at $(date)..."
echo "Log file: $LOG_FILE"
echo ""

# Run the experiment with comprehensive logging
{
    # Training phase
    echo "=============================================================="
    echo "PHASE 1: TRAINING"
    echo "=============================================================="

    python3 latentwire/train.py \
        --llama_id "$SOURCE_MODEL" \
        --qwen_id "$TARGET_MODEL" \
        --samples "$SAMPLES" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --latent_len "$LATENT_LEN" \
        --d_z "$D_Z" \
        --encoder_type byte \
        --dataset squad \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --output_dir "$OUTPUT_DIR/checkpoint" \
        --save_interval 500 \
        --eval_interval 100 \
        --gradient_checkpointing \
        2>&1

    TRAIN_EXIT=$?

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "ERROR: Training failed with exit code $TRAIN_EXIT"
        echo "Attempting to save partial results..."
    else
        echo "Training completed successfully"
    fi

    # Evaluation phase (even if training partially failed)
    echo ""
    echo "=============================================================="
    echo "PHASE 2: EVALUATION"
    echo "=============================================================="

    # Find the latest checkpoint
    LATEST_CKPT=$(ls -td ${OUTPUT_DIR}/checkpoint/epoch* 2>/dev/null | head -1)

    if [ -n "$LATEST_CKPT" ]; then
        echo "Evaluating checkpoint: $LATEST_CKPT"

        python3 latentwire/eval.py \
            --ckpt "$LATEST_CKPT" \
            --samples "$EVAL_SAMPLES" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --dataset squad \
            --sequential_eval \
            --fresh_eval \
            --calibration embed_rms \
            --latent_anchor_mode text \
            --latent_anchor_text "Answer: " \
            --append_bos_after_prefix yes \
            --output_file "${OUTPUT_DIR}/evaluation_results.json" \
            2>&1

        EVAL_EXIT=$?

        if [ $EVAL_EXIT -ne 0 ]; then
            echo "WARNING: Evaluation had issues (exit code $EVAL_EXIT)"
        else
            echo "Evaluation completed successfully"
        fi
    else
        echo "WARNING: No checkpoint found for evaluation"
    fi

    # Analysis phase
    echo ""
    echo "=============================================================="
    echo "PHASE 3: ANALYSIS"
    echo "=============================================================="

    # Run diagnostics if available
    if [ -f "${OUTPUT_DIR}/checkpoint/diagnostics.jsonl" ]; then
        echo "Analyzing training diagnostics..."
        python3 -c "
import json
import statistics

with open('${OUTPUT_DIR}/checkpoint/diagnostics.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]

if lines:
    last = lines[-1]
    print(f'Final epoch: {last.get(\"epoch\", \"?\")}')
    print(f'Final loss: {last.get(\"loss\", \"?\"):.4f}')

    losses = [l.get('loss', 0) for l in lines if 'loss' in l]
    if losses:
        print(f'Mean loss: {statistics.mean(losses):.4f}')
        print(f'Min loss: {min(losses):.4f}')
        print(f'Max loss: {max(losses):.4f}')
"
    fi

    # Summary statistics
    echo ""
    echo "=============================================================="
    echo "EXPERIMENT SUMMARY"
    echo "=============================================================="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Duration: $(($(date +%s) - $(date -d "$SLURM_JOB_START_TIME" +%s))) seconds"
    echo "Output directory: $OUTPUT_DIR"
    echo ""

    # Check for key output files
    for file in "checkpoint/model.pt" "evaluation_results.json" "gpu_usage.csv"; do
        if [ -f "${OUTPUT_DIR}/${file}" ]; then
            size=$(du -h "${OUTPUT_DIR}/${file}" | cut -f1)
            echo "✓ ${file} (${size})"
        else
            echo "✗ ${file} (missing)"
        fi
    done

} 2>&1 | tee "$LOG_FILE"

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true

# =============================================================================
# GIT COMMIT AND PUSH RESULTS
# =============================================================================

echo ""
echo "Saving results to git..."

# Add specific result files (not large checkpoints)
find "$OUTPUT_DIR" -type f \( \
    -name "*.log" -o \
    -name "*.err" -o \
    -name "*.json" -o \
    -name "*.jsonl" -o \
    -name "*.csv" -o \
    -name "*.txt" \
    \) -size -10M | xargs git add 2>/dev/null || true

# Also add any figures
git add figures/*.png figures/*.pdf 2>/dev/null || true

# Create detailed commit message
COMMIT_MSG="results: production experiment ${EXPERIMENT_NAME} (SLURM job $SLURM_JOB_ID)

Configuration:
- Models: ${SOURCE_MODEL} -> ${TARGET_MODEL}
- Dataset: squad
- Training: ${SAMPLES} samples, ${EPOCHS} epochs
- Latent: ${LATENT_LEN} tokens, dim=${D_Z}
- Node: $SLURMD_NODENAME
- GPUs: $SLURM_GPUS

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>"

# Commit and push with retry logic
if git diff --cached --quiet; then
    echo "No changes to commit"
else
    if git commit -m "$COMMIT_MSG"; then
        echo "Created commit, attempting to push..."

        # Push with retries
        RETRY_COUNT=0
        MAX_RETRIES=3

        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            if git push; then
                echo "Successfully pushed results to remote"
                break
            else
                RETRY_COUNT=$((RETRY_COUNT + 1))
                echo "Push attempt $RETRY_COUNT failed"

                if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                    sleep 5
                    # Try to resolve conflicts
                    git pull --rebase=false 2>/dev/null || true
                fi
            fi
        done

        if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
            echo "WARNING: Could not push after $MAX_RETRIES attempts"
            echo "Results saved locally in commit $(git rev-parse HEAD)"
        fi
    else
        echo "WARNING: Could not create commit"
    fi
fi

# =============================================================================
# FINAL CLEANUP
# =============================================================================

# Deactivate virtual environment
deactivate 2>/dev/null || true

# Final status
echo ""
echo "=============================================================="
echo "Job completed at $(date)"
echo "Total runtime: $(($(date +%s) - $(date -d "$SLURM_JOB_START_TIME" +%s 2>/dev/null || echo 0))) seconds"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================================="

# Exit with appropriate code
if [ "${TRAIN_EXIT:-0}" -ne 0 ] || [ "${EVAL_EXIT:-0}" -ne 0 ]; then
    exit 1
else
    exit 0
fi