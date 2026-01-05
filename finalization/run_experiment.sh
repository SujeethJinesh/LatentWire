#!/usr/bin/env bash
# =============================================================================
# MAIN ORCHESTRATOR SCRIPT FOR LATENTWIRE EXPERIMENTS
# =============================================================================
# This is THE script for running experiments on HPC with full resilience:
# - Handles preemption and automatic resumption
# - Works with elastic GPU allocation (1-4 GPUs)
# - Maximizes GPU utilization
# - Comprehensive logging and state management
# - Can be run with srun or sbatch
#
# Usage:
#   Direct execution: bash finalization/run_experiment.sh
#   Interactive SLURM: srun --gpus=4 --account=marlowe-m000066 --partition=preempt bash finalization/run_experiment.sh
#   Batch submission: sbatch finalization/submit_experiment.slurm
#
# Environment Variables:
#   EXP_NAME: Experiment name (default: auto-generated)
#   DATASET: Dataset to use (default: squad)
#   SAMPLES: Number of samples (default: 10000)
#   EPOCHS: Number of epochs (default: 3)
#   CHECKPOINT_DIR: Where to save checkpoints (default: runs/{EXP_NAME})
#   RESUME: Set to "yes" to resume from checkpoint
#   MAX_RETRIES: Maximum number of preemption retries (default: 10)
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

# Experiment configuration
EXP_NAME="${EXP_NAME:-exp_$(date +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-10000}"
EPOCHS="${EPOCHS:-3}"
LATENT_LEN="${LATENT_LEN:-32}"
D_Z="${D_Z:-256}"
LR="${LR:-1e-3}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# Hardware configuration
TARGET_GPU_UTIL="${TARGET_GPU_UTIL:-0.75}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

# Paths
WORK_DIR="${WORK_DIR:-/projects/m000066/sujinesh/LatentWire}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$WORK_DIR/runs/$EXP_NAME}"
STATE_FILE="$CHECKPOINT_DIR/.orchestrator_state"
LOG_DIR="$CHECKPOINT_DIR/logs"

# Resumption configuration
RESUME="${RESUME:-auto}"  # auto, yes, no
MAX_RETRIES="${MAX_RETRIES:-10}"
RETRY_COUNT_FILE="$CHECKPOINT_DIR/.retry_count"

# SLURM detection
IS_SLURM=0
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    IS_SLURM=1
    echo "Running under SLURM (Job ID: $SLURM_JOB_ID)"
fi

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Critical environment variables for optimal performance
export PYTHONUNBUFFERED=1  # Immediate output flushing
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # Memory fragmentation prevention
export CUDA_LAUNCH_BLOCKING=0  # Async kernel execution
export NCCL_DEBUG=WARN  # NCCL communication debugging
export TOKENIZERS_PARALLELISM=true  # Faster tokenization
export OMP_NUM_THREADS=8  # OpenMP threads
export MKL_NUM_THREADS=8  # MKL threads

# Path setup
cd "$WORK_DIR" || { echo "Failed to change to work directory: $WORK_DIR"; exit 1; }
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

# =============================================================================
# GPU DETECTION AND CONFIGURATION
# =============================================================================

detect_gpu_configuration() {
    local gpu_count=0
    local gpu_memory_gb=0
    local gpu_type="unknown"

    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 || echo 0)

        if [[ $gpu_count -gt 0 ]]; then
            # Get memory of first GPU (assumes homogeneous GPUs)
            gpu_memory_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
            gpu_memory_gb=$((gpu_memory_mb / 1024))

            # Get GPU type
            gpu_type=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
        fi
    fi

    echo "$gpu_count:$gpu_memory_gb:$gpu_type"
}

calculate_optimal_batch_size() {
    local gpu_count=$1
    local gpu_memory_gb=$2
    local model_size_gb=14  # Llama-8B ≈ 14GB

    # CRITICAL FIX: Account for optimizer state memory
    # Adam/AdamW optimizer needs 2x model params for momentum and variance
    local optimizer_state_gb=$((model_size_gb * 2))  # ~28GB for Adam
    local gradients_gb=$model_size_gb  # ~14GB for gradients
    local cuda_overhead_gb=2  # CUDA context and workspace
    local safety_margin_gb=2  # Additional safety buffer

    # Total fixed memory requirement
    local fixed_memory_gb=$((model_size_gb + optimizer_state_gb + gradients_gb + cuda_overhead_gb + safety_margin_gb))
    # Total: 14 + 28 + 14 + 2 + 2 = 60GB fixed overhead

    # Available memory for batch processing
    local available_gb=$((gpu_memory_gb - fixed_memory_gb))

    # If available memory is negative or very small, use minimum batch size
    if [[ $available_gb -lt 1 ]]; then
        log_message "WARN" "Very limited memory available. Using minimum batch size."
        batch_per_gpu=1
    else
        # Estimate batch size based on available memory
        # ~0.5-1GB per batch item for typical sequence lengths with activations
        batch_per_gpu=$((available_gb))  # Conservative: 1GB per sample

        # For very high memory GPUs, don't go too crazy
        if [[ $batch_per_gpu -gt 64 ]]; then
            batch_per_gpu=64
        fi
    fi

    # Ensure minimum batch size
    if [[ $batch_per_gpu -lt 1 ]]; then
        batch_per_gpu=1
    fi

    # Calculate gradient accumulation to reach effective batch size
    local effective_batch=$BASE_BATCH_SIZE
    local grad_accum=1

    if [[ $((batch_per_gpu * gpu_count)) -lt $effective_batch ]]; then
        grad_accum=$((effective_batch / (batch_per_gpu * gpu_count)))
        if [[ $grad_accum -eq 0 ]]; then
            grad_accum=1
        fi
    fi

    # Log memory calculation details
    log_message "INFO" "Memory calculation: GPU=${gpu_memory_gb}GB, Fixed=${fixed_memory_gb}GB, Available=${available_gb}GB"
    log_message "INFO" "Breakdown: Model=${model_size_gb}GB, Optimizer=${optimizer_state_gb}GB, Gradients=${gradients_gb}GB"

    echo "$batch_per_gpu:$grad_accum:$effective_batch"
}

# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

find_latest_checkpoint() {
    local checkpoint_pattern="$CHECKPOINT_DIR/epoch*"
    local latest=""
    local latest_epoch=-1

    for ckpt in $checkpoint_pattern; do
        if [[ -d "$ckpt" ]]; then
            # Extract epoch number from directory name
            epoch_num=$(basename "$ckpt" | sed 's/epoch//')
            if [[ $epoch_num =~ ^[0-9]+$ ]] && [[ $epoch_num -gt $latest_epoch ]]; then
                latest_epoch=$epoch_num
                latest="$ckpt"
            fi
        fi
    done

    echo "$latest"
}

save_orchestrator_state() {
    local status=$1
    local epoch=$2
    local checkpoint=$3

    cat > "$STATE_FILE" << EOF
{
    "experiment": "$EXP_NAME",
    "status": "$status",
    "epoch": $epoch,
    "checkpoint": "$checkpoint",
    "timestamp": "$(date -Iseconds)",
    "slurm_job_id": "${SLURM_JOB_ID:-}",
    "retry_count": $(cat "$RETRY_COUNT_FILE" 2>/dev/null || echo 0)
}
EOF
}

load_orchestrator_state() {
    if [[ -f "$STATE_FILE" ]]; then
        cat "$STATE_FILE"
    else
        echo "{}"
    fi
}

# =============================================================================
# LOGGING INFRASTRUCTURE
# =============================================================================

setup_logging() {
    mkdir -p "$LOG_DIR"

    # Create timestamped log file
    local timestamp=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/train_${timestamp}.log"

    # Also create a symlink to latest log
    ln -sf "train_${timestamp}.log" "$LOG_DIR/latest.log"

    echo "$LOG_FILE"
}

log_message() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] $message" | tee -a "${LOG_FILE:-/dev/stdout}"

    # Also log to SLURM output if running under SLURM
    if [[ $IS_SLURM -eq 1 ]]; then
        echo "[$timestamp] [$level] $message"
    fi
}

# =============================================================================
# TRAINING LAUNCHER
# =============================================================================

launch_training() {
    local resume_checkpoint=$1
    local batch_size=$2
    local grad_accum=$3
    local gpu_count=$4

    log_message "INFO" "Launching training with configuration:"
    log_message "INFO" "  GPUs: $gpu_count"
    log_message "INFO" "  Batch size per GPU: $batch_size"
    log_message "INFO" "  Gradient accumulation: $grad_accum"
    log_message "INFO" "  Effective batch size: $((batch_size * gpu_count * grad_accum))"

    # Build training command
    local train_cmd="python latentwire/train.py"
    train_cmd="$train_cmd --llama_id 'meta-llama/Meta-Llama-3.1-8B-Instruct'"
    train_cmd="$train_cmd --qwen_id 'Qwen/Qwen2.5-7B-Instruct'"
    train_cmd="$train_cmd --dataset $DATASET"
    train_cmd="$train_cmd --samples $SAMPLES"
    train_cmd="$train_cmd --epochs $EPOCHS"
    train_cmd="$train_cmd --batch_size $batch_size"
    train_cmd="$train_cmd --gradient_accumulation_steps $grad_accum"
    train_cmd="$train_cmd --latent_len $LATENT_LEN"
    train_cmd="$train_cmd --d_z $D_Z"
    train_cmd="$train_cmd --lr $LR"
    train_cmd="$train_cmd --max_grad_norm $MAX_GRAD_NORM"
    train_cmd="$train_cmd --output_dir $CHECKPOINT_DIR"
    train_cmd="$train_cmd --encoder_type byte"
    train_cmd="$train_cmd --sequential_models"
    train_cmd="$train_cmd --warm_anchor_text 'Answer: '"
    train_cmd="$train_cmd --first_token_ce_weight 0.5"

    # Add optimization flags
    train_cmd="$train_cmd --use_optimized_dataloader"
    train_cmd="$train_cmd --num_dataloader_workers $NUM_WORKERS"
    train_cmd="$train_cmd --dataloader_prefetch_factor $PREFETCH_FACTOR"
    train_cmd="$train_cmd --dataloader_cache_tokenization"
    train_cmd="$train_cmd --dataloader_pin_memory"

    # Add mixed precision if available
    if [[ "$gpu_count" -gt 0 ]]; then
        train_cmd="$train_cmd --mixed_precision bf16"
        train_cmd="$train_cmd --compile_model"
    fi

    # Add checkpoint resumption if specified
    if [[ -n "$resume_checkpoint" ]] && [[ -d "$resume_checkpoint" ]]; then
        log_message "INFO" "Resuming from checkpoint: $resume_checkpoint"
        train_cmd="$train_cmd --resume_from $resume_checkpoint"
    fi

    # Handle multi-GPU setup
    if [[ $gpu_count -gt 1 ]]; then
        # Use torchrun for distributed training
        local launch_cmd="torchrun"
        launch_cmd="$launch_cmd --nproc_per_node=$gpu_count"
        launch_cmd="$launch_cmd --master_port=$((29500 + RANDOM % 1000))"
        train_cmd="$launch_cmd $train_cmd --distributed"
    fi

    # Execute training with comprehensive logging
    log_message "INFO" "Executing: $train_cmd"

    # Run training and capture exit code
    set +e
    eval "$train_cmd" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    set -e

    return $exit_code
}

# =============================================================================
# PREEMPTION HANDLER
# =============================================================================

handle_preemption() {
    log_message "WARN" "Training interrupted (likely preemption)"

    # Check retry count
    local retry_count=$(cat "$RETRY_COUNT_FILE" 2>/dev/null || echo 0)
    retry_count=$((retry_count + 1))

    if [[ $retry_count -gt $MAX_RETRIES ]]; then
        log_message "ERROR" "Maximum retries ($MAX_RETRIES) exceeded. Giving up."
        exit 1
    fi

    echo "$retry_count" > "$RETRY_COUNT_FILE"
    log_message "INFO" "Retry attempt $retry_count of $MAX_RETRIES"

    # Find latest checkpoint for resumption
    local latest_ckpt=$(find_latest_checkpoint)

    if [[ -n "$latest_ckpt" ]]; then
        log_message "INFO" "Found checkpoint for resumption: $latest_ckpt"
        save_orchestrator_state "resuming" "" "$latest_ckpt"
        return 0
    else
        log_message "WARN" "No checkpoint found for resumption, starting fresh"
        return 1
    fi
}

# =============================================================================
# SIGNAL HANDLERS
# =============================================================================

cleanup() {
    log_message "INFO" "Cleanup handler triggered"

    # Save current state
    local latest_ckpt=$(find_latest_checkpoint)
    save_orchestrator_state "interrupted" "" "$latest_ckpt"

    # Upload logs if in SLURM environment
    if [[ $IS_SLURM -eq 1 ]]; then
        upload_logs
    fi

    log_message "INFO" "Cleanup complete"
}

# Set up signal handlers
trap cleanup EXIT SIGINT SIGTERM SIGHUP

# =============================================================================
# LOG UPLOAD TO GIT
# =============================================================================

upload_logs() {
    log_message "INFO" "Uploading logs to git"

    # Only upload logs and small files, not checkpoints
    git add -f "$LOG_DIR"/*.log "$LOG_DIR"/*.json 2>/dev/null || true
    git add -f "$CHECKPOINT_DIR"/*.json "$CHECKPOINT_DIR"/*.jsonl 2>/dev/null || true
    git add -f "$STATE_FILE" "$RETRY_COUNT_FILE" 2>/dev/null || true

    local commit_msg="logs: $EXP_NAME training logs ($(hostname))"
    if [[ $IS_SLURM -eq 1 ]]; then
        commit_msg="$commit_msg - SLURM job $SLURM_JOB_ID"
    fi

    if git diff --staged --quiet; then
        log_message "INFO" "No new logs to commit"
    else
        git commit -m "$commit_msg" || true

        # Try to push with retry logic
        local push_retries=0
        while [[ $push_retries -lt 3 ]]; do
            if git push; then
                log_message "INFO" "Successfully pushed logs to git"
                break
            else
                push_retries=$((push_retries + 1))
                log_message "WARN" "Push attempt $push_retries failed"
                sleep 5
                git pull --rebase=false || true
            fi
        done
    fi
}

# =============================================================================
# MAIN ORCHESTRATION LOGIC
# =============================================================================

main() {
    log_message "INFO" "="
    log_message "INFO" "LATENTWIRE EXPERIMENT ORCHESTRATOR"
    log_message "INFO" "="
    log_message "INFO" "Experiment: $EXP_NAME"
    log_message "INFO" "Working directory: $WORK_DIR"
    log_message "INFO" "Checkpoint directory: $CHECKPOINT_DIR"

    # Create directories
    mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

    # Initialize retry counter if needed
    if [[ ! -f "$RETRY_COUNT_FILE" ]]; then
        echo "0" > "$RETRY_COUNT_FILE"
    fi

    # Set up logging
    LOG_FILE=$(setup_logging)
    log_message "INFO" "Log file: $LOG_FILE"

    # Detect GPU configuration
    log_message "INFO" "Detecting GPU configuration..."
    gpu_info=$(detect_gpu_configuration)
    IFS=':' read -r gpu_count gpu_memory_gb gpu_type <<< "$gpu_info"

    if [[ $gpu_count -eq 0 ]]; then
        log_message "WARN" "No GPUs detected, using CPU mode"
        gpu_count=1  # Pretend we have 1 GPU for batch size calculation
        gpu_memory_gb=32  # Assume reasonable CPU memory
    else
        log_message "INFO" "Detected $gpu_count x $gpu_type GPUs with ${gpu_memory_gb}GB memory each"
    fi

    # Calculate optimal batch configuration
    batch_info=$(calculate_optimal_batch_size "$gpu_count" "$gpu_memory_gb")
    IFS=':' read -r batch_size grad_accum effective_batch <<< "$batch_info"

    log_message "INFO" "Calculated optimal configuration:"
    log_message "INFO" "  Batch size per GPU: $batch_size"
    log_message "INFO" "  Gradient accumulation: $grad_accum"
    log_message "INFO" "  Effective batch size: $((batch_size * gpu_count * grad_accum))"

    # Check for resumption
    resume_checkpoint=""
    if [[ "$RESUME" == "yes" ]] || [[ "$RESUME" == "auto" ]]; then
        resume_checkpoint=$(find_latest_checkpoint)
        if [[ -n "$resume_checkpoint" ]]; then
            log_message "INFO" "Will resume from: $resume_checkpoint"
        elif [[ "$RESUME" == "yes" ]]; then
            log_message "ERROR" "Resume requested but no checkpoint found"
            exit 1
        fi
    fi

    # Main training loop with preemption handling
    training_complete=0
    while [[ $training_complete -eq 0 ]]; do
        log_message "INFO" "Starting training run..."
        save_orchestrator_state "training" "" "$resume_checkpoint"

        # Launch training
        if launch_training "$resume_checkpoint" "$batch_size" "$grad_accum" "$gpu_count"; then
            log_message "INFO" "Training completed successfully"
            training_complete=1
            save_orchestrator_state "completed" "$EPOCHS" "$(find_latest_checkpoint)"

            # Reset retry counter on success
            echo "0" > "$RETRY_COUNT_FILE"
        else
            # Training failed - check if we should retry
            if handle_preemption; then
                # Update resume checkpoint for next iteration
                resume_checkpoint=$(find_latest_checkpoint)
                log_message "INFO" "Preparing to resume from $resume_checkpoint"

                # Brief pause before retry
                sleep 10
            else
                log_message "ERROR" "Training failed and cannot resume"
                save_orchestrator_state "failed" "" ""
                exit 1
            fi
        fi
    done

    # Final checkpoint discovery
    final_checkpoint=$(find_latest_checkpoint)
    log_message "INFO" "Final checkpoint: $final_checkpoint"

    # Upload final logs
    upload_logs

    # Print summary
    log_message "INFO" "="
    log_message "INFO" "EXPERIMENT COMPLETE"
    log_message "INFO" "="
    log_message "INFO" "Name: $EXP_NAME"
    log_message "INFO" "Dataset: $DATASET"
    log_message "INFO" "Samples: $SAMPLES"
    log_message "INFO" "Epochs: $EPOCHS"
    log_message "INFO" "Final checkpoint: $final_checkpoint"
    log_message "INFO" "Logs: $LOG_DIR"

    # Print next steps
    echo ""
    echo "To evaluate the model:"
    echo "  python latentwire/eval.py --ckpt $final_checkpoint --dataset $DATASET --samples 200"
    echo ""
    echo "To view training logs:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "To check training metrics:"
    echo "  python scripts/analyze_diagnostics.py $CHECKPOINT_DIR/diagnostics.jsonl"
}

# =============================================================================
# ENTRY POINT
# =============================================================================

# Run main function
main "$@"