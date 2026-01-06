#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=40GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/ablation-%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/ablation-%j.err
#SBATCH --signal=B:TERM@30

# LatentWire Ablation Experiments
# Priority 1: Source layer ablation (layers 8, 12, 16, 20, 24, 28)
# Priority 2: Multi-model pairs (Llama->Mistral, Qwen->Mistral)
# Priority 3: Training-free baseline (random projection)

WORK_DIR="/projects/m000066/sujinesh/LatentWire"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$WORK_DIR/runs/ablation_$TIMESTAMP"

# Create output dir FIRST so we can log there
mkdir -p "$OUTPUT_DIR"

# Log file goes IN the output directory so it gets pushed with results
LOG_FILE="$OUTPUT_DIR/experiment.log"

# Write directly to file FIRST (bypasses buffering)
echo "=== SLURM JOB ${SLURM_JOB_ID:-interactive} ===" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "Host: $(hostname)" >> "$LOG_FILE"
echo "Output dir: $OUTPUT_DIR" >> "$LOG_FILE"
sync

# Function to log with immediate flush
log() {
    echo "[$(date '+%H:%M:%S')] $1" >> "$LOG_FILE"
    echo "[$(date '+%H:%M:%S')] $1"
    sync
}

# Trap to push logs on ANY exit (including preemption)
cleanup() {
    log "=== CLEANUP TRIGGERED ==="
    cd "$WORK_DIR" 2>/dev/null || true

    # Add all results and logs (log file is inside OUTPUT_DIR)
    git add -f "$LOG_FILE" "$OUTPUT_DIR" 2>/dev/null || true
    git commit -m "results: ablation experiments (job ${SLURM_JOB_ID:-interactive}, cleanup)" 2>/dev/null || true
    git push 2>/dev/null || true
}
trap cleanup EXIT TERM INT

log "=== Job ${SLURM_JOB_ID:-interactive} starting on $(hostname) ==="

cd "$WORK_DIR" || { log "FATAL: Cannot cd to $WORK_DIR"; exit 1; }
log "Working directory: $(pwd)"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

log "Python: $(which python)"
log "Python version: $(python --version 2>&1)"

log "GPU check..."
nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1 | tee -a "$LOG_FILE" || log "nvidia-smi failed"

log "Git pull..."
git pull 2>&1 | tee -a "$LOG_FILE" || log "git pull failed"

# =============================================================================
# EXPERIMENT 1: Source Layer Ablation (Priority 1)
# =============================================================================
log "=== EXPERIMENT 1: Source Layer Ablation ==="

python telepathy/run_ablation_experiments.py \
    --experiment layer_ablation \
    --datasets agnews sst2 trec \
    --steps 1500 \
    --seed 42 \
    --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

LAYER_EXIT=${PIPESTATUS[0]}
log "Layer ablation exit code: $LAYER_EXIT"

if [ $LAYER_EXIT -eq 0 ]; then
    log "Layer ablation SUCCESS"
    # Push intermediate results
    git add -f "$LOG_FILE" "$OUTPUT_DIR" 2>/dev/null || true
    git commit -m "results: layer ablation complete (job ${SLURM_JOB_ID:-interactive})" 2>/dev/null || true
    git push 2>/dev/null || true
else
    log "Layer ablation FAILED"
    # Still push logs on failure
    git add -f "$LOG_FILE" "$OUTPUT_DIR" 2>/dev/null || true
    git commit -m "results: layer ablation FAILED (job ${SLURM_JOB_ID:-interactive})" 2>/dev/null || true
    git push 2>/dev/null || true
fi

# =============================================================================
# EXPERIMENT 2: Multi-Model Pairs (Priority 2)
# =============================================================================
log "=== EXPERIMENT 2: Multi-Model Pairs ==="

python telepathy/run_ablation_experiments.py \
    --experiment model_pairs \
    --model_pairs llama_mistral qwen_mistral \
    --datasets agnews sst2 trec \
    --steps 1500 \
    --seed 42 \
    --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

PAIRS_EXIT=${PIPESTATUS[0]}
log "Model pairs exit code: $PAIRS_EXIT"

if [ $PAIRS_EXIT -eq 0 ]; then
    log "Model pairs SUCCESS"
    # Push intermediate results
    git add -f "$LOG_FILE" "$OUTPUT_DIR" 2>/dev/null || true
    git commit -m "results: model pairs complete (job ${SLURM_JOB_ID:-interactive})" 2>/dev/null || true
    git push 2>/dev/null || true
else
    log "Model pairs FAILED"
    # Still push logs on failure
    git add -f "$LOG_FILE" "$OUTPUT_DIR" 2>/dev/null || true
    git commit -m "results: model pairs FAILED (job ${SLURM_JOB_ID:-interactive})" 2>/dev/null || true
    git push 2>/dev/null || true
fi

# =============================================================================
# EXPERIMENT 3: Training-Free Baseline (Priority 3)
# =============================================================================
log "=== EXPERIMENT 3: Training-Free Baseline ==="

python telepathy/run_ablation_experiments.py \
    --experiment training_free \
    --datasets agnews sst2 trec \
    --seed 42 \
    --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

FREE_EXIT=${PIPESTATUS[0]}
log "Training-free exit code: $FREE_EXIT"

if [ $FREE_EXIT -eq 0 ]; then
    log "Training-free SUCCESS"
    git add -f "$LOG_FILE" "$OUTPUT_DIR" 2>/dev/null || true
    git commit -m "results: training-free complete (job ${SLURM_JOB_ID:-interactive})" 2>/dev/null || true
    git push 2>/dev/null || true
else
    log "Training-free FAILED"
    git add -f "$LOG_FILE" "$OUTPUT_DIR" 2>/dev/null || true
    git commit -m "results: training-free FAILED (job ${SLURM_JOB_ID:-interactive})" 2>/dev/null || true
    git push 2>/dev/null || true
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================
log "=== FINAL SUMMARY ==="
log "Layer ablation: exit $LAYER_EXIT"
log "Model pairs: exit $PAIRS_EXIT"
log "Training-free: exit $FREE_EXIT"

# Show results
log "Results in $OUTPUT_DIR:"
find "$OUTPUT_DIR" -name "*.json" -exec echo {} \; -exec cat {} \; 2>&1 | tee -a "$LOG_FILE"

# Final push
log "Pushing all results..."
git add -f "$LOG_FILE" "$OUTPUT_DIR" 2>/dev/null || true
git commit -m "results: ablation experiments complete (job ${SLURM_JOB_ID:-interactive})" 2>/dev/null || true
git push 2>/dev/null || true

log "=== Job complete ==="
log "Log file saved to: $LOG_FILE"
