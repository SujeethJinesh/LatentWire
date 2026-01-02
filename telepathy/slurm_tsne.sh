#!/bin/bash
#SBATCH --job-name=tsne_viz
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=40GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/tsne-%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/tsne-%j.err
#SBATCH --signal=B:TERM@30

# IMMEDIATE file write - before anything else
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
LOG_FILE="$WORK_DIR/tsne_debug_$SLURM_JOB_ID.log"

# Write directly to file FIRST (bypasses buffering)
echo "=== SLURM JOB $SLURM_JOB_ID ===" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "Host: $(hostname)" >> "$LOG_FILE"
echo "PWD: $(pwd)" >> "$LOG_FILE"
sync  # Force flush to disk

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
    git add "$LOG_FILE" tsne-*.log tsne-*.err 2>/dev/null || true
    git commit -m "logs: t-SNE job $SLURM_JOB_ID (cleanup)" 2>/dev/null || true
    git push 2>/dev/null || true
}
trap cleanup EXIT TERM INT

mkdir -p "$WORK_DIR/runs"
log "=== Job $SLURM_JOB_ID starting on $(hostname) ==="

cd "$WORK_DIR" || { log "FATAL: Cannot cd to $WORK_DIR"; exit 1; }
log "Working directory: $(pwd)"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1

log "Python: $(which python)"
log "Python version: $(python --version 2>&1)"

log "GPU check..."
nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1 | tee -a "$LOG_FILE" || log "nvidia-smi failed"

log "Git pull..."
git pull 2>&1 | tee -a "$LOG_FILE" || log "git pull failed"

log "Searching for checkpoint..."
find runs -name "bridge_agnews*.pt" 2>/dev/null | tee -a "$LOG_FILE"

# Use the most recent agnews checkpoint (from phase3_multiseed)
CHECKPOINT=$(find runs -name "bridge_agnews_seed42.pt" 2>/dev/null | grep "phase3_multiseed" | sort -r | head -1)

# Fallback to any agnews checkpoint
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(find runs -name "bridge_agnews*.pt" 2>/dev/null | sort -r | head -1)
fi

if [ -z "$CHECKPOINT" ]; then
    log "ERROR: No AG News checkpoint found"
    log "Directory listing:"
    ls -la runs/ 2>&1 | tee -a "$LOG_FILE" || log "ls failed"

    # Push debug log
    git add "$LOG_FILE" tsne-*.err 2>/dev/null || true
    git commit -m "logs: t-SNE failed - no checkpoint (job $SLURM_JOB_ID)" 2>/dev/null || true
    git push 2>/dev/null || true
    exit 1
fi

log "Using checkpoint: $CHECKPOINT"
log "Starting Python t-SNE script..."

python telepathy/generate_tsne_visualization.py \
    --checkpoint "$CHECKPOINT" \
    --output figures/agnews_tsne.pdf \
    --samples_per_class 100 2>&1 | tee -a "$LOG_FILE"

PYTHON_EXIT=$?
log "Python exit code: $PYTHON_EXIT"

if [ $PYTHON_EXIT -eq 0 ]; then
    log "Success! Generated files:"
    ls -la figures/agnews_tsne.* 2>&1 | tee -a "$LOG_FILE"
    COMMIT_MSG="figures: AG News t-SNE (job $SLURM_JOB_ID)"
else
    log "Python script FAILED"
    COMMIT_MSG="logs: t-SNE failed (job $SLURM_JOB_ID)"
fi

log "Pushing to git..."
git add figures/*.pdf figures/*.png "$LOG_FILE" tsne-*.log tsne-*.err 2>/dev/null || true
git commit -m "$COMMIT_MSG" 2>/dev/null || log "Nothing to commit"
git push 2>/dev/null || log "Push failed"

log "=== Job complete ==="
