#!/bin/bash
#SBATCH --job-name=logged_experiment
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/slurm_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/slurm_%j.err
#SBATCH --signal=SIGUSR1@90  # Send signal 90 seconds before time limit

# =============================================================================
# SLURM Template with Comprehensive Logging for Preemptible Jobs
# =============================================================================
# This template demonstrates how to use the comprehensive logging system
# that ensures no log data is lost even during preemption.
#
# Submit with: sbatch telepathy/submit_with_logging.slurm
# Monitor with: squeue -u $USER
# View logs: tail -f runs/experiment_*/main_*.log
# =============================================================================

# Set working directory - MUST use /projects path
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
echo "Time limit: $SBATCH_TIMELIMIT"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1  # Disable Python buffering for immediate output
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Better memory management

# Create output directories
EXPERIMENT_NAME="experiment_${SLURM_JOB_ID}"
OUTPUT_DIR="runs/${EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR"

# =========================================================================
# Git Setup with Enhanced Error Handling
# =========================================================================
echo ""
echo "Configuring Git..."

# Configure git identity if not set
git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job $SLURM_JOB_ID"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@hpc.cluster"

# Stash any local changes to avoid conflicts
echo "Stashing local changes..."
git stash push -m "SLURM job $SLURM_JOB_ID pre-run stash" || true

# Pull latest code
echo "Pulling latest code..."
if ! git pull --rebase=false; then
    echo "WARNING: git pull failed. Continuing with existing code..."
fi

# =========================================================================
# Signal Handler for Graceful Preemption
# =========================================================================
handle_preemption() {
    echo ""
    echo "=============================================================="
    echo "PREEMPTION SIGNAL RECEIVED - Starting graceful shutdown"
    echo "Time: $(date)"
    echo "=============================================================="

    # Create preemption marker
    touch "$OUTPUT_DIR/PREEMPTED"

    # Save current state to a recovery file
    echo "{
        \"job_id\": \"$SLURM_JOB_ID\",
        \"preempted_at\": \"$(date -Iseconds)\",
        \"node\": \"$SLURMD_NODENAME\",
        \"output_dir\": \"$OUTPUT_DIR\"
    }" > "$OUTPUT_DIR/preemption_info.json"

    # Wait a moment for Python process to save checkpoint
    sleep 10

    # Force sync all files
    sync

    # Git commit logs immediately
    echo "Saving logs to git..."
    git add "$OUTPUT_DIR/*.log" "$OUTPUT_DIR/*.jsonl" "$OUTPUT_DIR/*.json" 2>/dev/null || true
    git commit -m "logs: preemption checkpoint for job $SLURM_JOB_ID" || true
    git push || echo "WARNING: Could not push to git"

    echo "Graceful shutdown complete"
    exit 0
}

# Register the signal handler
trap 'handle_preemption' SIGUSR1 SIGTERM

# =========================================================================
# Python Script with Comprehensive Logging
# =========================================================================
echo ""
echo "Starting Python script with comprehensive logging..."

# Create a Python wrapper that uses our logging system
cat << 'EOF' > "$OUTPUT_DIR/run_with_logging.py"
"""
Training wrapper with comprehensive logging for preemptible jobs.
"""
import sys
import os
import json
import signal
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/projects/m000066/sujinesh/LatentWire')

from telepathy.logging_utils import (
    LogConfig,
    setup_comprehensive_logging,
    log_metrics,
    recover_from_preemption
)

# Import your actual training script
# from latentwire.train import main as train_main

def handle_signal(signum, frame):
    """Handle preemption signal."""
    print(f"\nReceived signal {signum} - saving checkpoint and exiting gracefully...")
    # Your checkpoint saving logic here
    sys.exit(0)

def main():
    # Register signal handlers
    signal.signal(signal.SIGUSR1, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--resume', action='store_true')
    args, unknown = parser.parse_known_args()

    # Set up logging configuration
    config = LogConfig(
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        flush_interval=1.0,
        buffer_size=1,
        compress_old_logs=True,
        max_log_size_mb=100.0,
        enable_structured_logs=True,
        enable_git_backup=True,
        backup_interval=300  # 5 minutes
    )

    # Start comprehensive logging
    with setup_comprehensive_logging(config) as loggers:
        print("Comprehensive logging system initialized")
        print(f"Main log: {loggers['main_log_path']}")
        print(f"Metrics log: {loggers['metrics_log_path']}")

        # Check for recovery from preemption
        if args.resume:
            state = recover_from_preemption(args.output_dir)
            if state:
                print(f"Resuming from epoch {state.get('epoch', 0)}, step {state.get('step', 0)}")
            else:
                print("No checkpoint found, starting fresh")

        # Log initial configuration
        loggers['metrics'].log({
            'event': 'training_start',
            'job_id': os.getenv('SLURM_JOB_ID', 'local'),
            'node': os.getenv('SLURMD_NODENAME', 'local'),
            'gpus': os.getenv('CUDA_VISIBLE_DEVICES', 'none')
        })

        # ===================================================================
        # YOUR ACTUAL TRAINING CODE HERE
        # ===================================================================
        # Example training loop with logging
        import time
        import random

        for epoch in range(5):
            for step in range(100):
                # Simulate training
                time.sleep(0.1)
                loss = random.random()

                # Log metrics every 10 steps
                if step % 10 == 0:
                    metrics = {
                        'loss': loss,
                        'learning_rate': 1e-4,
                        'batch_size': 32
                    }
                    log_metrics(metrics, step=step, epoch=epoch, logger=loggers['metrics'])

                    # Save checkpoint state
                    loggers['checkpoint'].save_state({
                        'epoch': epoch,
                        'step': step,
                        'global_step': epoch * 100 + step,
                        'loss': loss,
                        'optimizer_state': 'dummy',  # Replace with actual state
                        'model_state': 'dummy'  # Replace with actual state
                    })

                    print(f"Epoch {epoch}, Step {step}: loss={loss:.4f}")

                # Check for log rotation
                if step % 50 == 0 and loggers['rotator'].should_rotate():
                    print("Rotating log file due to size limit...")
                    loggers['rotator'].rotate()

        # ===================================================================

        # Log completion
        loggers['metrics'].log({
            'event': 'training_complete',
            'final_epoch': epoch,
            'final_step': step
        })

        print("Training completed successfully")

if __name__ == "__main__":
    main()
EOF

# Run the Python script with comprehensive logging
python "$OUTPUT_DIR/run_with_logging.py" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "$EXPERIMENT_NAME" \
    --resume \
    &  # Run in background so we can handle signals

# Save the PID
PYTHON_PID=$!
echo "Python process PID: $PYTHON_PID"

# Wait for the Python process
wait $PYTHON_PID
PYTHON_EXIT_CODE=$?

echo ""
echo "Python process exited with code: $PYTHON_EXIT_CODE"

# =========================================================================
# Post-Processing and Final Git Push
# =========================================================================
echo ""
echo "Post-processing and saving results..."

# Create summary file
cat << EOF > "$OUTPUT_DIR/job_summary.json"
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$SLURMD_NODENAME",
    "start_time": "$(date -d @$SLURM_JOB_START_TIMESTAMP -Iseconds 2>/dev/null || echo 'unknown')",
    "end_time": "$(date -Iseconds)",
    "exit_code": $PYTHON_EXIT_CODE,
    "preempted": $([ -f "$OUTPUT_DIR/PREEMPTED" ] && echo "true" || echo "false")
}
EOF

# Compress large logs if needed
for log in "$OUTPUT_DIR"/*.log; do
    if [ -f "$log" ] && [ $(stat -c%s "$log" 2>/dev/null || stat -f%z "$log" 2>/dev/null) -gt 104857600 ]; then
        echo "Compressing large log: $log"
        gzip -9 "$log"
    fi
done

# Final git commit
echo "Pushing final results to git..."
git add "$OUTPUT_DIR/*.log" "$OUTPUT_DIR/*.log.gz" "$OUTPUT_DIR/*.jsonl" \
        "$OUTPUT_DIR/*.json" "$OUTPUT_DIR/**/*.json" 2>/dev/null || true

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    COMMIT_MSG="results: completed $EXPERIMENT_NAME (SLURM job $SLURM_JOB_ID)"
else
    COMMIT_MSG="results: incomplete $EXPERIMENT_NAME - exit code $PYTHON_EXIT_CODE (SLURM job $SLURM_JOB_ID)"
fi

git commit -m "$COMMIT_MSG

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true

# Try to push with retries
RETRY_COUNT=0
while [ $RETRY_COUNT -lt 3 ]; do
    if git push; then
        echo "Successfully pushed to remote"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "Push attempt $RETRY_COUNT failed"
        if [ $RETRY_COUNT -lt 3 ]; then
            sleep 5
            git pull --rebase=false || true
        fi
    fi
done

[ $RETRY_COUNT -eq 3 ] && echo "WARNING: Could not push after 3 attempts"

# =========================================================================
# Cleanup and Final Report
# =========================================================================
echo ""
echo "=============================================================="
echo "Job Summary"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Exit code: $PYTHON_EXIT_CODE"
echo "Output directory: $OUTPUT_DIR"
echo "Preempted: $([ -f "$OUTPUT_DIR/PREEMPTED" ] && echo "Yes" || echo "No")"
echo "End time: $(date)"
echo "=============================================================="

# List generated files
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.log "$OUTPUT_DIR"/*.jsonl "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No log files found"

exit $PYTHON_EXIT_CODE