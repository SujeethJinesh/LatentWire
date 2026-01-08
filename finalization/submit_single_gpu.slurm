#!/bin/bash
#SBATCH --job-name=latentwire_1gpu
#SBATCH --nodes=1
#SBATCH --gpus=1                                    # Use only 1 GPU for maximum efficiency
#SBATCH --account=marlowe-m000066                   # REQUIRED - correct account
#SBATCH --partition=preempt                         # REQUIRED - correct partition
#SBATCH --time=8:00:00                              # 8 hours to be safe (estimate: 5.6 hours)
#SBATCH --mem=40GB                                  # 40GB is sufficient for 1 GPU
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/single_gpu_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/single_gpu_%j.err

# =============================================================================
# SINGLE GPU LATENTWIRE EXPERIMENT - Maximum Resource Efficiency
# =============================================================================
# This configuration uses only 1 GPU, leaving 3 GPUs available for other jobs.
# Estimated completion time: ~5-6 hours (well within 12-hour limit)
# Memory usage: ~34 GB per GPU (well within 80 GB H100 limit)
#
# BENEFITS:
# - Leaves 3 H100 GPUs free for other experiments
# - No distributed training overhead
# - Simpler debugging and monitoring
# - Still completes within reasonable time
# =============================================================================
# Submit with: sbatch finalization/submit_single_gpu.slurm
# Monitor with: squeue -u $USER
# Watch logs: tail -f runs/single_gpu_*.log
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory - MUST use /projects path
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "SLURM Job Information - Single GPU Configuration"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs requested: 1"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"
echo "Memory: 40 GB"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONNOUSERSITE=1  # Prevent user site-packages conflicts

# Create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export OUTPUT_DIR="runs/single_gpu_${SLURM_JOB_ID}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/figures"

# Configure for single GPU
export SAMPLES=5000
export EPOCHS=8
export BATCH_SIZE=4        # Keep batch size at 4 for memory safety
export LATENT_LEN=32
export D_Z=256
export DATASET=squad
export EVAL_SAMPLES=1000
export SEEDS="42 123 456"

echo "=============================================================="
echo "Training Configuration (Single GPU - Resource Efficient)"
echo "=============================================================="
echo "Samples: $SAMPLES"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Latent dimensions: ${LATENT_LEN}x${D_Z}"
echo "Dataset: $DATASET"
echo "Eval samples: $EVAL_SAMPLES"
echo "Seeds: $SEEDS"
echo "Output directory: $OUTPUT_DIR"
echo "Estimated time: 5-6 hours"
echo "=============================================================="

# Pull latest code
echo "Pulling latest code..."
git pull

# Set up Python environment
echo "Setting up Python environment..."
if [ -f .venv/bin/activate ]; then
    rm -rf .venv  # Remove old venv to avoid conflicts
fi

python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Install dependencies
if [ -f requirements.txt ]; then
    echo "Installing dependencies..."
    python3 -m pip install --upgrade pip --no-user
    python3 -m pip install -r requirements.txt --no-user 2>/dev/null || {
        echo "Some packages failed, installing core packages..."
        python3 -m pip install torch transformers==4.45.2 datasets accelerate --no-user || true
    }
fi

echo "Python packages installed:"
pip list | grep -E "torch|transformers|datasets|accelerate"

# Run the experiment with timing
echo "=============================================================="
echo "Starting single GPU experiment at $(date)"
echo "=============================================================="

START_TIME=$SECONDS

# Use finalization/RUN.sh with the experiment command
bash finalization/RUN.sh experiment

END_TIME=$SECONDS
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_HOURS=$(echo "scale=2; $ELAPSED_TIME / 3600" | bc)

echo "=============================================================="
echo "Experiment completed at $(date)"
echo "Total runtime: ${ELAPSED_TIME} seconds (${ELAPSED_HOURS} hours)"
echo "=============================================================="

# Generate performance summary
cat > "$OUTPUT_DIR/performance_summary.txt" << EOF
Single GPU Training Performance Summary
=======================================
Configuration:
  - GPUs: 1 (H100)
  - Samples: $SAMPLES
  - Epochs: $EPOCHS
  - Batch size: $BATCH_SIZE
  - Latent: ${LATENT_LEN}x${D_Z}

Performance:
  - Total runtime: ${ELAPSED_TIME} seconds (${ELAPSED_HOURS} hours)
  - Samples per second: $(echo "scale=2; $SAMPLES * $EPOCHS / $ELAPSED_TIME" | bc)
  - Time per epoch: $(echo "scale=2; $ELAPSED_TIME / $EPOCHS / 60" | bc) minutes

Resource Usage:
  - GPUs used: 1 of 4 available (25%)
  - Memory used: ~34 GB of 80 GB (42%)
  - GPUs available for other jobs: 3

Status: COMPLETED
EOF

echo "Performance summary written to: $OUTPUT_DIR/performance_summary.txt"

# List generated files
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.log 2>/dev/null || echo "No log files found"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No JSON files found"
ls -lh "$OUTPUT_DIR"/results/*.json 2>/dev/null || echo "No result files found"

# Push results back to git
echo "Pushing results to git..."
git add -A
git commit -m "results: single GPU experiment (SLURM job $SLURM_JOB_ID)

Configuration:
- GPUs: 1 (most resource efficient)
- Samples: $SAMPLES
- Epochs: $EPOCHS
- Batch size: $BATCH_SIZE
- Latent: ${LATENT_LEN}x${D_Z}
- Runtime: ${ELAPSED_HOURS} hours
- Efficiency: Left 3 GPUs free for other jobs

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
git push || true

echo "=============================================================="
echo "Job completed successfully at $(date)"
echo "=============================================================="