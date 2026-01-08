#!/bin/bash
#SBATCH --job-name=latentwire_optimized
#SBATCH --nodes=1
#SBATCH --gpus=2                                    # Use 2 GPUs for balanced speed/resources
#SBATCH --account=marlowe-m000066                   # REQUIRED - correct account
#SBATCH --partition=preempt                         # REQUIRED - correct partition
#SBATCH --time=6:00:00                              # 6 hours should be sufficient
#SBATCH --mem=64GB                                  # 64GB is enough for 2 GPUs
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/optimized_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/optimized_%j.err

# =============================================================================
# OPTIMIZED LATENTWIRE EXPERIMENT - 2 GPU Configuration
# =============================================================================
# This configuration uses 2 GPUs for balanced speed and resource usage.
# Estimated completion time: ~3-4 hours (well within 12-hour limit)
# Memory usage: ~17 GB per GPU (well within 80 GB H100 limit)
# =============================================================================
# Submit with: sbatch finalization/submit_optimized.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory - MUST use /projects path
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "SLURM Job Information - Optimized Configuration"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs requested: 2"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Memory: 64 GB"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONNOUSERSITE=1  # Prevent user site-packages conflicts

# Create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export OUTPUT_DIR="runs/optimized_2gpu_${SLURM_JOB_ID}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/figures"

# Configure for 2 GPUs
export SAMPLES=5000
export EPOCHS=8
export BATCH_SIZE=4
export LATENT_LEN=32
export D_Z=256
export DATASET=squad

echo "=============================================================="
echo "Training Configuration (Optimized for 2 GPUs)"
echo "=============================================================="
echo "Samples: $SAMPLES"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Latent dimensions: ${LATENT_LEN}x${D_Z}"
echo "Dataset: $DATASET"
echo "Output directory: $OUTPUT_DIR"
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

# Run the experiment
echo "=============================================================="
echo "Starting experiment at $(date)"
echo "=============================================================="

# Use finalization/RUN.sh with the experiment command
bash finalization/RUN.sh experiment

# Check results
echo "=============================================================="
echo "Experiment completed at $(date)"
echo "=============================================================="

# List generated files
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.log 2>/dev/null || echo "No log files found"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No JSON files found"
ls -lh "$OUTPUT_DIR"/results/*.json 2>/dev/null || echo "No result files found"

# Push results back to git
echo "Pushing results to git..."
git add -A
git commit -m "results: optimized 2-GPU experiment (SLURM job $SLURM_JOB_ID)

Configuration:
- GPUs: 2
- Samples: $SAMPLES
- Epochs: $EPOCHS
- Batch size: $BATCH_SIZE
- Latent: ${LATENT_LEN}x${D_Z}
- Estimated time: 3-4 hours

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
git push || true

echo "=============================================================="
echo "Job completed successfully at $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=============================================================="