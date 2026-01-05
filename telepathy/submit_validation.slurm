#!/bin/bash
#SBATCH --job-name=telepathy_validation
#SBATCH --nodes=1
#SBATCH --gpus=1                                    # Only need 1 GPU for validation
#SBATCH --account=marlowe-m000066                   # REQUIRED - correct account
#SBATCH --partition=preempt                         # REQUIRED - correct partition
#SBATCH --time=00:30:00                             # 30 minutes should be plenty
#SBATCH --mem=64GB                                  # Minimal memory for validation
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/validation_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/validation_%j.err

# =============================================================================
# Validation check for telepathy experiments
# Performs dry runs to ensure everything works before actual training
# =============================================================================
# Submit with: sbatch telepathy/submit_validation.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory - MUST use /projects path
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "TELEPATHY VALIDATION - SLURM Job"
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

# Create output directories
mkdir -p runs/validation figures

# Pull latest code
echo "Pulling latest code..."
git pull

# Run FULL validation on HPC (we have the resources)
echo "Starting FULL validation..."
VALIDATION_MODE=full bash telepathy/run_validation.sh

# Check if validation passed
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ VALIDATION SUCCESSFUL - HPC environment ready for experiments!"
    VALIDATION_STATUS="PASSED"
else
    echo ""
    echo "❌ VALIDATION FAILED - Please check the logs and fix issues"
    VALIDATION_STATUS="FAILED"
fi

# Push validation results back to git
echo "Pushing validation results to git..."
git add -A
git commit -m "validation: telepathy experiment validation - $VALIDATION_STATUS (SLURM job $SLURM_JOB_ID)

Validation completed with status: $VALIDATION_STATUS
Check runs/validation/ for detailed reports

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
git push || true

echo "=============================================================="
echo "Validation job completed at $(date)"
echo "Status: $VALIDATION_STATUS"
echo "=============================================================="