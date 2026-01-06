#!/bin/bash
#SBATCH --job-name=validate_hpc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=00:10:00
#SBATCH --mem=16GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/validate_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/validate_%j.err

# =============================================================================
# HPC Validation Script - Checks all prerequisites on HPC cluster
# =============================================================================
# Submit with: sbatch telepathy/validate_hpc.slurm
# View output: cat runs/validate_*.log
# =============================================================================

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "HPC Validation Check"
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
export TOKENIZERS_PARALLELISM=false

# Create runs directory if needed
mkdir -p runs

# Pull latest code
echo "Pulling latest code..."
git pull || echo "Warning: Could not pull latest code"

echo ""
echo "Running validation..."
echo "=============================================================="

# Run validation
python3 finalization/validate.py

# Store exit code
VALIDATION_RESULT=$?

echo ""
echo "=============================================================="

# Report results
if [ $VALIDATION_RESULT -eq 0 ]; then
    echo "✓ VALIDATION PASSED - HPC environment is ready!"
    echo ""
    echo "You can now run experiments with:"
    echo "  sbatch telepathy/submit_enhanced_arxiv.slurm"
    echo "  sbatch telepathy/submit_unified_experiments.slurm"
else
    echo "✗ VALIDATION FAILED - Please fix issues before running experiments"
    echo ""
    echo "Check the log file for details:"
    echo "  cat runs/validate_${SLURM_JOB_ID}.log"
fi

echo "=============================================================="
echo "Validation completed at $(date)"
echo "=============================================================="

# Exit with validation result code
exit $VALIDATION_RESULT