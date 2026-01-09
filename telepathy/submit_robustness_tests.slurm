#!/bin/bash
#SBATCH --job-name=telepathy_robustness
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/robustness_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/robustness_%j.err

# =============================================================================
# Comprehensive Robustness Testing for Latent Telepathy
# =============================================================================
# This script runs a full robustness test suite covering:
# - Adversarial attacks
# - Noise resilience
# - Distribution shift
# - Model versioning
# - Stress testing
# - Security analysis
#
# Submit with: sbatch telepathy/submit_robustness_tests.slurm
# Monitor with: squeue -u $USER
# =============================================================================

# Set working directory
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
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directories
mkdir -p runs/robustness
mkdir -p runs/robustness/reports

# Pull latest code
echo "Pulling latest code..."
git pull

# Set up configuration
OUTPUT_DIR="runs/robustness/job_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Define checkpoint path (use latest checkpoint if not specified)
CHECKPOINT="${CHECKPOINT:-runs/telepathy_v15_gsm8k/checkpoint_epoch_10.pt}"

echo "=============================================================="
echo "Running Robustness Test Suite"
echo "=============================================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/robustness_suite_${TIMESTAMP}.log"

echo "Starting robustness tests..."
echo "Log file: $LOG_FILE"

# Run the comprehensive test suite
{
    echo "Phase 1: Quick smoke tests (fast mode)"
    echo "========================================="
    python telepathy/robustness_test_suite.py \
        --checkpoint "$CHECKPOINT" \
        --device cuda \
        --fast \
        2>&1 | tee "$OUTPUT_DIR/phase1_quick.log"

    echo ""
    echo "Phase 2: Full robustness suite"
    echo "========================================="
    python telepathy/robustness_test_suite.py \
        --checkpoint "$CHECKPOINT" \
        --device cuda \
        2>&1 | tee "$OUTPUT_DIR/phase2_full.log"

    echo ""
    echo "Phase 3: Generating summary report"
    echo "========================================="
    python -c "
import json
from pathlib import Path

# Load results
report_path = Path('runs/robustness_report.json')
if report_path.exists():
    with open(report_path) as f:
        report = json.load(f)

    # Print summary
    print('='*60)
    print('ROBUSTNESS TEST SUMMARY')
    print('='*60)
    print(f\"Overall Pass Rate: {report['overall']['pass_rate']*100:.1f}%\")
    print(f\"Total Tests: {report['overall']['total_tests']}\")
    print(f\"Average Score: {report['overall']['avg_score']:.3f}\")
    print()
    print('Category Results:')
    for cat, stats in report['categories'].items():
        print(f\"  {cat}: {stats['pass_rate']*100:.1f}% ({stats['passed']}/{stats['total']})\")

    # Save to output directory
    import shutil
    shutil.copy(report_path, '$OUTPUT_DIR/robustness_report.json')
    print(f\"\\nReport saved to: $OUTPUT_DIR/robustness_report.json\")
else:
    print('ERROR: No robustness report found!')
"

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================================="
echo "Test suite completed at $(date)"
echo "=============================================================="
echo "Results saved to:"
echo "  - $OUTPUT_DIR/robustness_report.json"
echo "  - $LOG_FILE"

# Push results back to git
echo ""
echo "Pushing results to git..."
git add -A
git commit -m "results: robustness test suite (SLURM job $SLURM_JOB_ID)

Comprehensive robustness testing for Latent Telepathy:
- Adversarial attacks
- Noise resilience
- Distribution shift
- Model versioning
- Stress testing
- Security analysis

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
git push || true

echo "=============================================================="
echo "Job completed successfully!"
echo "=============================================================="