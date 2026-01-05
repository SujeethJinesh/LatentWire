#!/bin/bash
#SBATCH --job-name=final_experiments
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=48:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/final_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/final_%j.err

# =============================================================================
# FINAL COMPREHENSIVE EXPERIMENT SUITE FOR LATENTWIRE PAPER
# =============================================================================
# This SLURM script runs the complete experiment suite required for the paper
# submission, including all baselines, ablations, and statistical analyses.
#
# Expected runtime: ~24-48 hours with 4 GPUs
# Memory usage: Up to 256GB for large model experiments
#
# Submit with: sbatch telepathy/submit_final_experiments.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# Watch logs: tail -f /projects/m000066/sujinesh/LatentWire/runs/final_*.log
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
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME=/projects/m000066/sujinesh/.cache/huggingface
export TRANSFORMERS_CACHE=/projects/m000066/sujinesh/.cache/huggingface
export CUDA_LAUNCH_BLOCKING=0  # Better performance
export OMP_NUM_THREADS=16      # Optimize CPU usage
export TOKENIZERS_PARALLELISM=false  # Avoid warnings

# Create necessary directories
mkdir -p runs/final_experiments
mkdir -p figures

# Pull latest code
echo ""
echo "=============================================================="
echo "Git Operations"
echo "=============================================================="
echo "Pulling latest code..."
git pull || {
    echo "WARNING: Failed to pull latest code, continuing with current version"
}
echo "Current commit: $(git rev-parse HEAD)"
echo "Branch: $(git branch --show-current)"
echo "=============================================================="

# Verify GPU availability
echo ""
echo "=============================================================="
echo "GPU Verification"
echo "=============================================================="
nvidia-smi || {
    echo "ERROR: No GPUs available!"
    exit 1
}
echo "=============================================================="

# Set experiment parameters
export OUTPUT_BASE_DIR="runs/final_experiments"
export EXPERIMENT_NAME="final_suite_job${SLURM_JOB_ID}"
export TEST_MODE=false
export SKIP_VALIDATION=false

# Create output directory for this job
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR/logs"

# Main log file
MAIN_LOG="$OUTPUT_DIR/logs/slurm_job_${SLURM_JOB_ID}.log"

echo ""
echo "=============================================================="
echo "Starting Final Experiment Suite"
echo "=============================================================="
echo "Experiment name: $EXPERIMENT_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Main log: $MAIN_LOG"
echo "=============================================================="

# Function to handle job interruption gracefully
cleanup() {
    echo ""
    echo "=============================================================="
    echo "Job interrupted or completed at $(date)"
    echo "=============================================================="

    # Save current state
    echo "Saving current state..."
    if [ -d "$OUTPUT_DIR" ]; then
        tar -czf "${OUTPUT_DIR}_partial.tar.gz" "$OUTPUT_DIR" \
            --exclude="*.pth" \
            --exclude="*.safetensors" \
            --exclude="*.bin" || true
    fi

    # Push any results to git
    echo "Pushing partial results to git..."
    git add -A
    git commit -m "results: partial results from job $SLURM_JOB_ID (interrupted/completed)

Job ended at $(date)
Status: ${1:-unknown}

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
    git push || true

    exit ${2:-0}
}

# Set up signal handlers
trap 'cleanup "SIGTERM" 130' SIGTERM
trap 'cleanup "SIGINT" 130' SIGINT
trap 'cleanup "completed" 0' EXIT

# Run the final experiment suite
echo "Launching experiment suite..." | tee -a "$MAIN_LOG"

# Use srun for proper GPU allocation across multiple processes
{
    # Export job ID for tracking
    export SLURM_JOB_ID

    # Run the main experiment script
    bash telepathy/run_final_experiments.sh 2>&1

    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ Experiment suite completed successfully"
        EXIT_STATUS=0
    else
        echo "✗ Experiment suite failed with errors"
        EXIT_STATUS=1
    fi

} 2>&1 | tee -a "$MAIN_LOG"

# Generate summary report
echo ""
echo "=============================================================="
echo "Generating Summary Report"
echo "=============================================================="
python -c "
import json
import glob
from pathlib import Path
from datetime import datetime

output_dir = Path('$OUTPUT_DIR')
results_dir = output_dir / 'results'

if results_dir.exists():
    # Count experiments
    result_files = list(results_dir.glob('*/results.json'))
    num_experiments = len(result_files)

    # Collect best results
    best_results = {}
    for result_file in result_files:
        with open(result_file) as f:
            result = json.load(f)
            exp_name = result_file.parent.name

            # Track best accuracy per task
            for task in ['sst2', 'agnews', 'trec', 'xsum']:
                if task in exp_name:
                    acc = result.get('accuracy', result.get('f1_score', 0))
                    if task not in best_results or acc > best_results[task][1]:
                        best_results[task] = (exp_name, acc)

    # Create summary
    summary = {
        'job_id': '$SLURM_JOB_ID',
        'experiment_name': '$EXPERIMENT_NAME',
        'completed_at': datetime.now().isoformat(),
        'num_experiments': num_experiments,
        'best_results': {task: {'experiment': exp, 'score': score}
                        for task, (exp, score) in best_results.items()}
    }

    # Save summary
    with open(output_dir / 'job_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Summary saved to {output_dir}/job_summary.json')
    print(f'Total experiments completed: {num_experiments}')
    print('\\nBest results:')
    for task, (exp, score) in best_results.items():
        print(f'  {task.upper()}: {score:.3f} ({exp})')
else:
    print('No results directory found')
" | tee -a "$MAIN_LOG"

# Archive final results
echo ""
echo "=============================================================="
echo "Archiving Results"
echo "=============================================================="
ARCHIVE_NAME="${OUTPUT_DIR}_complete.tar.gz"
tar -czf "$ARCHIVE_NAME" "$OUTPUT_DIR" \
    --exclude="*.pth" \
    --exclude="*.safetensors" \
    --exclude="*.bin" || true

echo "Results archived to: $ARCHIVE_NAME" | tee -a "$MAIN_LOG"

# Update REPORT.md
echo ""
echo "=============================================================="
echo "Updating Documentation"
echo "=============================================================="
python -c "
from datetime import datetime
from pathlib import Path

report_path = Path('telepathy/REPORT.md')
if report_path.exists():
    with open(report_path, 'a') as f:
        f.write(f'''

## Final Comprehensive Experiments - {datetime.now().strftime('%Y-%m-%d %H:%M')}
**SLURM Job ID**: $SLURM_JOB_ID
**Experiment**: $EXPERIMENT_NAME

### Configuration
- 3 seeds: 42, 123, 456
- Model families: Llama-3.2 (1B/3B), Llama-3.1 (8B), Mistral-7B, Qwen2.5 (1.5B)
- Tasks: SST-2, AG News, TREC, XSUM
- Baselines: Random, Majority, Linear Probe, LLMLingua, Direct Prompting
- Compression factors: 4, 8, 16, 32

### Phases Completed
1. Statistical Rigor (Full Test Sets)
2. Linear Probe Baseline
3. Fair Baseline Comparisons
4. Latency Measurements
5. Generation Task (XSUM)
6. Model Size Ablations

### Status
Completed at $(date)
Results: $OUTPUT_DIR
Archive: ${ARCHIVE_NAME}

''')
    print('Updated REPORT.md')
" | tee -a "$MAIN_LOG"

# Push final results to git
echo ""
echo "=============================================================="
echo "Pushing Final Results to Git"
echo "=============================================================="
git add -A
git commit -m "results: final comprehensive experiments (SLURM job $SLURM_JOB_ID)

Completed full experiment suite:
- Phase 1: Statistical rigor with 3 seeds
- Phase 2: Linear probe baselines
- Phase 3: Fair baseline comparisons
- Phase 4: Latency measurements
- Phase 5: Generation task (XSUM)
- Phase 6: Model size ablations

Results: $OUTPUT_DIR
Exit status: ${EXIT_STATUS:-unknown}

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true

git push || true

# Final summary
echo ""
echo "=============================================================="
echo "Job Summary"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date -d @${SLURM_JOB_START_TIME:-0})"
echo "End time: $(date)"
if [ -n "$SLURM_JOB_START_TIME" ]; then
    echo "Total runtime: $(($(date +%s) - SLURM_JOB_START_TIME)) seconds"
fi
echo "Results directory: $OUTPUT_DIR"
echo "Archive: $ARCHIVE_NAME"
echo "Main log: $MAIN_LOG"
echo "SLURM output: /projects/m000066/sujinesh/LatentWire/runs/final_${SLURM_JOB_ID}.log"
echo "Exit status: ${EXIT_STATUS:-0}"
echo "=============================================================="

exit ${EXIT_STATUS:-0}