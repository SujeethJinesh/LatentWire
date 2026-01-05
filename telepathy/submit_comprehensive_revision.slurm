#!/bin/bash
#SBATCH --job-name=comprehensive_revision
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/revision_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/revision_%j.err

# =============================================================================
# SLURM Submission Script for Comprehensive Revision Experiments
# =============================================================================
# This script runs a comprehensive suite of experiments testing multiple
# compression levels, loss functions, and model configurations to establish
# robust baselines and identify optimal settings for the LatentWire system.
#
# Submit with: sbatch telepathy/submit_comprehensive_revision.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# View logs: tail -f /projects/m000066/sujinesh/LatentWire/runs/revision_*.log
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
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME=/projects/m000066/sujinesh/.cache/huggingface
export TRANSFORMERS_CACHE=/projects/m000066/sujinesh/.cache/huggingface
export CUDA_LAUNCH_BLOCKING=0  # Disable for better performance

# Create necessary directories
mkdir -p runs/revision_suite
mkdir -p figures

# Pull latest code
echo ""
echo "=============================================================="
echo "Git Operations"
echo "=============================================================="
echo "Pulling latest code..."
git pull
echo "Current commit: $(git rev-parse HEAD)"
echo "Branch: $(git branch --show-current)"
echo "=============================================================="

# Verify GPU availability
echo ""
echo "=============================================================="
echo "GPU Verification"
echo "=============================================================="
nvidia-smi
echo "=============================================================="

# Run the comprehensive revision experiments
echo ""
echo "=============================================================="
echo "Starting Comprehensive Revision Experiments"
echo "=============================================================="
echo "This will run multiple configurations:"
echo "- Compression levels: 8, 16, 32, 64, 128 tokens"
echo "- Loss functions: CE-only, CE+KD, CE+Contrastive"
echo "- Models: Llama-3.1-8B, Qwen2.5-7B"
echo "- Datasets: SQuAD, HotpotQA, ArXiv"
echo "=============================================================="

# Set experiment parameters
export EXPERIMENT_NAME="revision_suite_${SLURM_JOB_ID}"
export OUTPUT_BASE_DIR="runs/revision_suite"
export MAX_SAMPLES=10000  # Adjust based on time constraints
export NUM_EVAL_SAMPLES=500
export BATCH_SIZE=32
export EPOCHS=12

# Run the comprehensive experiment suite
python telepathy/run_comprehensive_revision.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --output_dir "$OUTPUT_BASE_DIR" \
    --max_samples $MAX_SAMPLES \
    --num_eval_samples $NUM_EVAL_SAMPLES \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --use_wandb \
    --wandb_project "latentwire-revision" \
    --parallel_gpus 4 \
    --save_checkpoints \
    --run_baselines \
    --run_ablations \
    --generate_plots \
    2>&1 | tee "${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}_full.log"

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================================="
    echo "Experiment completed successfully!"
    echo "=============================================================="

    # Archive results
    echo "Archiving results..."
    tar -czf "${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}_results.tar.gz" \
        "${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}" \
        --exclude="*.pth" \
        --exclude="*.safetensors"

    # Generate summary report
    echo "Generating summary report..."
    python -c "
import json
import glob
from pathlib import Path

output_dir = Path('$OUTPUT_BASE_DIR/$EXPERIMENT_NAME')
results_files = list(output_dir.glob('*/results.json'))

summary = {
    'job_id': '$SLURM_JOB_ID',
    'experiment_name': '$EXPERIMENT_NAME',
    'num_experiments': len(results_files),
    'completed_at': '$(date)',
}

# Collect best results
best_f1 = 0
best_config = None
for result_file in results_files:
    with open(result_file) as f:
        result = json.load(f)
        if result.get('avg_f1', 0) > best_f1:
            best_f1 = result['avg_f1']
            best_config = str(result_file.parent.name)

summary['best_f1'] = best_f1
summary['best_config'] = best_config

with open(output_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Summary saved to {output_dir}/summary.json')
print(f'Best F1: {best_f1:.4f} from {best_config}')
"

else
    echo ""
    echo "=============================================================="
    echo "WARNING: Experiment may have encountered errors!"
    echo "Check logs at: ${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}_full.log"
    echo "=============================================================="
fi

# Update REPORT.md with results
echo ""
echo "=============================================================="
echo "Updating Documentation"
echo "=============================================================="
python -c "
from datetime import datetime
import json
from pathlib import Path

# Add entry to REPORT.md
report_path = Path('telepathy/REPORT.md')
if report_path.exists():
    with open(report_path, 'a') as f:
        f.write(f'''

## Comprehensive Revision Suite - {datetime.now().strftime('%Y-%m-%d %H:%M')}
**SLURM Job ID**: $SLURM_JOB_ID
**Experiment**: $EXPERIMENT_NAME

### Configuration
- Compression levels: 8, 16, 32, 64, 128
- Loss functions: CE, CE+KD, CE+Contrastive
- Models: Llama-3.1-8B, Qwen2.5-7B
- Datasets: SQuAD, HotpotQA, ArXiv
- Training samples: $MAX_SAMPLES
- Evaluation samples: $NUM_EVAL_SAMPLES
- Epochs: $EPOCHS
- Batch size: $BATCH_SIZE

### Status
Completed at $(date)
Results archived at: $OUTPUT_BASE_DIR/$EXPERIMENT_NAME

''')
    print('Updated REPORT.md with experiment details')
"

# Push results back to git
echo ""
echo "=============================================================="
echo "Pushing Results to Git"
echo "=============================================================="
git add -A
git commit -m "results: comprehensive revision experiments (SLURM job $SLURM_JOB_ID)

Ran comprehensive suite testing:
- Multiple compression levels (8-128 tokens)
- Different loss configurations
- Cross-model evaluation
- Baseline comparisons

Results saved to: $OUTPUT_BASE_DIR/$EXPERIMENT_NAME

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true

git push || true

echo ""
echo "=============================================================="
echo "Job Summary"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date -d @$SLURM_JOB_START_TIME)"
echo "End time: $(date)"
echo "Total runtime: $(($(date +%s) - SLURM_JOB_START_TIME)) seconds"
echo "Results directory: $OUTPUT_BASE_DIR/$EXPERIMENT_NAME"
echo "Log file: /projects/m000066/sujinesh/LatentWire/runs/revision_${SLURM_JOB_ID}.log"
echo "=============================================================="