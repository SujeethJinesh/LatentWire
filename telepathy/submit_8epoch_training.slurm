#!/bin/bash
#SBATCH --job-name=8epoch_training
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/8epoch_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/8epoch_%j.err

# =============================================================================
# 8-Epoch Training with Per-Epoch Evaluation on HPC
# =============================================================================
# This script runs 8 epochs of LatentWire training with comprehensive
# evaluation after each epoch. Designed for paper-quality results.
#
# Submit with: sbatch telepathy/submit_8epoch_training.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
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
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Configuration for 8-epoch training
EXPERIMENT_NAME="8epoch_pereval_job${SLURM_JOB_ID}"
OUTPUT_DIR="runs/$EXPERIMENT_NAME"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/epoch_evals"
mkdir -p "$OUTPUT_DIR/figures"
mkdir -p runs

# Pull latest code
echo "Pulling latest code..."
git pull

# Log configuration
echo "=============================================================="
echo "Training Configuration"
echo "=============================================================="
echo "Experiment name: $EXPERIMENT_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Total epochs: 8"
echo "Models: Llama 3.1 8B + Qwen 2.5 7B"
echo "Dataset: SQuAD"
echo "GPUs available: $(nvidia-smi -L | wc -l)"
echo "=============================================================="

# Run the 8-epoch training script
echo "Starting 8-epoch training with per-epoch evaluation..."
bash scripts/run_8epoch_training.sh

# After training completes, generate additional analysis
echo "=============================================================="
echo "Generating comprehensive analysis..."
echo "=============================================================="

python -c "
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

output_dir = Path('$OUTPUT_DIR')

# Load training summary
summary_file = output_dir / 'training_summary.json'
if summary_file.exists():
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    print('Training Summary:')
    print(f'  Total epochs: {summary[\"total_epochs\"]}')
    print(f'  Final latent F1: {summary[\"final_metrics\"][\"latent_f1\"]:.4f}')
    print(f'  Final text F1: {summary[\"final_metrics\"][\"text_f1\"]:.4f}')
    print(f'  F1 improvement: {summary[\"improvement\"][\"f1_absolute\"]:.4f} ({summary[\"improvement\"][\"f1_relative_pct\"]:.1f}%)')
    print(f'  Final first-token accuracy: {summary[\"final_metrics\"][\"first_tok_acc\"]:.4f}')

    # Calculate convergence metrics
    epoch_metrics = summary['epoch_metrics']
    if len(epoch_metrics['latent_f1']) > 3:
        # Check if converged (small change in last 3 epochs)
        recent_f1 = epoch_metrics['latent_f1'][-3:]
        f1_std = np.std(recent_f1)
        converged = f1_std < 0.01
        print(f'  Convergence: {\"Yes\" if converged else \"No\"} (std of last 3 epochs: {f1_std:.4f})')

    # Save LaTeX-ready table
    latex_table = []
    latex_table.append('\\\\begin{table}[h]')
    latex_table.append('\\\\centering')
    latex_table.append('\\\\begin{tabular}{lcccc}')
    latex_table.append('\\\\toprule')
    latex_table.append('Epoch & Latent F1 & Text F1 & First-Tok Acc & NLL/token \\\\\\\\')
    latex_table.append('\\\\midrule')

    for i, epoch in enumerate(epoch_metrics['epochs']):
        latex_table.append(f'{epoch} & {epoch_metrics[\"latent_f1\"][i]:.3f} & {epoch_metrics[\"text_f1\"][i]:.3f} & {epoch_metrics[\"first_tok_acc\"][i]:.3f} & {epoch_metrics[\"latent_nll\"][i]:.3f} \\\\\\\\')

    latex_table.append('\\\\bottomrule')
    latex_table.append('\\\\end{tabular}')
    latex_table.append('\\\\caption{8-Epoch Training Progression}')
    latex_table.append('\\\\label{tab:8epoch_training}')
    latex_table.append('\\\\end{table}')

    with open(output_dir / 'results_table.tex', 'w') as f:
        f.write('\\n'.join(latex_table))

    print(f'  LaTeX table saved to {output_dir}/results_table.tex')
"

# Copy key results to a central location for easy access
echo "Copying key results..."
RESULTS_SUMMARY="runs/8epoch_summary_${SLURM_JOB_ID}.json"
if [ -f "$OUTPUT_DIR/training_summary.json" ]; then
    cp "$OUTPUT_DIR/training_summary.json" "$RESULTS_SUMMARY"
    echo "Results summary copied to $RESULTS_SUMMARY"
fi

# Push results back to git
echo "=============================================================="
echo "Pushing results to git..."
echo "=============================================================="

git add -A
git commit -m "results: 8-epoch training with per-epoch evaluation (SLURM job $SLURM_JOB_ID)

Experiment: $EXPERIMENT_NAME
- 8 epochs with evaluation after each epoch
- Full checkpoints and metrics saved
- Training progression plots generated
- LaTeX-ready results table created

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
git push || true

echo "=============================================================="
echo "Job completed at $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=============================================================="
echo "Key outputs:"
echo "  - Checkpoints: $OUTPUT_DIR/epoch{1-8}/"
echo "  - Evaluations: $OUTPUT_DIR/epoch_evals/"
echo "  - Plots: $OUTPUT_DIR/figures/"
echo "  - Summary: $OUTPUT_DIR/training_summary.json"
echo "  - LaTeX table: $OUTPUT_DIR/results_table.tex"
echo "=============================================================="