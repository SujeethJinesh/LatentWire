#!/bin/bash
#SBATCH --job-name=enhanced_paper_eval
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/enhanced_paper_eval_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/enhanced_paper_eval_%j.err

# =============================================================================
# Enhanced Paper Evaluation - 12 Hour Comprehensive Suite
# =============================================================================
# This script runs the enhanced evaluation with:
# - 5 seeds (42, 123, 456, 789, 1011) for statistical power
# - 4 token configs (8, 16, 24, 32) for granular ablation
# - 3 classification datasets (SST-2, AG News, TREC)
# - 3 reasoning benchmarks (BoolQ, ARC-Easy, GSM8K)
# - Improved text-relay baseline implementation
#
# Time budget: 12 hours on 1 H100 GPU
#
# Submit with: sbatch telepathy/submit_enhanced_paper_eval.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "Enhanced Paper Evaluation"
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
mkdir -p runs/enhanced_paper_eval
mkdir -p runs/enhanced_paper_eval/bridges
mkdir -p runs/enhanced_paper_eval/checkpoints

# Pull latest code
echo "Pulling latest code..."
git pull

# Check GPU availability
echo ""
echo "Checking GPU..."
nvidia-smi

# Run enhanced evaluation
echo ""
echo "Starting enhanced evaluation suite..."
echo "This will take approximately 12 hours to complete."
echo ""

python telepathy/run_enhanced_paper_evaluation.py \
    --output_dir runs/enhanced_paper_eval \
    --gpu 0

EXIT_CODE=$?

# Check if evaluation completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================================="
    echo "Enhanced evaluation completed successfully!"
    echo "=============================================================="

    # Display summary if results file exists
    if [ -f "runs/enhanced_paper_eval/enhanced_evaluation_results.json" ]; then
        echo ""
        echo "Results summary:"
        python -c "
import json
with open('runs/enhanced_paper_eval/enhanced_evaluation_results.json') as f:
    results = json.load(f)

print(f\"Total time: {results['metadata']['total_time_hours']:.2f} hours\")
print(f\"Seeds: {results['metadata']['seeds']}\")
print(f\"Token configs: {results['metadata']['token_configs']}\")
print()

# Classification results
print('Classification Results (Bridge):')
for dataset in results['classification']['aggregated']['bridge'].keys():
    print(f'  {dataset}:')
    for tokens, stats in results['classification']['aggregated']['bridge'][dataset].items():
        print(f\"    {tokens} tokens: {stats['mean']:.3f} ± {stats['std']:.3f}\")

# Reasoning results
print()
print('Reasoning Benchmarks (Llama):')
for benchmark, stats in results['reasoning']['aggregated']['llama'].items():
    print(f\"  {benchmark}: {stats['mean']:.3f} ± {stats['std']:.3f}\")
"
    fi
else
    echo ""
    echo "=============================================================="
    echo "WARNING: Enhanced evaluation failed with exit code $EXIT_CODE"
    echo "=============================================================="
fi

# Push results back to git
echo ""
echo "Pushing results to git..."
git add -A
git commit -m "results: enhanced paper evaluation with 5 seeds and reasoning benchmarks (SLURM job $SLURM_JOB_ID)

Experimental configuration:
- Seeds: 5 (42, 123, 456, 789, 1011)
- Token configs: 4 (8, 16, 24, 32)
- Classification: SST-2, AG News, TREC (500 samples each)
- Reasoning: BoolQ, ARC-Easy, GSM8K (200 samples each)
- Baselines: Bridge, Text-Relay, Llama 0-shot, Mistral 0-shot

Time budget: 12 hours on 1 H100 GPU
Exit code: $EXIT_CODE

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true

git push || true

echo ""
echo "=============================================================="
echo "Job completed at $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=============================================================="
