#!/bin/bash
#SBATCH --job-name=unified_linear_probe
#SBATCH --nodes=1
#SBATCH --gpus=1                                    # Linear probe is CPU-heavy, 1 GPU sufficient
#SBATCH --account=marlowe-m000066                   # REQUIRED - correct account
#SBATCH --partition=preempt                         # REQUIRED - correct partition
#SBATCH --time=8:00:00                             # 8 hours should be enough
#SBATCH --mem=128GB                                # Need memory for feature extraction
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/unified_linear_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/unified_linear_%j.err

# =============================================================================
# Unified Comparison with Linear Probe Baseline
# =============================================================================
# This runs ALL baselines including the new Linear Probe baseline:
# 1. Bridge (Llama→Mistral) - Main method
# 2. Linear Probe (sklearn LogisticRegression) - CRITICAL reviewer baseline
# 3. Prompt-Tuning (Mistral only) - Proves sender is essential
# 4. Text-Relay (Llama summarizes → Mistral classifies) - Latency baseline
# 5. Few-shot Prompting (5-shot) - In-context learning baseline
# 6. Zero-shot baselines (Llama, Mistral direct)
# =============================================================================
# Submit with: sbatch telepathy/submit_unified_with_linear_probe.slurm
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

# Create runs directory if needed
mkdir -p runs/unified_linear figures

# Pull latest code
echo "Pulling latest code..."
git pull

# Run unified comparison with linear probe
echo "Starting unified comparison experiment with Linear Probe..."
echo "This includes: Bridge, Linear Probe, Prompt-Tuning, Zero-shot, Few-shot"
echo ""

python telepathy/run_unified_comparison.py \
    --datasets sst2 agnews trec \
    --seeds 42 123 456 \
    --soft_tokens 8 \
    --train_steps 2000 \
    --eval_samples 200 \
    --fewshot_shots 5 \
    --output_dir runs/unified_linear \
    --skip_text_relay  # Skip text-relay to save time (it's slow)

# Push results back to git
echo ""
echo "Pushing results to git..."
git add -A
git commit -m "results: unified comparison with linear probe baseline (SLURM job $SLURM_JOB_ID)

Includes Linear Probe baseline using sklearn LogisticRegression
Datasets: SST-2, AG News, TREC
Seeds: 42, 123, 456

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
git push || true

echo "=============================================================="
echo "Job completed at $(date)"
echo "Results saved to: runs/unified_linear/"
echo "=============================================================="