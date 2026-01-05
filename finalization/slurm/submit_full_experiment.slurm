#!/bin/bash
#SBATCH --job-name=latentwire-full
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/full_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/full_%j.err

# =============================================================================
# Full LatentWire Experiment Suite
# =============================================================================
# Runs complete training and evaluation pipeline
# Submit with: sbatch finalization/slurm/submit_full_experiment.slurm
# =============================================================================

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "Full Experiment Suite"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================================================="

# Set up environment
export PYTHONUNBUFFERED=1  # Critical: Immediate output flushing to prevent log loss
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# Pull latest code
git pull

# Create experiment directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_DIR="runs/full_experiment_${TIMESTAMP}"
mkdir -p "$EXP_DIR"

# ============================================================================
# Phase 1: Training with multiple configurations
# ============================================================================

echo -e "\n========== Phase 1: Training =========="

# Configuration 1: Standard
python latentwire/train.py \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --samples 87599 \
    --epochs 24 \
    --batch_size 64 \
    --latent_len 32 \
    --d_z 256 \
    --encoder_type byte \
    --dataset squad \
    --sequential_models \
    --warm_anchor_text "Answer: " \
    --first_token_ce_weight 0.5 \
    --output_dir "$EXP_DIR/standard" \
    2>&1 | tee "$EXP_DIR/train_standard.log"

# Configuration 2: High compression
python latentwire/train.py \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --samples 87599 \
    --epochs 24 \
    --batch_size 64 \
    --latent_len 16 \
    --d_z 128 \
    --encoder_type byte \
    --dataset squad \
    --sequential_models \
    --warm_anchor_text "Answer: " \
    --first_token_ce_weight 0.5 \
    --output_dir "$EXP_DIR/high_compression" \
    2>&1 | tee "$EXP_DIR/train_high_compression.log"

# ============================================================================
# Phase 2: Evaluation
# ============================================================================

echo -e "\n========== Phase 2: Evaluation =========="

# Evaluate standard model
python latentwire/eval.py \
    --ckpt "$EXP_DIR/standard/epoch23" \
    --samples 1000 \
    --max_new_tokens 12 \
    --dataset squad \
    --sequential_eval \
    --fresh_eval \
    --calibration embed_rms \
    --latent_anchor_mode text \
    --latent_anchor_text "Answer: " \
    --append_bos_after_prefix yes \
    --output_dir "$EXP_DIR/eval_standard" \
    2>&1 | tee "$EXP_DIR/eval_standard.log"

# Evaluate high compression model
python latentwire/eval.py \
    --ckpt "$EXP_DIR/high_compression/epoch23" \
    --samples 1000 \
    --max_new_tokens 12 \
    --dataset squad \
    --sequential_eval \
    --fresh_eval \
    --calibration embed_rms \
    --latent_anchor_mode text \
    --latent_anchor_text "Answer: " \
    --append_bos_after_prefix yes \
    --output_dir "$EXP_DIR/eval_high_compression" \
    2>&1 | tee "$EXP_DIR/eval_high_compression.log"

# ============================================================================
# Phase 3: Analysis
# ============================================================================

echo -e "\n========== Phase 3: Analysis =========="

python finalization/analysis/aggregate_results.py \
    --experiment_dir "$EXP_DIR" \
    --output "$EXP_DIR/results_summary.json" \
    2>&1 | tee "$EXP_DIR/analysis.log"

# ============================================================================
# Phase 4: Generate plots
# ============================================================================

echo -e "\n========== Phase 4: Visualization =========="

python finalization/analysis/plot_results.py \
    --results "$EXP_DIR/results_summary.json" \
    --output_dir "$EXP_DIR/figures" \
    2>&1 | tee "$EXP_DIR/plotting.log"

# ============================================================================
# Push results to git
# ============================================================================

echo -e "\n========== Pushing results to git =========="

git add -A
git commit -m "results: Full experiment suite (SLURM job $SLURM_JOB_ID)

Configurations tested:
- Standard: latent_len=32, d_z=256
- High compression: latent_len=16, d_z=128

Results saved to: $EXP_DIR

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true

git push || true

echo "=============================================================="
echo "Experiment completed at $(date)"
echo "Results directory: $EXP_DIR"
echo "=============================================================="export PYTHONUNBUFFERED=1
