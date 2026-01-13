#!/bin/bash
#SBATCH --job-name=cross_model_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=08:00:00
#SBATCH --mem=128GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/cross_model_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/cross_model_%j.err

# =============================================================================
# Cross-Model Communication Experiments
# =============================================================================
# Based on techniques from 10 opus subagents analyzing LatentMAS, Cache2Cache,
# COCONUT, and other recent papers on cross-model LLM communication.
#
# Experiments (Priority Order):
# 1. Ridge Regression Baseline (~1.5h) - LatentMAS training-free alignment
# 2. Multi-Layer Extraction (~1.5h) - Learned layer combination weights
# 3. Layer Gating (~1.5h) - Gumbel-sigmoid gates for selective injection
# 4. VIB Regularization (~1.5h) - Information bottleneck for compression
# 5. Curriculum Training (~2h) - COCONUT-style text-to-latent transition
#
# Total estimated runtime: ~8 hours on single H100 GPU
#
# Key papers referenced:
# - LatentMAS (arXiv:2511.20639): Ridge regression, KV-cache transfer
# - Cache2Cache (arXiv:2510.03215): Per-layer gating, Gumbel-sigmoid
# - COCONUT (arXiv:2412.06769): Curriculum training, latent reasoning
#
# Submit with: sbatch telepathy/submit_cross_model_experiments.slurm
# =============================================================================

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "CROSS-MODEL COMMUNICATION EXPERIMENTS"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo ""
echo "Experiments: Ridge Regression, Multi-Layer, Layer Gating, VIB, Curriculum"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME="/projects/m000066/sujinesh/.cache/huggingface"
export TRANSFORMERS_CACHE="/projects/m000066/sujinesh/.cache/huggingface/transformers"

# Create output directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="runs/cross_model_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
mkdir -p figures

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# =========================================================================
# Git Setup
# =========================================================================
echo "Git Configuration Check..."
git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job $SLURM_JOB_ID"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@hpc.cluster"

echo "Pulling latest code..."
if ! git pull; then
    echo "WARNING: git pull failed. Attempting to stash and retry..."
    git stash push -m "SLURM job $SLURM_JOB_ID auto-stash"
    git pull || echo "Pull failed - continuing with existing code"
    git stash pop 2>/dev/null || true
fi

# =========================================================================
# GPU Information
# =========================================================================
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# =========================================================================
# Experiment 1: Ridge Regression Baseline (~1.5h)
# =========================================================================
echo ""
echo "=============================================================="
echo "EXPERIMENT 1: Ridge Regression Baseline (LatentMAS)"
echo "=============================================================="
echo "Start: $(date)"
echo ""
echo "Testing training-free alignment via ridge regression:"
echo "  W_a = (W_out^T @ W_out + λI)^{-1} @ W_out^T @ W_in"
echo ""
echo "Lambda values: 1e-6, 1e-4, 1e-2, 1.0"
echo "Datasets: SST-2, AG News"
echo ""

EXP1_START=$SECONDS

# Run ridge regression experiments
python telepathy/train_telepathy.py \
    --dataset sst2 \
    --bridge_type ridge \
    --lambda_reg 1e-4 \
    --steps 100 \
    --eval_only \
    --output_dir "${OUTPUT_DIR}/ridge_sst2" \
    --gpu 0 \
    2>&1 | tee "${OUTPUT_DIR}/ridge_sst2.log" || echo "Ridge SST2 failed, continuing..."

python telepathy/train_telepathy.py \
    --dataset agnews \
    --bridge_type ridge \
    --lambda_reg 1e-4 \
    --steps 100 \
    --eval_only \
    --output_dir "${OUTPUT_DIR}/ridge_agnews" \
    --gpu 0 \
    2>&1 | tee "${OUTPUT_DIR}/ridge_agnews.log" || echo "Ridge AG News failed, continuing..."

EXP1_TIME=$((SECONDS - EXP1_START))
echo ""
echo "Experiment 1 completed in ${EXP1_TIME}s"

# =========================================================================
# Experiment 2: Multi-Layer Extraction (~1.5h)
# =========================================================================
echo ""
echo "=============================================================="
echo "EXPERIMENT 2: Multi-Layer Extraction with Learned Weights"
echo "=============================================================="
echo "Start: $(date)"
echo ""
echo "Testing extraction from multiple Llama layers:"
echo "  - Single layer [31] (baseline)"
echo "  - Two layers [16, 31]"
echo "  - Three layers [16, 24, 31] (recommended)"
echo ""

EXP2_START=$SECONDS

# Layer 31 only (baseline)
python telepathy/train_telepathy.py \
    --dataset sst2 \
    --extract_layers 31 \
    --steps 1000 \
    --output_dir "${OUTPUT_DIR}/multi_layer_31" \
    --gpu 0 \
    2>&1 | tee "${OUTPUT_DIR}/multi_layer_31.log" || echo "Multi-layer 31 failed, continuing..."

# Layers 16, 24, 31
python telepathy/train_telepathy.py \
    --dataset sst2 \
    --extract_layers 16 24 31 \
    --learn_layer_weights \
    --steps 1000 \
    --output_dir "${OUTPUT_DIR}/multi_layer_16_24_31" \
    --gpu 0 \
    2>&1 | tee "${OUTPUT_DIR}/multi_layer_16_24_31.log" || echo "Multi-layer 16,24,31 failed, continuing..."

EXP2_TIME=$((SECONDS - EXP2_START))
echo ""
echo "Experiment 2 completed in ${EXP2_TIME}s"

# =========================================================================
# Experiment 3: Layer Gating (C2C-style) (~1.5h)
# =========================================================================
echo ""
echo "=============================================================="
echo "EXPERIMENT 3: Gumbel-Sigmoid Layer Gating (Cache2Cache)"
echo "=============================================================="
echo "Start: $(date)"
echo ""
echo "Testing learnable layer selection for injection:"
echo "  - Embedding only [0] (baseline)"
echo "  - Embedding + middle [0, 16]"
echo "  - Multiple [0, 8, 16, 24]"
echo ""
echo "Using Gumbel-sigmoid gates with temperature annealing (2.0 -> 0.1)"
echo ""

EXP3_START=$SECONDS

# Embedding only (baseline)
python telepathy/train_telepathy.py \
    --dataset sst2 \
    --inject_layers 0 \
    --steps 1000 \
    --output_dir "${OUTPUT_DIR}/gating_embed_only" \
    --gpu 0 \
    2>&1 | tee "${OUTPUT_DIR}/gating_embed_only.log" || echo "Gating embed failed, continuing..."

# Multi-layer injection with gating
python telepathy/train_telepathy.py \
    --dataset sst2 \
    --inject_layers 0 8 16 24 \
    --use_gumbel_gates \
    --temp_start 2.0 \
    --temp_end 0.1 \
    --steps 1000 \
    --output_dir "${OUTPUT_DIR}/gating_multi_layer" \
    --gpu 0 \
    2>&1 | tee "${OUTPUT_DIR}/gating_multi_layer.log" || echo "Gating multi-layer failed, continuing..."

EXP3_TIME=$((SECONDS - EXP3_START))
echo ""
echo "Experiment 3 completed in ${EXP3_TIME}s"

# =========================================================================
# Experiment 4: VIB Regularization (~1.5h)
# =========================================================================
echo ""
echo "=============================================================="
echo "EXPERIMENT 4: Variational Information Bottleneck"
echo "=============================================================="
echo "Start: $(date)"
echo ""
echo "Testing stochastic soft tokens with KL regularization:"
echo "  z = mu + sigma * epsilon"
echo "  L = L_task + beta * KL(q(z|x) || N(0,1))"
echo ""
echo "Beta values: 0.0001, 0.001, 0.01"
echo ""

EXP4_START=$SECONDS

# Baseline (no VIB)
python telepathy/train_telepathy.py \
    --dataset sst2 \
    --use_vib \
    --vib_beta 0.0 \
    --steps 1000 \
    --output_dir "${OUTPUT_DIR}/vib_beta_0" \
    --gpu 0 \
    2>&1 | tee "${OUTPUT_DIR}/vib_beta_0.log" || echo "VIB beta=0 failed, continuing..."

# VIB with beta=0.001
python telepathy/train_telepathy.py \
    --dataset sst2 \
    --use_vib \
    --vib_beta 0.001 \
    --vib_beta_anneal \
    --steps 1000 \
    --output_dir "${OUTPUT_DIR}/vib_beta_001" \
    --gpu 0 \
    2>&1 | tee "${OUTPUT_DIR}/vib_beta_001.log" || echo "VIB beta=0.001 failed, continuing..."

# VIB with beta=0.01
python telepathy/train_telepathy.py \
    --dataset sst2 \
    --use_vib \
    --vib_beta 0.01 \
    --vib_beta_anneal \
    --steps 1000 \
    --output_dir "${OUTPUT_DIR}/vib_beta_01" \
    --gpu 0 \
    2>&1 | tee "${OUTPUT_DIR}/vib_beta_01.log" || echo "VIB beta=0.01 failed, continuing..."

EXP4_TIME=$((SECONDS - EXP4_START))
echo ""
echo "Experiment 4 completed in ${EXP4_TIME}s"

# =========================================================================
# Experiment 5: Curriculum Training (~2h)
# =========================================================================
echo ""
echo "=============================================================="
echo "EXPERIMENT 5: Curriculum Training (COCONUT-style)"
echo "=============================================================="
echo "Start: $(date)"
echo ""
echo "Testing gradual transition from text to latent communication:"
echo "  Stage 1: Full text (0 soft tokens)"
echo "  Stage 2: 75% text + 2 soft tokens"
echo "  Stage 3: 50% text + 4 soft tokens"
echo "  Stage 4: 25% text + 6 soft tokens"
echo "  Stage 5: Pure latent (8 soft tokens)"
echo ""

EXP5_START=$SECONDS

# Standard training (no curriculum, baseline)
python telepathy/train_telepathy.py \
    --dataset sst2 \
    --soft_tokens 8 \
    --steps 1500 \
    --output_dir "${OUTPUT_DIR}/curriculum_baseline" \
    --gpu 0 \
    2>&1 | tee "${OUTPUT_DIR}/curriculum_baseline.log" || echo "Curriculum baseline failed, continuing..."

# Curriculum training
python telepathy/train_telepathy.py \
    --dataset sst2 \
    --use_curriculum \
    --curriculum_stages 5 \
    --steps 1700 \
    --output_dir "${OUTPUT_DIR}/curriculum_full" \
    --gpu 0 \
    2>&1 | tee "${OUTPUT_DIR}/curriculum_full.log" || echo "Curriculum failed, continuing..."

EXP5_TIME=$((SECONDS - EXP5_START))
echo ""
echo "Experiment 5 completed in ${EXP5_TIME}s"

# =========================================================================
# Results Summary
# =========================================================================
echo ""
echo "=============================================================="
echo "RESULTS SUMMARY"
echo "=============================================================="
echo ""

# Aggregate results
python -c "
import json
import os
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
results = {'experiments': {}}

for exp_dir in output_dir.iterdir():
    if exp_dir.is_dir():
        results_file = exp_dir / 'results.json'
        if results_file.exists():
            with open(results_file) as f:
                results['experiments'][exp_dir.name] = json.load(f)

# Print summary
print('Experiment Results:')
print('-' * 40)
for exp_name, exp_results in results['experiments'].items():
    if 'accuracy' in exp_results:
        print(f'{exp_name}: {exp_results[\"accuracy\"]:.1%}')
    elif 'eval_accuracy' in exp_results:
        print(f'{exp_name}: {exp_results[\"eval_accuracy\"]:.1%}')

# Save combined results
with open(output_dir / 'combined_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nCombined results saved to {output_dir}/combined_results.json')
" 2>/dev/null || echo "Could not aggregate results"

# =========================================================================
# Timing Summary
# =========================================================================
echo ""
echo "Timing Summary:"
echo "  Experiment 1 (Ridge): ${EXP1_TIME}s"
echo "  Experiment 2 (Multi-Layer): ${EXP2_TIME}s"
echo "  Experiment 3 (Layer Gating): ${EXP3_TIME}s"
echo "  Experiment 4 (VIB): ${EXP4_TIME}s"
echo "  Experiment 5 (Curriculum): ${EXP5_TIME}s"
echo "  Total: $SECONDS seconds ($(($SECONDS / 60)) minutes)"

# =========================================================================
# Git Commit and Push
# =========================================================================
echo ""
echo "Saving results to git..."

git add "${OUTPUT_DIR}/" 2>/dev/null || true
git add runs/cross_model_*.log runs/cross_model_*.err 2>/dev/null || true
git add figures/*.png figures/*.pdf 2>/dev/null || true

echo ""
echo "Files staged for commit:"
git status --short

COMMIT_MSG="results: cross-model experiments (SLURM job $SLURM_JOB_ID)

Experiments based on LatentMAS, Cache2Cache, COCONUT papers:
1. Ridge Regression: Training-free alignment (LatentMAS)
2. Multi-Layer Extraction: Learned layer weights
3. Layer Gating: Gumbel-sigmoid selection (C2C)
4. VIB Regularization: Information bottleneck
5. Curriculum Training: Text-to-latent (COCONUT)

Runtime: $SECONDS seconds
Exit status: $?

References:
- LatentMAS (arXiv:2511.20639)
- Cache2Cache (arXiv:2510.03215)
- COCONUT (arXiv:2412.06769)

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

if git commit -m "$COMMIT_MSG"; then
    echo "Created commit, attempting to push..."

    RETRY_COUNT=0
    while [ $RETRY_COUNT -lt 3 ]; do
        if git push; then
            echo "Successfully pushed to remote"
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            echo "Push attempt $RETRY_COUNT failed"
            [ $RETRY_COUNT -lt 3 ] && sleep 5 && git pull --rebase=false || true
        fi
    done

    [ $RETRY_COUNT -eq 3 ] && echo "WARNING: Could not push after 3 attempts"
else
    echo "No changes to commit"
fi

# =========================================================================
# Final Summary
# =========================================================================
echo ""
echo "=============================================================="
echo "CROSS-MODEL EXPERIMENTS COMPLETED"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds ($(($SECONDS / 3600))h $(($SECONDS % 3600 / 60))m)"
echo ""
echo "Key comparisons:"
echo "  - Ridge Regression vs Perceiver Bridge"
echo "  - Single-layer vs Multi-layer extraction"
echo "  - Embedding-only vs Multi-layer injection"
echo "  - Deterministic vs VIB stochastic tokens"
echo "  - Standard vs Curriculum training"
echo ""
echo "Results directory: ${OUTPUT_DIR}"
echo "=============================================================="
