#!/bin/bash
#SBATCH --job-name=latentwire_final
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/final_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/final_%j.err

# =============================================================================
# MASTER SUBMISSION SCRIPT FOR LATENTWIRE PROJECT
# =============================================================================
# This script runs the complete experimental pipeline including:
#   - Training LatentWire models
#   - Multi-dataset evaluation (SST-2, AG News, GSM8K, TREC)
#   - Baseline comparisons (LLMLingua, Linear Probe)
#   - Statistical testing and visualization
# =============================================================================
# Submit with: sbatch finalization/SUBMIT.slurm
# Monitor with: squeue -u $USER
# Watch logs: tail -f /projects/m000066/sujinesh/LatentWire/runs/final_*.log
# Cancel with: scancel <job_id>
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION SECTION
# -----------------------------------------------------------------------------
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
HF_CACHE="/projects/m000066/sujinesh/.cache/huggingface"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="final_${TIMESTAMP}"

# Training hyperparameters (can override with env vars)
SAMPLES="${SAMPLES:-10000}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LATENT_LEN="${LATENT_LEN:-32}"
D_Z="${D_Z:-256}"

# -----------------------------------------------------------------------------
# JOB INITIALIZATION
# -----------------------------------------------------------------------------
cd "$WORK_DIR" || exit 1

echo "=============================================================="
echo "LATENTWIRE MASTER EXPERIMENT PIPELINE"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "Run name: $RUN_NAME"
echo "=============================================================="

# -----------------------------------------------------------------------------
# ENVIRONMENT SETUP
# -----------------------------------------------------------------------------
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_CACHE}"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN

# Create necessary directories
mkdir -p "runs/${RUN_NAME}"/{checkpoint,evaluations,baselines,analysis}
mkdir -p "figures/${RUN_NAME}"

# -----------------------------------------------------------------------------
# GIT CONFIGURATION
# -----------------------------------------------------------------------------
echo ""
echo "[Git Setup]"
echo "Configuring git identity..."
git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job $SLURM_JOB_ID"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@hpc.cluster"

echo "Pulling latest code..."
if ! git pull; then
    echo "WARNING: git pull failed. Attempting to stash and retry..."
    git stash push -m "SLURM job $SLURM_JOB_ID auto-stash"
    git pull || echo "Pull failed - continuing with existing code"
    git stash pop 2>/dev/null || true
fi

# Copy evaluation scripts if they exist in finalization/
if [ -d "finalization" ]; then
    echo "Copying finalization scripts..."
    cp -f finalization/*.py . 2>/dev/null || true
    cp -f finalization/telepathy/*.py telepathy/ 2>/dev/null || true
fi

# -----------------------------------------------------------------------------
# PHASE 1: TRAIN LATENTWIRE MODEL
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "PHASE 1: TRAINING LATENTWIRE MODEL"
echo "=============================================================="
echo "Samples: ${SAMPLES}, Epochs: ${EPOCHS}, Batch: ${BATCH_SIZE}"
echo "Latent: ${LATENT_LEN}x${D_Z}"

TRAIN_LOG="runs/${RUN_NAME}/train.log"
{
    python latentwire/train.py \
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
        --samples "${SAMPLES}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --latent_len "${LATENT_LEN}" \
        --d_z "${D_Z}" \
        --encoder_type byte \
        --dataset squad \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --output_dir "runs/${RUN_NAME}/checkpoint"
} 2>&1 | tee "$TRAIN_LOG"

# Check if training succeeded
if [ ! -d "runs/${RUN_NAME}/checkpoint/epoch$((EPOCHS-1))" ]; then
    echo "ERROR: Training failed - checkpoint not found"
    exit 1
fi

echo "Training complete. Using checkpoint: runs/${RUN_NAME}/checkpoint/epoch$((EPOCHS-1))"

# -----------------------------------------------------------------------------
# PHASE 2: MULTI-DATASET EVALUATION
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "PHASE 2: MULTI-DATASET EVALUATION"
echo "=============================================================="

CHECKPOINT="runs/${RUN_NAME}/checkpoint/epoch$((EPOCHS-1))"

# SST-2 Sentiment Classification
if [ -f "latentwire/eval_sst2.py" ]; then
    echo "[SST-2] Running sentiment classification evaluation..."
    SST2_LOG="runs/${RUN_NAME}/evaluations/sst2.log"
    {
        python latentwire/eval_sst2.py \
            --checkpoint_path "${CHECKPOINT}" \
            --output_dir "runs/${RUN_NAME}/evaluations/sst2" \
            --num_samples 500
    } 2>&1 | tee "$SST2_LOG"
else
    echo "[SST-2] Skipping - eval_sst2.py not found"
fi

# AG News Topic Classification
if [ -f "latentwire/eval_agnews.py" ]; then
    echo "[AG News] Running topic classification evaluation..."
    AGNEWS_LOG="runs/${RUN_NAME}/evaluations/agnews.log"
    {
        python latentwire/eval_agnews.py \
            --checkpoint_path "${CHECKPOINT}" \
            --output_dir "runs/${RUN_NAME}/evaluations/agnews" \
            --num_samples 500
    } 2>&1 | tee "$AGNEWS_LOG"
else
    echo "[AG News] Skipping - eval_agnews.py not found"
fi

# GSM8K Math Reasoning
if [ -f "latentwire/gsm8k_eval.py" ]; then
    echo "[GSM8K] Running math reasoning evaluation..."
    GSM8K_LOG="runs/${RUN_NAME}/evaluations/gsm8k.log"
    {
        python latentwire/gsm8k_eval.py \
            --checkpoint_path "${CHECKPOINT}" \
            --output_dir "runs/${RUN_NAME}/evaluations/gsm8k" \
            --num_samples 200
    } 2>&1 | tee "$GSM8K_LOG"
else
    echo "[GSM8K] Skipping - gsm8k_eval.py not found"
fi

# TREC Question Classification
if [ -f "telepathy/eval_telepathy_trec.py" ]; then
    echo "[TREC] Running question classification evaluation..."
    TREC_LOG="runs/${RUN_NAME}/evaluations/trec.log"
    {
        python telepathy/eval_telepathy_trec.py \
            --checkpoint_path "${CHECKPOINT}" \
            --output_dir "runs/${RUN_NAME}/evaluations/trec" \
            --num_samples 500
    } 2>&1 | tee "$TREC_LOG"
else
    echo "[TREC] Skipping - eval_telepathy_trec.py not found"
fi

# Standard SQuAD Evaluation
echo "[SQuAD] Running standard evaluation..."
SQUAD_LOG="runs/${RUN_NAME}/evaluations/squad.log"
{
    python latentwire/eval.py \
        --ckpt "${CHECKPOINT}" \
        --samples 500 \
        --max_new_tokens 12 \
        --dataset squad \
        --sequential_eval \
        --fresh_eval \
        --calibration embed_rms \
        --latent_anchor_mode text \
        --latent_anchor_text "Answer: " \
        --append_bos_after_prefix yes
} 2>&1 | tee "$SQUAD_LOG"

# -----------------------------------------------------------------------------
# PHASE 3: BASELINE COMPARISONS
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "PHASE 3: BASELINE COMPARISONS"
echo "=============================================================="

# LLMLingua Baseline
if [ -f "scripts/run_llmlingua_baseline.sh" ]; then
    echo "[LLMLingua] Running compression baseline..."
    LLMLINGUA_LOG="runs/${RUN_NAME}/baselines/llmlingua.log"
    {
        OUTPUT_DIR="runs/${RUN_NAME}/baselines/llmlingua" \
        bash scripts/run_llmlingua_baseline.sh
    } 2>&1 | tee "$LLMLINGUA_LOG"
else
    echo "[LLMLingua] Skipping - baseline script not found"
fi

# Linear Probe Baseline (if available)
if [ -f "telepathy/linear_probe_baseline.py" ]; then
    echo "[Linear Probe] Running baseline evaluation..."
    PROBE_LOG="runs/${RUN_NAME}/baselines/linear_probe.log"
    {
        python telepathy/linear_probe_baseline.py \
            --checkpoint_path "${CHECKPOINT}" \
            --output_dir "runs/${RUN_NAME}/baselines/linear_probe" \
            --num_samples 500
    } 2>&1 | tee "$PROBE_LOG"
else
    echo "[Linear Probe] Skipping - baseline script not found"
fi

# -----------------------------------------------------------------------------
# PHASE 4: STATISTICAL ANALYSIS
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "PHASE 4: STATISTICAL ANALYSIS"
echo "=============================================================="

if [ -f "scripts/statistical_testing.py" ]; then
    echo "Running statistical significance tests..."
    STATS_LOG="runs/${RUN_NAME}/analysis/statistics.log"

    # Find available result files
    LATENTWIRE_RESULTS=""
    BASELINE_RESULTS=""

    if [ -f "runs/${RUN_NAME}/evaluations/squad/eval_results.json" ]; then
        LATENTWIRE_RESULTS="runs/${RUN_NAME}/evaluations/squad/eval_results.json"
    elif [ -f "${CHECKPOINT}/eval_results.json" ]; then
        LATENTWIRE_RESULTS="${CHECKPOINT}/eval_results.json"
    fi

    if [ -f "runs/${RUN_NAME}/baselines/llmlingua/results.json" ]; then
        BASELINE_RESULTS="runs/${RUN_NAME}/baselines/llmlingua/results.json"
    fi

    if [ -n "$LATENTWIRE_RESULTS" ] && [ -n "$BASELINE_RESULTS" ]; then
        {
            python scripts/statistical_testing.py \
                --latentwire_results "$LATENTWIRE_RESULTS" \
                --baseline_results "$BASELINE_RESULTS" \
                --output_dir "runs/${RUN_NAME}/analysis"
        } 2>&1 | tee "$STATS_LOG"
    else
        echo "Skipping statistical tests - insufficient result files"
    fi
else
    echo "Skipping statistical analysis - script not found"
fi

# -----------------------------------------------------------------------------
# PHASE 5: VISUALIZATION AND REPORTING
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "PHASE 5: VISUALIZATION AND REPORTING"
echo "=============================================================="

# Generate summary report
python -c "
import json
import os
from datetime import datetime
import glob

run_name = '${RUN_NAME}'
base_dir = f'runs/{run_name}'

# Build report
report = []
report.append('='*70)
report.append('LATENTWIRE FINAL EXPERIMENT REPORT')
report.append('='*70)
report.append(f'Generated: {datetime.now().isoformat()}')
report.append(f'SLURM Job ID: ${SLURM_JOB_ID}')
report.append(f'Run Name: {run_name}')
report.append(f'Training: ${SAMPLES} samples, ${EPOCHS} epochs, batch ${BATCH_SIZE}')
report.append(f'Latent: ${LATENT_LEN} tokens × ${D_Z} dims')
report.append('')

# Training Summary
report.append('## TRAINING')
report.append('-'*40)
train_log = f'{base_dir}/train.log'
if os.path.exists(train_log):
    # Extract final loss from training log
    with open(train_log) as f:
        lines = f.readlines()
        for line in reversed(lines):
            if 'Loss:' in line or 'loss:' in line:
                report.append(f'Final: {line.strip()}')
                break
report.append('')

# Evaluation Results
report.append('## EVALUATION RESULTS')
report.append('-'*40)

eval_dirs = {
    'SST-2': f'{base_dir}/evaluations/sst2',
    'AG News': f'{base_dir}/evaluations/agnews',
    'GSM8K': f'{base_dir}/evaluations/gsm8k',
    'TREC': f'{base_dir}/evaluations/trec',
    'SQuAD': f'{base_dir}/evaluations/squad'
}

for dataset, eval_dir in eval_dirs.items():
    report.append(f'### {dataset}')

    # Look for results.json or eval_results.json
    result_files = glob.glob(f'{eval_dir}/*results.json')
    if result_files:
        with open(result_files[0]) as f:
            data = json.load(f)

        # Handle different result formats
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    report.append(f'  {key}:')
                    for metric, score in value.items():
                        if isinstance(score, (int, float)):
                            report.append(f'    {metric}: {score:.4f}')
                elif isinstance(value, (int, float)):
                    report.append(f'  {key}: {value:.4f}')
    else:
        report.append('  No results found')
    report.append('')

# Baseline Comparisons
report.append('## BASELINE COMPARISONS')
report.append('-'*40)

baseline_dirs = {
    'LLMLingua': f'{base_dir}/baselines/llmlingua',
    'Linear Probe': f'{base_dir}/baselines/linear_probe'
}

for baseline, baseline_dir in baseline_dirs.items():
    report.append(f'### {baseline}')
    result_files = glob.glob(f'{baseline_dir}/*results.json')
    if result_files:
        with open(result_files[0]) as f:
            data = json.load(f)
        for key, value in list(data.items())[:5]:  # Show top 5 metrics
            if isinstance(value, (int, float)):
                report.append(f'  {key}: {value:.4f}')
    else:
        report.append('  No results found')
    report.append('')

# Statistical Analysis
report.append('## STATISTICAL ANALYSIS')
report.append('-'*40)
stats_file = f'{base_dir}/analysis/significance_tests.json'
if os.path.exists(stats_file):
    with open(stats_file) as f:
        stats = json.load(f)
    for test, result in stats.items():
        if isinstance(result, dict) and 'p_value' in result:
            report.append(f'  {test}: p={result[\"p_value\"]:.4f}')
        else:
            report.append(f'  {test}: {result}')
else:
    report.append('  No statistical tests performed')
report.append('')

# Save report
report_text = '\\n'.join(report)
report_path = f'{base_dir}/FINAL_REPORT.txt'
with open(report_path, 'w') as f:
    f.write(report_text)

print(f'Report saved to: {report_path}')
print('')
print(report_text)
" || echo "Report generation failed"

# Generate performance plots
python -c "
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

run_name = '${RUN_NAME}'
base_dir = f'runs/{run_name}'
fig_dir = f'figures/{run_name}'

# Collect results from all evaluations
datasets = ['sst2', 'agnews', 'gsm8k', 'trec', 'squad']
results = {}

for dataset in datasets:
    eval_dir = f'{base_dir}/evaluations/{dataset}'
    result_files = glob.glob(f'{eval_dir}/*results.json')
    if result_files:
        with open(result_files[0]) as f:
            data = json.load(f)
            # Extract primary metric (accuracy, f1, or em)
            if 'latent' in data and isinstance(data['latent'], dict):
                for metric in ['accuracy', 'f1', 'em']:
                    if metric in data['latent']:
                        results[dataset.upper()] = data['latent'][metric]
                        break

if results:
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    datasets = list(results.keys())
    scores = list(results.values())

    bars = ax.bar(datasets, scores, color='steelblue', alpha=0.8)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('LatentWire Performance Across Datasets', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{score:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/performance_summary.png', dpi=150, bbox_inches='tight')
    print(f'Saved performance plot to {fig_dir}/performance_summary.png')

# Compression analysis plot if data available
compression_data = f'{base_dir}/analysis/compression_analysis.json'
if os.path.exists(compression_data):
    with open(compression_data) as f:
        comp_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Compression ratio vs performance
    if 'ratios' in comp_data and 'scores' in comp_data:
        ax1.plot(comp_data['ratios'], comp_data['scores'], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Compression Ratio')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Compression-Performance Trade-off')
        ax1.grid(True, alpha=0.3)

    # Latency comparison
    if 'latencies' in comp_data:
        methods = list(comp_data['latencies'].keys())
        latencies = list(comp_data['latencies'].values())
        ax2.bar(methods, latencies, color=['blue', 'green', 'orange'])
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Inference Latency Comparison')
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/compression_analysis.png', dpi=150, bbox_inches='tight')
    print(f'Saved compression analysis to {fig_dir}/compression_analysis.png')
" || echo "Visualization generation failed"

# -----------------------------------------------------------------------------
# GIT COMMIT AND PUSH
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "SAVING RESULTS TO GIT"
echo "=============================================================="

# Add all logs and results (but not checkpoints)
git add "runs/${RUN_NAME}/*.log" 2>/dev/null || true
git add "runs/${RUN_NAME}/**/*.log" 2>/dev/null || true
git add "runs/${RUN_NAME}/**/*.json" 2>/dev/null || true
git add "runs/${RUN_NAME}/**/*.txt" 2>/dev/null || true
git add "figures/${RUN_NAME}/*.png" 2>/dev/null || true

# Create detailed commit message
COMMIT_MSG="results: complete finalization experiment (job $SLURM_JOB_ID)

Training:
- Model: LatentWire with ${LATENT_LEN}×${D_Z} latent space
- Data: ${SAMPLES} samples from SQuAD
- Training: ${EPOCHS} epochs, batch size ${BATCH_SIZE}

Evaluation:
- Datasets: SST-2, AG News, GSM8K, TREC, SQuAD
- Baselines: LLMLingua compression, Linear probe
- Statistical testing performed

Run name: ${RUN_NAME}

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>"

# Commit and push with retry logic
if git commit -m "$COMMIT_MSG"; then
    echo "Created commit, attempting to push..."

    RETRY_COUNT=0
    MAX_RETRIES=3
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if git push; then
            echo "Successfully pushed results to remote"
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            echo "Push attempt $RETRY_COUNT of $MAX_RETRIES failed"
            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                sleep 5
                git pull --rebase=false 2>/dev/null || true
            fi
        fi
    done

    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "WARNING: Could not push after $MAX_RETRIES attempts"
        echo "Results are committed locally but not pushed to remote"
    fi
else
    echo "No changes to commit (or commit failed)"
fi

# -----------------------------------------------------------------------------
# JOB COMPLETION
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "JOB COMPLETION SUMMARY"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo ""
echo "Results saved to:"
echo "  - Checkpoint: runs/${RUN_NAME}/checkpoint/"
echo "  - Evaluations: runs/${RUN_NAME}/evaluations/"
echo "  - Baselines: runs/${RUN_NAME}/baselines/"
echo "  - Analysis: runs/${RUN_NAME}/analysis/"
echo "  - Report: runs/${RUN_NAME}/FINAL_REPORT.txt"
echo "  - Figures: figures/${RUN_NAME}/"
echo ""
echo "To retrieve results locally:"
echo "  git pull"
echo "=============================================================="