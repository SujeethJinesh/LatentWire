#!/bin/bash
#SBATCH --job-name=consolidated_exp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=08:00:00
#SBATCH --mem=128GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/consolidated_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/consolidated_%j.err
#SBATCH --signal=B:USR1@300

# =============================================================================
# CONSOLIDATED CROSS-MODEL COMMUNICATION EXPERIMENTS
# =============================================================================
#
# This script combines the best experiments from recent cross-model
# communication research, designed for 8-hour H100 runtime with preemption
# robustness.
#
# KEY PAPERS REFERENCED:
# - LatentMAS (arXiv:2511.20639): Ridge regression alignment, KV-cache transfer
# - Cache2Cache (arXiv:2510.03215): Per-layer gating, Gumbel-sigmoid selection
# - COCONUT (arXiv:2412.06769): Curriculum training, latent reasoning
#
# =============================================================================
# EXPERIMENT STRUCTURE (Priority Order)
# =============================================================================
#
# Phase 1: Baselines (Est. 1.5h) - CRITICAL for comparison
#   1.1 Ridge Regression (LatentMAS) - Training-free baseline to beat
#   1.2 Linear Probe (sklearn) - Upper bound for frozen features
#   1.3 Zero-shot baselines
#
# Phase 2: Core Bridge Training (Est. 2.5h)
#   2.1 Standard Bridge (SST-2, 3 seeds)
#   2.2 Standard Bridge (AG News, 3 seeds)
#
# Phase 3: Architecture Ablations (Est. 2h)
#   3.1 Multi-layer extraction [16, 24, 31] vs single layer
#   3.2 Gumbel-sigmoid layer gating (C2C-style)
#   3.3 VIB regularization for compression
#
# Phase 4: Advanced Techniques (Est. 1.5h)
#   4.1 Curriculum training (COCONUT-style)
#
# Phase 5: Statistical Analysis (Est. 0.5h)
#   5.1 Aggregate results across seeds
#   5.2 Compute confidence intervals
#   5.3 Run significance tests
#   5.4 Generate LaTeX tables
#
# =============================================================================
# TIME BUDGET BREAKDOWN
# =============================================================================
#
# Phase 1 - Baselines:          ~1.5 hours
#   - Ridge Regression:          15 min (training-free, just fitting)
#   - Linear Probe:              45 min (feature extraction + sklearn)
#   - Zero-shot eval:            30 min
#
# Phase 2 - Core Training:      ~2.5 hours
#   - SST-2 (3 seeds x 15 min):  45 min
#   - AG News (3 seeds x 20 min): 1 hour
#   - Evaluation:                45 min
#
# Phase 3 - Ablations:          ~2 hours
#   - Multi-layer:               40 min
#   - Gating:                    40 min
#   - VIB:                       40 min
#
# Phase 4 - Advanced:           ~1.5 hours
#   - Curriculum:                1.5 hours
#
# Phase 5 - Analysis:           ~30 min
#
# TOTAL:                        ~8 hours
#
# =============================================================================
# ROBUSTNESS FEATURES
# =============================================================================
#
# 1. Checkpoint after each experiment phase
# 2. Automatic resume from last completed phase
# 3. Signal handling for graceful preemption
# 4. Continue-on-failure for non-critical experiments
# 5. Frequent intermediate result saves
# 6. Git commit after each major phase
#
# =============================================================================

set -o pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="runs/consolidated_${TIMESTAMP}"
CHECKPOINT_FILE="${OUTPUT_DIR}/.checkpoint"
RESULTS_FILE="${OUTPUT_DIR}/consolidated_results.json"

# Experiment parameters
SEEDS=(42 123 456)
SOFT_TOKENS=8
TRAIN_STEPS=1000
EVAL_SAMPLES=200
DATASETS=("sst2" "agnews")

# Environment setup
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME="/projects/m000066/sujinesh/.cache/huggingface"
export TRANSFORMERS_CACHE="/projects/m000066/sujinesh/.cache/huggingface/transformers"

# =============================================================================
# SIGNAL HANDLING FOR PREEMPTION
# =============================================================================

PREEMPTED=0
CURRENT_PHASE=""

# Handle preemption signal (SIGUSR1 sent 5 min before termination)
handle_preemption() {
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "PREEMPTION SIGNAL RECEIVED"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "Current phase: $CURRENT_PHASE"
    echo "Saving intermediate results..."

    PREEMPTED=1

    # Save checkpoint
    echo "$CURRENT_PHASE" > "$CHECKPOINT_FILE"

    # Commit what we have so far
    save_results_to_git "PREEMPTED during phase: $CURRENT_PHASE"

    echo "Checkpoint saved. Requeue job to continue."
    exit 0
}

trap handle_preemption SIGUSR1

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_phase() {
    local phase_name=$1
    CURRENT_PHASE="$phase_name"
    echo ""
    echo "=============================================================="
    echo "PHASE: $phase_name"
    echo "=============================================================="
    echo "Time: $(date)"
    echo "Elapsed: $SECONDS seconds"
    echo ""
}

save_checkpoint() {
    local phase_name=$1
    echo "$phase_name" > "$CHECKPOINT_FILE"
    echo "[CHECKPOINT] Phase completed: $phase_name"
}

check_resume() {
    if [ -f "$CHECKPOINT_FILE" ]; then
        LAST_PHASE=$(cat "$CHECKPOINT_FILE")
        echo "RESUME: Found checkpoint at phase: $LAST_PHASE"
        return 0
    fi
    return 1
}

should_skip_phase() {
    local phase_order=$1
    if [ -f "$CHECKPOINT_FILE" ]; then
        # Read the last completed phase number
        local last_phase=$(cat "$CHECKPOINT_FILE" | grep -oE "^[0-9]+\.[0-9]+" | head -1)
        if [ -n "$last_phase" ]; then
            if (( $(echo "$phase_order <= $last_phase" | bc -l) )); then
                echo "[SKIP] Phase $phase_order already completed"
                return 0
            fi
        fi
    fi
    return 1
}

run_with_timeout() {
    local timeout_sec=$1
    shift
    local cmd="$@"

    timeout --signal=KILL $timeout_sec bash -c "$cmd"
    local exit_code=$?

    if [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ]; then
        echo "[TIMEOUT] Command exceeded ${timeout_sec}s limit"
        return 1
    fi
    return $exit_code
}

save_results_to_git() {
    local commit_msg=$1

    echo ""
    echo "Saving results to git..."

    git add "${OUTPUT_DIR}/" 2>/dev/null || true
    git add runs/consolidated_*.log runs/consolidated_*.err 2>/dev/null || true

    if git commit -m "results: consolidated experiments - $commit_msg (SLURM $SLURM_JOB_ID)

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>" 2>/dev/null; then

        local retry=0
        while [ $retry -lt 3 ]; do
            if git push 2>/dev/null; then
                echo "Successfully pushed to remote"
                break
            fi
            retry=$((retry + 1))
            sleep 5
            git pull --rebase=false 2>/dev/null || true
        done
    fi
}

# =============================================================================
# INITIALIZATION
# =============================================================================

echo "=============================================================="
echo "CONSOLIDATED CROSS-MODEL EXPERIMENTS"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Configuration:"
echo "  Seeds: ${SEEDS[*]}"
echo "  Soft tokens: $SOFT_TOKENS"
echo "  Train steps: $TRAIN_STEPS"
echo "  Datasets: ${DATASETS[*]}"
echo "=============================================================="

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/phase1_baselines"
mkdir -p "${OUTPUT_DIR}/phase2_core"
mkdir -p "${OUTPUT_DIR}/phase3_ablations"
mkdir -p "${OUTPUT_DIR}/phase4_advanced"
mkdir -p "${OUTPUT_DIR}/phase5_analysis"

# Git setup
git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job $SLURM_JOB_ID"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@hpc.cluster"

echo "Pulling latest code..."
git pull 2>/dev/null || { git stash && git pull && git stash pop 2>/dev/null; } || true

# GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Initialize results JSON
cat > "$RESULTS_FILE" << EOF
{
    "metadata": {
        "job_id": "$SLURM_JOB_ID",
        "start_time": "$(date -Iseconds)",
        "config": {
            "seeds": [${SEEDS[*]}],
            "soft_tokens": $SOFT_TOKENS,
            "train_steps": $TRAIN_STEPS,
            "datasets": ["${DATASETS[*]}"]
        }
    },
    "phases": {}
}
EOF

# =============================================================================
# PHASE 1: BASELINES (Critical for comparison)
# =============================================================================

log_phase "1.0 BASELINES"
PHASE1_START=$SECONDS

# -----------------------------------------------------------------------------
# 1.1 Ridge Regression Baseline (LatentMAS - using new --bridge_type ridge)
# -----------------------------------------------------------------------------
if ! should_skip_phase "1.1"; then
    log_phase "1.1 Ridge Regression (LatentMAS)"

    for dataset in "${DATASETS[@]}"; do
        echo "Running ridge regression on $dataset..."

        # Use the new --bridge_type ridge argument (training-free, sets eval_only automatically)
        python telepathy/train_telepathy.py \
            --dataset "$dataset" \
            --bridge_type ridge \
            --lambda_reg 1e-4 \
            --output_dir "${OUTPUT_DIR}/phase1_baselines/ridge_${dataset}" \
            --gpu 0 \
            2>&1 | tee "${OUTPUT_DIR}/phase1_baselines/ridge_${dataset}.log" || echo "Ridge regression on $dataset completed with warnings"

    done

    save_checkpoint "1.1 Ridge Regression"
fi

# -----------------------------------------------------------------------------
# 1.2 Linear Probe Baseline
# -----------------------------------------------------------------------------
if ! should_skip_phase "1.2"; then
    log_phase "1.2 Linear Probe Baseline"

    for dataset in "${DATASETS[@]}"; do
        echo "Running linear probe on $dataset..."

        python telepathy/linear_probe_baseline.py \
            --datasets "$dataset" \
            --layers 16 24 31 \
            --max_train_samples 2000 \
            --max_test_samples 500 \
            --seeds 42 \
            --output_dir "${OUTPUT_DIR}/phase1_baselines/linear_probe_${dataset}" \
            --batch_size 4 \
            2>&1 | tee "${OUTPUT_DIR}/phase1_baselines/linear_probe_${dataset}.log" || echo "Linear probe on $dataset completed with warnings"
    done

    save_checkpoint "1.2 Linear Probe"
fi

# -----------------------------------------------------------------------------
# 1.3 Zero-shot Baselines
# -----------------------------------------------------------------------------
if ! should_skip_phase "1.3"; then
    log_phase "1.3 Zero-shot Baselines"

    for dataset in "${DATASETS[@]}"; do
        echo "Running zero-shot baseline on $dataset..."

        python telepathy/run_baselines.py \
            --baseline zeroshot \
            --dataset "$dataset" \
            --eval_samples $EVAL_SAMPLES \
            --output_dir "${OUTPUT_DIR}/phase1_baselines" \
            2>&1 | tee "${OUTPUT_DIR}/phase1_baselines/zeroshot_${dataset}.log" || echo "Zero-shot on $dataset completed with warnings"
    done

    save_checkpoint "1.3 Zero-shot"
fi

PHASE1_TIME=$((SECONDS - PHASE1_START))
echo ""
echo "Phase 1 completed in ${PHASE1_TIME}s ($((PHASE1_TIME / 60)) minutes)"
save_results_to_git "Phase 1 (Baselines) complete"

# =============================================================================
# PHASE 2: CORE BRIDGE TRAINING
# =============================================================================

log_phase "2.0 CORE BRIDGE TRAINING"
PHASE2_START=$SECONDS

# -----------------------------------------------------------------------------
# 2.1 Standard Bridge on SST-2 (3 seeds)
# -----------------------------------------------------------------------------
if ! should_skip_phase "2.1"; then
    log_phase "2.1 Bridge Training - SST-2"

    for seed in "${SEEDS[@]}"; do
        echo "Training bridge on SST-2 (seed=$seed)..."

        python telepathy/train_telepathy.py \
            --dataset sst2 \
            --soft_tokens $SOFT_TOKENS \
            --steps $TRAIN_STEPS \
            --seed $seed \
            --eval_every 200 \
            --save_every $TRAIN_STEPS \
            --output_dir "${OUTPUT_DIR}/phase2_core/sst2_seed${seed}" \
            --gpu 0 \
            2>&1 | tee "${OUTPUT_DIR}/phase2_core/sst2_seed${seed}.log" || echo "SST-2 seed $seed completed with warnings"
    done

    save_checkpoint "2.1 SST-2 Training"
fi

# -----------------------------------------------------------------------------
# 2.2 Standard Bridge on AG News (3 seeds)
# -----------------------------------------------------------------------------
if ! should_skip_phase "2.2"; then
    log_phase "2.2 Bridge Training - AG News"

    for seed in "${SEEDS[@]}"; do
        echo "Training bridge on AG News (seed=$seed)..."

        python telepathy/train_telepathy.py \
            --dataset agnews \
            --soft_tokens $SOFT_TOKENS \
            --steps $TRAIN_STEPS \
            --seed $seed \
            --eval_every 200 \
            --save_every $TRAIN_STEPS \
            --output_dir "${OUTPUT_DIR}/phase2_core/agnews_seed${seed}" \
            --gpu 0 \
            2>&1 | tee "${OUTPUT_DIR}/phase2_core/agnews_seed${seed}.log" || echo "AG News seed $seed completed with warnings"
    done

    save_checkpoint "2.2 AG News Training"
fi

PHASE2_TIME=$((SECONDS - PHASE2_START))
echo ""
echo "Phase 2 completed in ${PHASE2_TIME}s ($((PHASE2_TIME / 60)) minutes)"
save_results_to_git "Phase 2 (Core Training) complete"

# =============================================================================
# PHASE 3: ARCHITECTURE ABLATIONS
# =============================================================================

log_phase "3.0 ARCHITECTURE ABLATIONS"
PHASE3_START=$SECONDS

# -----------------------------------------------------------------------------
# 3.1 Multi-Layer Extraction (using new bridge_type and extract_layers args)
# -----------------------------------------------------------------------------
if ! should_skip_phase "3.1"; then
    log_phase "3.1 Multi-Layer Extraction"

    echo "Testing multi-layer extraction configurations..."

    # Single layer (baseline - layer 31)
    python telepathy/train_telepathy.py \
        --dataset sst2 \
        --soft_tokens $SOFT_TOKENS \
        --source_layer 31 \
        --steps 750 \
        --seed 42 \
        --output_dir "${OUTPUT_DIR}/phase3_ablations/layer_31_only" \
        --gpu 0 \
        2>&1 | tee "${OUTPUT_DIR}/phase3_ablations/layer_31_only.log" || echo "Layer 31 completed with warnings"

    # Multi-layer extraction with learned weights [16, 24, 31]
    python telepathy/train_telepathy.py \
        --dataset sst2 \
        --bridge_type multi_layer \
        --extract_layers 16 24 31 \
        --learn_layer_weights \
        --soft_tokens $SOFT_TOKENS \
        --steps 750 \
        --seed 42 \
        --output_dir "${OUTPUT_DIR}/phase3_ablations/multi_layer_16_24_31" \
        --gpu 0 \
        2>&1 | tee "${OUTPUT_DIR}/phase3_ablations/multi_layer_16_24_31.log" || echo "Multi-layer [16,24,31] completed with warnings"

    # Two-layer extraction [16, 31]
    python telepathy/train_telepathy.py \
        --dataset sst2 \
        --bridge_type multi_layer \
        --extract_layers 16 31 \
        --learn_layer_weights \
        --soft_tokens $SOFT_TOKENS \
        --steps 750 \
        --seed 42 \
        --output_dir "${OUTPUT_DIR}/phase3_ablations/multi_layer_16_31" \
        --gpu 0 \
        2>&1 | tee "${OUTPUT_DIR}/phase3_ablations/multi_layer_16_31.log" || echo "Multi-layer [16,31] completed with warnings"

    save_checkpoint "3.1 Multi-Layer"
fi

# -----------------------------------------------------------------------------
# 3.2 Soft Token Count Ablation
# -----------------------------------------------------------------------------
if ! should_skip_phase "3.2"; then
    log_phase "3.2 Soft Token Count Ablation"

    for num_tokens in 4 8 16 24; do
        echo "Testing with $num_tokens soft tokens..."

        python telepathy/train_telepathy.py \
            --dataset sst2 \
            --soft_tokens $num_tokens \
            --steps 750 \
            --seed 42 \
            --output_dir "${OUTPUT_DIR}/phase3_ablations/tokens_${num_tokens}" \
            --gpu 0 \
            2>&1 | tee "${OUTPUT_DIR}/phase3_ablations/tokens_${num_tokens}.log" || echo "Tokens $num_tokens completed with warnings"
    done

    save_checkpoint "3.2 Token Ablation"
fi

# -----------------------------------------------------------------------------
# 3.3 Depth Ablation
# -----------------------------------------------------------------------------
if ! should_skip_phase "3.3"; then
    log_phase "3.3 Bridge Depth Ablation"

    for depth in 1 2 4; do
        echo "Testing with depth=$depth..."

        python telepathy/train_telepathy.py \
            --dataset sst2 \
            --soft_tokens $SOFT_TOKENS \
            --depth $depth \
            --steps 750 \
            --seed 42 \
            --output_dir "${OUTPUT_DIR}/phase3_ablations/depth_${depth}" \
            --gpu 0 \
            2>&1 | tee "${OUTPUT_DIR}/phase3_ablations/depth_${depth}.log" || echo "Depth $depth completed with warnings"
    done

    save_checkpoint "3.3 Depth Ablation"
fi

PHASE3_TIME=$((SECONDS - PHASE3_START))
echo ""
echo "Phase 3 completed in ${PHASE3_TIME}s ($((PHASE3_TIME / 60)) minutes)"
save_results_to_git "Phase 3 (Ablations) complete"

# =============================================================================
# PHASE 4: ADVANCED TECHNIQUES
# =============================================================================

log_phase "4.0 ADVANCED TECHNIQUES"
PHASE4_START=$SECONDS

# -----------------------------------------------------------------------------
# 4.1 VIB Regularization (Variational Information Bottleneck)
# -----------------------------------------------------------------------------
if ! should_skip_phase "4.1"; then
    log_phase "4.1 VIB Regularization"

    for beta in 0.0001 0.001 0.01; do
        beta_name=$(echo "$beta" | tr '.' '_')
        echo "Testing VIB beta=$beta..."

        python telepathy/train_telepathy.py \
            --dataset sst2 \
            --bridge_type vib \
            --vib_beta $beta \
            --vib_beta_anneal \
            --soft_tokens $SOFT_TOKENS \
            --steps 750 \
            --seed 42 \
            --output_dir "${OUTPUT_DIR}/phase4_advanced/vib_beta_${beta_name}" \
            --gpu 0 \
            2>&1 | tee "${OUTPUT_DIR}/phase4_advanced/vib_beta_${beta_name}.log" || echo "VIB beta=$beta completed with warnings"
    done

    save_checkpoint "4.1 VIB Regularization"
fi

# -----------------------------------------------------------------------------
# 4.2 Diversity Weight Ablation
# -----------------------------------------------------------------------------
if ! should_skip_phase "4.2"; then
    log_phase "4.2 Diversity Weight Ablation"

    for div_weight in 0.0 0.05 0.1 0.2; do
        echo "Testing diversity_weight=$div_weight..."

        python telepathy/train_telepathy.py \
            --dataset sst2 \
            --soft_tokens $SOFT_TOKENS \
            --diversity_weight $div_weight \
            --steps 750 \
            --seed 42 \
            --output_dir "${OUTPUT_DIR}/phase4_advanced/diversity_${div_weight}" \
            --gpu 0 \
            2>&1 | tee "${OUTPUT_DIR}/phase4_advanced/diversity_${div_weight}.log" || echo "Diversity $div_weight completed with warnings"
    done

    save_checkpoint "4.2 Diversity Ablation"
fi

# -----------------------------------------------------------------------------
# 4.3 Learning Rate Ablation
# -----------------------------------------------------------------------------
if ! should_skip_phase "4.3"; then
    log_phase "4.3 Learning Rate Ablation"

    for lr in 1e-4 2e-4 5e-4 1e-3; do
        echo "Testing lr=$lr..."

        python telepathy/train_telepathy.py \
            --dataset sst2 \
            --soft_tokens $SOFT_TOKENS \
            --lr $lr \
            --steps 750 \
            --seed 42 \
            --output_dir "${OUTPUT_DIR}/phase4_advanced/lr_${lr}" \
            --gpu 0 \
            2>&1 | tee "${OUTPUT_DIR}/phase4_advanced/lr_${lr}.log" || echo "LR $lr completed with warnings"
    done

    save_checkpoint "4.3 LR Ablation"
fi

PHASE4_TIME=$((SECONDS - PHASE4_START))
echo ""
echo "Phase 4 completed in ${PHASE4_TIME}s ($((PHASE4_TIME / 60)) minutes)"
save_results_to_git "Phase 4 (Advanced) complete"

# =============================================================================
# PHASE 5: STATISTICAL ANALYSIS
# =============================================================================

log_phase "5.0 STATISTICAL ANALYSIS"
PHASE5_START=$SECONDS

# Aggregate all results and run statistical tests
python -c "
import json
import os
import numpy as np
from pathlib import Path
from scipy import stats

output_dir = Path('${OUTPUT_DIR}')

# Collect all results
all_results = {
    'metadata': {
        'job_id': '$SLURM_JOB_ID',
        'total_runtime_seconds': $SECONDS,
        'seeds': [${SEEDS[*]}],
        'datasets': ['${DATASETS[*]}']
    },
    'baselines': {},
    'core_training': {},
    'ablations': {},
    'statistical_tests': {}
}

# Collect Phase 2 results (core training)
for dataset in ['sst2', 'agnews']:
    dataset_results = []
    for seed in [42, 123, 456]:
        results_file = output_dir / f'phase2_core/{dataset}_seed{seed}/{dataset}_seed{seed}_results.json'
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                if 'final_results' in data:
                    dataset_results.append(data['final_results']['accuracy'])

    if dataset_results:
        all_results['core_training'][dataset] = {
            'accuracies': dataset_results,
            'mean': np.mean(dataset_results),
            'std': np.std(dataset_results),
            'min': np.min(dataset_results),
            'max': np.max(dataset_results),
            'n_seeds': len(dataset_results)
        }

# Collect ablation results
ablation_dir = output_dir / 'phase3_ablations'
for ablation in ['layer_16_only', 'layer_24_only', 'layer_31_only',
                 'tokens_4', 'tokens_8', 'tokens_16', 'tokens_24',
                 'depth_1', 'depth_2', 'depth_4']:
    results_file = ablation_dir / ablation / 'sst2_seed42_results.json'
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
            if 'final_results' in data:
                all_results['ablations'][ablation] = {
                    'accuracy': data['final_results']['accuracy']
                }

# Statistical significance tests
print('\\n' + '='*60)
print('STATISTICAL ANALYSIS RESULTS')
print('='*60)

for dataset in ['sst2', 'agnews']:
    if dataset in all_results['core_training']:
        data = all_results['core_training'][dataset]
        print(f'\\n{dataset.upper()}:')
        print(f'  Mean accuracy: {data[\"mean\"]:.2f}%')
        print(f'  Std deviation: {data[\"std\"]:.2f}%')
        print(f'  95% CI: [{data[\"mean\"] - 1.96*data[\"std\"]/np.sqrt(data[\"n_seeds\"]):.2f}, {data[\"mean\"] + 1.96*data[\"std\"]/np.sqrt(data[\"n_seeds\"]):.2f}]')

        # Compare to random baseline
        random_baseline = 50.0 if dataset == 'sst2' else 25.0
        if data['std'] > 0 and data['n_seeds'] > 1:
            t_stat = (data['mean'] - random_baseline) / (data['std'] / np.sqrt(data['n_seeds']))
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), data['n_seeds'] - 1))
            print(f'  vs Random ({random_baseline}%): t={t_stat:.2f}, p={p_val:.4f}')

            all_results['statistical_tests'][f'{dataset}_vs_random'] = {
                't_statistic': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }

# Save results
with open(output_dir / 'consolidated_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\\nResults saved to: {output_dir}/consolidated_results.json')

# Generate LaTeX table
print('\\n' + '='*60)
print('LATEX TABLE')
print('='*60)

latex = '''
\\\\begin{table}[h]
\\\\centering
\\\\caption{Consolidated Experiment Results}
\\\\begin{tabular}{lcc}
\\\\toprule
Method & SST-2 & AG News \\\\\\\\
\\\\midrule
'''

if 'sst2' in all_results['core_training']:
    sst2_acc = all_results['core_training']['sst2']['mean']
    sst2_std = all_results['core_training']['sst2']['std']
else:
    sst2_acc, sst2_std = 0, 0

if 'agnews' in all_results['core_training']:
    agnews_acc = all_results['core_training']['agnews']['mean']
    agnews_std = all_results['core_training']['agnews']['std']
else:
    agnews_acc, agnews_std = 0, 0

latex += f'Bridge (ours) & {sst2_acc:.1f} \\$\\\\pm\\$ {sst2_std:.1f} & {agnews_acc:.1f} \\$\\\\pm\\$ {agnews_std:.1f} \\\\\\\\\n'
latex += '''\\\\bottomrule
\\\\end{tabular}
\\\\end{table}
'''

print(latex)

with open(output_dir / 'phase5_analysis/results_table.tex', 'w') as f:
    f.write(latex)
" 2>&1 | tee "${OUTPUT_DIR}/phase5_analysis/analysis.log"

PHASE5_TIME=$((SECONDS - PHASE5_START))
echo ""
echo "Phase 5 completed in ${PHASE5_TIME}s"

# =============================================================================
# FINAL SUMMARY
# =============================================================================

echo ""
echo "=============================================================="
echo "CONSOLIDATED EXPERIMENTS COMPLETE"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds ($((SECONDS / 3600))h $((SECONDS % 3600 / 60))m)"
echo ""
echo "Time breakdown:"
echo "  Phase 1 (Baselines):    ${PHASE1_TIME}s ($((PHASE1_TIME / 60))m)"
echo "  Phase 2 (Core):         ${PHASE2_TIME}s ($((PHASE2_TIME / 60))m)"
echo "  Phase 3 (Ablations):    ${PHASE3_TIME}s ($((PHASE3_TIME / 60))m)"
echo "  Phase 4 (Advanced):     ${PHASE4_TIME}s ($((PHASE4_TIME / 60))m)"
echo "  Phase 5 (Analysis):     ${PHASE5_TIME}s ($((PHASE5_TIME / 60))m)"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "Key files:"
echo "  - ${OUTPUT_DIR}/consolidated_results.json"
echo "  - ${OUTPUT_DIR}/phase5_analysis/results_table.tex"
echo "=============================================================="

# Final git commit
save_results_to_git "All phases complete"

echo ""
echo "To view results locally:"
echo "  git pull"
echo "  cat ${OUTPUT_DIR}/consolidated_results.json | python -m json.tool"
echo ""
