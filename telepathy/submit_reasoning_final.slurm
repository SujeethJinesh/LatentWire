#!/bin/bash
#SBATCH --job-name=reasoning_final
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --output=runs/reasoning_final_%j.log
#SBATCH --error=runs/reasoning_final_%j.err

set -o pipefail  # Catch pipe failures

# Ensure SLURM output directory exists (relative to submit directory)
mkdir -p runs

# Error tracking
declare -a FAILED_EXPERIMENTS=()
declare -a SKIPPED_EXPERIMENTS=()
TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0
SKIPPED_COUNT=0

# Source registry functions for experiment tracking
WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -f "${WORK_DIR}/scripts/registry_functions.sh" ]; then
    source "${WORK_DIR}/scripts/registry_functions.sh"
    REGISTRY_ENABLED=1
else
    echo "Warning: Registry functions not found, experiment tracking disabled"
    REGISTRY_ENABLED=0
fi

# =============================================================================
# REASONING-FOCUSED FINAL EXPERIMENTS (Publication Ready)
# =============================================================================
# This is the ONE SHOT script for MLSys 2025 - REASONING FOCUS
#
# REASONING BENCHMARKS:
# - ARC-Easy: Elementary science questions (4-way MC)
# - WinoGrande: Common sense pronoun resolution (2-way)
# - HellaSwag: Situational reasoning (4-way MC)
# - BoolQ: Yes/No reading comprehension (2-way)
#
# GPU WORKLOAD (Balanced ~4-5h each):
# - GPU 0: Standard bridge + baselines (zero-shot, few-shot, DoRA, prompt tuning, linear probe, text_relay, ensemble)
# - GPU 1: Novel bridges + HAIL MARY bridges + token capacity + ridge_regression + benchmarks
#
# EXPERIMENTS:
# 1. Standard bridge on all reasoning tasks (3 seeds) via hail_mary
# 2. Zero-shot baseline on all 4 reasoning datasets
# 3. Few-shot baseline (4-shot) on all 4 datasets
# 4. DoRA fine-tuning baseline on arc_easy (SOTA PEFT, 2024)
# 5. Prompt tuning baseline on arc_easy (proves sender helps)
# 6. Linear probe baseline (layer 31) on all datasets
# 7. Text relay baseline on arc_easy (fair text-based cross-model)
# 8. Ensemble baseline on arc_easy (alpha-weighted Llama+Mistral)
# 9. Novel bridges: flow_matching, optimal_transport, moe (1 seed each)
# 10. HAIL MARY bridges: cross_modal_distillation, mine, mixture_of_depths,
#     thalamic_relay, domain_adversarial, successive_refinement (1 seed each)
# 11. Token capacity: 4, 8, 16, 32 tokens (1 seed each)
# 12. Ridge regression baseline (training-free alignment)
# 13. Latency/memory/throughput/batched benchmarks
#
# Estimated runtime: ~4 hours on 2 H100 GPUs (balanced workload)
#
# Submit with: sbatch telepathy/submit_reasoning_final.slurm
# =============================================================================

WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORK_DIR"

export PYTHONPATH=.

# Use /scratch for ALL caches to avoid disk quota issues on /projects
# Set XDG_CACHE_HOME first to override default ~/.cache location
export XDG_CACHE_HOME="/scratch/m000066/sujinesh/.cache"
export HF_HOME="/scratch/m000066/sujinesh/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="runs/reasoning_final_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/standard_bridge"
mkdir -p "$OUTPUT_DIR/novel_bridges"
mkdir -p "$OUTPUT_DIR/baselines"
mkdir -p "$OUTPUT_DIR/benchmarks"

# Initialize experiment registry
REGISTRY_FILE="runs/experiment_registry.json"
if [ "$REGISTRY_ENABLED" = "1" ]; then
    log_msg "Initializing experiment registry at $REGISTRY_FILE"
    python telepathy/experiment_registry.py --registry "$REGISTRY_FILE" summary > /dev/null 2>&1 || {
        log_msg "Warning: Could not initialize registry"
    }
fi

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Timestamped logging
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Log experiment with error tracking and registry integration
run_experiment() {
    local name="$1"
    shift
    local cmd="$@"

    # Extract GPU from the command if present (look for --gpu N or cuda:N)
    local gpu=0
    if [[ "$cmd" =~ --gpu[[:space:]]+([0-9]+) ]]; then
        gpu="${BASH_REMATCH[1]}"
    elif [[ "$cmd" =~ cuda:([0-9]+) ]]; then
        gpu="${BASH_REMATCH[1]}"
    fi

    # Check registry if enabled
    if [ "$REGISTRY_ENABLED" = "1" ]; then
        local result=$(python telepathy/experiment_registry.py --registry "$REGISTRY_FILE" should_run "$name" 2>/dev/null)
        local should_run=$(echo "$result" | cut -d'|' -f1)
        local skip_reason=$(echo "$result" | cut -d'|' -f2-)

        if [ "$should_run" = "0" ]; then
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            SKIPPED_EXPERIMENTS+=("$name")
            log_msg "SKIP: $name ($skip_reason)"
            return 0
        fi

        # Mark as running in registry
        python telepathy/experiment_registry.py --registry "$REGISTRY_FILE" start "$name" --gpu "$gpu" 2>/dev/null || true
    fi

    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
    log_msg "START: $name"

    # Capture output for error extraction
    local log_capture="${OUTPUT_DIR}/.${name}_capture.log"

    if eval "$cmd" 2>&1 | tee "$log_capture"; then
        SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        log_msg "SUCCESS: $name"

        # Update registry with success
        if [ "$REGISTRY_ENABLED" = "1" ]; then
            # Try to find result file
            local result_file=""
            for pattern in "${OUTPUT_DIR}"/*"${name}"*_results.json "${OUTPUT_DIR}"/**/*"${name}"*_results.json; do
                if [ -f "$pattern" ] 2>/dev/null; then
                    result_file="$pattern"
                    break
                fi
            done
            python telepathy/experiment_registry.py --registry "$REGISTRY_FILE" complete "$name" ${result_file:+--result-file "$result_file"} 2>/dev/null || true
        fi

        rm -f "$log_capture"
        return 0
    else
        local exit_code=$?
        FAILED_EXPERIMENTS+=("$name")
        log_msg "FAILED: $name (exit code: $exit_code)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $name failed with exit code $exit_code" >> "${OUTPUT_DIR}/experiment_errors.log"

        # Update registry with failure
        if [ "$REGISTRY_ENABLED" = "1" ]; then
            # Extract error message from captured output
            local error_msg=$(tail -20 "$log_capture" 2>/dev/null | grep -E "(Error|Exception|CUDA|OOM|NaN|Failed|Traceback)" | tail -3 | tr '\n' ' ' | head -c 300)
            [ -z "$error_msg" ] && error_msg="Unknown error (exit code: $exit_code)"
            python telepathy/experiment_registry.py --registry "$REGISTRY_FILE" fail "$name" --error "$error_msg" --exit-code "$exit_code" 2>/dev/null || true
        fi

        rm -f "$log_capture"
        return $exit_code
    fi
}

# Git push with retry
push_with_retry() {
    local max_attempts=3
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        log_msg "Git push attempt $attempt of $max_attempts"
        if git push; then
            log_msg "Git push successful"
            return 0
        fi
        log_msg "Push failed, waiting 10s before retry..."
        attempt=$((attempt + 1))
        sleep 10
        git pull --rebase=false 2>/dev/null || true
    done

    log_msg "CRITICAL: Failed to push after $max_attempts attempts"
    return 1
}

# Cleanup trap
cleanup() {
    local exit_code=$?
    log_msg "Job ending with exit code: $exit_code"

    # Always try to save partial results
    if [ -d "$OUTPUT_DIR" ]; then
        git add "${OUTPUT_DIR}/" 2>/dev/null || true
        git commit -m "partial results: job $SLURM_JOB_ID (exit: $exit_code)" 2>/dev/null || true
        push_with_retry || true
    fi
}
trap cleanup EXIT

echo "=============================================================="
echo "REASONING-FOCUSED FINAL EXPERIMENTS (Publication Ready)"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""
echo "REASONING BENCHMARKS:"
echo "  - ARC-Easy (science)"
echo "  - WinoGrande (commonsense)"
echo "  - HellaSwag (situational)"
echo "  - BoolQ (reading comprehension)"
echo ""
echo "GPU ALLOCATION:"
echo "  - GPU 0: Standard bridge (hail mary) + baselines (zero-shot, few-shot, DoRA, prompt tuning, linear probe)"
echo "  - GPU 1: Novel bridges + token capacity + latency/memory/throughput/batched benchmarks"
echo "=============================================================="

# Git setup
git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job $SLURM_JOB_ID"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@hpc.cluster"
git pull || true

echo ""
nvidia-smi --query-gpu=index,name,memory.free --format=csv
echo ""

SEEDS="42 123 456"
REASONING_DATASETS="arc_easy winogrande hellaswag boolq"

# Create error log file
touch "${OUTPUT_DIR}/experiment_errors.log"

# =============================================================================
# GPU 0: STANDARD BRIDGE + BASELINES (zero-shot, few-shot, DoRA, prompt tuning, linear probe)
# =============================================================================
run_gpu0() {
    log_msg "[GPU 0] Starting: Standard Bridge + Baselines"

    # -----------------------------------------------------------------
    # HAIL MARY EVALUATION (Standard bridge on reasoning benchmarks)
    # -----------------------------------------------------------------
    log_msg "[GPU 0] Running Hail Mary Reasoning Evaluation"

    run_experiment "hail_mary_reasoning" "python telepathy/run_enhanced_paper_evaluation.py \
        --output_dir '${OUTPUT_DIR}/standard_bridge' \
        --gpu 0 \
        --hail_mary_mode \
        --fast_mode \
        --run_diagnostics \
        --log_level INFO \
        2>&1 | tee '${OUTPUT_DIR}/standard_bridge/hail_mary.log'"

    log_msg "[GPU 0] Hail mary evaluation completed"

    # -----------------------------------------------------------------
    # ZERO-SHOT BASELINE on all 4 reasoning datasets
    # Critical comparison: how do raw LLMs perform without any training?
    # -----------------------------------------------------------------
    echo ""
    log_msg "[GPU 0] Running Zero-Shot Baseline on all reasoning datasets"

    for DATASET in arc_easy winogrande hellaswag boolq; do
        log_msg "[GPU 0] Zero-shot on $DATASET"
        run_experiment "zeroshot_${DATASET}" "python telepathy/run_baselines.py \
            --baseline zeroshot \
            --dataset $DATASET \
            --gpu 0 \
            --max_samples 200 \
            --output_dir '${OUTPUT_DIR}/baselines/zeroshot' \
            2>&1 | tee '${OUTPUT_DIR}/baselines/zeroshot_${DATASET}.log'"
    done

    # -----------------------------------------------------------------
    # FEW-SHOT BASELINE (4-shot) on all reasoning datasets
    # -----------------------------------------------------------------
    echo ""
    log_msg "[GPU 0] Running Few-Shot Baseline (4-shot) on reasoning datasets"

    for DATASET in arc_easy winogrande hellaswag boolq; do
        log_msg "[GPU 0] Few-shot (4-shot) on $DATASET"
        run_experiment "fewshot_${DATASET}" "python telepathy/run_baselines.py \
            --baseline fewshot \
            --dataset $DATASET \
            --shots 4 \
            --seeds 42 123 456 \
            --gpu 0 \
            --max_samples 200 \
            --output_dir '${OUTPUT_DIR}/baselines/fewshot' \
            2>&1 | tee '${OUTPUT_DIR}/baselines/fewshot_${DATASET}.log'"
    done

    # -----------------------------------------------------------------
    # DORA FINE-TUNING BASELINE on ARC-Easy
    # DoRA = Weight-Decomposed LoRA, SOTA PEFT method (2024)
    # -----------------------------------------------------------------
    echo ""
    log_msg "[GPU 0] Running DoRA Baseline (rank=8) on ARC-Easy"

    run_experiment "dora_arc_easy" "python telepathy/run_baselines.py \
        --baseline dora \
        --dataset arc_easy \
        --rank 8 \
        --epochs 3 \
        --seeds 42 123 456 \
        --gpu 0 \
        --max_samples 200 \
        --max_train_samples 2000 \
        --output_dir '${OUTPUT_DIR}/baselines/dora' \
        2>&1 | tee '${OUTPUT_DIR}/baselines/dora_arc_easy.log'"

    # -----------------------------------------------------------------
    # PROMPT TUNING BASELINE on ARC-Easy
    # Proves that Llama (sender) actually helps - receiver alone can't match bridge
    # -----------------------------------------------------------------
    echo ""
    log_msg "[GPU 0] Running Prompt Tuning Baseline (8 tokens) on ARC-Easy"

    run_experiment "prompt_tuning_arc_easy" "python telepathy/run_baselines.py \
        --baseline prompt_tuning \
        --dataset arc_easy \
        --soft_tokens 8 \
        --steps 1500 \
        --seeds 42 123 456 \
        --gpu 0 \
        --max_samples 200 \
        --output_dir '${OUTPUT_DIR}/baselines/prompt_tuning' \
        2>&1 | tee '${OUTPUT_DIR}/baselines/prompt_tuning_arc_easy.log'"

    # -----------------------------------------------------------------
    # LINEAR PROBE BASELINE (Layer 31 - same as bridge)
    # -----------------------------------------------------------------
    echo ""
    log_msg "[GPU 0] Running Linear Probe Baseline (layer 31)"

    run_experiment "linear_probe" "python telepathy/linear_probe_baseline.py \
        --datasets arc_easy winogrande hellaswag boolq \
        --layers 31 \
        --seeds 42 123 456 \
        --output_dir '${OUTPUT_DIR}/baselines/linear_probe' \
        --device cuda:0 \
        2>&1 | tee '${OUTPUT_DIR}/baselines/linear_probe.log'"

    # -----------------------------------------------------------------
    # TEXT RELAY BASELINE (Fair text-based cross-model communication)
    # Llama generates a hint, Mistral classifies based on the hint
    # -----------------------------------------------------------------
    echo ""
    log_msg "[GPU 0] Running Text Relay Baseline on ARC-Easy"

    run_experiment "text_relay_arc_easy" "python telepathy/run_baselines.py \
        --baseline text_relay \
        --dataset arc_easy \
        --gpu 0 \
        --max_samples 200 \
        --output_dir '${OUTPUT_DIR}/baselines/text_relay' \
        2>&1 | tee '${OUTPUT_DIR}/baselines/text_relay_arc_easy.log'"

    # -----------------------------------------------------------------
    # ENSEMBLE BASELINE (Alpha-weighted Llama+Mistral ensemble)
    # Tests if simple model averaging beats the bridge
    # -----------------------------------------------------------------
    echo ""
    log_msg "[GPU 0] Running Ensemble Baseline on ARC-Easy"

    run_experiment "ensemble_arc_easy" "python telepathy/run_baselines.py \
        --baseline ensemble \
        --dataset arc_easy \
        --alphas 0.3 0.5 0.7 \
        --gpu 0 \
        --max_samples 200 \
        --output_dir '${OUTPUT_DIR}/baselines/ensemble' \
        2>&1 | tee '${OUTPUT_DIR}/baselines/ensemble_arc_easy.log'"

    # -----------------------------------------------------------------
    # GSM8K MATH REASONING BASELINES (Generative Task)
    # Tests whether bridge can transfer mathematical reasoning capability
    # -----------------------------------------------------------------
    echo ""
    log_msg "[GPU 0] Running GSM8K Math Reasoning Baselines"

    log_msg "[GPU 0] Zero-shot on GSM8K"
    run_experiment "zeroshot_gsm8k" "python telepathy/run_baselines.py \
        --baseline zeroshot \
        --dataset gsm8k \
        --gpu 0 \
        --max_samples 100 \
        --output_dir '${OUTPUT_DIR}/baselines/zeroshot' \
        2>&1 | tee '${OUTPUT_DIR}/baselines/zeroshot_gsm8k.log'"

    log_msg "[GPU 0] Few-shot (4-shot) on GSM8K"
    run_experiment "fewshot_gsm8k" "python telepathy/run_baselines.py \
        --baseline fewshot \
        --dataset gsm8k \
        --shots 4 \
        --seeds 42 \
        --gpu 0 \
        --max_samples 100 \
        --output_dir '${OUTPUT_DIR}/baselines/fewshot' \
        2>&1 | tee '${OUTPUT_DIR}/baselines/fewshot_gsm8k.log'"

    log_msg "[GPU 0] All evaluations completed"
}

# =============================================================================
# GPU 1: NOVEL BRIDGES + TOKEN CAPACITY + LATENCY/MEMORY/THROUGHPUT/BATCHED BENCHMARKS
# =============================================================================
run_gpu1() {
    log_msg "[GPU 1] Starting: Novel Bridges + Token Capacity + All Benchmarks"

    # -----------------------------------------------------------------
    # NOVEL BRIDGES on ARC-Easy (simplest reasoning task)
    # 3 original bridges: flow_matching, optimal_transport, moe (Mixtral/DeepSeek-style)
    # Multi-seed for statistical significance
    # -----------------------------------------------------------------
    NOVEL_BRIDGES="flow_matching optimal_transport moe"
    ABLATION_SEEDS="42"  # Single seed for now - verify reasoning benchmarks work first

    for BRIDGE in $NOVEL_BRIDGES; do
        for SEED in $ABLATION_SEEDS; do
            echo ""
            log_msg "[GPU 1] Testing $BRIDGE on ARC-Easy (seed=$SEED)"

            run_experiment "${BRIDGE}_arc_easy_seed${SEED}" "python telepathy/train_telepathy.py \
                --dataset arc_easy \
                --bridge_type $BRIDGE \
                --soft_tokens 8 \
                --steps 1500 \
                --eval_every 200 \
                --seed $SEED \
                --output_dir '${OUTPUT_DIR}/novel_bridges/${BRIDGE}_arc_easy_seed${SEED}' \
                --gpu 1 \
                2>&1 | tee '${OUTPUT_DIR}/novel_bridges/${BRIDGE}_arc_easy_seed${SEED}.log'"
        done
    done

    # -----------------------------------------------------------------
    # HAIL MARY BRIDGES on ARC-Easy (Literature-Informed Novel Experiments)
    # 6 new bridges from systematic literature review:
    # - cross_modal_distillation: KL divergence to sender logits
    # - mine: Mutual Information Neural Estimation (tighter than InfoNCE)
    # - mixture_of_depths: Adaptive compute via early exit
    # - thalamic_relay: Neuroscience-inspired inhibitory gating
    # - domain_adversarial: Gradient reversal for alignment
    # - successive_refinement: Progressive token generation
    # -----------------------------------------------------------------
    echo ""
    log_msg "[GPU 1] Starting HAIL MARY Bridge Experiments (6 literature-informed bridges)"

    HAIL_MARY_BRIDGES="cross_modal_distillation mine mixture_of_depths thalamic_relay domain_adversarial successive_refinement"

    for BRIDGE in $HAIL_MARY_BRIDGES; do
        for SEED in $ABLATION_SEEDS; do
            echo ""
            log_msg "[GPU 1] Testing HAIL MARY $BRIDGE on ARC-Easy (seed=$SEED)"

            run_experiment "hailmary_${BRIDGE}_arc_easy_seed${SEED}" "python telepathy/train_telepathy.py \
                --dataset arc_easy \
                --bridge_type $BRIDGE \
                --soft_tokens 8 \
                --steps 1500 \
                --eval_every 200 \
                --seed $SEED \
                --output_dir '${OUTPUT_DIR}/novel_bridges/hailmary_${BRIDGE}_arc_easy_seed${SEED}' \
                --gpu 1 \
                2>&1 | tee '${OUTPUT_DIR}/novel_bridges/hailmary_${BRIDGE}_arc_easy_seed${SEED}.log'"
        done
    done

    # -----------------------------------------------------------------
    # TOKEN CAPACITY EXPERIMENT (Does more tokens help reasoning?)
    # Full ablation curve: 4, 8, 16, 32 tokens
    # Multi-seed for statistical significance
    # -----------------------------------------------------------------
    echo ""
    log_msg "[GPU 1] Token Capacity Scaling on Reasoning (4, 8, 16, 32 tokens, 3 seeds)"

    for TOKENS in 4 8 16 32; do
        for SEED in $ABLATION_SEEDS; do
            log_msg "[GPU 1] Testing $TOKENS soft tokens on ARC-Easy (seed=$SEED)"
            run_experiment "token_capacity_${TOKENS}_seed${SEED}" "python telepathy/train_telepathy.py \
                --dataset arc_easy \
                --bridge_type standard \
                --soft_tokens $TOKENS \
                --steps 1500 \
                --eval_every 200 \
                --seed $SEED \
                --output_dir '${OUTPUT_DIR}/novel_bridges/token_capacity_${TOKENS}_seed${SEED}' \
                --gpu 1 \
                2>&1 | tee '${OUTPUT_DIR}/novel_bridges/token_capacity_${TOKENS}_seed${SEED}.log'"
        done
    done

    # -----------------------------------------------------------------
    # GSM8K BRIDGE TRAINING (Math Reasoning - Generative Task)
    # Test if bridge can transfer mathematical reasoning from Llama to Mistral
    # -----------------------------------------------------------------
    echo ""
    log_msg "[GPU 1] Training Bridge on GSM8K (Math Reasoning)"

    for SEED in $ABLATION_SEEDS; do
        log_msg "[GPU 1] Training Mixture of Depths bridge on GSM8K (seed=$SEED)"
        run_experiment "bridge_gsm8k_mod_seed${SEED}" "python telepathy/train_telepathy.py \
            --dataset gsm8k \
            --bridge_type mixture_of_depths \
            --soft_tokens 8 \
            --steps 1500 \
            --eval_every 200 \
            --seed $SEED \
            --output_dir '${OUTPUT_DIR}/novel_bridges/gsm8k_mod_seed${SEED}' \
            --gpu 1 \
            2>&1 | tee '${OUTPUT_DIR}/novel_bridges/gsm8k_mod_seed${SEED}.log'"
    done

    # -----------------------------------------------------------------
    # RIDGE REGRESSION BASELINE (Training-free linear alignment)
    # LatentMAS-style: W_align = (X'X + λI)^{-1} X'Y
    # -----------------------------------------------------------------
    echo ""
    log_msg "[GPU 1] Running Ridge Regression Baseline on ARC-Easy"

    run_experiment "ridge_regression_arc_easy" "python telepathy/run_baselines.py \
        --baseline ridge_regression \
        --dataset arc_easy \
        --lambda_values 0.1 1.0 10.0 100.0 \
        --gpu 1 \
        --max_samples 200 \
        --output_dir '${OUTPUT_DIR}/baselines/ridge_regression' \
        2>&1 | tee '${OUTPUT_DIR}/baselines/ridge_regression_arc_easy.log'"

    # -----------------------------------------------------------------
    # LATENCY BENCHMARKS
    # -----------------------------------------------------------------
    echo ""
    log_msg "[GPU 1] Running Latency Benchmarks"

    # Find a checkpoint to use for latency benchmark (from token_capacity_8, seed 42)
    CHECKPOINT="${OUTPUT_DIR}/novel_bridges/token_capacity_8_seed42/bridge_arc_easy.pt"

    # Wait a bit for checkpoint to be written
    sleep 5

    if [ -f "$CHECKPOINT" ]; then
        log_msg "[GPU 1] Running latency benchmark with checkpoint: $CHECKPOINT"
        run_experiment "latency_benchmark" "python telepathy/run_benchmarks.py \
            --benchmark latency \
            --checkpoint '$CHECKPOINT' \
            --soft_tokens 8 \
            --num_trials 50 \
            --warmup 5 \
            --gpu 1 \
            --output_dir '${OUTPUT_DIR}/benchmarks' \
            2>&1 | tee '${OUTPUT_DIR}/benchmarks/latency.log'"
    else
        log_msg "[GPU 1] No checkpoint found, running latency benchmark without bridge"
        run_experiment "latency_benchmark_no_checkpoint" "python telepathy/run_benchmarks.py \
            --benchmark latency \
            --soft_tokens 8 \
            --num_trials 50 \
            --warmup 5 \
            --gpu 1 \
            --output_dir '${OUTPUT_DIR}/benchmarks' \
            2>&1 | tee '${OUTPUT_DIR}/benchmarks/latency.log'"
    fi

    # -----------------------------------------------------------------
    # BATCHED LATENCY BENCHMARK (Critical for MLSys paper)
    # Tests throughput at different batch sizes: 1, 4, 8, 16
    # -----------------------------------------------------------------
    echo ""
    log_msg "[GPU 1] Running Batched Latency Benchmark (batch sizes: 1, 4, 8, 16)"
    run_experiment "batched_latency_benchmark" "python telepathy/run_benchmarks.py \
        --benchmark batched \
        --batch_sizes 1 4 8 16 \
        --soft_tokens 8 \
        --num_samples 100 \
        --num_runs 5 \
        --warmup 2 \
        --gpu 1 \
        --output_dir '${OUTPUT_DIR}/benchmarks' \
        2>&1 | tee '${OUTPUT_DIR}/benchmarks/batched_latency.log'"

    # Memory benchmark
    echo ""
    log_msg "[GPU 1] Running Memory Benchmark"
    run_experiment "memory_benchmark" "python telepathy/run_benchmarks.py \
        --benchmark memory \
        --gpu 1 \
        --output_dir '${OUTPUT_DIR}/benchmarks' \
        2>&1 | tee '${OUTPUT_DIR}/benchmarks/memory.log'"

    # Throughput benchmark
    echo ""
    log_msg "[GPU 1] Running Throughput Benchmark"
    if [ -f "$CHECKPOINT" ]; then
        run_experiment "throughput_benchmark" "python telepathy/run_benchmarks.py \
            --benchmark throughput \
            --checkpoint '$CHECKPOINT' \
            --soft_tokens 8 \
            --num_samples 100 \
            --gpu 1 \
            --output_dir '${OUTPUT_DIR}/benchmarks' \
            2>&1 | tee '${OUTPUT_DIR}/benchmarks/throughput.log'"
    else
        run_experiment "throughput_benchmark_no_checkpoint" "python telepathy/run_benchmarks.py \
            --benchmark throughput \
            --soft_tokens 8 \
            --num_samples 100 \
            --gpu 1 \
            --output_dir '${OUTPUT_DIR}/benchmarks' \
            2>&1 | tee '${OUTPUT_DIR}/benchmarks/throughput.log'"
    fi

    log_msg "[GPU 1] All experiments completed"
}

# =============================================================================
# RUN BOTH GPUS IN PARALLEL
# =============================================================================
echo ""
log_msg "Starting parallel execution on 2 GPUs..."
echo ""

run_gpu0 &
GPU0_PID=$!

run_gpu1 &
GPU1_PID=$!

log_msg "GPU 0 PID: $GPU0_PID (Standard Bridge + Baselines)"
log_msg "GPU 1 PID: $GPU1_PID (Novel Bridges + Benchmarks)"
echo ""

wait $GPU0_PID
log_msg "GPU 0 finished"

wait $GPU1_PID
log_msg "GPU 1 finished"

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================
echo ""
echo "=============================================================="
echo "AGGREGATING REASONING RESULTS"
echo "=============================================================="

python -c "
import json
from pathlib import Path
import os

output_dir = Path('${OUTPUT_DIR}')

print('='*70)
print('REASONING BENCHMARK RESULTS (Publication Ready)')
print('='*70)

# Standard bridge results (from hail mary)
hail_mary_results = list(output_dir.glob('standard_bridge/run_*/enhanced_evaluation_results.json'))
if hail_mary_results:
    with open(hail_mary_results[0]) as f:
        results = json.load(f)

    print()
    print('STANDARD BRIDGE (3 seeds):')
    print('-'*40)

    for dataset in ['arc_easy', 'winogrande', 'hellaswag', 'boolq']:
        if 'classification' in results and 'aggregated' in results['classification']:
            for method in ['bridge_8', 'zeroshot_llama', 'zeroshot_mistral', 'linear_probe']:
                stats = results['classification']['aggregated'].get(method, {}).get(dataset, {})
                if stats and 'mean' in stats:
                    print(f'  {dataset:12} | {method:18} | {stats[\"mean\"]:.1f}% +/- {stats.get(\"std\", 0):.1f}%')

# Zero-shot baseline results (NEW)
print()
print('ZERO-SHOT BASELINE (no training):')
print('-'*40)

for zeroshot_file in sorted(output_dir.glob('baselines/zeroshot/*.json')):
    with open(zeroshot_file) as f:
        data = json.load(f)
    if 'results' in data:
        dataset = data.get('config', {}).get('dataset', 'unknown')
        for model, stats in data['results'].items():
            if isinstance(stats, dict) and 'accuracy' in stats:
                print(f'  {dataset:12} | {model:10} | {stats[\"accuracy\"]:.1f}%')

# Few-shot baseline results
print()
print('FEW-SHOT BASELINE (4-shot):')
print('-'*40)

for fewshot_file in sorted(output_dir.glob('baselines/fewshot/*.json')):
    with open(fewshot_file) as f:
        data = json.load(f)
    if 'results' in data:
        dataset = data.get('config', {}).get('dataset', 'unknown')
        for model, stats in data['results'].items():
            if isinstance(stats, dict) and 'mean_accuracy' in stats:
                print(f'  {dataset:12} | {model:10} | {stats[\"mean_accuracy\"]:.1f}% +/- {stats.get(\"std_accuracy\", 0):.1f}%')

# DoRA baseline results
print()
print('DORA BASELINE (rank=8, SOTA 2024):')
print('-'*40)

for dora_file in sorted(output_dir.glob('baselines/dora/*.json')):
    with open(dora_file) as f:
        data = json.load(f)
    if 'results' in data and 'summary' in data['results']:
        dataset = data.get('config', {}).get('dataset', 'unknown')
        summary = data['results']['summary']
        print(f'  {dataset:12} | DoRA r=8   | {summary[\"mean_accuracy\"]:.1f}% +/- {summary.get(\"std_accuracy\", 0):.1f}%')

# Prompt tuning baseline results
print()
print('PROMPT TUNING BASELINE (8 tokens, no sender):')
print('-'*40)

for pt_file in sorted(output_dir.glob('baselines/prompt_tuning/*.json')):
    with open(pt_file) as f:
        data = json.load(f)
    if 'results' in data and 'summary' in data['results']:
        dataset = data.get('config', {}).get('dataset', 'unknown')
        summary = data['results']['summary']
        print(f'  {dataset:12} | Prompt Tuning | {summary[\"mean_accuracy\"]:.1f}% +/- {summary.get(\"std_accuracy\", 0):.1f}%')

# Novel bridge results (fixed glob to include seed suffix)
print()
print('NOVEL BRIDGES (ARC-Easy):')
print('-'*40)

for bridge_dir in sorted(output_dir.glob('novel_bridges/*_arc_easy_seed*')):
    # Skip hail mary bridges (handled separately) and token capacity
    if 'hailmary_' in bridge_dir.name or 'token_capacity' in bridge_dir.name:
        continue
    for json_file in bridge_dir.glob('*_results.json'):
        with open(json_file) as f:
            data = json.load(f)
        if 'final_results' in data:
            acc = data['final_results'].get('accuracy', 0)
            # Extract bridge name: flow_matching_arc_easy_seed42 -> flow_matching
            bridge_name = bridge_dir.name.replace('_arc_easy_seed42', '').replace('_arc_easy_seed123', '').replace('_arc_easy_seed456', '')
            print(f'  {bridge_name:25} | {acc:.1f}%')

# HAIL MARY bridge results (6 literature-informed experiments)
print()
print('HAIL MARY BRIDGES (ARC-Easy):')
print('-'*40)

for bridge_dir in sorted(output_dir.glob('novel_bridges/hailmary_*_arc_easy_seed*')):
    for json_file in bridge_dir.glob('*_results.json'):
        with open(json_file) as f:
            data = json.load(f)
        if 'final_results' in data:
            acc = data['final_results'].get('accuracy', 0)
            # Extract bridge name: hailmary_thalamic_relay_arc_easy_seed42 -> thalamic_relay
            bridge_name = bridge_dir.name.replace('hailmary_', '').replace('_arc_easy_seed42', '').replace('_arc_easy_seed123', '').replace('_arc_easy_seed456', '')
            print(f'  {bridge_name:30} | {acc:.1f}%')

# Token capacity results (fixed token extraction)
print()
print('TOKEN CAPACITY (ARC-Easy):')
print('-'*40)

for cap_dir in sorted(output_dir.glob('novel_bridges/token_capacity_*_seed*')):
    for json_file in cap_dir.glob('*_results.json'):
        with open(json_file) as f:
            data = json.load(f)
        if 'final_results' in data:
            acc = data['final_results'].get('accuracy', 0)
            # Extract tokens: token_capacity_8_seed42 -> 8 (index 2)
            tokens = cap_dir.name.split('_')[2]
            print(f'  {tokens} tokens | {acc:.1f}%')

# Latency benchmark results
print()
print('LATENCY BENCHMARKS:')
print('-'*40)

for bench_file in sorted(output_dir.glob('benchmarks/latency_*.json')):
    with open(bench_file) as f:
        data = json.load(f)
    if 'results' in data:
        results = data['results']
        if 'direct_text' in results:
            print(f'  Direct Text:  {results[\"direct_text\"][\"avg_ms\"]:.1f} ms')
        if 'text_relay' in results:
            print(f'  Text-Relay:   {results[\"text_relay\"][\"avg_ms\"]:.1f} ms')
        if 'bridge' in results:
            print(f'  Bridge:       {results[\"bridge\"][\"avg_ms\"]:.1f} ms')
            if 'text_relay' in results:
                speedup = results['text_relay']['avg_ms'] / results['bridge']['avg_ms']
                print(f'  Speedup:      {speedup:.1f}x vs Text-Relay')

# Batched latency benchmark results (NEW - Critical for MLSys)
print()
print('BATCHED LATENCY (samples/sec by batch size):')
print('-'*40)

for bench_file in sorted(output_dir.glob('benchmarks/batched_*.json')):
    with open(bench_file) as f:
        data = json.load(f)
    if 'results' in data:
        results = data['results']
        print(f'  {\"Batch\":<8} {\"Bridge\":<15} {\"Direct\":<15}')
        print(f'  {\"-----\":<8} {\"------\":<15} {\"------\":<15}')
        if 'bridge' in results and 'direct_mistral' in results:
            for i, bridge_res in enumerate(results['bridge']):
                bs = bridge_res['batch_size']
                bridge_tp = bridge_res['throughput_samples_per_s']
                direct_tp = results['direct_mistral'][i]['throughput_samples_per_s']
                print(f'  {bs:<8} {bridge_tp:<15.1f} {direct_tp:<15.1f}')

print('='*70)

# Save combined results
combined = {
    'hail_mary': str(hail_mary_results[0]) if hail_mary_results else None,
    'novel_bridges': {},
    'token_capacity': {},
    'baselines': {
        'zeroshot': {},
        'fewshot': {},
        'dora': {},
        'prompt_tuning': {}
    },
    'benchmarks': {
        'latency': {},
        'batched': {},
        'memory': {},
        'throughput': {}
    }
}
with open(output_dir / 'reasoning_results_summary.json', 'w') as f:
    json.dump(combined, f, indent=2)
" || echo "Could not aggregate (some experiments may have failed)"

# =============================================================================
# EXPERIMENT SUMMARY AND GIT COMMIT
# =============================================================================
echo ""
echo "=============================================================="
echo "EXPERIMENT SUMMARY"
echo "=============================================================="
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo "Successful: $SUCCESSFUL_EXPERIMENTS"
echo "Failed: ${#FAILED_EXPERIMENTS[@]}"
echo "Skipped (already completed): $SKIPPED_COUNT"

if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed experiments:"
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - $exp"
    done
fi

if [ ${#SKIPPED_EXPERIMENTS[@]} -gt 0 ] && [ ${#SKIPPED_EXPERIMENTS[@]} -le 10 ]; then
    echo ""
    echo "Skipped experiments:"
    for exp in "${SKIPPED_EXPERIMENTS[@]}"; do
        echo "  - $exp"
    done
elif [ ${#SKIPPED_EXPERIMENTS[@]} -gt 10 ]; then
    echo ""
    echo "Skipped ${#SKIPPED_EXPERIMENTS[@]} experiments (showing first 10):"
    for exp in "${SKIPPED_EXPERIMENTS[@]:0:10}"; do
        echo "  - $exp"
    done
    echo "  ... and $((${#SKIPPED_EXPERIMENTS[@]} - 10)) more"
fi

# Print registry summary if enabled
if [ "$REGISTRY_ENABLED" = "1" ]; then
    echo ""
    echo "Registry summary:"
    python telepathy/experiment_registry.py --registry "$REGISTRY_FILE" summary 2>/dev/null || echo "  (could not get registry summary)"
fi
echo "=============================================================="

# Git commit and push (include registry file)
git add "$OUTPUT_DIR"
[ -f "$REGISTRY_FILE" ] && git add "$REGISTRY_FILE"
COMMIT_MSG="Reasoning experiments: job $SLURM_JOB_ID - ${SUCCESSFUL_EXPERIMENTS}/${TOTAL_EXPERIMENTS} succeeded, ${SKIPPED_COUNT} skipped

REASONING BENCHMARKS:
- ARC-Easy, WinoGrande, HellaSwag, BoolQ

EXPERIMENTS:
1. Standard bridge (3 seeds) via hail_mary evaluation
2. Zero-shot baseline on all 4 reasoning datasets (Llama + Mistral)
3. Few-shot baseline (4-shot) on arc_easy, winogrande, hellaswag, boolq
4. DoRA baseline (rank=8) on arc_easy (SOTA PEFT 2024)
5. Prompt tuning baseline (8 tokens, no sender) on arc_easy
6. Linear probe baseline (layer 31) on all reasoning datasets
7. Novel bridges (3):
   - Flow Matching, Optimal Transport, MoE (Mixtral/DeepSeek-style)
8. Token capacity scaling (4, 8, 16, 32 tokens)
9. Latency/memory/throughput benchmarks
10. Batched latency benchmark (batch sizes: 1, 4, 8, 16) - Critical for MLSys

KEY QUESTIONS:
- Can novel compression techniques improve reasoning?
- Does more tokens help reasoning?
- What's the latency advantage over text-relay?
- Does the sender (Llama) actually help? (prompt_tuning vs bridge)
- How does throughput scale with batch size?

Runtime: $SECONDS seconds ($(($SECONDS / 3600))h $(($SECONDS % 3600 / 60))m)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

git commit -m "$COMMIT_MSG"
push_with_retry

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "=============================================================="
echo "REASONING EXPERIMENTS COMPLETED (Publication Ready)"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds ($(($SECONDS / 3600))h $(($SECONDS % 3600 / 60))m)"
echo ""
echo "KEY EXPERIMENTS:"
echo "  1. Standard bridge on reasoning? (hail_mary)"
echo "  2. Zero-shot baseline? (baselines/zeroshot/)"
echo "  3. Few-shot baseline (4-shot)? (baselines/fewshot/)"
echo "  4. DoRA baseline (SOTA 2024)? (baselines/dora/)"
echo "  5. Prompt tuning (does sender help)? (baselines/prompt_tuning/)"
echo "  6. Linear probe upper bound? (baselines/linear_probe/)"
echo "  7. Novel bridges help? (novel_bridges/)"
echo "  8. More tokens help? (token_capacity: 4, 8, 16, 32)"
echo "  9. Latency advantage? (benchmarks/latency)"
echo " 10. Batched throughput scaling? (benchmarks/batched) - MLSys Critical"
echo ""
echo "Results: ${OUTPUT_DIR}"
echo "=============================================================="
