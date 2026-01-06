#!/usr/bin/env bash
# =============================================================================
# UNIFIED EXPERIMENT RUNNER WITH PREEMPTION HANDLING
# =============================================================================
# This script handles ALL experiment scenarios:
# - Automatic preemption recovery and job resumption
# - GPU utilization maximization
# - Comprehensive logging and progress tracking
# - Works with both srun (interactive) and sbatch (batch)
#
# Usage (simple - user just runs this):
#   srun --account=marlowe-m000066 --partition=preempt --gpus=1 --mem=40G --time=04:00:00 --pty bash telepathy/run_experiments.sh
#
# Or with sbatch:
#   Create a SLURM script that just calls: bash telepathy/run_experiments.sh
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths (using /projects for HPC)
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
OUTPUT_DIR="${WORK_DIR}/runs/unified_experiments"
STATE_FILE="${OUTPUT_DIR}/experiment_state.json"
LOG_DIR="${OUTPUT_DIR}/logs"

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/main_experiment_${TIMESTAMP}.log"

# Experiment configuration
DATASETS=("sst2" "agnews" "trec")
SEEDS=(42 123 456)
EXPERIMENTS=("bridge" "prompt_tuning" "text_relay" "fewshot" "zeroshot" "linear_probe" "llmlingua")

# Training hyperparameters (optimized for preemption)
TRAIN_SAMPLES=1000      # Quick iterations for preemptible jobs
EVAL_SAMPLES=500        # Enough for statistical significance
BATCH_SIZE=32           # Balance speed vs memory
LATENT_DIM=32          # Standard compression factor
EPOCHS=10              # Short training cycles
SAVE_INTERVAL=100      # Frequent checkpointing

# =============================================================================
# SETUP AND INITIALIZATION
# =============================================================================

# Ensure we're in the right directory
cd "$WORK_DIR"

# Create output directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "${OUTPUT_DIR}/checkpoints" "${OUTPUT_DIR}/results"

# Initialize logging
exec > >(tee -a "$MAIN_LOG")
exec 2>&1

echo "=============================================================="
echo "EXPERIMENT RUNNER STARTED"
echo "=============================================================="
echo "Timestamp: $(date)"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Working directory: $WORK_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "State file: $STATE_FILE"
echo "=============================================================="

# Pull latest code
echo "Pulling latest code..."
git pull || true

# Set up Python environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

# Initialize or load experiment state
initialize_state() {
    if [ ! -f "$STATE_FILE" ]; then
        echo "Initializing new experiment state..."
        cat > "$STATE_FILE" << EOF
{
    "start_time": "$(date -Iseconds)",
    "completed": [],
    "in_progress": null,
    "failed": [],
    "remaining": [],
    "checkpoints": {},
    "results": {},
    "preemption_count": 0
}
EOF
        # Initialize remaining experiments
        python3 -c "
import json
experiments = []
for dataset in ${DATASETS[@]@Q}.split():
    for seed in ${SEEDS[@]@Q}.split():
        for exp_type in ${EXPERIMENTS[@]@Q}.split():
            experiments.append(f'{dataset}_{exp_type}_{seed}')

with open('$STATE_FILE', 'r') as f:
    state = json.load(f)
state['remaining'] = experiments
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
"
    else
        echo "Loading existing experiment state..."
        # Handle preemption recovery
        python3 -c "
import json
with open('$STATE_FILE', 'r') as f:
    state = json.load(f)

# Move any in-progress back to remaining (preemption recovery)
if state.get('in_progress'):
    state['remaining'].insert(0, state['in_progress'])
    state['in_progress'] = None
    state['preemption_count'] = state.get('preemption_count', 0) + 1

with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)

print(f\"Recovered from preemption #{state['preemption_count']}\")
print(f\"Completed: {len(state['completed'])} experiments\")
print(f\"Remaining: {len(state['remaining'])} experiments\")
"
    fi
}

# Update state for an experiment
update_state() {
    local exp_name="$1"
    local status="$2"  # started, completed, failed
    local result_file="${3:-}"

    python3 -c "
import json
with open('$STATE_FILE', 'r') as f:
    state = json.load(f)

if '$status' == 'started':
    state['in_progress'] = '$exp_name'
elif '$status' == 'completed':
    state['completed'].append('$exp_name')
    if state.get('in_progress') == '$exp_name':
        state['in_progress'] = None
    if '$exp_name' in state['remaining']:
        state['remaining'].remove('$exp_name')
    if '$result_file':
        state['results']['$exp_name'] = '$result_file'
elif '$status' == 'failed':
    state['failed'].append('$exp_name')
    if state.get('in_progress') == '$exp_name':
        state['in_progress'] = None
    # Put back at end of queue for retry
    if '$exp_name' not in state['remaining']:
        state['remaining'].append('$exp_name')

with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
"
}

# Get next experiment to run
get_next_experiment() {
    python3 -c "
import json
with open('$STATE_FILE', 'r') as f:
    state = json.load(f)
if state['remaining']:
    print(state['remaining'][0])
else:
    print('NONE')
"
}

# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

# Run Bridge experiment (Llama → Mistral)
run_bridge_experiment() {
    local dataset="$1"
    local seed="$2"
    local exp_dir="${OUTPUT_DIR}/bridge_${dataset}_${seed}"
    local log_file="${LOG_DIR}/bridge_${dataset}_${seed}_${TIMESTAMP}.log"

    echo "Running Bridge experiment: dataset=$dataset, seed=$seed"

    {
        python telepathy/train_telepathy_${dataset}.py \
            --latent_dim $LATENT_DIM \
            --train_samples $TRAIN_SAMPLES \
            --eval_samples $EVAL_SAMPLES \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --seed $seed \
            --output_dir "$exp_dir" \
            --save_every $SAVE_INTERVAL \
            --use_perceiver_bridge \
            --compress_embeds \
            2>&1 | tee "$log_file"

        # Run evaluation
        python telepathy/eval_telepathy_${dataset}.py \
            --checkpoint "${exp_dir}/best_model.pt" \
            --eval_samples $EVAL_SAMPLES \
            --output_file "${exp_dir}/results.json" \
            2>&1 | tee -a "$log_file"

    } && return 0 || return 1
}

# Run Prompt Tuning experiment (Mistral only)
run_prompt_tuning_experiment() {
    local dataset="$1"
    local seed="$2"
    local exp_dir="${OUTPUT_DIR}/prompt_tuning_${dataset}_${seed}"
    local log_file="${LOG_DIR}/prompt_tuning_${dataset}_${seed}_${TIMESTAMP}.log"

    echo "Running Prompt Tuning experiment: dataset=$dataset, seed=$seed"

    {
        python telepathy/train_prompt_tuning_baseline.py \
            --dataset $dataset \
            --num_virtual_tokens $LATENT_DIM \
            --train_samples $TRAIN_SAMPLES \
            --eval_samples $EVAL_SAMPLES \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --seed $seed \
            --output_dir "$exp_dir" \
            2>&1 | tee "$log_file"

    } && return 0 || return 1
}

# Run Text Relay experiment
run_text_relay_experiment() {
    local dataset="$1"
    local seed="$2"
    local exp_dir="${OUTPUT_DIR}/text_relay_${dataset}_${seed}"
    local log_file="${LOG_DIR}/text_relay_${dataset}_${seed}_${TIMESTAMP}.log"

    echo "Running Text Relay experiment: dataset=$dataset, seed=$seed"

    {
        python telepathy/eval_text_relay_baseline.py \
            --dataset $dataset \
            --eval_samples $EVAL_SAMPLES \
            --seed $seed \
            --output_file "${exp_dir}/results.json" \
            2>&1 | tee "$log_file"

    } && return 0 || return 1
}

# Run Few-shot experiment
run_fewshot_experiment() {
    local dataset="$1"
    local seed="$2"
    local exp_dir="${OUTPUT_DIR}/fewshot_${dataset}_${seed}"
    local log_file="${LOG_DIR}/fewshot_${dataset}_${seed}_${TIMESTAMP}.log"

    echo "Running Few-shot experiment: dataset=$dataset, seed=$seed"

    {
        python telepathy/eval_fewshot_baselines.py \
            --dataset $dataset \
            --num_shots 5 \
            --eval_samples $EVAL_SAMPLES \
            --seed $seed \
            --output_file "${exp_dir}/results.json" \
            2>&1 | tee "$log_file"

    } && return 0 || return 1
}

# Run Zero-shot experiment
run_zeroshot_experiment() {
    local dataset="$1"
    local seed="$2"
    local exp_dir="${OUTPUT_DIR}/zeroshot_${dataset}_${seed}"
    local log_file="${LOG_DIR}/zeroshot_${dataset}_${seed}_${TIMESTAMP}.log"

    echo "Running Zero-shot experiment: dataset=$dataset, seed=$seed"

    {
        python telepathy/eval_zeroshot_baselines.py \
            --dataset $dataset \
            --eval_samples $EVAL_SAMPLES \
            --seed $seed \
            --output_file "${exp_dir}/results.json" \
            2>&1 | tee "$log_file"

    } && return 0 || return 1
}

# Run Linear Probe experiment
run_linear_probe_experiment() {
    local dataset="$1"
    local seed="$2"
    local exp_dir="${OUTPUT_DIR}/linear_probe_${dataset}_${seed}"
    local log_file="${LOG_DIR}/linear_probe_${dataset}_${seed}_${TIMESTAMP}.log"

    echo "Running Linear Probe experiment: dataset=$dataset, seed=$seed"

    {
        python telepathy/linear_probe_baseline.py \
            --dataset $dataset \
            --train_samples $TRAIN_SAMPLES \
            --eval_samples $EVAL_SAMPLES \
            --seed $seed \
            --output_dir "$exp_dir" \
            2>&1 | tee "$log_file"

    } && return 0 || return 1
}

# Run LLMLingua experiment
run_llmlingua_experiment() {
    local dataset="$1"
    local seed="$2"
    local exp_dir="${OUTPUT_DIR}/llmlingua_${dataset}_${seed}"
    local log_file="${LOG_DIR}/llmlingua_${dataset}_${seed}_${TIMESTAMP}.log"

    echo "Running LLMLingua experiment: dataset=$dataset, seed=$seed"

    {
        python scripts/run_llmlingua_baseline.py \
            --dataset $dataset \
            --compression_ratio 0.25 \
            --eval_samples $EVAL_SAMPLES \
            --seed $seed \
            --output_file "${exp_dir}/results.json" \
            2>&1 | tee "$log_file"

    } && return 0 || return 1
}

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================

# Initialize state
initialize_state

# Signal handler for graceful shutdown
trap 'echo "Received interrupt signal. Saving state..."; exit 0' SIGINT SIGTERM

# Main experiment loop
while true; do
    # Get next experiment
    NEXT_EXP=$(get_next_experiment)

    if [ "$NEXT_EXP" = "NONE" ]; then
        echo "All experiments completed!"
        break
    fi

    echo ""
    echo "=============================================================="
    echo "Starting experiment: $NEXT_EXP"
    echo "Time: $(date)"
    echo "=============================================================="

    # Parse experiment name (format: dataset_type_seed)
    IFS='_' read -r dataset exp_type seed <<< "$NEXT_EXP"

    # Mark as started
    update_state "$NEXT_EXP" "started"

    # Run the appropriate experiment
    case "$exp_type" in
        "bridge")
            run_bridge_experiment "$dataset" "$seed" && STATUS="completed" || STATUS="failed"
            ;;
        "prompt")
            run_prompt_tuning_experiment "$dataset" "$seed" && STATUS="completed" || STATUS="failed"
            ;;
        "text")
            run_text_relay_experiment "$dataset" "$seed" && STATUS="completed" || STATUS="failed"
            ;;
        "fewshot")
            run_fewshot_experiment "$dataset" "$seed" && STATUS="completed" || STATUS="failed"
            ;;
        "zeroshot")
            run_zeroshot_experiment "$dataset" "$seed" && STATUS="completed" || STATUS="failed"
            ;;
        "linear")
            run_linear_probe_experiment "$dataset" "$seed" && STATUS="completed" || STATUS="failed"
            ;;
        "llmlingua")
            run_llmlingua_experiment "$dataset" "$seed" && STATUS="completed" || STATUS="failed"
            ;;
        *)
            echo "Unknown experiment type: $exp_type"
            STATUS="failed"
            ;;
    esac

    # Update state
    RESULT_FILE="${OUTPUT_DIR}/${exp_type}_${dataset}_${seed}/results.json"
    update_state "$NEXT_EXP" "$STATUS" "$RESULT_FILE"

    echo "Experiment $NEXT_EXP: $STATUS"

    # Check GPU memory periodically
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "GPU Memory Status:"
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    fi

    # Brief pause between experiments
    sleep 2
done

# =============================================================================
# FINAL AGGREGATION
# =============================================================================

echo ""
echo "=============================================================="
echo "AGGREGATING RESULTS"
echo "=============================================================="

# Aggregate all results
python3 telepathy/aggregate_results.py \
    --input_dir "${OUTPUT_DIR}/results" \
    --output_file "${OUTPUT_DIR}/final_results.json" \
    2>&1 | tee "${LOG_DIR}/aggregation_${TIMESTAMP}.log"

# Generate summary report
python3 -c "
import json
import glob
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
results_files = list(output_dir.glob('*/results.json'))

print('\\nEXPERIMENT SUMMARY')
print('=' * 60)

# Load state for statistics
with open('$STATE_FILE', 'r') as f:
    state = json.load(f)

print(f\"Total experiments: {len(state['completed']) + len(state['failed'])}\")
print(f\"Completed: {len(state['completed'])}\")
print(f\"Failed: {len(state['failed'])}\")
print(f\"Preemptions recovered: {state.get('preemption_count', 0)}\")
print()

# Show results summary
if (output_dir / 'final_results.json').exists():
    with open(output_dir / 'final_results.json', 'r') as f:
        results = json.load(f)

    for dataset in results:
        print(f\"\\n{dataset.upper()} Results:\")
        print('-' * 40)
        for method in results[dataset]:
            acc = results[dataset][method].get('accuracy', {})
            if isinstance(acc, dict):
                mean = acc.get('mean', 0) * 100
                std = acc.get('std', 0) * 100
                print(f\"  {method:15s}: {mean:5.1f} ± {std:4.1f}%\")
            else:
                print(f\"  {method:15s}: {acc*100:5.1f}%\")

print()
print('=' * 60)
print(f\"Results saved to: {output_dir}\")
print(f\"Logs saved to: {output_dir}/logs/\")
"

# Push results to git
echo ""
echo "Pushing results to git..."
git add -A
git commit -m "results: unified experiments complete (${TIMESTAMP})

Datasets: ${DATASETS[@]}
Seeds: ${SEEDS[@]}
Experiments: ${EXPERIMENTS[@]}

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
git push || true

echo ""
echo "=============================================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=============================================================="
echo "End time: $(date)"
echo "Total runtime: $((SECONDS/3600))h $(((SECONDS%3600)/60))m $((SECONDS%60))s"
echo "Results: ${OUTPUT_DIR}/final_results.json"
echo "Logs: ${LOG_DIR}/"
echo "=============================================================="