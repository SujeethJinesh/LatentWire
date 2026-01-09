#!/usr/bin/env bash
# Statistical Validation Suite for Telepathy
# Runs existing experiments with proper statistical rigor
# Usage: bash telepathy/scripts/run_statistical_validation.sh

set -e

# Configuration
OUTPUT_DIR="runs/statistical_validation_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT="${CHECKPOINT:-runs/telepathy_checkpoint/best_model.pt}"
SEEDS="42 123 456 789 2024"
DATASETS="sst2 agnews trec banking77"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/validation.log"

echo "==============================================================" | tee "$LOG_FILE"
echo "Statistical Validation Suite for Telepathy" | tee -a "$LOG_FILE"
echo "==============================================================" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Checkpoint: $CHECKPOINT" | tee -a "$LOG_FILE"
echo "Seeds: $SEEDS" | tee -a "$LOG_FILE"
echo "Datasets: $DATASETS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to run experiment with seed
run_experiment() {
    local dataset=$1
    local seed=$2
    local output_file="$OUTPUT_DIR/${dataset}_seed${seed}.json"

    echo "[$(date +%H:%M:%S)] Running $dataset with seed $seed..." | tee -a "$LOG_FILE"

    {
        python telepathy/eval_classification.py \
            --checkpoint "$CHECKPOINT" \
            --dataset "$dataset" \
            --seed "$seed" \
            --full_test \
            --output_json "$output_file" \
            --batch_size 32 \
            --max_examples -1
    } 2>&1 | tee -a "$LOG_FILE"

    if [ -f "$output_file" ]; then
        echo "[$(date +%H:%M:%S)] ✓ Completed $dataset seed $seed" | tee -a "$LOG_FILE"
    else
        echo "[$(date +%H:%M:%S)] ✗ Failed $dataset seed $seed" | tee -a "$LOG_FILE"
    fi
}

# Main loop - run all experiments
echo "Starting validation runs..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for dataset in $DATASETS; do
    echo "==============================================================" | tee -a "$LOG_FILE"
    echo "Dataset: $dataset" | tee -a "$LOG_FILE"
    echo "==============================================================" | tee -a "$LOG_FILE"

    for seed in $SEEDS; do
        run_experiment "$dataset" "$seed"
    done

    # Compute statistics after all seeds
    echo "" | tee -a "$LOG_FILE"
    echo "Computing statistics for $dataset..." | tee -a "$LOG_FILE"

    {
        python -c "
import json
import numpy as np
from scipy import stats
import glob

# Load all results for this dataset
files = glob.glob('$OUTPUT_DIR/${dataset}_seed*.json')
accuracies = []
f1_scores = []

for f in files:
    with open(f) as fp:
        data = json.load(fp)
        accuracies.append(data.get('accuracy', 0))
        f1_scores.append(data.get('f1', 0))

if accuracies:
    # Compute statistics
    acc_mean = np.mean(accuracies)
    acc_std = np.std(accuracies)
    acc_ci = stats.t.interval(0.95, len(accuracies)-1,
                               loc=acc_mean,
                               scale=stats.sem(accuracies))

    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    f1_ci = stats.t.interval(0.95, len(f1_scores)-1,
                             loc=f1_mean,
                             scale=stats.sem(f1_scores))

    print(f'Results for {dataset.upper()}:')
    print(f'  Accuracy: {acc_mean:.3f} ± {acc_std:.3f}')
    print(f'  95% CI: [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]')
    print(f'  F1 Score: {f1_mean:.3f} ± {f1_std:.3f}')
    print(f'  95% CI: [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]')
    print(f'  Seeds run: {len(accuracies)}')

    # Save summary
    summary = {
        'dataset': dataset,
        'accuracy': {'mean': acc_mean, 'std': acc_std, 'ci': list(acc_ci)},
        'f1': {'mean': f1_mean, 'std': f1_std, 'ci': list(f1_ci)},
        'n_seeds': len(accuracies),
        'raw_accuracies': accuracies,
        'raw_f1_scores': f1_scores
    }

    with open('$OUTPUT_DIR/${dataset}_summary.json', 'w') as fp:
        json.dump(summary, fp, indent=2)
"
    } 2>&1 | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
done

# Generate final report
echo "==============================================================" | tee -a "$LOG_FILE"
echo "Generating Final Report" | tee -a "$LOG_FILE"
echo "==============================================================" | tee -a "$LOG_FILE"

{
    python -c "
import json
import glob
import pandas as pd

# Load all summaries
summaries = []
for f in glob.glob('$OUTPUT_DIR/*_summary.json'):
    with open(f) as fp:
        summaries.append(json.load(fp))

# Create table
if summaries:
    rows = []
    for s in summaries:
        rows.append({
            'Dataset': s['dataset'].upper(),
            'Accuracy': f\"{s['accuracy']['mean']:.3f} ± {s['accuracy']['std']:.3f}\",
            '95% CI': f\"[{s['accuracy']['ci'][0]:.3f}, {s['accuracy']['ci'][1]:.3f}]\",
            'F1 Score': f\"{s['f1']['mean']:.3f} ± {s['f1']['std']:.3f}\",
            'Seeds': s['n_seeds']
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Save as CSV for paper
    df.to_csv('$OUTPUT_DIR/statistical_validation_results.csv', index=False)

    # Save as LaTeX table
    with open('$OUTPUT_DIR/results_table.tex', 'w') as f:
        f.write(df.to_latex(index=False))
"
} 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "==============================================================" | tee -a "$LOG_FILE"
echo "Statistical validation complete!" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "==============================================================" | tee -a "$LOG_FILE"