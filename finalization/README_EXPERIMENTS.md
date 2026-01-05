# LatentWire Paper Revision Experiments

## Overview

This directory contains the comprehensive experiment infrastructure for the LatentWire paper revision. The main script (`run_all_experiments.sh`) orchestrates all four phases of experiments required to address reviewer concerns.

## Quick Start

### Local Development (MacBook)

```bash
# Run all experiments locally (small scale for testing)
export MAX_EVAL_SAMPLES=100  # Use small samples for testing
bash finalization/run_all_experiments.sh

# Monitor progress
bash finalization/monitor_experiments.sh
```

### HPC Execution (Full Scale)

```bash
# On HPC cluster
cd /projects/m000066/sujinesh/LatentWire
git pull

# Submit job for all experiments
sbatch finalization/submit_all_experiments.slurm

# Monitor job
squeue -u $USER

# Check progress
bash finalization/monitor_experiments.sh runs/paper_revision_latest
```

## Experiment Phases

The script runs four sequential phases:

### Phase 1: Statistical Rigor
- Evaluates on full test sets (SST-2, AG News, TREC, XSUM)
- Runs with 3 random seeds (42, 123, 456)
- Computes bootstrap confidence intervals (n=10,000)
- Performs McNemar's test for paired comparisons
- Calculates Cohen's d effect sizes

### Phase 2: Linear Probe Baseline
- Tests if simple linear probe on Llama embeddings suffices
- Uses 5-fold cross-validation
- Compares against LatentWire's cross-model transfer
- Proves value of learned compression

### Phase 3: Fair Baseline Comparisons
- **Text-relay**: Full text baseline (upper bound)
- **Token-budget**: Text truncated to same token count
- **LLMLingua-2**: State-of-the-art prompt compression
- **Zero-shot**: Direct prompting without examples
- **Few-shot**: 3-shot prompting baseline

### Phase 4: Efficiency Measurements
- Latency: p50, p95, p99 percentiles
- Memory: Peak GPU memory usage
- Throughput: Samples per second
- Compression: Different quantization levels (fp16, int8, int4)

## Key Features

### Preemption Resilience
- Automatic checkpoint saving every 5 minutes
- Graceful handling of SIGTERM (120s grace period)
- Automatic resumption from latest checkpoint
- State preservation across interruptions

### Single Checkpoint Efficiency
- Trains one checkpoint (24 epochs on SQuAD)
- Reuses same checkpoint for all evaluations
- Minimizes compute requirements
- Ensures consistent comparisons

### Comprehensive Logging
- All output captured with timestamps
- Separate logs for each phase
- Automatic git commit/push of results
- Progress monitoring dashboard

## Configuration

### Environment Variables

```bash
# Training configuration
export SKIP_TRAINING=yes       # Skip training, use existing checkpoint
export CHECKPOINT_PATH=/path/to/checkpoint  # Path to pre-trained checkpoint

# Evaluation configuration
export SEEDS="42 123 456"      # Random seeds for statistical rigor
export DATASETS="sst2 agnews"  # Datasets to evaluate (subset for testing)
export MAX_EVAL_SAMPLES=100    # Limit samples per dataset (for testing)
export BOOTSTRAP_SAMPLES=1000  # Number of bootstrap samples

# Hardware configuration
export NUM_GPUS=1               # Number of GPUs to use
export BATCH_SIZE=8            # Training batch size
export EVAL_BATCH_SIZE=16     # Evaluation batch size
```

### Directory Structure

```
runs/
└── paper_revision_job12345/
    ├── checkpoint/              # Trained model checkpoint
    │   ├── encoder.pt
    │   ├── llama_adapter.pt
    │   └── qwen_adapter.pt
    ├── results/
    │   ├── phase1_statistical/  # Statistical evaluation results
    │   ├── phase2_linear_probe/ # Linear probe baseline results
    │   ├── phase3_baselines/    # Baseline comparison results
    │   └── phase4_efficiency/   # Efficiency measurement results
    ├── logs/                    # All experiment logs
    ├── paper_assets/           # Generated tables and figures
    ├── final_results.json      # Aggregated results
    └── experiment_summary.md   # Human-readable summary
```

## Output Files

### Main Results
- `final_results.json`: Aggregated results from all phases
- `experiment_summary.md`: Human-readable summary with key findings

### Phase-Specific Results
- `phase1_statistical/statistical_summary.json`: Statistical significance tests
- `phase2_linear_probe/comparison_report.json`: Linear probe vs LatentWire
- `phase3_baselines/baseline_comparison.json`: All baseline comparisons
- `phase4_efficiency/efficiency_summary.json`: Performance metrics

### Paper Assets
- `paper_assets/main_results_table.tex`: LaTeX table for paper
- `paper_assets/compression_quality_plot.pdf`: Pareto frontier plot
- `paper_assets/efficiency_comparison.pdf`: Speed/memory comparison

## Monitoring and Debugging

### Real-time Monitoring

```bash
# Monitor experiment progress
bash finalization/monitor_experiments.sh

# Monitor specific run
bash finalization/monitor_experiments.sh runs/paper_revision_job12345

# Single status check (no loop)
MONITOR_MODE=once bash finalization/monitor_experiments.sh
```

### Debugging Failed Runs

```bash
# Check logs for errors
grep -i error runs/paper_revision_latest/logs/*.log

# Check phase completion
ls runs/paper_revision_latest/results/*/

# View training progress
tail -f runs/paper_revision_latest/logs/training_*.log

# Check checkpoint validity
python -c "import torch; torch.load('runs/paper_revision_latest/checkpoint/encoder.pt')"
```

### Resuming Interrupted Experiments

The script automatically resumes from the latest checkpoint:

```bash
# Resume from existing checkpoint
export SKIP_TRAINING=yes
export CHECKPOINT_PATH=runs/paper_revision_latest/checkpoint
bash finalization/run_all_experiments.sh
```

## Common Issues and Solutions

### Out of Memory
- Reduce `EVAL_BATCH_SIZE` (default: 16)
- Use gradient checkpointing
- Enable mixed precision (automatically enabled)

### Slow Evaluation
- Reduce `MAX_EVAL_SAMPLES` for testing
- Use fewer seeds for initial runs
- Skip expensive baselines with phase-specific scripts

### Preemption on HPC
- Script handles automatically via SIGTERM
- Results pushed to git before termination
- Resubmit with same command to resume

### Git Push Failures
- Script retries 3 times automatically
- Manual push: `git add runs/*/logs && git commit -m "logs" && git push`

## Results Interpretation

### Statistical Significance
- p < 0.05: Statistically significant difference
- CI overlap: Check 95% confidence intervals
- Effect size: Cohen's d > 0.5 indicates medium effect

### Performance Targets
- LatentWire: 85-90% of text baseline
- Compression: 4-8× reduction
- Latency: 20-30% faster than text
- First-token accuracy: >12% for viability

## Contact

For issues or questions about the experiments, check:
1. Logs in `runs/paper_revision_latest/logs/`
2. Error messages in SLURM output
3. Git history for recent changes