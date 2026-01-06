# Scripts Inventory for Finalization Directory

This document lists all scripts copied or created for the LatentWire finalization pipeline.

## Core Scripts

### RUN_ALL.sh
- **Purpose**: Master orchestration script for all experiments
- **Location**: `/finalization/RUN_ALL.sh`
- **Features**:
  - Runs all 4 phases of experiments
  - Handles both local and HPC execution
  - SLURM job submission support
  - Complete pipeline from training to paper compilation

### MAIN_EXPERIMENT.py
- **Purpose**: Python-based experiment runner
- **Location**: `/finalization/MAIN_EXPERIMENT.py`
- **Features**:
  - Programmatic experiment execution
  - Checkpoint management
  - Progress tracking

## Evaluation Scripts (Copied from `/latentwire/`)

### eval.py
- **Purpose**: Main evaluation script for SQuAD dataset
- **Source**: `/latentwire/eval.py`
- **Used in**: Phase 1 (Statistical Rigor), general evaluation
- **Features**:
  - Handles checkpoint loading
  - Computes F1/EM scores for SQuAD
  - Supports multiple evaluation modes (latent, text baseline, token-budget)

### eval_sst2.py
- **Purpose**: SST-2 sentiment classification evaluation
- **Source**: `/latentwire/eval_sst2.py`
- **Used in**: Phase 1 experiments for SST-2 dataset
- **Features**:
  - Binary sentiment classification
  - Accuracy computation
  - Multiple seed support

### eval_agnews.py
- **Purpose**: AG News classification evaluation
- **Source**: `/latentwire/eval_agnews.py`
- **Used in**: Phase 1 experiments for AG News dataset
- **Features**:
  - 4-class news categorization
  - Accuracy and confusion matrix

### eval_telepathy_trec.py
- **Purpose**: TREC question classification evaluation
- **Source**: `/telepathy/eval_telepathy_trec.py`
- **Used in**: Phase 1 experiments for TREC dataset
- **Features**:
  - 6-class question type classification
  - Fine-grained evaluation metrics

## Analysis Scripts

### statistical_testing.py
- **Purpose**: Statistical significance testing and bootstrap confidence intervals
- **Source**: `/scripts/statistical_testing.py`
- **Used in**: Phase 1 (Statistical Rigor)
- **Features**:
  - Bootstrap confidence intervals
  - Multiple hypothesis correction
  - Paired statistical tests
  - Result aggregation across seeds

### linear_probe_baseline.py
- **Purpose**: Linear probe baseline experiments
- **Source**: `/latentwire/linear_probe_baseline.py`
- **Used in**: Phase 2 (Linear Probe Baselines)
- **Features**:
  - Extracts hidden states from frozen LLMs
  - Trains linear classifiers on representations
  - Cross-validation support
  - Layer-wise analysis

## Efficiency and Aggregation Scripts (Created)

### benchmark_efficiency.py
- **Purpose**: Measure efficiency metrics (latency, throughput, memory)
- **Created**: New file for finalization
- **Used in**: Phase 4 (Efficiency Measurements)
- **Features**:
  - Latency benchmarking (ms per sample)
  - Throughput measurement (samples/second)
  - Memory footprint analysis
  - Compression ratio calculations

### aggregate_results.py
- **Purpose**: Aggregate all experimental results for paper
- **Created**: New file for finalization
- **Used in**: Final results compilation
- **Features**:
  - JSON result aggregation
  - LaTeX table generation
  - Plot generation for paper figures
  - Summary statistics computation

## Test Scripts

### test_checkpoint_integration.py
- **Purpose**: Test checkpoint loading and model integration
- **Location**: `/finalization/test_checkpoint_integration.py`

### test_consolidated_system.py
- **Purpose**: Test the consolidated experiment system
- **Location**: `/finalization/test_consolidated_system.py`

### test_phase_invocation.py
- **Purpose**: Test individual phase execution
- **Location**: `/finalization/test_phase_invocation.py`

### test_phases_direct.py
- **Purpose**: Direct phase testing without full pipeline
- **Location**: `/finalization/test_phases_direct.py`

### test_preemption_resume.py
- **Purpose**: Test experiment resumption after preemption
- **Location**: `/finalization/test_preemption_resume.py`

## Usage

To run the complete finalization pipeline:

```bash
# Local execution
bash RUN_ALL.sh experiment

# HPC execution with SLURM
bash RUN_ALL.sh slurm experiment

# Run specific phase
bash RUN_ALL.sh experiment --phase 1 --dataset sst2

# Quick test
bash RUN_ALL.sh quick
```

## Dependencies

The scripts expect the following directory structure:
```
finalization/
├── RUN_ALL.sh                  # Master script
├── MAIN_EXPERIMENT.py           # Python runner
├── eval*.py                     # Evaluation scripts
├── statistical_testing.py       # Statistical analysis
├── linear_probe_baseline.py    # Linear probe baseline
├── benchmark_efficiency.py      # Efficiency metrics
├── aggregate_results.py         # Result aggregation
└── test_*.py                   # Test scripts
```

## Notes

- All scripts are designed to work with the LatentWire checkpoint format
- Results are saved in JSON format for easy aggregation
- Scripts support both CPU and GPU execution
- Proper PYTHONPATH setup is required: `export PYTHONPATH=.`