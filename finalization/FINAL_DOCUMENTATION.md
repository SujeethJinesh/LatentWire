# LatentWire - Final Documentation

## What This Is

LatentWire is a research framework for creating compressed representations that enable heterogeneous LLMs (Llama, Qwen) to communicate through a learned "latent wire". It achieves ≥4× prompt compression while maintaining task performance.

## Quick Start (3 Commands)

```bash
# 1. Submit experiment on HPC
cd /projects/m000066/sujinesh/LatentWire
sbatch finalization/slurm/submit_preemptible.slurm

# 2. Monitor progress
tail -f runs/preempt_JOBID.log

# 3. Analyze results
python finalization/analysis/aggregate_results.py --experiment_dirs runs/preemptible
```

## Core Experiments

### 1. Main Pipeline - Latent Wire Training & Evaluation
**Purpose**: Train encoder/adapter to create compressed representations that both models can decode

```bash
python latentwire/train.py \
  --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
  --samples 87599 --epochs 24 --batch_size 64 \
  --latent_len 32 --d_z 256 \
  --encoder_type byte --dataset squad \
  --sequential_models \
  --warm_anchor_text "Answer: " \
  --first_token_ce_weight 0.5
```

### 2. Statistical Testing - Bootstrap CI & Significance
**Purpose**: Rigorous statistical validation with 95% confidence intervals

```python
from scripts.statistical_testing import comprehensive_comparison_report

report = comprehensive_comparison_report(
    'Text Baseline', text_scores,
    {'Latent M=32': latent_scores, 'LLMLingua': llmlingua_scores},
    correction='fdr_bh'  # Multiple comparison correction
)
```

### 3. Linear Probe Baseline
**Purpose**: Test if sender model already contains task information

```bash
python latentwire/baselines/linear_probe_baseline.py \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset sst2 --layer 24 \
  --pooling mean --cv_folds 5
```

### 4. LLMLingua Comparison
**Purpose**: Compare against state-of-the-art prompt compression

```bash
bash scripts/run_llmlingua_baseline.sh \
  --dataset sst2 \
  --compression_ratios 32 64 128 \
  --llmlingua_version 2  # Bidirectional version
```

### 5. Telepathy Cross-Model Bridge
**Purpose**: Enable model-to-model communication without shared vocabulary

```bash
python telepathy/train_bridge.py \
  --sender llama --receiver mistral \
  --dataset agnews --samples 10000 \
  --bridge_dim 256 --epochs 10
```

## Key Requirements

### HPC Environment (Marlowe Cluster)
- **Account**: `marlowe-m000066` (NOT just `marlowe`)
- **Partition**: `preempt` (NOT `gpu`)
- **Working Dir**: `/projects/m000066/sujinesh/LatentWire`
- **Resources**: 4× H100 GPUs, 256GB memory
- **SLURM**: All experiments must use proper SLURM scripts

### Software Dependencies
```bash
# Core
python>=3.9
torch>=2.0
transformers>=4.35
datasets>=2.14

# Baselines
llmlingua>=0.2.2
scikit-learn>=1.3

# Statistics
scipy>=1.11
statsmodels>=0.14
```

### Critical Configuration
```python
# Latent dimensions
LATENT_LEN = 32       # Compression vs capacity tradeoff
D_Z = 256            # Latent dimension per token

# Training objectives
K = 4                # K-token supervision
FIRST_TOKEN_CE = 0.5 # First-token emphasis
KD_TAU = 1.0        # Knowledge distillation temperature

# Calibration
CALIBRATION = "embed_rms"      # Match embedding statistics
WARM_ANCHOR_TEXT = "Answer: "  # Between prefix and answer
APPEND_BOS_AFTER_PREFIX = "yes" # BOS handling
```

## Key Metrics Tracked

- **Task Performance**: F1, Exact Match, ROUGE-L, Accuracy
- **Statistical**: 95% Bootstrap CI, p-values, Cohen's d
- **Compression**: Wire bytes (fp16/int8/int4), compression ratio
- **Efficiency**: Latency (ms/sample), memory (GB), throughput

## File Organization

```
LatentWire/
├── latentwire/          # Core training/evaluation
│   ├── train.py         # Main training loop
│   ├── eval.py          # Evaluation pipeline
│   └── baselines/       # Linear probe, LLMLingua
├── telepathy/           # Cross-model bridge experiments
├── finalization/        # Production-ready infrastructure
│   ├── slurm/          # HPC submission scripts
│   ├── training/       # Preemption-safe trainer
│   └── analysis/       # Result aggregation
├── scripts/            # Utility scripts
│   ├── statistical_testing.py  # Bootstrap, McNemar
│   └── run_*.sh        # Experiment runners
└── runs/               # Experiment outputs (auto-created)
```

## Results Location

- **Logs**: `runs/experiment_*/pipeline_*.log`
- **Checkpoints**: `runs/experiment_*/epoch*/` (HPC only)
- **Results JSON**: `runs/experiment_*/results.json`
- **Figures**: `figures/*.pdf`
- **Statistical Reports**: `runs/experiment_*/statistical_report.json`

## Troubleshooting

### Out of Memory
```bash
--batch_size 32  # Reduce from 64
--gradient_accumulation_steps 2
--use_fp16 yes
```

### Slow Training
```bash
--use_optimized_dataloader
--num_dataloader_workers 4
--dataloader_pin_memory
```

### Poor Results
```bash
--first_token_ce_weight 0.8  # Increase emphasis
--k_token_ce_from_prefix 8   # More supervision
--epochs 48                   # Train longer
```

### Git Push Failures
```bash
git config user.email "you@example.com"
git config user.name "Your Name"
git pull --rebase
```

## Monitoring Commands

```bash
# Job status
squeue -u $USER
scontrol show job JOBID

# Live logs
tail -f runs/experiment_*.log

# GPU usage (on compute node)
nvidia-smi -l 1

# Cancel job
scancel JOBID
```

## Critical Workflow Rules

1. **Always use SLURM** for HPC experiments - never run directly
2. **Always git pull** before analyzing results (logs sync via git)
3. **Always use tee** to capture outputs: `{ cmd } 2>&1 | tee log.txt`
4. **Never skip features** - fix root causes (use IncrementalPCA for OOM, not skip)
5. **Scripts must be self-contained** - no reliance on pre-existing checkpoints

## References

- Paper: "LatentWire: Learned Compression for Cross-Model Communication" (2025)
- Baselines: LLMLingua (Microsoft), Telepathy (cross-model bridge)
- Statistical Methods: Dietterich (1998), Colas et al. (2018)