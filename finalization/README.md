# LatentWire - Learned Compression for Cross-Model Communication

## What It Does

LatentWire enables heterogeneous LLMs (Llama, Qwen) to communicate through learned compressed representations, achieving ≥4× prompt compression while maintaining task performance. The system learns a shared "latent wire" - soft tokens that both models can decode without retokenization.

**Key Features:**
- **Cross-model bridge**: Single encoder produces latents consumable by multiple LLMs
- **4× compression**: Reduces prompt size from ~128 tokens to 32 latent vectors
- **Frozen base models**: Only small adapters trained, preserving original capabilities
- **Preemption-safe**: Automatic checkpoint resumption for HPC environments
- **Statistical rigor**: Bootstrap CI, McNemar's test, multiple comparison correction

## Requirements

### Software
```bash
python>=3.9
torch>=2.0
transformers>=4.35
datasets>=2.14
llmlingua>=0.2.2  # For baseline comparison
scipy>=1.11       # For statistical testing
```

### Hardware (HPC Cluster)
- **Account**: `marlowe-m000066` (critical - NOT just `marlowe`)
- **Partition**: `preempt` (critical - NOT `gpu`)
- **Resources**: 1-4× H100 GPUs, 256GB memory
- **Working dir**: `/projects/m000066/sujinesh/LatentWire`

## How to Run

### Quick Start (3 Commands)
```bash
# 1. Navigate to HPC project directory
cd /projects/m000066/sujinesh/LatentWire

# 2. Pull latest code
git pull

# 3. Submit main experiment
sbatch finalization/slurm/submit_main.slurm
```

### Core Training Command
```bash
python latentwire/train.py \
  --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
  --samples 87599 --epochs 24 --batch_size 64 \
  --latent_len 32 --d_z 256 \
  --dataset squad \
  --K 4 --first_token_ce_weight 0.5 \
  --warm_anchor_text "Answer: "
```

### Evaluation
```bash
python latentwire/eval.py \
  --ckpt runs/experiment_*/epoch24 \
  --samples 200 --dataset squad \
  --calibration embed_rms \
  --latent_anchor_text "Answer: "
```

### Statistical Testing
```python
from scripts.statistical_testing import comprehensive_comparison_report
report = comprehensive_comparison_report(
    'Text Baseline', text_scores,
    {'Latent M=32': latent_scores},
    correction='fdr_bh'
)
```

### Monitor Progress
```bash
# Job status
squeue -u $USER

# Live logs
tail -f runs/preempt_*.log

# GPU usage
nvidia-smi -l 1

# Cancel if needed
scancel <job_id>
```

## Key Configuration

### Critical Parameters
```python
LATENT_LEN = 32              # Number of soft tokens (compression ratio)
D_Z = 256                    # Latent dimension per token
K = 4                        # K-token supervision depth
FIRST_TOKEN_CE_WEIGHT = 0.5  # First-token loss emphasis
CALIBRATION = "embed_rms"    # Match embedding statistics
WARM_ANCHOR_TEXT = "Answer: " # Anchor between prefix and answer
```

### SLURM Script Template
```bash
#!/bin/bash
#SBATCH --job-name=latentwire
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066    # CRITICAL: exact string
#SBATCH --partition=preempt          # CRITICAL: exact string
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --signal=TERM@120            # Preemption warning
#SBATCH --requeue                    # Auto-restart on preemption
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/%j.log

cd /projects/m000066/sujinesh/LatentWire
export PYTHONPATH=.
git pull

# Run with automatic resume
python finalization/training/preemptible_trainer.py \
    --auto_resume \
    --checkpoint_interval 300 \
    --save_dir runs/experiment_$SLURM_JOB_ID \
    [training args...]

# Push results
git add -A && git commit -m "results: job $SLURM_JOB_ID" && git push
```

## Directory Structure

```
LatentWire/
├── latentwire/           # Core training/evaluation
│   ├── train.py         # Main training loop
│   ├── eval.py          # Evaluation pipeline
│   └── baselines/       # Linear probe, LLMLingua
├── telepathy/           # Cross-model bridge experiments
├── finalization/        # Production infrastructure
│   ├── slurm/          # HPC submission scripts
│   ├── training/       # Preemption-safe trainer
│   └── analysis/       # Result aggregation
├── scripts/            # Utilities and runners
│   └── statistical_testing.py  # Bootstrap CI, significance tests
└── runs/              # Experiment outputs (auto-created)
```

## Troubleshooting

### Out of Memory
```bash
--batch_size 32                    # Reduce from 64
--gradient_accumulation_steps 2    # Split batches
--use_fp16 yes                     # Mixed precision
```

### Poor Results
```bash
--first_token_ce_weight 0.8   # Increase first-token emphasis
--k_token_ce_from_prefix 8    # More supervision tokens
--epochs 48                    # Train longer
```

### Preemption Issues
```bash
--checkpoint_interval 180      # Save more frequently (3 min)
--auto_resume                  # Enable automatic resumption
ls runs/*/preempt_checkpoint/  # Check for saved checkpoints
```

### Git Push Failures
```bash
git config user.email "you@example.com"
git config user.name "Your Name"
git pull --rebase
```

### Slow Training
```bash
--num_dataloader_workers 4    # Parallel data loading
--dataloader_pin_memory       # Pin memory for GPU transfer
--use_optimized_dataloader    # Enable optimizations
```

## Expected Results

- **Text baseline**: F1 ~0.80-0.85 (upper bound)
- **Latent M=32**: Target F1 ~0.10-0.20 with 4× compression
- **Linear probe**: ~60-70% accuracy (validates sender representations)
- **LLMLingua-2**: F1 ~0.40-0.50 at similar compression ratios

## Critical Workflow Rules

1. **ALWAYS use SLURM** for HPC - never run directly on login nodes
2. **ALWAYS git pull** before analyzing results (logs sync via git)
3. **ALWAYS use tee** for logging: `{ cmd } 2>&1 | tee log.txt`
4. **Scripts must be self-contained** - no reliance on pre-existing checkpoints
5. **Fix root causes** - use IncrementalPCA for OOM, not skip features

## Support

- **Logs**: Check `runs/experiment_*/pipeline_*.log`
- **Results**: JSON summaries in `runs/experiment_*/results.json`
- **Statistics**: Reports in `runs/experiment_*/statistical_report.json`
- **Monitoring**: Use `squeue`, `nvidia-smi`, `tail -f` commands above

For issues, first check logs, then verify SLURM settings match exactly as shown.