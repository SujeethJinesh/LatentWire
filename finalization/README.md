# LatentWire Finalization Pipeline

Complete guide for running experiments and generating paper results.

## Quick Start (3 Commands)

```bash
# 1. Submit the main experiment
cd /projects/m000066/sujinesh/LatentWire
sbatch finalization/run_all_experiments.slurm

# 2. Monitor progress (job ID from step 1)
tail -f runs/finalization_JOBID.log

# 3. Generate paper after completion
sbatch finalization/generate_paper_results.slurm
```

## Prerequisites

### System Requirements
- Access to Marlowe HPC cluster
- 4× H100 GPUs available
- ~256GB memory for full experiments
- Git configured for push access

### Environment Setup
```bash
# One-time setup on HPC
cd /projects/m000066/sujinesh/LatentWire
git pull
mkdir -p runs figures
```

## Configuration Options

### Main Experiment Parameters
Edit `finalization/run_all_experiments.slurm` to adjust:

```bash
# Latent dimensions (compression vs capacity)
--latent_len 32      # Number of soft tokens (16/32/48/64)
--d_z 256           # Latent dimension per token

# Training scale
--samples 10000     # Dataset size (10K/50K/87599)
--epochs 12         # Training epochs

# Objectives (critical for performance)
--k_token_ce_from_prefix 4        # Supervise K tokens
--first_token_ce_weight 0.5       # First-token emphasis
--kd_first_k_prefix_vs_text yes   # Knowledge distillation

# Models to evaluate
--sequential_models    # Train Llama then Qwen
# OR
--skip_qwen           # Llama only (faster)
```

### Resource Allocation
```bash
#SBATCH --gpus=4         # Use 1 for debugging, 4 for full runs
#SBATCH --time=12:00:00  # Adjust based on experiment scale
#SBATCH --mem=256GB      # 64GB for small, 256GB for large
```

## Job Submission

### Submit Main Experiments
```bash
# Full pipeline (12-24 hours)
sbatch finalization/run_all_experiments.slurm

# Quick test (2-4 hours)
sbatch finalization/run_quick_test.slurm

# Paper generation only (30 min)
sbatch finalization/generate_paper_results.slurm
```

### Submit Specific Components
```bash
# Just baselines
sbatch finalization/run_baselines.slurm

# Just attention analysis
sbatch finalization/analyze_attention.slurm

# Just statistical tests
sbatch finalization/run_statistical_tests.slurm
```

## Monitor Progress

### Check Job Status
```bash
# Your jobs
squeue -u $USER

# Detailed job info
scontrol show job JOBID

# Estimated completion
squeue -u $USER --start
```

### Watch Logs in Real-Time
```bash
# Main experiment log
tail -f runs/finalization_JOBID.log

# Watch for errors
tail -f runs/finalization_JOBID.err

# Monitor GPU usage (on compute node)
nvidia-smi -l 1
```

### Progress Indicators
Look for these milestones in logs:
```
"Starting epoch 1/12"          # Training begun
"Epoch 6/12 complete"          # Halfway point
"Starting evaluation"          # Training done
"F1 Score: 0.XX"              # Quality metrics
"Compression ratio: X.XX"     # Efficiency metrics
"Pushing results to git"      # Near completion
```

## Handle Preemptions

### Automatic Recovery
The pipeline automatically:
- Saves checkpoints every epoch
- Pushes intermediate results to git
- Can resume from last checkpoint

### Manual Recovery
If job gets preempted:
```bash
# 1. Check what completed
ls -la runs/finalization_*/
git pull  # Get pushed results

# 2. Resume from checkpoint (if implemented)
sbatch finalization/resume_experiments.slurm

# 3. Or restart specific phase
sbatch finalization/run_evaluation_only.slurm
```

### Prevent Preemptions
```bash
# Use higher priority (if available)
#SBATCH --qos=high

# Request specific nodes
#SBATCH --nodelist=node001

# Shorter jobs less likely preempted
#SBATCH --time=04:00:00
```

## Analyze Results

### Pull Results Locally
```bash
# On your local machine
cd ~/Desktop/LatentWire
git pull

# View results
cat runs/finalization_*/results.json | jq '.'
```

### Generate Visualizations
```bash
# Automated figures
python finalization/generate_figures.py \
    --results_dir runs/finalization_latest/

# Custom analysis
python finalization/analyze_results.py \
    --metric f1 \
    --breakdown_by model
```

### Key Metrics to Check
```python
# In results.json
{
  "latent_f1": 0.XX,        # Main quality metric
  "text_baseline_f1": 0.XX,  # Upper bound
  "compression_ratio": X.XX, # Efficiency
  "first_token_acc": 0.XX,   # Generation quality
  "wire_bytes": XXX,         # Communication cost
}
```

## Generate Paper

### After Experiments Complete
```bash
# Generate all paper materials
sbatch finalization/generate_paper_results.slurm

# This creates:
# - figures/*.pdf (all plots)
# - tables/*.tex (LaTeX tables)
# - paper_results.json (all metrics)
```

### Update Paper Text
```latex
% In paper.tex, update with:
\input{tables/main_results.tex}
\includegraphics{figures/compression_quality.pdf}

% Key numbers from paper_results.json:
Our method achieves \resultsLatentF{} F1 score...
With \resultsCompressionRatio{}× compression...
```

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)
```bash
# Reduce batch size
--batch_size 32  # Instead of 64

# Use gradient accumulation
--gradient_accumulation_steps 2

# Reduce model precision
--use_fp16 yes
```

#### Slow Training
```bash
# Check GPU utilization
nvidia-smi

# Use all GPUs
--world_size 4

# Increase batch size if memory allows
--batch_size 128
```

#### Poor Results
```bash
# Verify data quality
python finalization/validate_data.py

# Check loss curves
python finalization/plot_training.py \
    --log runs/finalization_*/diagnostics.jsonl

# Try different hyperparameters
--first_token_ce_weight 0.8  # More emphasis
--k_token_ce_from_prefix 8   # More supervision
```

#### Git Issues
```bash
# Can't push results
git config user.email "you@example.com"
git config user.name "Your Name"

# Conflicts
git stash
git pull --rebase
git stash pop
```

## FAQ

### Q: How long do experiments take?
**A:**
- Quick test (1K samples): 2-4 hours
- Medium (10K samples): 6-12 hours
- Full (87K samples): 12-24 hours
- Paper generation: 30 minutes

### Q: Which experiments are most important?
**A:** Priority order:
1. Main latent wire results (compression + quality)
2. Baseline comparisons (text, token-budget)
3. Ablations (K-token, calibration)
4. Attention visualizations

### Q: Can I run multiple jobs?
**A:** Yes, but coordinate GPU usage:
```bash
# Check available resources
sinfo -p preempt

# Run complementary jobs
sbatch --gpus=2 job1.slurm  # Uses 2 GPUs
sbatch --gpus=2 job2.slurm  # Uses other 2
```

### Q: How to ensure reproducibility?
**A:** The pipeline:
- Sets random seeds
- Logs all configurations
- Saves git commit hash
- Archives exact commands

### Q: What if results look wrong?
**A:** Check:
1. Training completed without errors
2. Evaluation used correct checkpoints
3. Baselines ran successfully
4. No data corruption (validate JSONs)

### Q: How to speed up iteration?
**A:**
- Use `--samples 1000` for development
- Skip Qwen with `--skip_qwen`
- Run single latent size first
- Use `--debug` flag for verbose output

## Support

### Getting Help
1. Check logs for error messages
2. Review this README
3. Check CLAUDE.md for design principles
4. Review recent git commits for examples

### File Locations
- Logs: `runs/finalization_*/`
- Checkpoints: `runs/finalization_*/epoch*/`
- Figures: `figures/`
- Results: `runs/finalization_*/results.json`

### Key Scripts
- `finalization/run_all_experiments.slurm` - Main pipeline
- `finalization/analyze_results.py` - Result analysis
- `finalization/generate_figures.py` - Plot generation
- `finalization/generate_paper_results.slurm` - Paper materials

## Next Steps

After successful run:
1. Review results in `paper_results.json`
2. Check generated figures in `figures/`
3. Update paper with new numbers
4. Run additional ablations if needed
5. Generate final camera-ready version

Remember: Always `git pull` before starting and `git push` after completion!