# Preemptible Training Guide

A practical guide for running experiments on Marlowe HPC's preemptible partition with automatic resumption.

## Quick Start

### 1. Submit Your Job

```bash
# On HPC:
cd /projects/m000066/sujinesh/LatentWire
git pull
sbatch telepathy/submit_preemptible_experiment.slurm
```

### 2. Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch live output
tail -f runs/preemptible_training_*.log

# Check checkpoint progress
ls -lah runs/exp*/checkpoint_epoch_*
```

### 3. Verify Completion

```bash
# Check for completion marker
cat runs/exp*/training_complete.marker

# Review final metrics
cat runs/exp*/final_metrics.json
```

## How Preemption Works

### What Happens During Preemption

1. **SIGTERM Signal** → Job receives 60-second warning
2. **Graceful Save** → Current epoch completes, checkpoint saved
3. **Metadata Update** → Progress tracked in `training_state.json`
4. **Job Requeues** → Automatically submitted again via `--requeue`
5. **Resume Training** → Picks up from last checkpoint seamlessly

### Checkpoint Strategy

```
runs/exp{timestamp}/
├── checkpoint_epoch_1/          # Full model state
├── checkpoint_epoch_2/          # Saved every epoch
├── training_state.json          # Progress tracker
├── metrics_history.json         # All metrics
├── optimizer_state.pt           # For exact resumption
└── training_complete.marker     # Success indicator
```

**Key Features:**
- **Atomic Saves**: Checkpoints written to temp, then moved
- **State Tracking**: Epoch, sample count, best metrics preserved
- **Auto-cleanup**: Only keeps last 3 checkpoints (configurable)
- **Smart Resume**: Detects incomplete epochs, continues exactly

## GPU Utilization Optimization

### Multi-GPU Strategy

```python
# Automatic distribution across 4 H100s
--num_gpus 4                    # Use all available
--batch_size 128                # Per-GPU batch size
--gradient_accumulation 2       # Effective batch = 128 * 4 * 2 = 1024
```

### Memory Optimization

```python
# For large models or OOM issues
--use_gradient_checkpointing    # Trade compute for memory
--mixed_precision fp16          # Faster training, less memory
--offload_optimizer             # Move optimizer to CPU if needed
```

### Throughput Monitoring

The system logs GPU utilization every 30 seconds:
```
[GPU Stats] Memory: 45.2/80GB | Util: 92% | Samples/sec: 1024
```

## Common Scenarios

### Scenario 1: Training Interrupted at Epoch 7/10

**What happens:**
1. Checkpoint saved at epoch 6 (last complete)
2. Job requeues automatically
3. Resumes from epoch 7
4. Continues to epoch 10

**No action needed** - fully automatic

### Scenario 2: Multiple Preemptions

**What happens:**
1. Each preemption saves state
2. Job requeues up to 5 times (configurable)
3. Progress accumulates across runs

**To check cumulative progress:**
```bash
grep "Resuming from epoch" runs/preemptible_training_*.log
```

### Scenario 3: Job Needs More Time

**Before submission:**
```bash
# Edit the SLURM script
#SBATCH --time=24:00:00  # Increase from default 12h
```

**After submission:**
```bash
# Extend running job (if allowed)
scontrol update job=$JOBID TimeLimit=24:00:00
```

## Troubleshooting

### Issue: "No checkpoint found" but Training Started

**Cause**: First preemption happened before epoch 1 completed

**Fix**: Automatic - job will restart from beginning

---

### Issue: Job Keeps Failing Immediately

**Check**:
```bash
# Look for Python errors
tail -100 runs/preemptible_training_*.err

# Common issues:
# - OOM: Reduce batch_size or enable gradient_checkpointing
# - Import error: Ensure git pull succeeded
# - Data error: Check dataset availability
```

---

### Issue: Job Not Requeuing After Preemption

**Check SLURM settings**:
```bash
scontrol show job $JOBID | grep Requeue
# Should show: Requeue=1

# If not, future jobs need:
#SBATCH --requeue
```

---

### Issue: Checkpoints Taking Too Much Space

**Configure cleanup**:
```python
# In unified_cross_model_experiments.py
MAX_CHECKPOINTS = 2  # Reduce from default 3
```

**Manual cleanup**:
```bash
# Keep only latest checkpoint
cd runs/exp*/
ls -t checkpoint_epoch_* | tail -n +2 | xargs rm -rf
```

## FAQ

### Q: How long can jobs run on preemptible?

**A**: Up to 24 hours per submission, unlimited requeues. Most experiments complete within 12 hours even with interruptions.

---

### Q: What's the preemption rate?

**A**: Typically 1-2 preemptions per 12-hour job. Varies with cluster load.

---

### Q: Can I disable auto-resumption?

**A**: Yes, remove `--requeue` from SLURM script. Job will stop after first preemption.

---

### Q: How do I know training is truly complete?

**A**: Check for `runs/exp*/training_complete.marker`. This file is only created after all epochs finish successfully.

---

### Q: Can I resume manually after job timeout?

**A**: Yes:
```bash
# Find latest checkpoint
CKPT=$(ls -t runs/exp*/checkpoint_epoch_* | head -1)

# Resume with same config
python experimental/learning/unified_cross_model_experiments.py \
    --resume_from $CKPT \
    [... original arguments ...]
```

---

### Q: What about evaluation after training?

**A**: The script automatically runs evaluation if training completes:
```python
# Happens automatically after training:
if training_complete:
    run_evaluation(best_checkpoint)
    save_results("final_results.json")
```

---

### Q: How do I run multiple experiments?

**A**: Submit multiple jobs with different configs:
```bash
# Experiment 1: Baseline
sbatch telepathy/submit_preemptible_experiment.slurm

# Experiment 2: Different latent size
sbatch --export=LATENT_LEN=64 telepathy/submit_preemptible_experiment.slurm

# Experiment 3: Different model
sbatch --export=MODEL_TYPE=qwen telepathy/submit_preemptible_experiment.slurm
```

## Best Practices

1. **Always use `--requeue`** for long experiments
2. **Save checkpoints every epoch** for minimal loss
3. **Monitor first epoch** to catch config errors early
4. **Use `git pull`** in SLURM script for latest code
5. **Check `training_complete.marker`** before analyzing results
6. **Keep logs** with `tee` for debugging
7. **Set reasonable time limits** (12-24h typically sufficient)

## Summary

The preemptible training system is designed to be **fully automatic**:
- Submit once, completes even with interruptions
- No manual intervention needed
- Progress preserved across preemptions
- Automatic evaluation on completion

Just submit and monitor - the system handles the rest.

---

**For issues not covered here**, check:
1. `runs/preemptible_training_*.err` for Python errors
2. `runs/exp*/training_state.json` for progress
3. SLURM docs: `man sbatch` for queue options