# Preemptible Training for LatentWire

This directory contains preemption-safe training infrastructure for LatentWire that handles SLURM preemption signals gracefully.

## Overview

When running on SLURM clusters with preemptible partitions, jobs can be terminated with short notice (typically 120 seconds). This infrastructure ensures:

1. **Immediate checkpoint saving** when SIGTERM is received
2. **Periodic checkpoints** at configurable intervals
3. **Exact resumption** from the interrupted point
4. **Full state preservation** including RNG states

## Components

### Core Files

- `preemptible_trainer.py` - Main preemptible training wrapper
- `checkpoint_manager.py` - Robust checkpoint management with atomic writes
- `logging_utils.py` - Thread-safe logging for preemptible jobs
- `gpu_monitor.py` - GPU monitoring and profiling
- `test_preemption.py` - Test script to verify preemption handling

### Support Files

- `preemptible_train.py` - Direct integration approach (template)
- `README_PREEMPTION.md` - This documentation

## Usage

### Basic Training with Preemption Support

```bash
python preemptible_trainer.py \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --samples 10000 \
    --epochs 10 \
    --batch_size 32 \
    --save_dir runs/preemptible \
    --checkpoint_interval 300  # Save every 5 minutes
```

### Auto-Resume from Preemption

```bash
python preemptible_trainer.py \
    --auto_resume \
    --save_dir runs/preemptible \
    [same training args as before]
```

### SLURM Integration

Create a SLURM script with preemption support:

```bash
#!/bin/bash
#SBATCH --job-name=latentwire_preemptible
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --signal=TERM@120    # Send SIGTERM 120 seconds before termination
#SBATCH --requeue            # Allow job to be requeued after preemption
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/preempt_%j.log

# Set working directory
cd /projects/m000066/sujinesh/LatentWire

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Pull latest code
git pull

# Run preemptible training
python finalization/training/preemptible_trainer.py \
    --auto_resume \
    --checkpoint_interval 300 \
    --save_dir runs/preemptible_exp \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
    --samples 87599 \
    --epochs 24 \
    --batch_size 64 \
    --latent_len 32 \
    --d_z 256 \
    --dataset squad \
    --K 4 \
    --first_token_ce_weight 0.5 \
    --warm_anchor_text "Answer: "

# Push results if completed normally
if [ $? -eq 0 ]; then
    git add -A
    git commit -m "results: preemptible training completed (job $SLURM_JOB_ID)" || true
    git push || true
fi
```

## Testing Preemption Handling

### Local Test

Test the preemption handler locally:

```bash
# Run the test script
python test_preemption.py &

# Note the PID printed, then send SIGTERM
kill -TERM <PID>

# Or use timeout for automatic termination
timeout --signal=TERM 10 python test_preemption.py
```

### Verify Checkpoint Saving

After running the test, check the saved checkpoints:

```bash
ls -la test_checkpoints/
cat test_checkpoints/checkpoint_*.json
```

## How It Works

### Signal Handling

1. **SIGTERM Registration**: On startup, registers a signal handler for SIGTERM
2. **Grace Period**: When SIGTERM is received, immediately saves checkpoint
3. **Clean Exit**: Exits with status 0 to allow SLURM requeuing

### Checkpoint Management

1. **Atomic Writes**: Saves to temporary file then renames (prevents corruption)
2. **State Preservation**: Saves model weights, optimizer state, RNG states
3. **Metadata Tracking**: Records epoch, batch index, timestamp
4. **Auto-Pruning**: Keeps only the latest checkpoint to save space

### Resumption Logic

1. **Auto-Discovery**: Finds latest valid checkpoint on startup
2. **State Restoration**: Loads model weights, optimizer, RNG states
3. **Exact Continuation**: Resumes from exact batch within epoch
4. **Validation**: Checksums ensure checkpoint integrity

## Configuration Options

### Preemption Settings

- `--checkpoint_interval`: Seconds between periodic checkpoints (default: 300)
- `--auto_resume`: Automatically resume from latest checkpoint
- `--save_dir`: Directory for checkpoints and logs
- `--monitor_gpu`: Enable GPU monitoring for profiling

### Training Settings

All standard LatentWire training arguments are supported:

- `--llama_id`, `--qwen_id`: Model identifiers
- `--samples`, `--epochs`, `--batch_size`: Training parameters
- `--latent_len`, `--d_z`: Latent space configuration
- `--dataset`, `--K`, `--first_token_ce_weight`: Loss configuration

## Monitoring

### Check Job Status

```bash
# View running jobs
squeue -u $USER

# Check job details
scontrol show job <JOB_ID>

# Monitor training progress
tail -f runs/preempt_*.log
```

### GPU Monitoring

Enable GPU monitoring for resource tracking:

```bash
python preemptible_trainer.py --monitor_gpu ...
```

This creates `gpu_stats.json` with utilization metrics.

## Troubleshooting

### Common Issues

1. **Checkpoint not found on resume**
   - Check `save_dir` path is correct
   - Verify checkpoint files exist: `ls -la <save_dir>/preempt_checkpoint/`

2. **SIGTERM not handled**
   - Ensure `--signal=TERM@120` is in SLURM script
   - Check process has permission to handle signals

3. **Out of memory during checkpoint**
   - Reduce `--checkpoint_interval` to save more frequently
   - Consider using gradient checkpointing

### Debug Mode

Run with debug output:

```bash
python preemptible_trainer.py --debug ...
```

## Best Practices

1. **Set appropriate grace period**: Use `--signal=TERM@120` for 2-minute warning
2. **Regular checkpoints**: Save every 5-10 minutes for long jobs
3. **Monitor disk usage**: Checkpoints can be large (10-20GB)
4. **Test locally first**: Use `test_preemption.py` before cluster deployment
5. **Keep logs**: Output files help debug preemption issues

## Implementation Notes

### Current Limitations

The current `preemptible_trainer.py` uses a wrapper approach that:
- Delegates to the original `latentwire/train.py`
- Intercepts epoch loops for checkpoint opportunities
- Has limited visibility into batch-level progress

### Future Improvements

For production use, consider:
1. Direct integration into `latentwire/train.py`
2. Batch-level checkpoint granularity
3. Distributed training support (DDP)
4. Checkpoint compression
5. Cloud storage backup

## Support

For issues or questions:
1. Check this README first
2. Review test output from `test_preemption.py`
3. Examine logs in `runs/preempt_*.log`
4. Check SLURM documentation for cluster-specific settings