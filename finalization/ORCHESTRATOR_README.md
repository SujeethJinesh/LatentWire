# LatentWire Experiment Orchestrator

This directory contains the main orchestration scripts for running LatentWire experiments on HPC clusters with full resilience and optimization.

## 🚀 Quick Start

### Running on HPC Cluster

```bash
# 1. Push code to git
git add -A && git commit -m "Ready to run experiment" && git push

# 2. On HPC, pull latest code
cd /projects/m000066/sujinesh/LatentWire
git pull

# 3. Submit experiment (will use 4 GPUs by default)
sbatch finalization/submit_experiment.slurm

# 4. Monitor job
squeue -u $USER
bash finalization/monitor_experiment.sh
```

### Running Interactively

```bash
# Request interactive session with GPUs
srun --gpus=4 --account=marlowe-m000066 --partition=preempt --time=02:00:00 --pty bash

# Run experiment
bash finalization/run_experiment.sh
```

## 📁 Scripts Overview

### `run_experiment.sh` - Main Orchestrator
The core script that handles everything:
- **Elastic GPU allocation**: Automatically adapts to 1-4 GPUs
- **Preemption handling**: Saves checkpoints and resumes automatically
- **Optimal batching**: Calculates best batch size for available memory
- **Comprehensive logging**: Captures all output with timestamps
- **State tracking**: Maintains experiment state across resumptions

### `submit_experiment.slurm` - SLURM Wrapper
SLURM submission script that:
- Configures proper HPC settings
- Handles preemption signals gracefully
- Automatically pushes logs to git
- Supports job requeuing

### `monitor_experiment.sh` - Real-time Monitor
Interactive monitoring tool that displays:
- Current training status
- Loss trends and metrics
- GPU utilization
- Checkpoint progress
- Recent log output

### `test_orchestrator.sh` - Test Suite
Verification script to test:
- Script syntax and setup
- GPU detection
- State management
- Signal handling
- Quick training test (optional)

## ⚙️ Configuration

### Environment Variables

Configure experiments via environment variables:

```bash
# Experiment settings
export EXP_NAME="my_experiment"         # Experiment name
export DATASET="squad"                  # Dataset: squad, hotpotqa
export SAMPLES="10000"                  # Number of training samples
export EPOCHS="3"                       # Number of epochs
export LATENT_LEN="32"                 # Latent sequence length
export D_Z="256"                       # Latent dimension

# Hardware optimization
export TARGET_GPU_UTIL="0.75"          # Target GPU memory utilization
export BASE_BATCH_SIZE="64"            # Target effective batch size
export NUM_WORKERS="4"                 # DataLoader workers
export PREFETCH_FACTOR="2"             # DataLoader prefetch factor

# Resumption settings
export RESUME="auto"                   # auto, yes, or no
export MAX_RETRIES="10"               # Maximum preemption retries
```

### Custom SLURM Configuration

Modify SLURM parameters:

```bash
# Request different GPU count
sbatch --gpus=2 finalization/submit_experiment.slurm

# Shorter time limit
sbatch --time=06:00:00 finalization/submit_experiment.slurm

# Different partition
sbatch --partition=gpu finalization/submit_experiment.slurm
```

## 🔄 Preemption Handling

The orchestrator automatically handles preemption:

1. **Signal detection**: Catches SLURM preemption signals (SIGUSR1)
2. **Checkpoint saving**: Immediately saves current model state
3. **State persistence**: Saves training state to `.orchestrator_state`
4. **Automatic requeue**: Job requeues itself with `--requeue` flag
5. **Smart resumption**: Finds latest checkpoint and continues training

### Manual Recovery

If automatic recovery fails:

```bash
# Check experiment state
cat runs/YOUR_EXP/.orchestrator_state

# Manually resume
export RESUME=yes
export EXP_NAME="YOUR_EXP"
sbatch finalization/submit_experiment.slurm
```

## 📊 Monitoring

### During Training

```bash
# Real-time monitoring with UI
bash finalization/monitor_experiment.sh [EXP_NAME]

# Watch SLURM output
tail -f runs/slurm_*.log

# Check latest training log
tail -f runs/YOUR_EXP/logs/latest.log

# View metrics
python scripts/analyze_diagnostics.py runs/YOUR_EXP/diagnostics.jsonl
```

### After Training

```bash
# Evaluate checkpoint
python latentwire/eval.py \
    --ckpt runs/YOUR_EXP/epoch3 \
    --dataset squad \
    --samples 200

# Analyze results
python scripts/statistical_testing.py runs/YOUR_EXP/
```

## 🐛 Troubleshooting

### Common Issues

**GPU not detected:**
```bash
# Check GPU availability
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Force specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

**Out of memory:**
```bash
# Reduce batch size
export BASE_BATCH_SIZE=32

# Increase gradient accumulation
# (handled automatically by orchestrator)
```

**Git push fails:**
```bash
# Configure git credentials
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Or use token authentication
git config credential.helper store
```

**Checkpoint not found:**
```bash
# List available checkpoints
ls -la runs/YOUR_EXP/epoch*

# Check orchestrator state
cat runs/YOUR_EXP/.orchestrator_state
```

## 📈 Performance Optimization

The orchestrator automatically optimizes for:

### GPU Utilization
- Dynamic batch sizing based on available memory
- Gradient accumulation for effective batch sizes
- Mixed precision training (bf16) when available
- Model compilation with torch.compile

### Data Loading
- Optimized DataLoader with caching
- Multiple workers for preprocessing
- Prefetching for GPU feeding
- Memory pinning for faster transfers

### Checkpoint Management
- Incremental checkpointing (only saves changed parts)
- Async checkpoint saving (doesn't block training)
- Automatic old checkpoint cleanup

## 🔍 Advanced Usage

### Multi-Node Training

For multi-node training (future support):

```bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1

# Orchestrator will detect and configure distributed training
```

### Custom Training Scripts

To use orchestrator with custom training:

1. Ensure your script accepts standard arguments
2. Supports `--resume_from` flag
3. Saves checkpoints to `--output_dir`
4. Outputs metrics to `diagnostics.jsonl`

## 📝 Example Workflows

### Standard Experiment

```bash
# Configure experiment
export EXP_NAME="llama_squad_baseline"
export DATASET="squad"
export SAMPLES="87599"
export EPOCHS="24"

# Submit job
sbatch finalization/submit_experiment.slurm

# Monitor progress
bash finalization/monitor_experiment.sh llama_squad_baseline
```

### Quick Test

```bash
# Test with minimal settings
export SAMPLES="1000"
export EPOCHS="1"
srun --gpus=1 --time=00:30:00 bash finalization/run_experiment.sh
```

### Production Run

```bash
# Full dataset, long training
export SAMPLES="87599"
export EPOCHS="24"
export MAX_RETRIES="20"  # More retries for long runs
sbatch --time=24:00:00 finalization/submit_experiment.slurm
```

## 📚 Files Created

The orchestrator creates the following structure:

```
runs/
└── YOUR_EXP_NAME/
    ├── .orchestrator_state     # Current training state
    ├── .retry_count            # Preemption retry counter
    ├── diagnostics.jsonl       # Training metrics
    ├── logs/
    │   ├── train_*.log        # Training output logs
    │   └── latest.log         # Symlink to current log
    ├── epoch0/                # Checkpoint directories
    ├── epoch1/
    └── epoch2/
```

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in `runs/YOUR_EXP/logs/`
3. Check SLURM output: `runs/slurm_*.log`
4. Verify GPU status: `nvidia-smi`

## 📄 License

Part of the LatentWire project - see main repository for license details.