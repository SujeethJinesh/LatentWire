# COCONUT Training on HPC - Complete Setup Guide

## Prerequisites

- HPC cluster with CUDA GPUs
- Access to conda module
- Git repository cloned

## One-Time Setup

### 1. Setup Conda Environment

```bash
# Load conda module
module load conda/24.3.0-0

# Initialize conda (one-time only)
conda init bash

# Reload bashrc
source ~/.bashrc

# Create Python 3.11 environment
conda create -n 3_11 python=3.11 -y

# Activate environment
conda activate 3_11

# Verify Python version
python --version  # Should show Python 3.11.x
```

### 2. Configure Auto-Activation (Optional)

Edit `~/.bashrc` and add these lines **after** the conda initialize block:

```bash
# Auto-load conda and activate 3_11 environment
module load conda/24.3.0-0
conda activate 3_11
```

### 3. Pull Latest Code

```bash
cd /path/to/LatentWire
git pull
cd experimental/learning/reproduce_coconut/coconut
```

### 4. Download Data (One-Time)

```bash
# Download GSM8K Internalize CoT dataset
bash preprocessing/gsm_icot.bash

# Verify data
ls -lh data/
# Should see: gsm_train.json (99M), gsm_valid.json (162K), gsm_test.json
```

### 5. Install Dependencies

```bash
# Make sure you're in the coconut directory
cd /path/to/LatentWire/experimental/learning/reproduce_coconut/coconut

# Activate environment if not already active
conda activate 3_11

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Running Training

### Quick Test (3 epochs, ~1-2 hours)

```bash
# Navigate to coconut directory
cd /path/to/LatentWire/experimental/learning/reproduce_coconut/coconut

# Activate conda environment
conda activate 3_11

# Single GPU
torchrun --nproc_per_node=1 run.py args/gsm_cot_test.yaml

# Or 4 GPUs (faster)
torchrun --nproc_per_node=4 run.py args/gsm_cot_test.yaml
```

### Full Training (25 epochs, ~10-12 hours)

```bash
# Single GPU
torchrun --nproc_per_node=1 run.py args/gsm_cot.yaml

# Or 4 GPUs (recommended)
torchrun --nproc_per_node=4 run.py args/gsm_cot.yaml
```

## Configuration Files

### Test Config: `args/gsm_cot_test.yaml`
- **Purpose**: Quick 3-epoch test run
- **Epochs**: 3 (reduced from 25)
- **Debug mode**: True (skips wandb)
- **Batch size**: 8
- **Gradient accumulation**: 4 (effective batch = 32)

### Full Config: `args/gsm_cot.yaml`
- **Purpose**: Full 25-epoch training
- **Epochs**: 25
- **Debug mode**: False (uses wandb)
- **Batch size**: 32
- **Gradient accumulation**: 4 (effective batch = 128 with 4 GPUs)

## Expected Output

### During Training
- Model loading messages
- Dataset mapping progress bars
- Training progress: `Training Epoch: 1: X%|█████| N/M [time, it/s]`
- Loss values printed per batch
- Validation metrics after each epoch

### Checkpoints
Saved to: `../runs/stage0/gsm-cot-test/checkpoint_N/`

Contains:
- `pytorch_model.bin` - Model weights
- `config.json` - Model configuration
- `optimizer.pt` - Optimizer state

## Monitoring Training

### Check GPU Usage
```bash
# While training is running
nvidia-smi
watch -n 1 nvidia-smi  # Update every second
```

### Check Training Logs
If using `tee` (recommended):
```bash
# Training command with logging
torchrun --nproc_per_node=1 run.py args/gsm_cot_test.yaml 2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log

# View log in real-time
tail -f training_*.log
```

## Troubleshooting

### Python Version Issues
```bash
# Check Python version
python --version

# Should be 3.11.x
# If not, ensure conda environment is activated:
conda activate 3_11
```

### CUDA Not Available
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, check if CUDA module needs loading
module avail cuda
module load cuda/XX.X  # Load appropriate version
```

### Out of Memory (OOM)
Edit config file and reduce batch size:
```yaml
batch_size_training: 4  # Reduce from 8
gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

### Data Not Found
```bash
# Re-download data
cd /path/to/LatentWire/experimental/learning/reproduce_coconut/coconut
bash preprocessing/gsm_icot.bash
```

### Module Not Found Errors
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

## Quick Reference Commands

```bash
# Setup (once)
module load conda/24.3.0-0
conda activate 3_11
cd /path/to/LatentWire/experimental/learning/reproduce_coconut/coconut
pip install -r requirements.txt

# Run training (every time)
conda activate 3_11
cd /path/to/LatentWire/experimental/learning/reproduce_coconut/coconut
torchrun --nproc_per_node=1 run.py args/gsm_cot_test.yaml

# Monitor
nvidia-smi
tail -f training_*.log
```

## Expected Results

### Stage 0 (CoT Baseline)
- **Validation accuracy**: ~40% (at 25 epochs)
- **Test accuracy**: Target ~40%
- **Purpose**: Establishes baseline with full chain-of-thought reasoning

### Stage 1-3 (COCONUT - if running full pipeline)
After Stage 0, train with continuous thoughts:
```bash
# Update load_model_path in args/gsm_coconut.yaml first
torchrun --nproc_per_node=4 run.py args/gsm_coconut.yaml
```

## File Structure

```
experimental/learning/reproduce_coconut/coconut/
├── args/
│   ├── gsm_cot.yaml              # Full 25-epoch config
│   ├── gsm_cot_test.yaml         # Quick 3-epoch test
│   └── gsm_coconut.yaml          # Stage 1-3 config (after Stage 0)
├── data/
│   ├── gsm_train.json            # 385K training examples
│   ├── gsm_valid.json            # 500 validation examples
│   └── gsm_test.json             # 1319 test examples
├── preprocessing/
│   └── gsm_icot.bash             # Data download script
├── run.py                        # Main training script
├── coconut.py                    # COCONUT model
├── dataset.py                    # Data loading
└── requirements.txt              # Dependencies
```

## Next Steps After Training

1. **Check validation accuracy** in training output
2. **Locate best checkpoint**: `../runs/stage0/gsm-cot-test/checkpoint_N/`
3. **Run evaluation** (if needed):
   ```bash
   # Edit args/gsm_coconut_eval.yaml to set load_model_path
   torchrun --nproc_per_node=1 run.py args/gsm_coconut_eval.yaml
   ```
4. **Analyze results** and compare with paper metrics

## Contact & Support

- COCONUT Paper: https://arxiv.org/abs/2412.06769
- Official Repo: https://github.com/facebookresearch/coconut
- Issues: Check training logs and troubleshooting section above
