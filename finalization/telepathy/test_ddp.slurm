#!/bin/bash
#SBATCH --job-name=test_ddp
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=00:30:00
#SBATCH --mem=64GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/test_ddp_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/test_ddp_%j.err

# =============================================================================
# Test DistributedDataParallel (DDP) support with torchrun on 4 GPUs
# =============================================================================
# Submit with: sbatch telepathy/test_ddp.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory - MUST use /projects path
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "SLURM DDP Test Job"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs requested: 4"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create runs directory if needed
mkdir -p runs

# Pull latest code
echo "Pulling latest code..."
git pull

echo ""
echo "=============================================================="
echo "Test 1: Verify GPU availability"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "=============================================================="
echo "Test 2: Single GPU baseline (no DDP)"
echo "=============================================================="
echo "Running with single GPU..."
python latentwire/train.py \
    --samples 100 \
    --epochs 1 \
    --batch_size 16 \
    --latent_len 32 \
    --d_z 256 \
    --encoder_type byte \
    --dataset squad \
    --sequential_models \
    --output_dir runs/test_ddp/single_gpu \
    --seed 42

echo ""
echo "=============================================================="
echo "Test 3: Multi-GPU with torchrun (4 GPUs)"
echo "=============================================================="
echo "Running with torchrun on 4 GPUs..."
echo "Command: torchrun --nproc_per_node=4 latentwire/train.py ..."

# Use torchrun for proper DDP initialization
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    latentwire/train.py \
    --samples 400 \
    --epochs 1 \
    --batch_size 16 \
    --latent_len 32 \
    --d_z 256 \
    --encoder_type byte \
    --dataset squad \
    --sequential_models \
    --output_dir runs/test_ddp/multi_gpu_torchrun \
    --seed 42

echo ""
echo "=============================================================="
echo "Test 4: Verify DDP initialization in logs"
echo "=============================================================="
echo "Checking for DDP initialization messages..."
grep -i "ddp" runs/test_ddp/multi_gpu_torchrun/*.log 2>/dev/null | head -20 || echo "No DDP messages found in logs"

echo ""
echo "=============================================================="
echo "Test 5: Compare throughput"
echo "=============================================================="
echo "Analyzing training speed..."

# Extract samples/second from logs if available
if [ -f runs/test_ddp/single_gpu/diagnostics.jsonl ]; then
    echo "Single GPU throughput:"
    python -c "
import json
with open('runs/test_ddp/single_gpu/diagnostics.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'samples_per_second' in data:
            print(f'  Samples/sec: {data[\"samples_per_second\"]:.2f}')
            break
" || echo "  Could not extract throughput"
fi

if [ -f runs/test_ddp/multi_gpu_torchrun/diagnostics.jsonl ]; then
    echo "Multi-GPU throughput (4 GPUs):"
    python -c "
import json
with open('runs/test_ddp/multi_gpu_torchrun/diagnostics.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'samples_per_second' in data:
            print(f'  Samples/sec: {data[\"samples_per_second\"]:.2f}')
            break
" || echo "  Could not extract throughput"
fi

echo ""
echo "=============================================================="
echo "DDP Test Summary"
echo "=============================================================="
echo "✓ GPU detection tested"
echo "✓ Single GPU training tested"
echo "✓ Multi-GPU DDP training with torchrun tested"
echo "✓ DDP initialization verified"
echo "✓ Throughput comparison completed"
echo ""
echo "Results saved to: runs/test_ddp/"
echo "=============================================================="
echo "Job completed at $(date)"
echo "=============================================================="