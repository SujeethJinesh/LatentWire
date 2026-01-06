#!/bin/bash
#SBATCH --job-name=gpu_setup_test
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=00:30:00
#SBATCH --mem=64GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/gpu_setup_test_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/gpu_setup_test_%j.err

# =============================================================================
# Test Multi-GPU Setup on HPC
# =============================================================================
# This script tests multi-GPU configuration, CUDA_VISIBLE_DEVICES handling,
# and batch size scaling on the HPC cluster with 4 H100 GPUs.
#
# Submit with: sbatch telepathy/submit_gpu_setup_test.slurm
# Monitor with: tail -f /projects/m000066/sujinesh/LatentWire/runs/gpu_setup_test_*.log
# =============================================================================

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "GPU Setup Test on HPC"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs allocated: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory
OUTPUT_DIR="runs/gpu_setup_test_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Pull latest code
echo "Pulling latest code..."
git pull

echo ""
echo "=============================================================="
echo "System Information"
echo "=============================================================="

# Check nvidia-smi
echo "NVIDIA GPUs:"
nvidia-smi -L

echo ""
echo "NVIDIA-SMI Summary:"
nvidia-smi

echo ""
echo "=============================================================="
echo "Python Environment"
echo "=============================================================="

python -c "
import sys
import torch
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'cuDNN Version: {torch.backends.cudnn.version()}')
    print(f'Device Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)')
"

echo ""
echo "=============================================================="
echo "Test 1: Default Configuration (All 4 GPUs)"
echo "=============================================================="

python scripts/test_multi_gpu_setup.py \
    --output-json "$OUTPUT_DIR/test1_all_gpus.json"

echo ""
echo "=============================================================="
echo "Test 2: Single GPU Configurations"
echo "=============================================================="

for gpu in 0 1 2 3; do
    echo "Testing GPU $gpu only..."
    CUDA_VISIBLE_DEVICES=$gpu python scripts/test_multi_gpu_setup.py \
        --output-json "$OUTPUT_DIR/test2_gpu${gpu}.json" 2>&1 | head -30
    echo "---"
done

echo ""
echo "=============================================================="
echo "Test 3: Dual GPU Configurations"
echo "=============================================================="

echo "Testing GPUs 0,1..."
CUDA_VISIBLE_DEVICES=0,1 python scripts/test_multi_gpu_setup.py \
    --output-json "$OUTPUT_DIR/test3_gpu01.json" 2>&1 | head -30

echo ""
echo "Testing GPUs 2,3..."
CUDA_VISIBLE_DEVICES=2,3 python scripts/test_multi_gpu_setup.py \
    --output-json "$OUTPUT_DIR/test3_gpu23.json" 2>&1 | head -30

echo ""
echo "=============================================================="
echo "Test 4: All GPUs with Subprocess Tests"
echo "=============================================================="

python scripts/test_multi_gpu_setup.py \
    --subprocess-tests \
    --output-json "$OUTPUT_DIR/test4_subprocess.json"

echo ""
echo "=============================================================="
echo "Test 5: Quick Training Test (Single vs Multi-GPU)"
echo "=============================================================="

echo "Single GPU training (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 python latentwire/train.py \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --samples 100 \
    --epochs 1 \
    --batch_size 16 \
    --latent_len 16 \
    --d_z 128 \
    --encoder_type byte \
    --dataset squad \
    --output_dir "$OUTPUT_DIR/train_single_gpu" \
    --no_save_checkpoint 2>&1 | tail -20

echo ""
echo "Dual GPU training (GPUs 0,1)..."
CUDA_VISIBLE_DEVICES=0,1 python latentwire/train.py \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --samples 100 \
    --epochs 1 \
    --batch_size 32 \
    --latent_len 16 \
    --d_z 128 \
    --encoder_type byte \
    --dataset squad \
    --output_dir "$OUTPUT_DIR/train_dual_gpu" \
    --no_save_checkpoint 2>&1 | tail -20

echo ""
echo "Quad GPU training (All GPUs)..."
python latentwire/train.py \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --samples 100 \
    --epochs 1 \
    --batch_size 64 \
    --latent_len 16 \
    --d_z 128 \
    --encoder_type byte \
    --dataset squad \
    --output_dir "$OUTPUT_DIR/train_quad_gpu" \
    --no_save_checkpoint 2>&1 | tail -20

echo ""
echo "=============================================================="
echo "Test 6: DDP Test with torchrun"
echo "=============================================================="

if command -v torchrun &> /dev/null; then
    echo "Testing DDP with 4 GPUs..."
    torchrun --nproc_per_node=4 \
        --master_port=29500 \
        scripts/test_multi_gpu_setup.py \
        --test-ddp-worker
else
    echo "torchrun not available, skipping DDP test"
fi

echo ""
echo "=============================================================="
echo "Test 7: Memory Scaling Test"
echo "=============================================================="

python -c "
import torch
import json

results = {
    'gpu_count': torch.cuda.device_count(),
    'memory_per_gpu': {},
    'batch_size_recommendations': {}
}

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    total_gb = props.total_memory / (1024**3)
    results['memory_per_gpu'][i] = {
        'name': props.name,
        'total_gb': total_gb
    }

# Calculate batch size recommendations
base_batch = 64
gpu_count = results['gpu_count']

# Conservative: Same per GPU
results['batch_size_recommendations']['conservative'] = {
    'per_gpu': base_batch,
    'total': base_batch * gpu_count
}

# Memory-aware: Scale based on H100 80GB
results['batch_size_recommendations']['h100_optimized'] = {
    'per_gpu': base_batch * 2,  # H100 can handle 2x
    'total': base_batch * 2 * gpu_count
}

# Aggressive: Maximum throughput
results['batch_size_recommendations']['aggressive'] = {
    'per_gpu': base_batch * 4,
    'total': base_batch * 4 * gpu_count
}

print(json.dumps(results, indent=2))

# Save to file
with open('$OUTPUT_DIR/memory_scaling.json', 'w') as f:
    json.dump(results, f, indent=2)
"

echo ""
echo "=============================================================="
echo "Test Summary"
echo "=============================================================="

# Generate summary report
python -c "
import json
import os
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
print('Test Results Summary:')
print('-' * 60)

# Count test result files
json_files = list(output_dir.glob('*.json'))
print(f'Generated {len(json_files)} test result files')

# Check for errors in log
log_file = '/projects/m000066/sujinesh/LatentWire/runs/gpu_setup_test_${SLURM_JOB_ID}.log'
if os.path.exists(log_file):
    with open(log_file) as f:
        content = f.read()
        error_count = content.count('Error') + content.count('Traceback')
        if error_count > 0:
            print(f'⚠️  Found {error_count} potential errors in log')
        else:
            print('✅ No errors detected')

# Load and summarize memory scaling
if (output_dir / 'memory_scaling.json').exists():
    with open(output_dir / 'memory_scaling.json') as f:
        data = json.load(f)
        print(f'')
        print(f'GPU Configuration:')
        print(f'  Total GPUs: {data['gpu_count']}')
        for gpu_id, info in data['memory_per_gpu'].items():
            print(f'  GPU {gpu_id}: {info['name']} ({info['total_gb']:.1f} GB)')
        print(f'')
        print(f'Batch Size Recommendations:')
        for strategy, config in data['batch_size_recommendations'].items():
            print(f'  {strategy}: {config['per_gpu']} per GPU, {config['total']} total')
"

echo ""
echo "=============================================================="
echo "Pushing results to git..."
git add -A
git commit -m "results: GPU setup test on HPC (SLURM job $SLURM_JOB_ID)

Test configurations:
- Single GPU (0, 1, 2, 3)
- Dual GPU (0,1 and 2,3)
- Quad GPU (all 4)
- DDP with torchrun
- Memory scaling analysis

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
git push || true

echo ""
echo "=============================================================="
echo "Job completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================================="