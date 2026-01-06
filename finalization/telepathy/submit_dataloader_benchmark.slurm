#!/bin/bash
#SBATCH --job-name=dataloader_benchmark
#SBATCH --nodes=1
#SBATCH --gpus=1                                    # Only need 1 GPU for benchmarking
#SBATCH --account=marlowe-m000066                   # REQUIRED - correct account
#SBATCH --partition=preempt                         # REQUIRED - correct partition
#SBATCH --time=02:00:00                             # 2 hours should be plenty for benchmark
#SBATCH --mem=64GB                                  # 64GB for benchmark
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/dataloader_benchmark_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/dataloader_benchmark_%j.err

# =============================================================================
# Benchmark optimized dataloader performance on HPC
# =============================================================================
# This script compares the original manual batch indexing with the optimized
# multi-worker DataLoader implementation, measuring:
# - Data loading throughput (batches/second)
# - GPU utilization during loading
# - Speedup from caching and prefetching
#
# Submit with: sbatch telepathy/submit_dataloader_benchmark.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory - MUST use /projects path
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "SLURM Job Information"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create runs directory if needed
mkdir -p runs/dataloader_benchmark

# Pull latest code
echo "Pulling latest code..."
git pull

# Run benchmark comparing original vs optimized
echo ""
echo "=============================================================="
echo "Running DataLoader Benchmark"
echo "=============================================================="

python scripts/benchmark_dataloader.py \
    --dataset squad \
    --samples 5000 \
    --batch_size 64 \
    --num_batches 200 \
    --num_workers 8 \
    --device cuda \
    --output_dir runs/dataloader_benchmark \
    --compare_original \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Run a real training comparison (small scale)
echo ""
echo "=============================================================="
echo "Running Training Comparison (1 epoch, 1000 samples)"
echo "=============================================================="

# First with optimized dataloader
echo "Testing WITH optimized dataloader..."
python latentwire/train.py \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --models "llama" \
    --samples 1000 \
    --epochs 1 \
    --batch_size 64 \
    --latent_len 32 \
    --d_z 256 \
    --encoder_type byte \
    --dataset squad \
    --sequential_models \
    --output_dir runs/dataloader_benchmark/optimized \
    --use_optimized_dataloader \
    --num_dataloader_workers 8 \
    --dataloader_prefetch_factor 4 \
    --dataloader_cache_tokenization \
    --dataloader_pin_memory 2>&1 | tee runs/dataloader_benchmark/train_optimized.log

# Extract timing from optimized run
echo ""
echo "Extracting optimized timing..."
grep -E "(Epoch time|batch time|throughput)" runs/dataloader_benchmark/train_optimized.log || true

# Then without optimization for comparison
echo ""
echo "Testing WITHOUT optimized dataloader (original)..."
python latentwire/train.py \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --models "llama" \
    --samples 1000 \
    --epochs 1 \
    --batch_size 64 \
    --latent_len 32 \
    --d_z 256 \
    --encoder_type byte \
    --dataset squad \
    --sequential_models \
    --output_dir runs/dataloader_benchmark/original 2>&1 | tee runs/dataloader_benchmark/train_original.log

# Extract timing from original run
echo ""
echo "Extracting original timing..."
grep -E "(Epoch time|batch time|throughput)" runs/dataloader_benchmark/train_original.log || true

# Summary analysis
echo ""
echo "=============================================================="
echo "Creating performance summary..."
echo "=============================================================="

python -c "
import json
from pathlib import Path

results_dir = Path('runs/dataloader_benchmark')
benchmark_files = list(results_dir.glob('benchmark_results_*.json'))

if benchmark_files:
    latest = sorted(benchmark_files)[-1]
    with open(latest) as f:
        data = json.load(f)

    print('DATALOADER BENCHMARK SUMMARY')
    print('='*60)

    if 'original' in data:
        orig = data['original']['mean_time_ms']
        print(f'Original (manual indexing): {orig:.2f}ms/batch')

        for config in ['optimized_no_cache', 'optimized_with_cache', 'optimized_max']:
            if config in data:
                opt = data[config]['mean_time_ms']
                speedup = orig / opt
                print(f'{config}: {opt:.2f}ms/batch (speedup: {speedup:.2f}x)')

    print('')
    print('RECOMMENDATIONS:')
    print('- Use 4-8 workers for optimal performance')
    print('- Enable tokenization caching for multi-epoch training')
    print('- Pin memory when GPU training')
    print('- Prefetch 2-4 batches per worker')
"

# Push results back to git
echo ""
echo "Pushing results to git..."
git add -A
git commit -m "results: dataloader benchmark (SLURM job $SLURM_JOB_ID)

Benchmarked optimized dataloader vs original implementation.
Measured throughput, GPU utilization, and speedup factors.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
git push || true

echo ""
echo "=============================================================="
echo "Job completed at $(date)"
echo "Results saved to: runs/dataloader_benchmark/"
echo "=============================================================="