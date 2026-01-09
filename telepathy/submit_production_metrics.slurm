#!/bin/bash
#SBATCH --job-name=telepathy_production_metrics
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/production_metrics_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/production_metrics_%j.err

# =============================================================================
# Production Metrics Benchmark for Telepathy Paper (MLSys)
# =============================================================================
# This script runs comprehensive benchmarks measuring:
# - Throughput at various batch sizes [1, 4, 8, 16, 32, 64, 128]
# - Latency breakdown (p50, p95, p99) for each component
# - Memory usage and scaling
# - Bridge vs text generation speed comparison
# - Quantization impact (fp32, fp16, int8, int4)
#
# Submit with: sbatch telepathy/submit_production_metrics.slurm
# Monitor with: squeue -u $USER
# View logs: tail -f runs/production_metrics_*.log
# =============================================================================

# Set working directory
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

# Create output directories
mkdir -p runs figures
OUTPUT_DIR="runs/production_metrics_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Pull latest code
echo ""
echo "Pulling latest code..."
git pull

# Print GPU info
echo ""
echo "GPU Information:"
nvidia-smi

# Run the production metrics benchmark
echo ""
echo "=============================================================="
echo "Running Production Metrics Benchmark"
echo "=============================================================="

python telepathy/production_metrics.py \
    --output_dir "$OUTPUT_DIR" \
    --test all \
    --batch_sizes 1 4 8 16 32 64 128 \
    --seq_lengths 128 256 512 1024 \
    --num_warmup 10 \
    --num_iterations 50 \
    --gpu 0

echo ""
echo "=============================================================="
echo "Benchmark Complete"
echo "=============================================================="

# List output files
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"

# Show summary of results
if [ -f "$OUTPUT_DIR/production_metrics.json" ]; then
    echo ""
    echo "Key Results:"
    python -c "
import json
with open('$OUTPUT_DIR/production_metrics.json') as f:
    data = json.load(f)

if 'headline_speedup' in data:
    hs = data['headline_speedup']
    print(f'  Headline Speedup: {hs[\"mean\"]:.1f}x (range: {hs[\"min\"]:.1f}x - {hs[\"max\"]:.1f}x)')

if 'throughput_batch_sweep' in data:
    bridge = data['throughput_batch_sweep']['bridge']
    print(f'  Bridge throughput at batch=1: {bridge[0][\"latency\"][\"samples_per_second\"]:.1f} samples/s')
    print(f'  Bridge throughput at batch=32: {bridge[-2][\"latency\"][\"samples_per_second\"]:.1f} samples/s')

if 'memory_models' in data:
    mem = data['memory_models']
    print(f'  Total GPU memory: {mem[\"total_system\"][\"gpu_allocated_mb\"]:.0f} MB')
"
fi

# Copy figures to main figures directory
if ls "$OUTPUT_DIR"/*.png 1> /dev/null 2>&1; then
    cp "$OUTPUT_DIR"/*.png figures/
    cp "$OUTPUT_DIR"/*.pdf figures/ 2>/dev/null || true
    echo ""
    echo "Figures copied to figures/"
fi

# Push results back to git
echo ""
echo "Pushing results to git..."
git add -A
git commit -m "results: production metrics benchmark (SLURM job $SLURM_JOB_ID)

Comprehensive benchmarks for MLSys paper:
- Throughput at batch sizes [1, 4, 8, 16, 32, 64, 128]
- Latency breakdown (p50, p95, p99) by component
- Memory usage and scaling analysis
- Quantization impact (fp32, fp16, int8, int4)

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>" || true
git push || true

echo ""
echo "=============================================================="
echo "Job completed at $(date)"
echo "=============================================================="
