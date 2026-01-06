#!/bin/bash
#SBATCH --job-name=latentwire_h100_optimized
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/h100_optimized_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/h100_optimized_%j.err

# =============================================================================
# H100-Optimized LatentWire Training
# Implements all optimizations for >90% GPU utilization
# =============================================================================
# Submit with: sbatch telepathy/submit_h100_optimized.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "H100 OPTIMIZED TRAINING JOB"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "=============================================================="

# Environment setup for H100 optimization
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# H100-specific optimizations
export OMP_NUM_THREADS=16  # Match cpus-per-task
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Better memory management
export CUDA_LAUNCH_BLOCKING=0  # Enable async operations
export NCCL_DEBUG=INFO  # Monitor multi-GPU communication
export NCCL_SOCKET_IFNAME=eth0  # Network interface for NCCL

# Enable TF32 for H100 Tensor Cores
export PYTORCH_ALLOW_TF32=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Create directories
mkdir -p runs/h100_optimized figures

# Pull latest code
echo "Pulling latest code..."
git pull

# Configuration
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-87599}"
EPOCHS="${EPOCHS:-24}"
LATENT_LEN="${LATENT_LEN:-32}"
D_Z="${D_Z:-256}"

# H100-optimized settings
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"  # Per GPU
GRAD_ACCUM="${GRAD_ACCUM:-16}"  # Effective batch = 64
NUM_WORKERS="${NUM_WORKERS:-16}"  # Data loading optimization
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"  # BF16 for stability

OUTPUT_DIR="runs/h100_optimized/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=============================================================="
echo "OPTIMIZATION CONFIGURATION"
echo "=============================================================="
echo "Model: $MODEL_ID"
echo "Dataset: $DATASET ($SAMPLES samples)"
echo "Epochs: $EPOCHS"
echo "Latent: ${LATENT_LEN}x${D_Z}"
echo ""
echo "H100 Optimizations:"
echo "  Micro batch size (per GPU): $MICRO_BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM steps"
echo "  Effective batch size: $((MICRO_BATCH_SIZE * 4 * GRAD_ACCUM))"
echo "  Mixed precision: $MIXED_PRECISION"
echo "  Data workers: $NUM_WORKERS"
echo "  GPUs: 4x H100"
echo "=============================================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start GPU monitoring in background
echo "Starting GPU monitoring..."
nvidia-smi dmon -s pucvmet -i 0,1,2,3 -f "$OUTPUT_DIR/gpu_monitor.log" &
MONITOR_PID=$!

# Function to cleanup monitoring on exit
cleanup() {
    echo "Stopping GPU monitor..."
    kill $MONITOR_PID 2>/dev/null || true
}
trap cleanup EXIT

# Run benchmark first
echo ""
echo "Running optimization benchmark..."
{
    python telepathy/optimize_h100_training.py \
        --model "$MODEL_ID" \
        --benchmark \
        --output "$OUTPUT_DIR/benchmark_results.json"
} 2>&1 | tee "$OUTPUT_DIR/benchmark.log"

# Main training with DDP
echo ""
echo "=============================================================="
echo "STARTING OPTIMIZED TRAINING"
echo "=============================================================="

# Use torchrun for distributed training
{
    torchrun \
        --nproc_per_node=4 \
        --master_port=29500 \
        --master_addr=localhost \
        latentwire/train_h100_optimized.py \
        --llama_id "$MODEL_ID" \
        --samples "$SAMPLES" \
        --epochs "$EPOCHS" \
        --batch_size "$MICRO_BATCH_SIZE" \
        --gradient_accumulation "$GRAD_ACCUM" \
        --latent_len "$LATENT_LEN" \
        --d_z "$D_Z" \
        --encoder_type byte \
        --dataset "$DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --lr 1e-4 \
        --mixed_precision "$MIXED_PRECISION" \
        --num_workers "$NUM_WORKERS" \
        --pin_memory \
        --persistent_workers \
        --compile_model \
        --compile_mode "max-autotune" \
        --use_flash_attention \
        --profile_gpu \
        --log_interval 10 \
        --save_interval 1000
} 2>&1 | tee "$LOG_FILE"

# Analyze GPU utilization
echo ""
echo "=============================================================="
echo "ANALYZING GPU UTILIZATION"
echo "=============================================================="

if [ -f "$OUTPUT_DIR/gpu_monitor.log" ]; then
    python -c "
import re

with open('$OUTPUT_DIR/gpu_monitor.log', 'r') as f:
    lines = f.readlines()

gpu_utils = {0: [], 1: [], 2: [], 3: []}
mem_utils = {0: [], 1: [], 2: [], 3: []}

for line in lines:
    parts = line.strip().split()
    if len(parts) >= 4 and parts[0].isdigit():
        gpu_id = int(parts[0])
        if gpu_id in gpu_utils:
            gpu_utils[gpu_id].append(float(parts[2]))
            mem_utils[gpu_id].append(float(parts[3]))

print('GPU Utilization Summary:')
print('-' * 40)

total_samples = 0
for gpu_id in range(4):
    if gpu_utils[gpu_id]:
        avg_gpu = sum(gpu_utils[gpu_id]) / len(gpu_utils[gpu_id])
        max_gpu = max(gpu_utils[gpu_id])
        avg_mem = sum(mem_utils[gpu_id]) / len(mem_utils[gpu_id])
        max_mem = max(mem_utils[gpu_id])
        total_samples += len(gpu_utils[gpu_id])

        print(f'GPU {gpu_id}:')
        print(f'  Compute: Avg={avg_gpu:.1f}%, Max={max_gpu:.1f}%')
        print(f'  Memory:  Avg={avg_mem:.1f}%, Max={max_mem:.1f}%')

if total_samples > 0:
    overall_avg = sum(sum(v) for v in gpu_utils.values()) / total_samples
    print(f'')
    print(f'Overall Average GPU Utilization: {overall_avg:.1f}%')

    if overall_avg >= 90:
        print('✅ TARGET ACHIEVED: >90% GPU utilization!')
    elif overall_avg >= 80:
        print('⚠️  Good utilization (>80%), but room for improvement')
    else:
        print('❌ Below target (<80%), review optimizations')

print('-' * 40)
" | tee "$OUTPUT_DIR/gpu_utilization_summary.txt"
fi

# Run evaluation on best checkpoint
echo ""
echo "=============================================================="
echo "RUNNING EVALUATION"
echo "=============================================================="

BEST_CKPT=$(ls -t "$OUTPUT_DIR"/epoch* 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    python latentwire/eval.py \
        --ckpt "$BEST_CKPT" \
        --samples 200 \
        --max_new_tokens 12 \
        --dataset "$DATASET" \
        --fresh_eval \
        --calibration embed_rms \
        --latent_anchor_mode text \
        --latent_anchor_text "Answer: " \
        --append_bos_after_prefix yes \
        2>&1 | tee "$OUTPUT_DIR/evaluation.log"
else
    echo "No checkpoint found for evaluation"
fi

# Generate performance report
echo ""
echo "=============================================================="
echo "GENERATING PERFORMANCE REPORT"
echo "=============================================================="

python -c "
import json
import glob

report = {
    'job_id': '$SLURM_JOB_ID',
    'configuration': {
        'model': '$MODEL_ID',
        'gpus': 4,
        'micro_batch_size': $MICRO_BATCH_SIZE,
        'gradient_accumulation': $GRAD_ACCUM,
        'effective_batch_size': $MICRO_BATCH_SIZE * 4 * $GRAD_ACCUM,
        'mixed_precision': '$MIXED_PRECISION',
        'num_workers': $NUM_WORKERS,
    },
    'optimizations_enabled': [
        'Mixed precision ($MIXED_PRECISION)',
        'Gradient accumulation ($GRAD_ACCUM steps)',
        'Multi-GPU DDP (4x H100)',
        'Optimized DataLoader ($NUM_WORKERS workers)',
        'torch.compile (max-autotune)',
        'Flash Attention',
        'Memory-efficient allocator',
    ],
}

# Load benchmark results if available
benchmark_file = '$OUTPUT_DIR/benchmark_results.json'
try:
    with open(benchmark_file, 'r') as f:
        report['benchmark'] = json.load(f)
except:
    pass

# Load GPU utilization summary
util_file = '$OUTPUT_DIR/gpu_utilization_summary.txt'
try:
    with open(util_file, 'r') as f:
        report['gpu_utilization'] = f.read()
except:
    pass

# Save report
with open('$OUTPUT_DIR/performance_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('Performance report saved to $OUTPUT_DIR/performance_report.json')
" | tee -a "$LOG_FILE"

# Push results to git
echo ""
echo "=============================================================="
echo "PUSHING RESULTS TO GIT"
echo "=============================================================="

git add -A
git commit -m "results: H100-optimized training (SLURM job $SLURM_JOB_ID)

Configuration:
- Model: $MODEL_ID
- GPUs: 4x H100
- Batch: ${MICRO_BATCH_SIZE}x4x${GRAD_ACCUM} = $((MICRO_BATCH_SIZE * 4 * GRAD_ACCUM))
- Mixed Precision: $MIXED_PRECISION
- Optimizations: DDP, torch.compile, Flash Attention
- Dataset: $DATASET ($SAMPLES samples, $EPOCHS epochs)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true

git push || true

echo ""
echo "=============================================================="
echo "JOB COMPLETE"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Results: $OUTPUT_DIR"
echo "End time: $(date)"
echo ""
echo "To analyze results:"
echo "  cat $OUTPUT_DIR/gpu_utilization_summary.txt"
echo "  cat $OUTPUT_DIR/performance_report.json"
echo "=============================================================="