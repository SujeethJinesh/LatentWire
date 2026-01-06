#!/bin/bash
#SBATCH --job-name=gpu_monitor_test
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=02:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/gpu_monitor_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/gpu_monitor_%j.err

# =============================================================================
# GPU Monitoring Test - Verify Maximum GPU Utilization
# =============================================================================
# This script runs training with GPU monitoring to identify bottlenecks
# and ensure we're maximizing GPU usage on 4x H100s.
#
# Submit with: sbatch telepathy/submit_gpu_monitoring.slurm
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
mkdir -p runs figures

# Pull latest code
echo "Pulling latest code..."
git pull

# Configuration for monitoring test
OUTPUT_DIR="runs/gpu_monitor_test_${SLURM_JOB_ID}"
MONITOR_DIR="${OUTPUT_DIR}/metrics"
LOG_FILE="${OUTPUT_DIR}/training.log"

mkdir -p "$OUTPUT_DIR" "$MONITOR_DIR"

# Install pynvml if not present (for better GPU monitoring)
echo "Checking for pynvml..."
python -c "import pynvml" 2>/dev/null || pip install --user nvidia-ml-py

echo ""
echo "=============================================================="
echo "Starting GPU Monitoring Test"
echo "=============================================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Start GPU monitor in background
echo "Starting GPU monitor..."
python telepathy/gpu_monitor.py \
    --output_dir "$MONITOR_DIR" \
    --interval 1.0 \
    --alert_threshold 80.0 \
    --duration 7200 &  # Monitor for up to 2 hours

MONITOR_PID=$!
echo "Monitor started (PID: $MONITOR_PID)"

# Function to ensure monitor is stopped
cleanup_monitor() {
    echo "Stopping GPU monitor..."
    kill $MONITOR_PID 2>/dev/null || true
    wait $MONITOR_PID 2>/dev/null || true
}
trap cleanup_monitor EXIT

# Wait for monitor to initialize
sleep 3

echo ""
echo "Running training workload with monitoring..."
echo "=============================================================="

# Run main training with different configurations to test GPU utilization
{
    # Test 1: Baseline training
    echo "[$(date '+%H:%M:%S')] Test 1: Baseline configuration"
    python latentwire/train.py \
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --samples 5000 \
        --epochs 2 \
        --batch_size 64 \
        --latent_len 32 \
        --d_z 256 \
        --encoder_type byte \
        --dataset squad \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --output_dir "${OUTPUT_DIR}/baseline" \
        --llama_only

    echo ""
    echo "[$(date '+%H:%M:%S')] Test 2: Large batch size"
    python latentwire/train.py \
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --samples 5000 \
        --epochs 2 \
        --batch_size 128 \
        --latent_len 32 \
        --d_z 256 \
        --encoder_type byte \
        --dataset squad \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --output_dir "${OUTPUT_DIR}/large_batch" \
        --llama_only

    echo ""
    echo "[$(date '+%H:%M:%S')] Test 3: Multi-GPU data parallel"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python latentwire/train.py \
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --samples 20000 \
        --epochs 2 \
        --batch_size 256 \
        --latent_len 32 \
        --d_z 256 \
        --encoder_type byte \
        --dataset squad \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --output_dir "${OUTPUT_DIR}/multi_gpu" \
        --llama_only

} 2>&1 | tee "$LOG_FILE"

# Stop monitor
cleanup_monitor

echo ""
echo "=============================================================="
echo "Analyzing GPU Utilization Results"
echo "=============================================================="

# Analyze monitoring data
python -c "
import json
import glob
from pathlib import Path

monitor_dir = Path('$MONITOR_DIR')

# Find summary file
summary_files = list(monitor_dir.glob('gpu_summary_*.json'))
if not summary_files:
    print('No monitoring summary found')
    exit(1)

with open(summary_files[-1]) as f:
    summary = json.load(f)

print('GPU UTILIZATION REPORT')
print('='*60)
print(f\"Monitoring Duration: {summary['monitoring_duration_seconds']:.1f}s\")
print(f\"Total Samples: {summary['total_samples']}\")
print('')
print(f\"OVERALL AVERAGE GPU UTILIZATION: {summary['overall_avg_utilization']:.1f}%\")
print('')

# Per-GPU statistics
for gpu_id, stats in summary['gpu_stats'].items():
    print(f\"GPU {gpu_id}:\")
    print(f\"  Avg Utilization: {stats['avg_utilization']:.1f}%\")
    print(f\"  Min/Max: {stats['min_utilization']:.1f}% / {stats['max_utilization']:.1f}%\")
    print(f\"  Avg Memory: {stats['avg_memory_percent']:.1f}%\")
    print(f\"  Max Memory: {stats['max_memory_gb']:.2f}GB\")
    print(f\"  Avg Temperature: {stats['avg_temperature']:.1f}°C\")
    print(f\"  Time Below 80%: {stats['time_below_threshold']:.1f}s\")
    print('')

# Bottleneck analysis
if summary['bottleneck_counts']:
    print('BOTTLENECKS DETECTED:')
    for bottleneck, count in summary['bottleneck_counts'].items():
        print(f\"  {bottleneck}: {count} occurrences\")
    print('')

# Recommendations
if summary['recommendations']:
    print('OPTIMIZATION RECOMMENDATIONS:')
    for rec in summary['recommendations']:
        print(f\"  {rec}\")
    print('')

# Performance grade
avg_util = summary['overall_avg_utilization']
if avg_util >= 90:
    grade = '✅ EXCELLENT (90%+)'
elif avg_util >= 80:
    grade = '✓ GOOD (80-90%)'
elif avg_util >= 70:
    grade = '⚠️  MODERATE (70-80%)'
elif avg_util >= 60:
    grade = '⚠️  POOR (60-70%)'
else:
    grade = '❌ VERY POOR (<60%)'

print(f\"PERFORMANCE GRADE: {grade}\")
print('='*60)

# Save detailed report
report_path = Path('$OUTPUT_DIR') / 'gpu_utilization_report.txt'
with open(report_path, 'w') as f:
    f.write(f\"GPU Utilization Report - Job {$SLURM_JOB_ID}\\n\")
    f.write('='*60 + '\\n')
    f.write(f\"Average GPU Utilization: {avg_util:.1f}%\\n\")
    f.write(f\"Performance Grade: {grade}\\n\")
    if summary['recommendations']:
        f.write('\\nRecommendations:\\n')
        for rec in summary['recommendations']:
            f.write(f\"  - {rec}\\n\")

print(f\"\\nDetailed report saved to: {report_path}\")
"

echo ""
echo "=============================================================="
echo "Creating optimization recommendations..."
echo "=============================================================="

# Generate actionable recommendations based on results
python -c "
import json
from pathlib import Path

# Read summary
monitor_dir = Path('$MONITOR_DIR')
summary_files = list(monitor_dir.glob('gpu_summary_*.json'))
if summary_files:
    with open(summary_files[-1]) as f:
        summary = json.load(f)

    avg_util = summary['overall_avg_utilization']
    bottlenecks = summary.get('bottleneck_counts', {})

    print('ACTIONABLE NEXT STEPS:')
    print('')

    if avg_util < 80:
        print('1. GPU utilization is below target (80%). Key actions:')

        if 'data_loading' in bottlenecks:
            print('   - Increase DataLoader num_workers (try 8-16)')
            print('   - Enable persistent_workers=True')
            print('   - Use pin_memory=True for faster GPU transfers')
            print('   - Consider prefetch_factor=2 or higher')

        if 'cpu_bound' in bottlenecks:
            print('   - Profile with torch.profiler to find slow CPU ops')
            print('   - Use torch.compile() for graph optimization')
            print('   - Move preprocessing to GPU where possible')

        if 'memory_bound' in bottlenecks:
            print('   - Implement gradient checkpointing')
            print('   - Use mixed precision training (fp16/bf16)')
            print('   - Optimize batch size for memory efficiency')

    else:
        print('✓ GPU utilization is good (>80%)')
        print('  Continue monitoring during full training runs')

    print('')
    print('2. Next experiment to run:')
    print('   sbatch telepathy/submit_optimized_training.slurm')
    print('   (This will apply the optimizations above)')
"

# Push results back to git
echo ""
echo "Pushing results to git..."
git add -A
git commit -m "results: GPU monitoring test - avg utilization analysis (SLURM job $SLURM_JOB_ID)

- Tested GPU utilization across different configurations
- Identified bottlenecks and optimization opportunities
- Generated recommendations for improving throughput

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
git push || true

echo ""
echo "=============================================================="
echo "GPU Monitoring Test Complete!"
echo "=============================================================="
echo "Results saved to: $OUTPUT_DIR"
echo "Monitor logs: $MONITOR_DIR"
echo ""
echo "To view results locally:"
echo "  git pull"
echo "  cat $OUTPUT_DIR/gpu_utilization_report.txt"
echo ""
echo "Job completed at $(date)"
echo "=============================================================="