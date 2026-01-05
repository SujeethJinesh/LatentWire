#!/usr/bin/env bash
set -e

# =============================================================================
# Monitored Training Script with GPU Utilization Tracking
# =============================================================================
# This script runs training while monitoring GPU utilization to identify
# bottlenecks and ensure maximum resource usage.
#
# Usage:
#   bash telepathy/run_monitored_training.sh
#
# Features:
#   - Real-time GPU monitoring during training
#   - Automatic bottleneck detection
#   - Post-training optimization recommendations
# =============================================================================

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/monitored_exp}"
MONITOR_DIR="${OUTPUT_DIR}/gpu_monitor"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-1.0}"  # Sample every 1 second
ALERT_THRESHOLD="${ALERT_THRESHOLD:-80.0}"    # Alert if util < 80%

# Training configuration
SAMPLES="${SAMPLES:-10000}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LATENT_LEN="${LATENT_LEN:-32}"
D_Z="${D_Z:-256}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directories
mkdir -p "$OUTPUT_DIR" "$MONITOR_DIR"

# Create timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/training_monitored_${TIMESTAMP}.log"

echo "=============================================================="
echo "Monitored Training Session"
echo "=============================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Monitor directory: $MONITOR_DIR"
echo "Log file: $LOG_FILE"
echo "Alert threshold: ${ALERT_THRESHOLD}%"
echo "=============================================================="

# Start GPU monitoring in background
echo ""
echo "Starting GPU monitor..."
python telepathy/gpu_monitor.py \
    --output_dir "$MONITOR_DIR" \
    --interval "$MONITOR_INTERVAL" \
    --alert_threshold "$ALERT_THRESHOLD" \
    --quiet &

MONITOR_PID=$!
echo "Monitor started (PID: $MONITOR_PID)"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping GPU monitor..."
    kill $MONITOR_PID 2>/dev/null || true
    wait $MONITOR_PID 2>/dev/null || true

    # Analyze monitoring results
    echo ""
    echo "Analyzing GPU utilization..."
    python -c "
import json
from pathlib import Path

monitor_dir = Path('$MONITOR_DIR')
summary_files = list(monitor_dir.glob('gpu_summary_*.json'))

if summary_files:
    with open(summary_files[-1]) as f:
        summary = json.load(f)

    print('='*60)
    print('GPU UTILIZATION ANALYSIS')
    print('='*60)
    print(f\"Average GPU Utilization: {summary['overall_avg_utilization']:.1f}%\")
    print(f\"Total Alerts: {summary['alerts_count']}\")

    if summary['bottleneck_counts']:
        print(\"\\nBottlenecks Detected:\")
        for bottleneck, count in summary['bottleneck_counts'].items():
            print(f\"  - {bottleneck}: {count} times\")

    if summary['recommendations']:
        print(\"\\nOptimization Recommendations:\")
        for rec in summary['recommendations']:
            print(f\"  {rec}\")

    # Check if we achieved good utilization
    avg_util = summary['overall_avg_utilization']
    if avg_util >= 90:
        print(\"\\n✅ Excellent GPU utilization!\")
    elif avg_util >= 80:
        print(\"\\n✓ Good GPU utilization\")
    elif avg_util >= 70:
        print(\"\\n⚠️  Moderate GPU utilization - room for improvement\")
    else:
        print(\"\\n❌ Poor GPU utilization - significant optimization needed\")

    print(f\"\\nDetailed metrics saved to: {summary_files[-1]}\")
else:
    print('No monitoring summary found')
"
}

# Set trap to ensure cleanup happens
trap cleanup EXIT INT TERM

# Wait a moment for monitor to start
sleep 2

# Run main training
echo ""
echo "Starting training with GPU monitoring..."
echo "=============================================================="
{
    # Training command with monitoring hooks
    python -c "
import sys
import time
from pathlib import Path

# Add monitoring integration
sys.path.append('.')
from telepathy.gpu_monitor import TrainingMonitor

# Import training modules
from latentwire.train import main as train_main
from latentwire.train import parse_args

# Parse arguments
args = parse_args([
    '--llama_id', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    '--samples', '$SAMPLES',
    '--epochs', '$EPOCHS',
    '--batch_size', '$BATCH_SIZE',
    '--latent_len', '$LATENT_LEN',
    '--d_z', '$D_Z',
    '--encoder_type', 'byte',
    '--dataset', 'squad',
    '--sequential_models',
    '--warm_anchor_text', 'Answer: ',
    '--first_token_ce_weight', '0.5',
    '--output_dir', '$OUTPUT_DIR/checkpoint',
    '--llama_only',  # Focus on single model for monitoring test
])

# Run training with monitoring context
print('\\nStarting monitored training...')
start_time = time.time()

# Create a secondary monitor for detailed tracking
with TrainingMonitor(output_dir='$MONITOR_DIR', interval=0.5) as monitor:
    # Periodically check GPU health during training
    def check_callback():
        stats = monitor.get_current_stats()
        if stats['utilization'] < 50:
            print(f\"\\n⚠️  Low GPU util: {stats['utilization']:.1f}%\", file=sys.stderr)
        return stats

    # Inject monitoring callback (this is pseudo-code, actual integration depends on train.py)
    # For now, just run training normally
    train_main(args)

    # Get final stats
    final_stats = monitor.get_current_stats()
    print(f\"\\nFinal GPU stats: Util={final_stats['utilization']:.1f}%, Mem={final_stats['memory_percent']:.1f}%\")

elapsed = time.time() - start_time
print(f\"\\nTraining completed in {elapsed:.1f}s\")
"

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================================="
echo "Training complete!"
echo ""
echo "Results saved to:"
echo "  - Training: $OUTPUT_DIR/checkpoint/"
echo "  - Logs: $LOG_FILE"
echo "  - GPU metrics: $MONITOR_DIR/"
echo "=============================================================="