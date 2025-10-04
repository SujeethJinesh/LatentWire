#!/usr/bin/env bash
# monitor_gpu.sh - Continuous GPU memory monitoring
# Usage: bash scripts/monitor_gpu.sh [output_file] [interval_seconds]

OUTPUT_FILE="${1:-gpu_monitor.log}"
INTERVAL="${2:-5}"  # Default 5 seconds

# Write header to log file only
{
    echo "GPU Monitoring started at $(date)"
    echo "Logging to: $OUTPUT_FILE every ${INTERVAL}s"
    echo "---"
} >> "$OUTPUT_FILE"

while true; do
    {
        echo ""
        echo "=== $(date) ==="
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
            --format=csv,noheader,nounits 2>&1
    } >> "$OUTPUT_FILE"

    sleep "$INTERVAL"
done
