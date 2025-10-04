#!/usr/bin/env bash
# monitor_gpu.sh - Continuous GPU memory monitoring
# Usage: bash scripts/monitor_gpu.sh [output_file] [interval_seconds]

OUTPUT_FILE="${1:-gpu_monitor.log}"
INTERVAL="${2:-5}"  # Default 5 seconds

echo "GPU Monitoring started at $(date)" | tee -a "$OUTPUT_FILE"
echo "Logging to: $OUTPUT_FILE every ${INTERVAL}s" | tee -a "$OUTPUT_FILE"
echo "Press Ctrl+C to stop" | tee -a "$OUTPUT_FILE"
echo "---" | tee -a "$OUTPUT_FILE"

while true; do
    echo -e "\n=== $(date) ===" >> "$OUTPUT_FILE"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits >> "$OUTPUT_FILE" 2>&1

    # Also show a summary to stdout every 30 seconds
    if (( $(date +%s) % 30 == 0 )); then
        echo "$(date): GPU Status"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    fi

    sleep "$INTERVAL"
done
