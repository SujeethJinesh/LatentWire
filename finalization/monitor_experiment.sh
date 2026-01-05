#!/usr/bin/env bash
# =============================================================================
# Experiment Monitoring Script
# =============================================================================
# Monitor running experiments with real-time updates on progress, GPU usage,
# and training metrics.
#
# Usage:
#   bash finalization/monitor_experiment.sh [EXP_NAME]
#
# If EXP_NAME is not provided, monitors the most recent experiment.
# =============================================================================

set -e

# Critical environment variable
export PYTHONUNBUFFERED=1  # Immediate output flushing

# Configuration
EXP_NAME="${1:-}"
WORK_DIR="${WORK_DIR:-/projects/m000066/sujinesh/LatentWire}"
RUNS_DIR="$WORK_DIR/runs"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-5}"  # Seconds between updates

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'  # No Color

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

find_experiment() {
    if [[ -n "$EXP_NAME" ]]; then
        echo "$RUNS_DIR/$EXP_NAME"
    else
        # Find most recent experiment
        local latest=""
        local latest_time=0

        for exp_dir in "$RUNS_DIR"/*/; do
            if [[ -d "$exp_dir" ]] && [[ -f "$exp_dir/.orchestrator_state" ]]; then
                local mod_time=$(stat -c %Y "$exp_dir/.orchestrator_state" 2>/dev/null || \
                                 stat -f %m "$exp_dir/.orchestrator_state" 2>/dev/null || echo 0)
                if [[ $mod_time -gt $latest_time ]]; then
                    latest_time=$mod_time
                    latest="$exp_dir"
                fi
            fi
        done

        echo "$latest"
    fi
}

parse_state_file() {
    local state_file="$1"
    if [[ -f "$state_file" ]]; then
        python -c "
import json
with open('$state_file', 'r') as f:
    state = json.load(f)
    print(f\"status:{state.get('status', 'unknown')}\")
    print(f\"epoch:{state.get('epoch', 0)}\")
    print(f\"checkpoint:{state.get('checkpoint', '')}\")
    print(f\"timestamp:{state.get('timestamp', '')}\")
    print(f\"retry_count:{state.get('retry_count', 0)}\")
    print(f\"slurm_job_id:{state.get('slurm_job_id', '')}\")
"
    fi
}

get_latest_metrics() {
    local diagnostics_file="$1"
    if [[ -f "$diagnostics_file" ]]; then
        # Get last 5 lines of metrics
        tail -5 "$diagnostics_file" 2>/dev/null | python -c "
import sys, json
metrics = []
for line in sys.stdin:
    try:
        data = json.loads(line.strip())
        metrics.append(data)
    except:
        pass

if metrics:
    latest = metrics[-1]
    print(f\"epoch:{latest.get('epoch', 0)}\")
    print(f\"step:{latest.get('step', 0)}\")
    print(f\"loss:{latest.get('loss', 0.0):.4f}\")
    print(f\"lr:{latest.get('lr', 0.0):.6f}\")
    print(f\"grad_norm:{latest.get('grad_norm', 0.0):.2f}\")

    # Calculate average loss over last 5 entries
    avg_loss = sum(m.get('loss', 0.0) for m in metrics) / len(metrics)
    print(f\"avg_loss:{avg_loss:.4f}\")

    # Check if loss is decreasing
    if len(metrics) > 1:
        trend = 'decreasing' if metrics[-1].get('loss', 0) < metrics[0].get('loss', 0) else 'increasing'
        print(f\"trend:{trend}\")
" 2>/dev/null || echo ""
    fi
}

format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

check_gpu_usage() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
                   --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' '{
            printf "GPU%s: %s | Util: %3d%% | Mem: %5.1f/%5.1f GB | Temp: %d°C\n",
                   $1, substr($2, 1, 20), $3, $4/1024, $5/1024, $6
        }'
    else
        echo "GPU monitoring not available"
    fi
}

display_header() {
    clear
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}   LATENTWIRE EXPERIMENT MONITOR${NC}"
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# =============================================================================
# MAIN MONITORING LOOP
# =============================================================================

main() {
    # Find experiment directory
    EXP_DIR=$(find_experiment)

    if [[ -z "$EXP_DIR" ]] || [[ ! -d "$EXP_DIR" ]]; then
        echo -e "${RED}Error: No experiment found${NC}"
        echo "Usage: $0 [experiment_name]"
        exit 1
    fi

    EXP_NAME=$(basename "$EXP_DIR")
    STATE_FILE="$EXP_DIR/.orchestrator_state"
    DIAGNOSTICS_FILE="$EXP_DIR/diagnostics.jsonl"
    LOG_DIR="$EXP_DIR/logs"

    echo -e "${GREEN}Monitoring experiment: ${BOLD}$EXP_NAME${NC}"
    echo -e "Press ${BOLD}Ctrl+C${NC} to exit"
    sleep 2

    # Main monitoring loop
    while true; do
        display_header

        # Display experiment info
        echo -e "\n${BOLD}Experiment:${NC} $EXP_NAME"
        echo -e "${BOLD}Directory:${NC} $EXP_DIR"
        echo -e "${BOLD}Time:${NC} $(date '+%Y-%m-%d %H:%M:%S')"

        # Parse and display state
        echo -e "\n${BOLD}${BLUE}═══ Orchestrator State ═══${NC}"
        if [[ -f "$STATE_FILE" ]]; then
            while IFS=':' read -r key value; do
                case "$key" in
                    "status")
                        case "$value" in
                            "training") color="${GREEN}" ;;
                            "completed") color="${GREEN}${BOLD}" ;;
                            "failed") color="${RED}${BOLD}" ;;
                            "resuming") color="${YELLOW}" ;;
                            *) color="${NC}" ;;
                        esac
                        echo -e "Status: ${color}${value}${NC}"
                        ;;
                    "epoch")
                        echo -e "Epoch: ${BOLD}${value}${NC}"
                        ;;
                    "retry_count")
                        if [[ "$value" -gt 0 ]]; then
                            echo -e "Retries: ${YELLOW}${value}${NC}"
                        fi
                        ;;
                    "slurm_job_id")
                        if [[ -n "$value" ]]; then
                            echo -e "SLURM Job: ${value}"
                        fi
                        ;;
                esac
            done < <(parse_state_file "$STATE_FILE")
        else
            echo -e "${YELLOW}State file not found${NC}"
        fi

        # Display training metrics
        echo -e "\n${BOLD}${BLUE}═══ Training Metrics ═══${NC}"
        if [[ -f "$DIAGNOSTICS_FILE" ]]; then
            while IFS=':' read -r key value; do
                case "$key" in
                    "loss")
                        echo -e "Current Loss: ${BOLD}${value}${NC}"
                        ;;
                    "avg_loss")
                        echo -e "Average Loss (5-step): ${value}"
                        ;;
                    "lr")
                        echo -e "Learning Rate: ${value}"
                        ;;
                    "grad_norm")
                        echo -e "Gradient Norm: ${value}"
                        ;;
                    "trend")
                        if [[ "$value" == "decreasing" ]]; then
                            echo -e "Trend: ${GREEN}↓ ${value}${NC}"
                        else
                            echo -e "Trend: ${YELLOW}↑ ${value}${NC}"
                        fi
                        ;;
                    "step")
                        echo -e "Step: ${value}"
                        ;;
                esac
            done < <(get_latest_metrics "$DIAGNOSTICS_FILE")

            # Count total lines in diagnostics
            local total_steps=$(wc -l < "$DIAGNOSTICS_FILE" 2>/dev/null || echo 0)
            echo -e "Total Steps Logged: ${total_steps}"
        else
            echo -e "${YELLOW}No metrics available yet${NC}"
        fi

        # Display GPU usage
        echo -e "\n${BOLD}${BLUE}═══ GPU Status ═══${NC}"
        check_gpu_usage

        # Display latest log entries
        echo -e "\n${BOLD}${BLUE}═══ Recent Log Output ═══${NC}"
        if [[ -d "$LOG_DIR" ]]; then
            LATEST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
            if [[ -f "$LATEST_LOG" ]]; then
                tail -5 "$LATEST_LOG" | sed 's/^/  /'
            else
                echo -e "${YELLOW}No logs available${NC}"
            fi
        else
            echo -e "${YELLOW}Log directory not found${NC}"
        fi

        # Display checkpoints
        echo -e "\n${BOLD}${BLUE}═══ Checkpoints ═══${NC}"
        CHECKPOINTS=$(ls -d "$EXP_DIR"/epoch* 2>/dev/null | sort -V | tail -3)
        if [[ -n "$CHECKPOINTS" ]]; then
            echo "$CHECKPOINTS" | while read -r ckpt; do
                if [[ -d "$ckpt" ]]; then
                    local ckpt_name=$(basename "$ckpt")
                    local ckpt_size=$(du -sh "$ckpt" 2>/dev/null | cut -f1)
                    echo -e "  ${GREEN}✓${NC} $ckpt_name (${ckpt_size})"
                fi
            done
        else
            echo -e "  ${YELLOW}No checkpoints saved yet${NC}"
        fi

        # Footer
        echo -e "\n${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "Refreshing every ${REFRESH_INTERVAL} seconds... Press ${BOLD}Ctrl+C${NC} to exit"

        # Check if experiment is still running
        if [[ -f "$STATE_FILE" ]]; then
            STATUS=$(parse_state_file "$STATE_FILE" | grep "^status:" | cut -d':' -f2)
            if [[ "$STATUS" == "completed" ]] || [[ "$STATUS" == "failed" ]]; then
                echo -e "\n${BOLD}${GREEN}Experiment has finished with status: $STATUS${NC}"
                echo -e "Press Enter to exit or wait to continue monitoring..."
                read -t 5 -n 1 && break || true
            fi
        fi

        sleep "$REFRESH_INTERVAL"
    done
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${BOLD}Monitoring stopped${NC}"; exit 0' INT

# Run main function
main