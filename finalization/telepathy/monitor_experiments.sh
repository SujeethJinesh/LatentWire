#!/bin/bash

# =============================================================================
# Experiment Monitoring Script for HPC/SLURM
# =============================================================================
# Purpose: Monitor long-running experiments with GPU, memory, and progress tracking
# Usage: bash telepathy/monitor_experiments.sh [options]
# =============================================================================

set -e

# Configuration
REFRESH_INTERVAL="${REFRESH_INTERVAL:-30}"  # Seconds between updates
LOG_DIR="${LOG_DIR:-/projects/m000066/sujinesh/LatentWire/runs}"
ALERT_ON_STALL="${ALERT_ON_STALL:-yes}"
STALL_THRESHOLD="${STALL_THRESHOLD:-300}"  # Alert if no progress for 5 minutes
COLOR_OUTPUT="${COLOR_OUTPUT:-yes}"

# Color codes for output
if [ "$COLOR_OUTPUT" = "yes" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    MAGENTA=''
    CYAN=''
    BOLD=''
    NC=''
fi

# Parse command line arguments
MODE="auto"  # auto, job_id, or log_file
TARGET=""
WATCH_MODE="no"
SUMMARY_ONLY="no"

usage() {
    cat << EOF
${BOLD}Experiment Monitoring Script${NC}

${BOLD}Usage:${NC}
    $(basename $0) [options]

${BOLD}Options:${NC}
    -j, --job <job_id>     Monitor specific SLURM job
    -l, --log <log_file>   Monitor specific log file
    -a, --all              Monitor all user's jobs (default)
    -w, --watch            Continuous watch mode (updates every ${REFRESH_INTERVAL}s)
    -s, --summary          Show summary only (no continuous monitoring)
    -r, --refresh <sec>    Set refresh interval (default: ${REFRESH_INTERVAL})
    -h, --help            Show this help message

${BOLD}Examples:${NC}
    # Monitor all your running jobs
    $(basename $0)

    # Monitor specific job in watch mode
    $(basename $0) -j 12345 -w

    # Monitor specific log file
    $(basename $0) -l runs/experiment_12345.log

    # Quick summary of all jobs
    $(basename $0) -s

    # Watch with custom refresh interval
    $(basename $0) -w -r 10

${BOLD}Environment Variables:${NC}
    REFRESH_INTERVAL    Seconds between updates (default: 30)
    LOG_DIR            Directory containing logs (default: /projects/.../runs)
    ALERT_ON_STALL     Alert if training stalls (default: yes)
    STALL_THRESHOLD    Seconds without progress to trigger alert (default: 300)
    COLOR_OUTPUT       Use colored output (default: yes)

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--job)
            MODE="job_id"
            TARGET="$2"
            shift 2
            ;;
        -l|--log)
            MODE="log_file"
            TARGET="$2"
            shift 2
            ;;
        -a|--all)
            MODE="auto"
            shift
            ;;
        -w|--watch)
            WATCH_MODE="yes"
            shift
            ;;
        -s|--summary)
            SUMMARY_ONLY="yes"
            shift
            ;;
        -r|--refresh)
            REFRESH_INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Function to format time duration
format_duration() {
    local seconds=$1
    local days=$((seconds / 86400))
    local hours=$(( (seconds % 86400) / 3600 ))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))

    if [ $days -gt 0 ]; then
        printf "%dd %02dh %02dm" $days $hours $minutes
    elif [ $hours -gt 0 ]; then
        printf "%dh %02dm %02ds" $hours $minutes $secs
    else
        printf "%dm %02ds" $minutes $secs
    fi
}

# Function to get GPU utilization for a job
get_gpu_stats() {
    local job_id=$1
    local node=$(squeue -j "$job_id" -h -o "%N" 2>/dev/null || echo "")

    if [ -z "$node" ]; then
        echo "N/A"
        return
    fi

    # Try to get GPU stats from the node
    local gpu_info=$(srun --jobid="$job_id" nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "")

    if [ -z "$gpu_info" ]; then
        echo "Unable to query"
        return
    fi

    echo "$gpu_info" | while IFS=',' read -r idx name util mem_used mem_total temp; do
        # Trim whitespace
        util=$(echo $util | xargs)
        mem_used=$(echo $mem_used | xargs)
        mem_total=$(echo $mem_total | xargs)
        temp=$(echo $temp | xargs)

        # Calculate memory percentage
        if [ "$mem_total" -gt 0 ] 2>/dev/null; then
            mem_pct=$((100 * mem_used / mem_total))
        else
            mem_pct=0
        fi

        # Color code utilization
        if [ "$util" -gt 80 ]; then
            util_color="$GREEN"
        elif [ "$util" -gt 50 ]; then
            util_color="$YELLOW"
        else
            util_color="$RED"
        fi

        printf "  GPU %s: %s%3d%%%s util | Mem: %d/%d MB (%d%%) | Temp: %d°C\n" \
            "$idx" "$util_color" "$util" "$NC" "$mem_used" "$mem_total" "$mem_pct" "$temp"
    done
}

# Function to parse training metrics from log
parse_training_metrics() {
    local log_file=$1
    local last_lines="${2:-100}"

    if [ ! -f "$log_file" ]; then
        echo "Log file not found"
        return
    fi

    # Get last modification time
    local last_mod=$(stat -c %Y "$log_file" 2>/dev/null || stat -f %m "$log_file" 2>/dev/null || echo 0)
    local current_time=$(date +%s)
    local time_since_update=$((current_time - last_mod))

    # Check for stall
    if [ "$time_since_update" -gt "$STALL_THRESHOLD" ] && [ "$ALERT_ON_STALL" = "yes" ]; then
        echo -e "${RED}${BOLD}⚠ WARNING: No updates for $(format_duration $time_since_update)${NC}"
    fi

    # Extract metrics from recent log entries
    local recent_logs=$(tail -n "$last_lines" "$log_file" 2>/dev/null)

    # Look for loss values (adapt patterns based on your logging format)
    local latest_loss=$(echo "$recent_logs" | grep -oE "loss[: ]+[0-9]+\.[0-9]+" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
    local latest_acc=$(echo "$recent_logs" | grep -oE "acc(uracy)?[: ]+[0-9]+\.[0-9]+" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
    local latest_f1=$(echo "$recent_logs" | grep -oE "f1[: ]+[0-9]+\.[0-9]+" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")

    # Look for epoch/step information
    local epoch=$(echo "$recent_logs" | grep -oE "[Ee]poch[: ]+[0-9]+" | tail -1 | grep -oE "[0-9]+" || echo "N/A")
    local step=$(echo "$recent_logs" | grep -oE "[Ss]tep[: ]+[0-9]+" | tail -1 | grep -oE "[0-9]+" || echo "N/A")

    # Look for ETA or progress
    local progress=$(echo "$recent_logs" | grep -oE "[0-9]+/[0-9]+" | tail -1 || echo "N/A")
    local eta=$(echo "$recent_logs" | grep -oE "ETA[: ]+[0-9]+:[0-9]+" | tail -1 | grep -oE "[0-9]+:[0-9]+" || echo "N/A")

    # Display metrics
    echo -e "${CYAN}Training Metrics:${NC}"
    [ "$epoch" != "N/A" ] && echo "  Epoch: $epoch"
    [ "$step" != "N/A" ] && echo "  Step: $step"
    [ "$progress" != "N/A" ] && echo "  Progress: $progress"
    [ "$latest_loss" != "N/A" ] && echo "  Loss: $latest_loss"
    [ "$latest_acc" != "N/A" ] && echo "  Accuracy: $latest_acc"
    [ "$latest_f1" != "N/A" ] && echo "  F1: $latest_f1"
    [ "$eta" != "N/A" ] && echo "  ETA: $eta"

    # Show recent errors if any
    local errors=$(echo "$recent_logs" | grep -iE "error|exception|traceback" | tail -3)
    if [ ! -z "$errors" ]; then
        echo -e "${RED}Recent Errors:${NC}"
        echo "$errors" | sed 's/^/  /'
    fi
}

# Function to monitor a specific job
monitor_job() {
    local job_id=$1

    # Get job info
    local job_info=$(squeue -j "$job_id" -h -o "%T|%M|%N|%j" 2>/dev/null)

    if [ -z "$job_info" ]; then
        echo -e "${RED}Job $job_id not found or completed${NC}"
        return 1
    fi

    IFS='|' read -r state runtime node jobname <<< "$job_info"

    echo -e "${BOLD}=== Job $job_id: $jobname ===${NC}"
    echo -e "Status: ${GREEN}$state${NC}"
    echo -e "Runtime: $(format_duration $(echo $runtime | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }'))"
    echo -e "Node: $node"
    echo ""

    # Get GPU stats
    echo -e "${MAGENTA}GPU Utilization:${NC}"
    get_gpu_stats "$job_id"
    echo ""

    # Find and parse log file
    local log_file="${LOG_DIR}/$(echo $jobname | sed 's/[^a-zA-Z0-9_-]/_/g')_${job_id}.log"
    if [ ! -f "$log_file" ]; then
        # Try alternative log naming patterns
        log_file=$(find "$LOG_DIR" -name "*${job_id}*.log" -type f 2>/dev/null | head -1)
    fi

    if [ -f "$log_file" ]; then
        echo -e "${BLUE}Log: $log_file${NC}"
        parse_training_metrics "$log_file"
    else
        echo -e "${YELLOW}Log file not found${NC}"
    fi
}

# Function to monitor all user's jobs
monitor_all_jobs() {
    local user_jobs=$(squeue -u "$USER" -h -o "%i" 2>/dev/null)

    if [ -z "$user_jobs" ]; then
        echo -e "${YELLOW}No running jobs found for user $USER${NC}"
        return
    fi

    echo -e "${BOLD}Monitoring all jobs for user $USER${NC}"
    echo "=================================================="

    for job_id in $user_jobs; do
        monitor_job "$job_id"
        echo ""
        echo "--------------------------------------------------"
        echo ""
    done
}

# Function to monitor a specific log file
monitor_log_file() {
    local log_file=$1

    if [ ! -f "$log_file" ]; then
        echo -e "${RED}Log file not found: $log_file${NC}"
        return 1
    fi

    echo -e "${BOLD}=== Monitoring: $(basename $log_file) ===${NC}"

    # Try to extract job ID from filename
    local job_id=$(basename "$log_file" | grep -oE "[0-9]{5,}" | head -1)

    if [ ! -z "$job_id" ]; then
        # Check if job is still running
        local job_state=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)
        if [ ! -z "$job_state" ]; then
            echo -e "Job Status: ${GREEN}$job_state${NC}"
            echo ""
            echo -e "${MAGENTA}GPU Utilization:${NC}"
            get_gpu_stats "$job_id"
            echo ""
        fi
    fi

    parse_training_metrics "$log_file"

    # Show tail of log
    echo ""
    echo -e "${CYAN}Recent Output:${NC}"
    tail -n 10 "$log_file" | sed 's/^/  /'
}

# Function for summary view
show_summary() {
    echo -e "${BOLD}Job Summary for $USER${NC}"
    echo "=================================================="

    # Get all jobs with details
    local jobs=$(squeue -u "$USER" -o "%i|%j|%T|%M|%N" --noheader 2>/dev/null)

    if [ -z "$jobs" ]; then
        echo -e "${YELLOW}No running jobs${NC}"
        return
    fi

    echo "$jobs" | while IFS='|' read -r id name state runtime node; do
        local runtime_sec=$(echo $runtime | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')
        echo -e "${GREEN}[$id]${NC} $name"
        echo -e "  State: $state | Runtime: $(format_duration $runtime_sec) | Node: $node"

        # Quick GPU check
        local gpu_util=$(srun --jobid="$id" nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "N/A")
        [ "$gpu_util" != "N/A" ] && echo -e "  GPU Util: ${GREEN}${gpu_util}%${NC}"
        echo ""
    done
}

# Main monitoring loop
main() {
    # Clear screen for watch mode
    [ "$WATCH_MODE" = "yes" ] && clear

    if [ "$SUMMARY_ONLY" = "yes" ]; then
        show_summary
        exit 0
    fi

    while true; do
        # Clear screen for watch mode updates
        [ "$WATCH_MODE" = "yes" ] && [ "$FIRST_RUN" != "yes" ] && clear
        FIRST_RUN="no"

        # Header
        echo -e "${BOLD}LatentWire Experiment Monitor${NC}"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=================================================="
        echo ""

        # Execute based on mode
        case $MODE in
            job_id)
                monitor_job "$TARGET"
                ;;
            log_file)
                monitor_log_file "$TARGET"
                ;;
            auto|*)
                monitor_all_jobs
                ;;
        esac

        # Exit or wait for next update
        if [ "$WATCH_MODE" = "yes" ]; then
            echo ""
            echo -e "${CYAN}Refreshing in ${REFRESH_INTERVAL} seconds... (Ctrl+C to stop)${NC}"
            sleep "$REFRESH_INTERVAL"
        else
            break
        fi
    done
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Monitoring stopped${NC}"; exit 0' INT

# Run main function
main