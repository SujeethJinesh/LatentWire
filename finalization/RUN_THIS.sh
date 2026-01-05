#!/bin/bash
# =============================================================================
# LATENTWIRE FINALIZATION EXPERIMENTS - MASTER LAUNCH SCRIPT
# =============================================================================
# This is THE ONLY script you need to run. It handles everything:
# - Environment setup and validation
# - SLURM job submission with proper settings
# - Automatic monitoring and log tailing
# - Clear status updates and instructions
# =============================================================================
# Usage: Just run this script!
#   cd finalization
#   bash RUN_THIS.sh
# =============================================================================


# =============================================================================
# LOGGING SETUP
# =============================================================================

# Ensure output directory exists
OUTPUT_DIR="${OUTPUT_DIR:-runs/RUN_THIS}"
mkdir -p "$OUTPUT_DIR"

# Create timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/RUN_THIS_${TIMESTAMP}.log"

echo "Starting RUN_THIS at $(date)" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Wrapper function for logging commands
run_with_logging() {
    echo "Running: $*" | tee -a "$LOG_FILE"
    { "$@"; } 2>&1 | tee -a "$LOG_FILE"
    return ${PIPESTATUS[0]}
}

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SLURM_SCRIPT="submit_finalization.slurm"
EXPECTED_RUNTIME="8-12 hours"
REQUIRED_GPUS=4

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Function to check if we're on HPC
check_hpc() {
    if [[ ! -d "/projects/m000066" ]]; then
        print_error "Not on HPC cluster! This script must be run on the Marlowe HPC."
        echo ""
        echo "Please SSH to the HPC cluster first:"
        echo "  ssh your_username@hpc_hostname"
        echo "  cd /projects/m000066/sujinesh/LatentWire/finalization"
        echo "  bash RUN_THIS.sh"
        exit 1
    fi
}

# Function to validate environment
validate_environment() {
    echo "=============================================================="
    echo "VALIDATING ENVIRONMENT"
    echo "=============================================================="

    # Check if on HPC
    check_hpc
    print_status "Running on HPC cluster"

    # Check current directory
    if [[ ! -f "$SLURM_SCRIPT" ]]; then
        print_error "Cannot find $SLURM_SCRIPT in current directory!"
        echo "Please ensure you're in the finalization directory:"
        echo "  cd /projects/m000066/sujinesh/LatentWire/finalization"
        exit 1
    fi
    print_status "SLURM script found: $SLURM_SCRIPT"

    # Check Python scripts exist
    if [[ ! -f "unified_cross_model_experiments.py" ]]; then
        print_error "Cannot find unified_cross_model_experiments.py!"
        echo "Please ensure all files are present in finalization/"
        exit 1
    fi
    print_status "Python experiment script found"

    # Check git status
    if command -v git &> /dev/null; then
        if git diff --quiet; then
            print_status "Git working directory is clean"
        else
            print_warning "You have uncommitted changes. Consider committing before running."
        fi
    fi

    # Check SLURM availability
    if ! command -v sbatch &> /dev/null; then
        print_error "SLURM not available! Are you on the HPC login node?"
        exit 1
    fi
    print_status "SLURM commands available"

    echo ""
}

# Function to show experiment summary
show_experiment_summary() {
    echo "=============================================================="
    echo "EXPERIMENT SUMMARY"
    echo "=============================================================="
    echo ""
    echo "This will run comprehensive finalization experiments:"
    echo ""
    echo "📊 EXPERIMENTS TO RUN:"
    echo "  • Baseline: Text-only performance (upper bound)"
    echo "  • Token Budget: Text truncated to latent token count"
    echo "  • Latent (Main): Our compression method"
    echo "  • LLMLingua: Prompt compression baseline"
    echo "  • Linear Probe: Direct linear mapping baseline"
    echo ""
    echo "🎯 DATASETS:"
    echo "  • SQuAD (question answering)"
    echo "  • AG News (classification)"
    echo "  • SST-2 (sentiment analysis)"
    echo ""
    echo "🤖 MODELS:"
    echo "  • Llama 3.1 8B Instruct"
    echo "  • Qwen 2.5 7B Instruct"
    echo ""
    echo "⚙️ CONFIGURATIONS:"
    echo "  • Compression ratios: 2x, 4x, 8x, 16x"
    echo "  • Latent dimensions: 128, 64, 32, 16 tokens"
    echo "  • Training: 10k samples, 3 epochs each"
    echo "  • Evaluation: 500 samples per experiment"
    echo ""
    echo "📦 RESOURCES:"
    echo "  • GPUs: $REQUIRED_GPUS H100s"
    echo "  • Expected runtime: $EXPECTED_RUNTIME"
    echo "  • Memory: 256GB"
    echo ""
}

# Function to submit job
submit_job() {
    echo "=============================================================="
    echo "SUBMITTING SLURM JOB"
    echo "=============================================================="
    echo ""

    # Submit the job and capture job ID
    JOB_OUTPUT=$(sbatch "$SLURM_SCRIPT" 2>&1)

    if [[ $? -eq 0 ]]; then
        # Extract job ID from output like "Submitted batch job 12345"
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+' | head -1)

        if [[ -n "$JOB_ID" ]]; then
            print_status "Job submitted successfully!"
            echo ""
            echo "  Job ID: ${GREEN}$JOB_ID${NC}"
            echo "  Script: $SLURM_SCRIPT"
            echo ""
            return 0
        else
            print_error "Could not extract job ID from: $JOB_OUTPUT"
            return 1
        fi
    else
        print_error "Job submission failed!"
        echo "Error: $JOB_OUTPUT"
        return 1
    fi
}

# Function to show monitoring instructions
show_monitoring_instructions() {
    local job_id=$1

    echo "=============================================================="
    echo "MONITORING YOUR JOB"
    echo "=============================================================="
    echo ""
    echo "Your job is now running! Here's how to monitor it:"
    echo ""
    echo "📊 CHECK JOB STATUS:"
    echo "  ${BLUE}squeue -j $job_id${NC}"
    echo ""
    echo "📝 WATCH LIVE OUTPUT:"
    echo "  ${BLUE}tail -f /projects/m000066/sujinesh/LatentWire/runs/finalization_${job_id}.log${NC}"
    echo ""
    echo "❌ CANCEL JOB (if needed):"
    echo "  ${BLUE}scancel $job_id${NC}"
    echo ""
    echo "📂 RESULTS LOCATION:"
    echo "  /projects/m000066/sujinesh/LatentWire/runs/finalization/"
    echo ""
    echo "🔄 CHECK ALL YOUR JOBS:"
    echo "  ${BLUE}squeue -u \$USER${NC}"
    echo ""
}

# Function to start automatic monitoring
start_monitoring() {
    local job_id=$1
    local log_file="/projects/m000066/sujinesh/LatentWire/runs/finalization_${job_id}.log"

    echo "=============================================================="
    echo "STARTING AUTOMATIC MONITORING"
    echo "=============================================================="
    echo ""
    print_info "Waiting for log file to be created..."
    echo ""

    # Wait for log file to appear (max 30 seconds)
    local waited=0
    while [[ ! -f "$log_file" ]] && [[ $waited -lt 30 ]]; do
        sleep 1
        waited=$((waited + 1))
        echo -ne "\rWaiting for job to start... ${waited}s"
    done
    echo ""

    if [[ -f "$log_file" ]]; then
        print_status "Log file created! Starting live output..."
        echo ""
        echo "Press Ctrl+C to stop watching (job will continue running)"
        echo "=============================================================="
        echo ""

        # Start tailing the log
        tail -f "$log_file"
    else
        print_warning "Log file not created yet. Job may be queued."
        echo ""
        echo "Check job status with:"
        echo "  ${BLUE}squeue -j $job_id${NC}"
        echo ""
        echo "Once running, watch output with:"
        echo "  ${BLUE}tail -f $log_file${NC}"
    fi
}

# Main execution
main() {
    clear

    echo "=============================================================="
    echo "🚀 LATENTWIRE FINALIZATION EXPERIMENTS LAUNCHER"
    echo "=============================================================="
    echo ""

    # Step 1: Validate environment
    validate_environment

    # Step 2: Show experiment summary
    show_experiment_summary

    # Step 3: Confirm with user
    echo "=============================================================="
    echo ""
    read -p "Do you want to proceed with submission? (y/N): " -n 1 -r
    echo ""
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Submission cancelled by user."
        exit 0
    fi

    # Step 4: Submit job
    if submit_job; then
        # Step 5: Show monitoring instructions
        show_monitoring_instructions "$JOB_ID"

        # Step 6: Ask if user wants automatic monitoring
        echo "=============================================================="
        echo ""
        read -p "Start automatic monitoring? (Y/n): " -n 1 -r
        echo ""
        echo ""

        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            start_monitoring "$JOB_ID"
        else
            print_info "You can monitor manually using the commands above."
        fi
    else
        print_error "Job submission failed. Please check the error above."
        exit 1
    fi
}

# Handle Ctrl+C gracefully
trap 'echo ""; print_info "Monitoring stopped. Job $JOB_ID continues running in background."; exit 0' INT

# Run main function
main