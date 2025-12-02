#!/bin/bash
#SBATCH --job-name=code-server
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=40GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/code-server-%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/code-server-%j.err
#SBATCH --signal=B:TERM@30

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

# Cleanup function - removes ALL logs and temporary files
cleanup() {
    echo "Starting comprehensive cleanup at $(date)..."
    
    # Remove this job's logs
    rm -f "${WORK_DIR}/code-server-${SLURM_JOB_ID}.log"
    rm -f "${WORK_DIR}/code-server-${SLURM_JOB_ID}.err"
    rm -f "${WORK_DIR}/code-server-${SLURM_JOB_ID}-info.txt"
    
    # Remove ALL code-server logs in project directory
    rm -f ${WORK_DIR}/code-server-*.log
    rm -f ${WORK_DIR}/code-server-*.err
    rm -f ${WORK_DIR}/code-server-*-info.txt
    
    # Clean up temp directories
    rm -rf /tmp/code-server-${SLURM_JOB_ID}
    rm -rf /tmp/code-server-ext-${SLURM_JOB_ID}
    
    # Clean up any code-server session files
    rm -rf ~/.local/share/code-server/logs/*
    rm -rf ~/.cache/code-server/*
    
    echo "Cleanup complete - all logs removed at $(date)"
}

# Trap signals for cleanup
trap cleanup EXIT TERM INT

# Clean up any old logs at start
rm -f ${WORK_DIR}/code-server-*.log 2>/dev/null
rm -f ${WORK_DIR}/code-server-*.err 2>/dev/null
rm -f ${WORK_DIR}/code-server-*-info.txt 2>/dev/null

# Load module
module load code-server/4.93.1

# Generate port and hostname
PORT=8800
HOSTNAME=$(hostname)

# Write connection info to temporary file
INFO_FILE="${WORK_DIR}/code-server-${SLURM_JOB_ID}-info.txt"

{
echo "========================================="
echo "CODE-SERVER CONNECTION INFO"
echo "========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $HOSTNAME"
echo "Port: $PORT"
echo "Started at: $(date)"
echo ""
echo "STEP 1: Run this on your LOCAL machine:"
echo ""
echo "ssh -L $PORT:$HOSTNAME:$PORT marlowe"
echo ""
echo "STEP 2: Open browser to:"
echo ""
echo "http://localhost:$PORT"
echo ""
echo "Working directory: $WORK_DIR"
echo ""
echo "========================================="
echo "To stop: scancel $SLURM_JOB_ID"
echo "========================================="
} | tee "$INFO_FILE"

# Main loop
while true; do
    echo "Starting code-server at $(date)..."
    
    # Start code-server with all data in /tmp
    code-server \
        --bind-addr 0.0.0.0:$PORT \
        --auth none \
        --disable-workspace-trust \
        --disable-telemetry \
        --disable-update-check \
        --user-data-dir /tmp/code-server-${SLURM_JOB_ID} \
        --extensions-dir /tmp/code-server-ext-${SLURM_JOB_ID} \
        "$WORK_DIR" \
        2>&1
    
    EXIT_CODE=$?
    echo "Code-server exited with code $EXIT_CODE at $(date)"
    
    # Check time limit
    if [ -n "$SLURM_JOB_END_TIME" ]; then
        TIME_LEFT=$((SLURM_JOB_END_TIME - $(date +%s)))
        if [ $TIME_LEFT -lt 30 ]; then
            echo "Less than 30 seconds left, exiting..."
            break
        fi
    fi
    
    echo "Restarting in 5 seconds..."
    sleep 5
done

echo "Job ending at $(date)"
# Cleanup will run automatically via trap
