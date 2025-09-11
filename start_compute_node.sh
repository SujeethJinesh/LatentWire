#!/bin/bash
#SBATCH --job-name=code-server
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=40GB
#SBATCH --output=code-server-%j.log
#SBATCH --error=code-server-%j.err

# Load code-server module
module load code-server/4.93.1

# Generate a random port to avoid conflicts
PORT=$(shuf -i 8000-9999 -n 1)
HOSTNAME=$(hostname)

# Print connection instructions
echo "========================================="
echo "Code-server started at: $(date)"
echo "Running on node: $HOSTNAME"
echo "Port: $PORT"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "To connect, run this from your LOCAL machine:"
echo "ssh -L 8080:$HOSTNAME:$PORT marlowe"
echo ""
echo "Then open in your browser:"
echo "http://localhost:8080"
echo ""
echo "To check job status: squeue -j $SLURM_JOB_ID"
echo "To cancel: scancel $SLURM_JOB_ID"
echo "========================================="

# Start code-server
code-server --bind-addr 0.0.0.0:$PORT --auth none

# Job will end when code-server stops or time limit is reached
