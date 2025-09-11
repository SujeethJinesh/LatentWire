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
#SBATCH --signal=B:TERM@30    # Send SIGTERM 30 seconds before time limit

# Cleanup function
cleanup() {
	    echo "Cleaning up code-server files..."
	        rm -f /users/sujinesh/code-server-${SLURM_JOB_ID}.log
		    rm -f /users/sujinesh/code-server-${SLURM_JOB_ID}.err
		        echo "Cleanup complete"
		}

		# Trap signals for cleanup
		trap cleanup EXIT TERM INT

		module load code-server/4.93.1
		PORT=$(shuf -i 8000-9999 -n 1)
		HOSTNAME=$(hostname)

		echo "========================================="
		echo "COPY AND RUN THIS ON YOUR LOCAL MACHINE:"
		echo ""
		echo "ssh -L $PORT:$HOSTNAME:$PORT marlowe"
		echo ""
		echo "THEN OPEN: http://localhost:$PORT"
		echo "========================================="

		# Start code-server
		code-server --bind-addr 0.0.0.0:$PORT --auth none

		# Cleanup happens automatically when job ends
