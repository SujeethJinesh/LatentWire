#!/bin/bash
# =============================================================================
# EXACT HPC COMMAND TO RUN LATENTWIRE EXPERIMENTS ON MARLOWE
# =============================================================================
# This file contains the exact command to run on Marlowe HPC cluster
# Copy-paste this command after connecting to the cluster
# =============================================================================

cat << 'EOF'

############################################################
# COPY AND RUN THIS COMMAND ON MARLOWE HPC
############################################################

# Step 1: Connect to HPC and navigate to project directory
cd /projects/m000066/sujinesh/LatentWire

# Step 2: Pull latest code from git
git pull

# Step 3: Run the full experiment (choose ONE option below)

# ===== OPTION A: Interactive Session (Recommended - you can monitor progress) =====
srun --job-name=latentwire_paper \
     --nodes=1 \
     --gpus=4 \
     --account=marlowe-m000066 \
     --partition=preempt \
     --time=4:00:00 \
     --mem=256GB \
     --pty bash -c "
         # Ensure dependencies are installed
         if [ -f requirements.txt ]; then
             echo 'Installing dependencies...'
             pip install -q -r requirements.txt 2>/dev/null || true
         fi

         # Set environment variables
         export PYTHONPATH=.
         export PYTHONUNBUFFERED=1
         export PYTORCH_ENABLE_MPS_FALLBACK=1

         # Configuration for paper
         export SAMPLES=5000
         export EPOCHS=8
         export EVAL_SAMPLES=1000
         export OUTPUT_DIR=runs/paper_final_$(date +%Y%m%d_%H%M%S)

         # Use the FIXED script that addresses all reviewer concerns
         if [ -f finalization/RUN_FIXED.sh ]; then
             echo 'Running fixed experiment script...'
             bash finalization/RUN_FIXED.sh experiment
         else
             echo 'Running original experiment script...'
             bash finalization/RUN.sh experiment
         fi

         # Push results to git
         git add -A
         git commit -m 'results: paper final experiment' || true
         git push || true
     "

# ===== OPTION B: Quick Test (For debugging - 30 minutes, 1 GPU) =====
srun --job-name=latentwire_test \
     --nodes=1 \
     --gpus=1 \
     --account=marlowe-m000066 \
     --partition=preempt \
     --time=0:30:00 \
     --mem=64GB \
     --pty bash -c "
         export PYTHONPATH=.
         pip install -q -r requirements.txt 2>/dev/null || true
         bash finalization/RUN_FIXED.sh quick_test
     "

# ===== OPTION C: Batch Submission (Runs in background) =====
# First create the SLURM script
cat > submit_paper.slurm << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=latentwire_paper
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=4:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/paper_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/paper_%j.err

cd /projects/m000066/sujinesh/LatentWire
git pull

# Install dependencies
pip install -q -r requirements.txt 2>/dev/null || true

# Set environment
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export SAMPLES=5000
export EPOCHS=8
export EVAL_SAMPLES=1000
export OUTPUT_DIR=runs/paper_final_${SLURM_JOB_ID}

# Run experiment
bash finalization/RUN_FIXED.sh experiment

# Push results
git add -A
git commit -m "results: paper experiment job ${SLURM_JOB_ID}" || true
git push || true
SLURM_EOF

# Then submit it
sbatch submit_paper.slurm

# Monitor the job
squeue -u $USER
tail -f runs/paper_*.log

############################################################
# MONITORING COMMANDS
############################################################

# Check job status:
squeue -u $USER

# Watch output log (replace JOB_ID with actual job ID):
tail -f runs/paper_JOB_ID.log

# Cancel job if needed (replace JOB_ID with actual job ID):
scancel JOB_ID

# Pull results back to your local machine:
git pull

############################################################
# NOTES
############################################################

1. The experiment will take approximately 3-4 hours with 4 GPUs
2. Results will be automatically pushed to git when complete
3. Use Option A (interactive) if you want to monitor progress
4. Use Option C (batch) if you want to submit and disconnect
5. The RUN_FIXED.sh script addresses ALL reviewer concerns

EOF