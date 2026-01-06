#!/bin/bash
# =============================================================================
# HPC COMMAND TO RUN LATENTWIRE EXPERIMENTS ON MARLOWE
# =============================================================================
# Copy-paste these commands after connecting to the HPC cluster
# =============================================================================

cat << 'EOF'

############################################################
# EXACT COMMAND FOR MARLOWE HPC (40GB MEMORY)
############################################################

# Step 1: Connect to HPC and navigate to project
cd /projects/m000066/sujinesh/LatentWire

# Step 2: Pull latest code
git pull

# Step 3: Run the full experiment

# ===== OPTION A: Interactive Session (Recommended) =====
srun --job-name=latentwire_paper \
     --nodes=1 \
     --gpus=4 \
     --account=marlowe-m000066 \
     --partition=preempt \
     --time=4:00:00 \
     --mem=40GB \
     --pty bash -c "
         # Install dependencies
         pip install -q -r requirements.txt 2>/dev/null || true

         # Set environment
         export PYTHONPATH=.
         export PYTHONUNBUFFERED=1
         export SAMPLES=5000
         export EPOCHS=8
         export EVAL_SAMPLES=1000
         export BATCH_SIZE=4  # Reduced for 40GB
         export OUTPUT_DIR=runs/paper_final_\$(date +%Y%m%d_%H%M%S)

         # Run experiment with comprehensive logging
         bash finalization/RUN.sh experiment

         # Push results to git
         git add -A
         git commit -m 'results: paper final experiment' || true
         git push || true
     "

# ===== OPTION B: Quick Test (30 minutes, 1 GPU) =====
srun --job-name=latentwire_test \
     --nodes=1 \
     --gpus=1 \
     --account=marlowe-m000066 \
     --partition=preempt \
     --time=0:30:00 \
     --mem=40GB \
     --pty bash finalization/RUN.sh quick_test

# ===== OPTION C: Batch Submission =====
# Create and submit SLURM script
bash finalization/RUN.sh slurm experiment
sbatch runs/exp_*/submit_*.slurm

############################################################
# MONITORING & RESULTS
############################################################

# Monitor job status
squeue -u $USER

# Watch logs in real-time
tail -f runs/*/master_*.log
tail -f runs/*/training_*.log

# After completion, pull results locally
git pull

# View results
cat runs/*/LOG_INDEX.md  # See all log files
cat runs/*/evaluation_summary_*.log  # See metrics
cat runs/*/results/*.json | jq '.metrics'  # Parse JSON results

############################################################
# CONFIGURATION OPTIONS
############################################################

# All parameters can be customized via environment variables:
export SAMPLES=10000       # More training samples
export EPOCHS=12           # More epochs
export EVAL_SAMPLES=full   # Use full test set
export BATCH_SIZE=2        # Smaller batch for memory
export SEEDS="42 123 456 789 999"  # More seeds

# Then run with your custom configuration:
srun --mem=40GB ... bash finalization/RUN.sh experiment

############################################################
# NOTES
############################################################

1. The consolidated RUN.sh includes:
   - All reviewer fixes (path resolution, output paths, etc.)
   - Comprehensive logging to timestamped files
   - Automatic dependency installation
   - Configuration saved to JSON
   - Log index generation

2. Memory configuration:
   - 40GB is sufficient with BATCH_SIZE=4
   - If OOM, reduce to BATCH_SIZE=2

3. All outputs are logged:
   - master_*.log: Everything
   - training_*.log: Training progress
   - eval_seed*_*.log: Per-seed evaluation
   - config.json: Experiment configuration
   - LOG_INDEX.md: Guide to all logs

4. Results automatically pushed to git when complete

EOF