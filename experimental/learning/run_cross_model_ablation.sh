#!/bin/bash
#SBATCH --job-name=cross_model_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=cross_model_ablation_%j.log

# Load required modules
module load cudatoolkit/12.5
module load gcc/13.1.0

# Activate virtual environment
source .venv/bin/activate

# Set up environment
export PYTHONPATH=.
export HF_HOME=/projects/m000066/sujinesh/.cache/huggingface

# Verify CUDA is available
echo "=== CUDA Check ==="
python experimental/learning/check_cuda.py

# Run the experiment
echo ""
echo "=== Running Cross-Model Ablation ==="
python experimental/learning/cross_model_ablation.py
