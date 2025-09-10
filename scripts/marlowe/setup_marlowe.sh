#!/usr/bin/env bash
set -euo pipefail

# Load available modules
module load python3
module load cudatoolkit/12.5

# Create venv only if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
else
    echo "Virtual environment already exists, skipping creation"
fi

source .venv/bin/activate

# Upgrade pip first
python -m pip install --upgrade pip wheel

# Install packages (pip will skip already installed ones)
python -m pip install -r requirements.txt

echo "✅ Setup complete. Python: $(python --version)"
echo "📍 Environment: $(which python)"

# Check GPU

echo "=== CUDA Version Check ==="
echo "1. Driver supports up to:"
nvidia-smi | grep "CUDA Version" | awk '{print $9}'

echo -e "\n2. Toolkit version:"
nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//'

echo -e "\n3. Loaded module:"
module list 2>&1 | grep cuda

echo -e "\n4. PyTorch using:"
python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'No CUDA')"
