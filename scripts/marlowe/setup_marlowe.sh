#!/usr/bin/env bash
set -euo pipefail

# Load available modules
module load python3
module load cudatoolkit/12.5

# Create venv only if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists, skipping creation"
fi

source .venv/bin/activate

# Upgrade pip first
python3 -m pip install --upgrade pip wheel

# Update torch with compatible cuda
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu125

# Install packages (pip will skip already installed ones)
python3 -m pip install -r requirements.txt

echo "✅ Setup complete. python3: $(python3 --version)"
echo "📍 Environment: $(which python3)"

# Check GPU

echo "=== CUDA Version Check ==="
echo "1. Driver supports up to:"
nvidia-smi | grep "CUDA Version" | awk '{print $9}'

echo -e "\n2. Toolkit version:"
nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//'

echo -e "\n3. Loaded module:"
module list 2>&1 | grep cuda

echo -e "\n4. PyTorch using:"
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
    print(f'Current GPU: {torch.cuda.current_device()}')
    print(f'CUDA version: {torch.version.cuda}')
"
