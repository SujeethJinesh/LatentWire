#!/usr/bin/env bash
# Environment setup script for LatentWire project.
#
# This script sets up the correct environment variables for running LatentWire.
# Source this script before running any training or evaluation:
#     source setup_env.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export PYTHONPATH to include the project root
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Enable MPS fallback for Apple Silicon Macs
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Optional: Set CUDA visible devices if on GPU system
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Print confirmation
echo "LatentWire environment configured:"
echo "  PYTHONPATH includes: $SCRIPT_DIR"
echo "  PYTORCH_ENABLE_MPS_FALLBACK: $PYTORCH_ENABLE_MPS_FALLBACK"

# Check for critical dependencies
echo ""
echo "Checking dependencies:"

# Check Python version
python_version=$(python3 --version 2>&1)
echo "  Python: $python_version"

# Check PyTorch
if python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null; then
    :  # Success, message already printed
else
    echo "  ⚠ PyTorch: NOT INSTALLED - Install with: pip install torch"
fi

# Check Transformers
if python3 -c "import transformers; print(f'  Transformers: {transformers.__version__}')" 2>/dev/null; then
    :  # Success, message already printed
else
    echo "  ⚠ Transformers: NOT INSTALLED - Install with: pip install transformers"
fi

# Check other key dependencies
if python3 -c "import datasets; print(f'  Datasets: {datasets.__version__}')" 2>/dev/null; then
    :  # Success
else
    echo "  ⚠ Datasets: NOT INSTALLED - Install with: pip install datasets"
fi

if python3 -c "import numpy; print(f'  NumPy: {numpy.__version__}')" 2>/dev/null; then
    :  # Success
else
    echo "  ⚠ NumPy: NOT INSTALLED - Install with: pip install numpy"
fi

if python3 -c "import scipy; print(f'  SciPy: {scipy.__version__}')" 2>/dev/null; then
    :  # Success
else
    echo "  ⚠ SciPy: NOT INSTALLED - Install with: pip install scipy"
fi

echo ""
echo "To run the import test:"
echo "  python3 test_imports.py"
echo ""
echo "To run training:"
echo "  bash scripts/run_pipeline.sh"