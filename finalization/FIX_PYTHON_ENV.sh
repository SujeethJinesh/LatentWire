#!/bin/bash
# =============================================================================
# FIX PYTHON ENVIRONMENT CONFLICTS ON HPC
# =============================================================================
# This script fixes the dataclasses import error caused by conflicting
# Python packages between user site-packages and system packages
# =============================================================================

set -e

echo "=============================================================="
echo "Python Environment Fix Script"
echo "=============================================================="
echo "This will fix the dataclasses import error by:"
echo "1. Creating a clean virtual environment"
echo "2. Installing packages without user site-packages"
echo "3. Setting environment variables to prevent conflicts"
echo "=============================================================="

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo ""
echo "Step 1: Checking current Python environment..."
echo "Python version:"
python3 --version
echo "Python executable:"
which python3

echo ""
echo "Step 2: Removing old virtual environment if exists..."
if [ -d .venv ]; then
    rm -rf .venv
    echo "Removed old .venv directory"
fi

echo ""
echo "Step 3: Creating fresh virtual environment..."
python3 -m venv .venv --system-site-packages
echo "Created new virtual environment"

echo ""
echo "Step 4: Activating virtual environment..."
source .venv/bin/activate
echo "Activated virtual environment"
echo "New Python executable:"
which python3
which pip

echo ""
echo "Step 5: Setting environment variables..."
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
echo "PYTHONNOUSERSITE=1 (ignoring user site-packages)"
echo "PYTHONPATH=$PYTHONPATH"

echo ""
echo "Step 6: Installing packages..."
python3 -m pip install --upgrade pip --no-user

# Install core packages first
echo "Installing core packages..."
python3 -m pip install torch --no-user || echo "torch install had issues"
python3 -m pip install transformers==4.45.2 --no-user || echo "transformers install had issues"
python3 -m pip install datasets --no-user || echo "datasets install had issues"
python3 -m pip install accelerate --no-user || echo "accelerate install had issues"

# Try full requirements
echo "Installing remaining requirements..."
python3 -m pip install -r requirements.txt --no-user 2>&1 | grep -v "Requirement already satisfied" || true

echo ""
echo "Step 7: Testing imports..."
python3 -c "
import sys
print('Python:', sys.version)
print('Executable:', sys.executable)
print()
print('Testing critical imports...')
try:
    import dataclasses
    print('✓ dataclasses')
except Exception as e:
    print('✗ dataclasses:', e)

try:
    import torch
    print('✓ torch')
except Exception as e:
    print('✗ torch:', e)

try:
    import transformers
    print('✓ transformers')
except Exception as e:
    print('✗ transformers:', e)

try:
    import datasets
    print('✓ datasets')
except Exception as e:
    print('✗ datasets:', e)

try:
    from datasets import load_dataset
    print('✓ datasets.load_dataset')
except Exception as e:
    print('✗ datasets.load_dataset:', e)
"

echo ""
echo "=============================================================="
echo "Environment fix complete!"
echo ""
echo "To use this environment in your session:"
echo "  source .venv/bin/activate"
echo "  export PYTHONNOUSERSITE=1"
echo ""
echo "The RUN.sh script has also been updated to use these fixes."
echo "=============================================================="