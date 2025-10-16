#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "CUDA/MPS Diagnostic and Fix"
echo "=========================================="
echo ""

echo "Step 1: Check nvidia-smi..."
nvidia-smi
echo ""

echo "Step 2: Kill MPS daemon if running..."
echo quit | nvidia-cuda-mps-control 2>&1 || echo "MPS not running or already stopped"
sleep 2
echo ""

echo "Step 3: Export CUDA_VISIBLE_DEVICES..."
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

echo "Step 4: Test PyTorch CUDA access..."
python3 <<'PYTHON'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Try to allocate a small tensor
    try:
        x = torch.randn(10, 10).cuda()
        print(f"\nSUCCESS: Created tensor on GPU: {x.device}")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nERROR: Failed to allocate tensor on GPU: {e}")
        sys.exit(1)
else:
    print("\nERROR: CUDA not available to PyTorch!")
    print("\nDebugging info:")
    import os
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')}")

    # Check for common issues
    try:
        import torch.cuda
        print(f"  torch.cuda module loaded: True")
    except Exception as e:
        print(f"  torch.cuda module error: {e}")

    sys.exit(1)
PYTHON

echo ""
echo "=========================================="
echo "CUDA is working! You can now run:"
echo "  export CUDA_VISIBLE_DEVICES=0,1,2,3"
echo "  PYTHONPATH=. bash scripts/sweep_sequence_compression.sh"
echo "=========================================="
