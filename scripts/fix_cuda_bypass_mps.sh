#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "CUDA Fix: Bypass Broken MPS Daemon"
echo "=========================================="
echo ""

# Disable MPS entirely
unset CUDA_MPS_PIPE_DIRECTORY
unset CUDA_MPS_LOG_DIRECTORY
export CUDA_MPS_PIPE_DIRECTORY=/dev/null
export CUDA_MPS_LOG_DIRECTORY=/dev/null

# Set visible devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Disabled MPS (set pipe/log to /dev/null)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# Clear any cached CUDA state
echo "Clearing Python CUDA cache..."
python3 <<'PYTHON'
import torch
import gc

# Force garbage collection
gc.collect()

# Try to clear CUDA cache if available
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()

print("Cache cleared")
PYTHON

echo ""
echo "Testing CUDA access with MPS disabled..."
python3 <<'PYTHON'
import torch
import os
import sys

# Double-check MPS is disabled
os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/dev/null'
os.environ['CUDA_MPS_LOG_DIRECTORY'] = '/dev/null'

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA compiled version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\nERROR: CUDA still not available!")
    print("\nThis might be a deeper issue. Try:")
    print("1. Check if other users are hogging GPUs:")
    print("   nvidia-smi")
    print("2. Try a different node")
    print("3. Contact cluster admin about broken MPS")
    sys.exit(1)

print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Try to allocate a tensor
try:
    print("\nTesting GPU allocation...")
    x = torch.randn(1000, 1000, device='cuda:0')
    print(f"✓ Successfully allocated tensor on {x.device}")
    print(f"✓ Tensor shape: {x.shape}")
    del x
    torch.cuda.empty_cache()
    print("✓ Freed tensor and cleared cache")
except Exception as e:
    print(f"\n✗ Failed to allocate tensor: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("SUCCESS! CUDA is working with MPS bypassed")
print("="*50)
PYTHON

if [ $? -eq 0 ]; then
    echo ""
    echo "CUDA is working! Export these variables and run:"
    echo ""
    echo "  export CUDA_MPS_PIPE_DIRECTORY=/dev/null"
    echo "  export CUDA_MPS_LOG_DIRECTORY=/dev/null"
    echo "  export CUDA_VISIBLE_DEVICES=0,1,2,3"
    echo "  PYTHONPATH=. bash scripts/sweep_sequence_compression.sh"
    echo ""
else
    echo ""
    echo "CUDA still not working. This node may have issues."
    echo "Try running on a different node or contact cluster admin."
    exit 1
fi
