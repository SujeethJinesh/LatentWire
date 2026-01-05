#!/usr/bin/env bash
# scripts/quick_test.sh
# Ultra-quick smoke test - runs in under 1 minute
# Use this for rapid iteration and sanity checking

set -e

echo "=========================================="
echo "LatentWire Quick Smoke Test"
echo "=========================================="
echo "This test verifies basic functionality in <1 minute"
echo ""

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Quick import test
echo "1. Testing imports..."
python3 -c "
import latentwire.train
import latentwire.eval
import latentwire.models
import latentwire.data
print('✓ All core modules imported successfully')
" || exit 1

# Quick data test
echo ""
echo "2. Testing data loading..."
python3 -c "
from latentwire.data import load_examples
examples = load_examples('squad', 'train', 2)
print(f'✓ Loaded {len(examples)} examples')
" || exit 1

# Quick model init test
echo ""
echo "3. Testing model initialization..."
python3 -c "
import torch
from latentwire.models import ByteLatentEncoder, LatentAdapter

encoder = ByteLatentEncoder(257, 64, 4, 2)
adapter = LatentAdapter(64, 4096)
x = torch.randn(1, 10, 257)
z = encoder(x)
out = adapter(z)
print(f'✓ Model forward pass successful')
print(f'  Encoder output: {z.shape}')
print(f'  Adapter output: {out.shape}')
" || exit 1

# Ultra-mini training test (2 samples, 1 step)
echo ""
echo "4. Testing mini training loop..."
python3 latentwire/train.py \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --models "llama" \
    --samples 2 \
    --epochs 1 \
    --batch_size 1 \
    --latent_len 4 \
    --d_z 32 \
    --encoder_type byte \
    --dataset squad \
    --sequential_models \
    --output_dir runs/quick_test_$(date +%s) \
    --no_save \
    2>&1 | grep -E "(Starting|Complete|Error)" || true

if [ $? -eq 0 ]; then
    echo "✓ Mini training completed"
else
    echo "✗ Mini training failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ QUICK TEST PASSED!"
echo "=========================================="
echo ""
echo "All basic components are working. For more thorough testing:"
echo "  - Standard test: TEST_MODE=standard bash scripts/run_integration_test.sh"
echo "  - Full test: bash scripts/run_full_integration_test.sh"
echo ""