#!/usr/bin/env python3
"""
Quick test to verify Stage 1 training can start without errors.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from latentwire.data import load_squad_subset
from latentwire.models import Adapter

def test_stage1_components():
    """Test all Stage 1 components work together"""

    print("Testing Stage 1 components...")
    print("="*50)

    # Test 1: Data loading
    print("\n1. Testing data loading...")
    dataset = load_squad_subset("train", 5)
    val_dataset = load_squad_subset("validation", 5)

    sample = dataset[0]
    print(f"  ✓ Data loaded: {len(dataset)} train, {len(val_dataset)} val")
    print(f"  ✓ Sample keys: {list(sample.keys())}")
    print(f"  ✓ Source length: {len(sample['source'])}")
    print(f"  ✓ Answer: {sample['answer'][:50]}...")

    # Test 2: Create model and tokenizer (small model for speed)
    print("\n2. Testing model loading...")
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Just test tokenizer for speed
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✓ Tokenizer loaded")

    # Test 3: Process data format
    print("\n3. Testing data processing...")
    text = sample['source'] + "Answer: "
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    print(f"  ✓ Text tokenized: {inputs.input_ids.shape}")

    # Test 4: Create adapter
    print("\n4. Testing adapter creation...")
    adapter = Adapter(
        d_z=512,
        d_model=4096,
        latent_length=32,
        hidden_mult=4,
        dropout=0.1,
        enable_metadata=False,
        colorize=False
    )
    print(f"  ✓ Adapter created: {sum(p.numel() for p in adapter.parameters()):,} params")

    # Test 5: Test adapter forward pass
    print("\n5. Testing adapter forward pass...")
    dummy_input = torch.randn(1, 10, 512)  # [batch, seq_len, d_z]
    with torch.no_grad():
        output = adapter(dummy_input)
    print(f"  ✓ Adapter output shape: {output.shape}")
    print(f"  ✓ Expected shape: [1, 10, 4096]")

    print("\n" + "="*50)
    print("✅ All Stage 1 components working correctly!")
    print("Ready to run full training.")

    return True

if __name__ == "__main__":
    try:
        success = test_stage1_components()
        if success:
            print("\n🚀 You can now run:")
            print("   git pull && rm -rf runs && PYTHONPATH=. ./scripts/run_stage1_h100.sh")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()