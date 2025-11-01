#!/usr/bin/env python3
"""Test loading SQuAD using the actual latentwire.data function."""

import sys
sys.path.insert(0, '/Users/sujeethjinesh/Desktop/LatentWire')

from latentwire.data import load_squad_subset
import traceback

print("Testing load_squad_subset function...")
print("=" * 60)

try:
    # Call it exactly as used in unified_cross_model_experiments.py line 3788
    squad_samples = load_squad_subset(split="validation", samples=20, seed=42)

    print(f"OK - Loaded {len(squad_samples)} samples")
    print(f"\nFirst sample:")
    print(f"  Keys: {squad_samples[0].keys()}")
    print(f"  Source: {squad_samples[0]['source'][:100]}...")
    print(f"  Answer: {squad_samples[0]['answer']}")

except Exception as e:
    print(f"FAIL - Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")
