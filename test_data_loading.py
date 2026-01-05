#!/usr/bin/env python
"""Test script to verify dataset loading works correctly."""

from latentwire.data import load_examples

def test_dataset_loading():
    """Test that all datasets load without errors."""

    datasets_to_test = [
        ("hotpot", {"samples": 5}),
        ("squad", {"samples": 5}),
        ("squad_v2", {"samples": 5}),
        ("gsm8k", {"samples": 5}),
        ("trec", {"samples": 5}),
        ("agnews", {"samples": 5}),
        ("sst2", {"samples": 5}),
        ("xsum", {"samples": 5}),
    ]

    print("Testing dataset loading...")
    print("=" * 60)

    for dataset_name, kwargs in datasets_to_test:
        try:
            print(f"\nTesting {dataset_name}...")
            examples = load_examples(dataset_name, **kwargs)

            if len(examples) > 0:
                print(f"✓ {dataset_name}: Loaded {len(examples)} examples")
                print(f"  Sample source: {examples[0]['source'][:100]}...")
                print(f"  Sample answer: {examples[0]['answer'][:50]}...")
            else:
                print(f"✗ {dataset_name}: No examples loaded")

        except Exception as e:
            print(f"✗ {dataset_name}: Error - {e}")

    print("\n" + "=" * 60)
    print("Testing complete!")

if __name__ == "__main__":
    test_dataset_loading()