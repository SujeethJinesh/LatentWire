#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
            print("\nTesting {}...".format(dataset_name))
            examples = load_examples(dataset_name, **kwargs)

            if len(examples) > 0:
                print("[PASS] {}: Loaded {} examples".format(dataset_name, len(examples)))
                print("  Sample source: {}...".format(examples[0]['source'][:100]))
                print("  Sample answer: {}...".format(examples[0]['answer'][:50] if examples[0]['answer'] else "None"))
            else:
                print("[FAIL] {}: No examples loaded".format(dataset_name))

        except Exception as e:
            print("[FAIL] {}: Error - {}".format(dataset_name, e))

    print("\n" + "=" * 60)
    print("Testing complete!")

if __name__ == "__main__":
    test_dataset_loading()