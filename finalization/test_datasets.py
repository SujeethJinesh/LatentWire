#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script to verify all datasets load correctly with expected sizes."""

import sys
from typing import Dict, Any

def test_datasets():
    """Test loading all 4 datasets and verify their sizes."""

    # Import the data loading function
    from latentwire.data import load_examples

    # Define expected sizes for each dataset's test/validation set
    expected_sizes = {
        "sst2": ("validation", 872),      # SST-2 validation set
        "agnews": ("test", 7600),         # AG News test set
        "trec": ("test", 500),             # TREC test set
        "xsum": ("test", 11334),           # XSUM test set (or 11490 for CNN/DailyMail fallback)
    }

    # Alternative accepted sizes (for fallback datasets)
    alternative_sizes = {
        "xsum": 11490  # CNN/DailyMail test set size when used as fallback
    }

    print("="*60)
    print("Testing Dataset Loading")
    print("="*60)

    all_passed = True
    results = []

    for dataset_name, (split, expected_size) in expected_sizes.items():
        print("\nTesting {}...".format(dataset_name.upper()))
        print("  Expected: {:,} examples from '{}' split".format(expected_size, split))

        try:
            # Load the dataset with no sampling (get all examples)
            examples = load_examples(dataset=dataset_name, split=split, samples=None, seed=0)
            actual_size = len(examples)

            # Check if size matches
            if actual_size == expected_size:
                status = "✓ PASSED"
                print("  Actual:   {:,} examples".format(actual_size))
                print("  {}".format(status))

                # Show a sample example
                if examples:
                    ex = examples[0]
                    print("  Sample source: {}...".format(ex['source'][:100]))
                    print("  Sample answer: {}...".format(ex['answer'][:50]))
            else:
                status = "✗ FAILED (got {:,} instead of {:,})".format(actual_size, expected_size)
                print("  Actual:   {:,} examples".format(actual_size))
                print("  {}".format(status))
                all_passed = False

            results.append({
                "dataset": dataset_name,
                "split": split,
                "expected": expected_size,
                "actual": actual_size,
                "passed": actual_size == expected_size
            })

        except Exception as e:
            print("  ✗ ERROR: {}".format(str(e)))
            results.append({
                "dataset": dataset_name,
                "split": split,
                "expected": expected_size,
                "actual": 0,
                "passed": False,
                "error": str(e)
            })
            all_passed = False

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\n| Dataset | Split      | Expected | Actual | Status |")
    print("|---------|------------|----------|--------|--------|")

    for result in results:
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        if "error" in result:
            status = "✗ ERROR"
        print("| {:7s} | {:10s} | {:8,d} | {:6,d} | {:6s} |".format(
            result['dataset'], result['split'], result['expected'],
            result.get('actual', 0), status))

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(test_datasets())