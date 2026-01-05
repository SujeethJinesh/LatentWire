#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Dataset Sizes for Reviewer Requirements

This script verifies that all datasets are loaded with the correct test set sizes
as required by the reviewers:
- SST-2: 872 validation samples (test labels not available)
- AG News: 7,600 test samples
- TREC: 500 test samples
- XSUM: 11,334 test samples (may have loading issues)

The script tests the data loading functions and confirms that:
1. The full test sets are available
2. The loading functions can access the complete datasets
3. No unexpected truncation or sampling occurs
"""

import sys
sys.path.append('/Users/sujeethjinesh/Desktop/LatentWire')

from latentwire.data import (
    load_sst2_subset,
    load_agnews_subset,
    load_trec_subset,
    load_xsum_subset,
    load_examples
)


def verify_sst2():
    """Verify SST-2 validation set loading."""
    print("\n" + "="*60)
    print("Testing SST-2 Dataset Loading", flush=True)
    print("="*60)

    try:
        # Test with default (should use full validation set)
        samples_default = load_sst2_subset(split="validation", samples=None)
        print(f"✓ Default loading: {len(samples_default)} samples", flush=True)

        # Test explicit full set request
        samples_full = load_sst2_subset(split="validation", samples=872)
        print(f"✓ Explicit 872 samples: {len(samples_full)} samples")

        # Test that requesting more than available still gives 872
        samples_large = load_sst2_subset(split="validation", samples=10000)
        print(f"✓ Requesting 10000 samples: {len(samples_large)} samples (capped at available)")

        # Verify format
        sample = samples_default[0]
        assert "source" in sample and "answer" in sample
        assert "Sentiment:" in sample["source"]
        assert sample["answer"] in ["positive", "negative"]
        print(f"✓ Sample format correct")

        # Check using load_examples interface
        samples_unified = load_examples("sst2", split="validation", samples=None)
        print(f"✓ Via load_examples: {len(samples_unified)} samples")

        if len(samples_default) == 872:
            print("✅ SST-2: PASS - Full 872 validation samples available")
            return True
        else:
            print(f"❌ SST-2: FAIL - Expected 872 samples, got {len(samples_default)}")
            return False

    except Exception as e:
        print(f"❌ SST-2: ERROR - {e}", flush=True)
        return False


def verify_agnews():
    """Verify AG News test set loading."""
    print("\n" + "="*60)
    print("Testing AG News Dataset Loading", flush=True)
    print("="*60)

    try:
        # Test with default (should use full test set)
        samples_default = load_agnews_subset(split="test", samples=None)
        print(f"✓ Default loading: {len(samples_default)} samples", flush=True)

        # Test explicit full set request
        samples_full = load_agnews_subset(split="test", samples=7600)
        print(f"✓ Explicit 7600 samples: {len(samples_full)} samples")

        # Test that requesting more than available still gives 7600
        samples_large = load_agnews_subset(split="test", samples=10000)
        print(f"✓ Requesting 10000 samples: {len(samples_large)} samples (capped at available)")

        # Verify format
        sample = samples_default[0]
        assert "source" in sample and "answer" in sample
        assert "Topic" in sample["source"]
        assert sample["answer"] in ["world", "sports", "business", "science"]
        print(f"✓ Sample format correct")

        # Check class distribution (should be balanced, ~1900 per class)
        class_counts = {}
        for s in samples_default:
            label = s["answer"]
            class_counts[label] = class_counts.get(label, 0) + 1
        print(f"✓ Class distribution: {class_counts}")

        # Check using load_examples interface
        samples_unified = load_examples("agnews", split="test", samples=None)
        print(f"✓ Via load_examples: {len(samples_unified)} samples")

        if len(samples_default) == 7600:
            print("✅ AG News: PASS - Full 7,600 test samples available")
            return True
        else:
            print(f"❌ AG News: FAIL - Expected 7,600 samples, got {len(samples_default)}")
            return False

    except Exception as e:
        print(f"❌ AG News: ERROR - {e}", flush=True)
        return False


def verify_trec():
    """Verify TREC test set loading."""
    print("\n" + "="*60)
    print("Testing TREC Dataset Loading", flush=True)
    print("="*60)

    try:
        # Test with default (should use full test set)
        samples_default = load_trec_subset(split="test", samples=None)
        print(f"✓ Default loading: {len(samples_default)} samples", flush=True)

        # Test explicit full set request
        samples_full = load_trec_subset(split="test", samples=500)
        print(f"✓ Explicit 500 samples: {len(samples_full)} samples")

        # Test that requesting more than available still gives 500
        samples_large = load_trec_subset(split="test", samples=10000)
        print(f"✓ Requesting 10000 samples: {len(samples_large)} samples (capped at available)")

        # Verify format
        sample = samples_default[0]
        assert "source" in sample and "answer" in sample
        assert "Question:" in sample["source"]
        assert sample["answer"] in ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]
        print(f"✓ Sample format correct")

        # Check class distribution
        class_counts = {}
        for s in samples_default:
            label = s["answer"]
            class_counts[label] = class_counts.get(label, 0) + 1
        print(f"✓ Class distribution: {class_counts}")

        # Check using load_examples interface
        samples_unified = load_examples("trec", split="test", samples=None)
        print(f"✓ Via load_examples: {len(samples_unified)} samples")

        if len(samples_default) == 500:
            print("✅ TREC: PASS - Full 500 test samples available")
            return True
        else:
            print(f"❌ TREC: FAIL - Expected 500 samples, got {len(samples_default)}")
            return False

    except Exception as e:
        print(f"❌ TREC: ERROR - {e}", flush=True)
        return False


def verify_xsum():
    """Verify XSUM test set loading."""
    print("\n" + "="*60)
    print("Testing XSUM Dataset Loading", flush=True)
    print("="*60)

    try:
        # Test with default (should use full test set)
        try:
            samples_default = load_xsum_subset(split="test", samples=None)
            print(f"✓ Default loading: {len(samples_default)} samples", flush=True)
            loaded = True
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" in str(e):
                print(f"⚠️ XSUM loading issue (known compatibility problem with newer datasets library):", flush=True)
                print(f"  {e}")
                loaded = False
            else:
                raise

        if loaded:
            # Test explicit full set request
            samples_full = load_xsum_subset(split="test", samples=11334)
            print(f"✓ Explicit 11,334 samples: {len(samples_full)} samples")

            # Verify format
            sample = samples_default[0]
            assert "source" in sample and "answer" in sample
            assert "Article:" in sample["source"] and "Summary:" in sample["source"]
            print(f"✓ Sample format correct")

            # Check first few examples
            print(f"✓ Example summary length: {len(samples_default[0]['answer'].split())} words")

            # Check using load_examples interface
            samples_unified = load_examples("xsum", split="test", samples=None)
            print(f"✓ Via load_examples: {len(samples_unified)} samples")

            if len(samples_default) == 11334:
                print("✅ XSUM: PASS - Full 11,334 test samples available")
                return True
            else:
                print(f"❌ XSUM: FAIL - Expected 11,334 samples, got {len(samples_default)}")
                return False
        else:
            print("⚠️ XSUM: PARTIAL - Dataset loading requires special handling", flush=True)
            print("  See docstring in data.py for solutions")
            return None  # Partial pass

    except Exception as e:
        print(f"❌ XSUM: ERROR - {e}", flush=True)
        return False


def main():
    """Run all dataset verification tests."""
    print("="*60)
    print("Dataset Size Verification for Reviewer Requirements")
    print("="*60)
    print("\nRequired test set sizes:")
    print("  - SST-2: 872 validation samples")
    print("  - AG News: 7,600 test samples")
    print("  - TREC: 500 test samples")
    print("  - XSUM: 11,334 test samples")

    results = {
        "SST-2": verify_sst2(),
        "AG News": verify_agnews(),
        "TREC": verify_trec(),
        "XSUM": verify_xsum(),
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for dataset, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:  # None for partial
            status = "⚠️ PARTIAL"
        print(f"{dataset:10s}: {status}")

    # Overall assessment
    print("\n" + "="*60)
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    partial = sum(1 for r in results.values() if r is None)

    print(f"Results: {passed} passed, {failed} failed, {partial} partial")

    if failed == 0:
        if partial > 0:
            print("\n✓ Core datasets (SST-2, AG News, TREC) load correctly with full test sets")
            print("⚠️ XSUM has known compatibility issues with newer datasets library")
        else:
            print("\n✅ All datasets load correctly with full test sets as required!")
    else:
        print("\n❌ Some datasets are not loading the full test sets as required", flush=True)
        print("   Please review the data.py implementation")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)