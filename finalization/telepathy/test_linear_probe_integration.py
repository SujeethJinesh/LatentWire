#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify Linear Probe integration with run_unified_comparison.py
"""

import sys

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        from linear_probe_baseline import LinearProbeBaseline, train_linear_probe, eval_linear_probe
        print("✓ Linear probe modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import linear probe modules: {e}")
        return False

    try:
        from telepathy import run_unified_comparison
        print("✓ run_unified_comparison imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import run_unified_comparison: {e}")
        return False

    return True

def test_linear_probe_instantiation():
    """Test that LinearProbeBaseline can be instantiated."""
    print("\nTesting LinearProbeBaseline instantiation...")

    try:
        from linear_probe_baseline import LinearProbeBaseline

        probe = LinearProbeBaseline(
            hidden_dim=4096,
            num_classes=2,
            layer_idx=24,
            pooling="mean",
            normalize=True,
            C=1.0,
            max_iter=1000,
            n_jobs=-1,
            random_state=42,
        )

        print(f"✓ Created LinearProbeBaseline with config:")
        print(f"  - Hidden dim: {probe.hidden_dim}")
        print(f"  - Num classes: {probe.num_classes}")
        print(f"  - Layer idx: {probe.layer_idx}")
        print(f"  - Pooling: {probe.pooling}")

        return True
    except Exception as e:
        print(f"✗ Failed to instantiate LinearProbeBaseline: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("LINEAR PROBE INTEGRATION TEST")
    print("="*60)

    tests_passed = []

    # Test imports
    tests_passed.append(test_imports())

    # Test instantiation
    tests_passed.append(test_linear_probe_instantiation())

    # Summary
    print("\n" + "="*60)
    print(f"Tests passed: {sum(tests_passed)}/{len(tests_passed)}")

    if all(tests_passed):
        print("✓ All tests passed! Integration is ready.")
        print("\nTo run the full experiment with linear probe:")
        print("  python telepathy/run_unified_comparison.py --datasets sst2 --seeds 42")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()