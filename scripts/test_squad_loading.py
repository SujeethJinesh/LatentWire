#!/usr/bin/env python3
"""
Test script to diagnose SQuAD dataset loading issues with Python 3.11 and datasets library
"""
import sys
import os
import platform
import dataclasses

print("=" * 60)
print("ENVIRONMENT DIAGNOSTICS")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Python executable: {sys.executable}")
print()

# Check datasets version
try:
    import datasets
    print(f"datasets version: {datasets.__version__}")
    print(f"datasets location: {datasets.__file__}")
except ImportError as e:
    print(f"ERROR: Could not import datasets: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("TESTING SQUAD LOADING")
print("=" * 60)

# Test 1: Basic load
print("\nTest 1: Basic load_dataset('squad')")
try:
    from datasets import load_dataset
    ds = load_dataset('squad', split='train[:10]')
    print(f"✓ SUCCESS: Loaded {len(ds)} examples")
    print(f"  First example keys: {list(ds[0].keys())}")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Load with validation split
print("\nTest 2: Load validation split")
try:
    ds_val = load_dataset('squad', split='validation[:10]')
    print(f"✓ SUCCESS: Loaded {len(ds_val)} validation examples")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")

# Test 3: Alternative loading method (rajpurkar/squad)
print("\nTest 3: Alternative load with 'rajpurkar/squad'")
try:
    ds_alt = load_dataset('rajpurkar/squad', split='train[:10]')
    print(f"✓ SUCCESS: Loaded {len(ds_alt)} examples via rajpurkar/squad")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")

# Test 4: Check dataclasses compatibility
print("\nTest 4: Check dataclasses.fields compatibility")
try:
    from datasets.info import DatasetInfo
    # Try to access fields like the error shows
    field_names = set(f.name for f in dataclasses.fields(DatasetInfo))
    print(f"✓ SUCCESS: DatasetInfo has {len(field_names)} fields")
    print(f"  Fields: {sorted(field_names)[:5]}...")  # Show first 5
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Load from cache vs fresh download
print("\nTest 5: Check cache behavior")
try:
    import tempfile
    import shutil

    # Save original cache dir
    original_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

    # Try with temp cache to force fresh download
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['HF_HOME'] = tmpdir
        print(f"  Using temp cache: {tmpdir}")

        # This might trigger the error if it's related to fresh downloads
        ds_fresh = load_dataset('squad', split='train[:10]')
        print(f"✓ SUCCESS: Loaded {len(ds_fresh)} examples with fresh cache")

        # Restore original cache
        if original_cache:
            os.environ['HF_HOME'] = original_cache
        else:
            del os.environ['HF_HOME']
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")
    # Restore cache dir on error
    if original_cache:
        os.environ['HF_HOME'] = original_cache
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

# Check if we need to upgrade
if hasattr(datasets, '__version__'):
    major, minor, patch = datasets.__version__.split('.')[:3]
    major, minor = int(major), int(minor)

    if major == 4 and minor < 4:
        print("⚠️  Consider upgrading datasets to 4.4.2 or later:")
        print("    pip install --upgrade datasets")
        print()

    if sys.version_info >= (3, 11):
        print("ℹ️  Using Python 3.11+ which has stricter dataclass rules.")
        print("    If you encounter mutable default errors, upgrading datasets may help.")

print("\nTesting complete!")