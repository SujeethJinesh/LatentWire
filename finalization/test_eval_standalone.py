#!/usr/bin/env python3
"""
Test script to verify that evaluation modules work standalone in finalization directory.
"""

import sys
import os
import importlib.util
from pathlib import Path

def test_module_import(module_path: str, module_name: str):
    """Test if a module can be imported successfully."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✓ Successfully imported {module_name} from {module_path}")
            return True
    except Exception as e:
        print(f"✗ Failed to import {module_name}: {e}")
        return False

def main():
    """Test all evaluation modules."""
    finalization_dir = Path(__file__).parent

    # Add finalization directory to Python path
    sys.path.insert(0, str(finalization_dir))

    print("Testing evaluation modules in finalization directory...")
    print(f"Directory: {finalization_dir}")
    print("-" * 60)

    # Test core latentwire modules
    latentwire_modules = [
        ("latentwire/eval.py", "eval"),
        ("latentwire/eval_sst2.py", "eval_sst2"),
        ("latentwire/eval_agnews.py", "eval_agnews"),
        ("latentwire/gsm8k_eval.py", "gsm8k_eval"),
        ("latentwire/models.py", "models"),
        ("latentwire/core_utils.py", "core_utils"),
        ("latentwire/data.py", "data"),
    ]

    print("\n1. Testing core latentwire modules:")
    all_passed = True
    for rel_path, module_name in latentwire_modules:
        full_path = finalization_dir / rel_path
        if full_path.exists():
            if not test_module_import(str(full_path), f"latentwire.{module_name}"):
                all_passed = False
        else:
            print(f"✗ Module not found: {rel_path}")
            all_passed = False

    # Test telepathy evaluation modules
    telepathy_modules = [
        ("telepathy/eval_telepathy.py", "eval_telepathy"),
        ("telepathy/eval_telepathy_sst2.py", "eval_telepathy_sst2"),
        ("telepathy/eval_telepathy_agnews.py", "eval_telepathy_agnews"),
        ("telepathy/eval_telepathy_gsm8k.py", "eval_telepathy_gsm8k"),
        ("telepathy/eval_telepathy_trec.py", "eval_telepathy_trec"),
    ]

    print("\n2. Testing telepathy evaluation modules:")
    for rel_path, module_name in telepathy_modules:
        full_path = finalization_dir / rel_path
        if full_path.exists():
            if not test_module_import(str(full_path), f"telepathy.{module_name}"):
                all_passed = False
        else:
            print(f"✗ Module not found: {rel_path}")
            all_passed = False

    # Test if we can actually use the main eval module
    print("\n3. Testing main eval module functionality:")
    try:
        from latentwire.eval import _parse_device_map, EVAL_FIXED_SEED
        print(f"✓ Successfully imported functions from latentwire.eval")
        print(f"  - EVAL_FIXED_SEED = {EVAL_FIXED_SEED}")
        result = _parse_device_map("0")
        print(f"  - _parse_device_map('0') = {result}")
    except Exception as e:
        print(f"✗ Failed to use latentwire.eval: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Evaluation modules are ready for standalone use.")
    else:
        print("✗ Some tests failed. Please check the error messages above.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())