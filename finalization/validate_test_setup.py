#!/usr/bin/env python
"""
Validate that the test infrastructure is set up correctly.
This can be run locally without GPUs to verify imports and basic functionality.
"""

import sys
import importlib
from pathlib import Path

def check_module(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - FAILED: {e}")
        return False

def check_file(file_path: str) -> bool:
    """Check if a file exists."""
    path = Path(file_path)
    if path.exists():
        print(f"✅ {file_path} - EXISTS")
        return True
    else:
        print(f"❌ {file_path} - NOT FOUND")
        return False

def main():
    print("="*60)
    print("VALIDATING TEST SETUP")
    print("="*60)
    print()

    all_ok = True

    # Check required Python modules
    print("Checking Python modules:")
    print("-"*40)
    modules = [
        "latentwire.train",
        "latentwire.eval",
        "latentwire.models",
        "latentwire.data",
        "latentwire.metrics",
        "latentwire.error_handling",
        "latentwire.core_utils",
    ]

    for module in modules:
        if not check_module(module):
            all_ok = False

    print()

    # Check required files
    print("Checking required files:")
    print("-"*40)
    files = [
        "test_end_to_end.py",
        "run_end_to_end_test.sh",
        "latentwire/__init__.py",
        "latentwire/train.py",
        "latentwire/eval.py",
    ]

    for file in files:
        if not check_file(file):
            all_ok = False

    print()

    # Check optional but useful files
    print("Checking optional files:")
    print("-"*40)
    optional_files = [
        "CLAUDE.md",
        "LOG.md",
        "scripts/run_pipeline.sh",
    ]

    for file in optional_files:
        check_file(file)  # Don't affect all_ok

    print()
    print("="*60)

    if all_ok:
        print("✅ VALIDATION PASSED")
        print("You can now run: bash run_end_to_end_test.sh")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print("Please fix the issues above before running tests")
        return 1

if __name__ == "__main__":
    sys.exit(main())