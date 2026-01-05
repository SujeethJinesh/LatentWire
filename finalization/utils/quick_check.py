#!/usr/bin/env python3
"""
Quick validation check for critical requirements.
Can be called from other scripts to ensure environment is ready.

Usage:
    python finalization/quick_check.py

    # Or from Python:
    from finalization.quick_check import validate_critical
    if not validate_critical():
        sys.exit(1)
"""

import sys
import os
from pathlib import Path


def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available() or torch.backends.mps.is_available()
    except:
        return False


def check_packages():
    """Check critical packages"""
    required = ['torch', 'transformers', 'datasets', 'numpy']
    try:
        for pkg in required:
            __import__(pkg)
        return True
    except ImportError as e:
        print(f"Missing package: {e}", file=sys.stderr)
        return False


def check_memory():
    """Check if we have enough memory"""
    try:
        import torch
        if torch.cuda.is_available():
            # Check GPU memory
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                if props.total_memory < 8 * (1024**3):  # Less than 8GB
                    print(f"Warning: GPU {i} has only {props.total_memory/(1024**3):.1f}GB", flush=True)
                          file=sys.stderr)
                    return False
        return True
    except:
        return True  # Don't fail if we can't check


def check_working_dir():
    """Check if we're in the right directory"""
    # Should have these key files/dirs
    expected = ['latentwire', 'telepathy', 'scripts', 'finalization']
    cwd = Path.cwd()

    for item in expected:
        if not (cwd / item).exists():
            print(f"Error: Not in LatentWire root (missing {item})", file=sys.stderr, flush=True)
            return False
    return True


def validate_critical(verbose=True):
    """Run critical validation checks"""
    checks = [
        ("Working Directory", check_working_dir),
        ("Required Packages", check_packages),
        ("GPU Available", check_gpu),
        ("GPU Memory", check_memory),
    ]

    all_passed = True
    for name, check_func in checks:
        try:
            passed = check_func()
            if verbose:
                status = "✓" if passed else "✗"
                print(f"{status} {name}")
            if not passed:
                all_passed = False
        except Exception as e:
            if verbose:
                print(f"✗ {name}: {e}")
            all_passed = False

    return all_passed


def main():
    """Run quick validation"""
    print("Quick Validation Check")
    print("-" * 30)

    if validate_critical(verbose=True):
        print("\n✓ Critical checks passed")
        sys.exit(0)
    else:
        print("\n✗ Critical checks failed")
        print("\nRun full validation for details:")
        print("  python finalization/validate.py")
        sys.exit(1)


if __name__ == "__main__":
    main()