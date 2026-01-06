#!/usr/bin/env python3
"""
Validation script for run_comprehensive_revision.py

Checks that all dependencies exist and are properly configured before running
the comprehensive experiment orchestrator.

Usage:
    python telepathy/validate_comprehensive_revision.py
"""

import os
import sys
import importlib
from pathlib import Path

def check_file(path, description):
    """Check if a file exists."""
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} MISSING: {path}")
        return False

def check_import(module_name, description):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {description} FAILED: {e}")
        return False

def main():
    print("=" * 70)
    print("VALIDATING COMPREHENSIVE REVISION DEPENDENCIES")
    print("=" * 70)

    all_ok = True

    # Check main orchestrator
    print("\n[Main Script]")
    all_ok &= check_file("telepathy/run_comprehensive_revision.py", "Orchestrator")

    # Check statistical testing
    print("\n[Statistical Testing]")
    all_ok &= check_file("scripts/statistical_testing.py", "Statistical utilities")

    # Check Phase 1 dependencies
    print("\n[Phase 1: Statistical Rigor]")
    all_ok &= check_file("telepathy/run_unified_comparison.py", "Unified comparison")

    # Check Phase 2 dependencies
    print("\n[Phase 2: Linear Probe]")
    all_ok &= check_file("telepathy/linear_probe_baseline.py", "Linear probe baseline")

    # Check Phase 3 dependencies
    print("\n[Phase 3: Fair Baselines]")
    all_ok &= check_file("telepathy/train_prompt_tuning_baseline.py", "Prompt tuning")

    # Check Phase 4 dependencies
    print("\n[Phase 4: Latency]")
    all_ok &= check_file("telepathy/benchmark_batched_latency.py", "Batched latency")

    # Check Phase 5 dependencies
    print("\n[Phase 5: Generation/XSUM]")
    all_ok &= check_file("telepathy/train_xsum_bridge.py", "Train XSUM bridge")
    all_ok &= check_file("telepathy/eval_xsum_bridge.py", "Eval XSUM bridge")

    # Check Python imports
    print("\n[Python Dependencies]")
    all_ok &= check_import("torch", "PyTorch")
    all_ok &= check_import("transformers", "Transformers")
    all_ok &= check_import("datasets", "Datasets")
    all_ok &= check_import("numpy", "NumPy")
    all_ok &= check_import("scipy", "SciPy (for stats)")
    all_ok &= check_import("statsmodels", "StatsModels (for McNemar)")
    all_ok &= check_import("rouge_score", "ROUGE scorer")

    # Check for CUDA
    print("\n[Hardware]")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} devices")
            print(f"  Device: {torch.cuda.get_device_name()}")
        else:
            print("⚠ CUDA not available (CPU mode)")
    except:
        print("✗ Cannot check CUDA availability")

    # Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("VALIDATION PASSED - All dependencies found")
        print("\nYou can now run:")
        print("  python telepathy/run_comprehensive_revision.py --run_all")
        print("\nOr for specific phases:")
        print("  python telepathy/run_comprehensive_revision.py --phase 1")
    else:
        print("VALIDATION FAILED - Missing dependencies")
        print("\nPlease install missing dependencies before running experiments.")
        print("For HPC execution, ensure all modules are loaded:")
        print("  module load python pytorch transformers")
        return 1

    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())