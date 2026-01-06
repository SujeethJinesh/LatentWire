#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct test of phase functionality without going through RUN_ALL.sh
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_phase1_imports():
    """Test Phase 1: Statistical testing imports"""
    print("\n[Phase 1] Statistical Rigor")
    try:
        # Check if statistical_testing.py exists and can be imported
        script_path = Path(__file__).parent.parent / "scripts" / "statistical_testing.py"
        if script_path.exists():
            print("  ✓ statistical_testing.py exists")
            # Try to import the required libraries for statistical testing
            try:
                import numpy as np
                print("  ✓ numpy available")
            except ImportError:
                print("  ✗ numpy not available")

            try:
                from scipy import stats
                print("  ✓ scipy.stats available")
            except ImportError:
                print("  ✗ scipy.stats not available")
            return True
        else:
            print("  ✗ statistical_testing.py not found")
            return False
    except Exception as e:
        print("  ✗ Error: {}".format(e))
        return False

def test_phase2_imports():
    """Test Phase 2: Linear probe baseline"""
    print("\n[Phase 2] Linear Probe Baseline")
    try:
        # Check if we have LinearProbeBaseline in latentwire
        try:
            from latentwire.linear_probe_baseline import LinearProbeBaseline
            print("  ✓ LinearProbeBaseline class available")
            return True
        except ImportError as e:
            # Check alternative location
            script_path = Path(__file__).parent.parent / "latentwire" / "linear_probe_baseline.py"
            if script_path.exists():
                print("  ⚠ linear_probe_baseline.py exists but can't import: {}".format(e))
            else:
                print("  ✗ linear_probe_baseline.py not found")
            return False
    except Exception as e:
        print("  ✗ Error: {}".format(e))
        return False

def test_phase3_imports():
    """Test Phase 3: Fair baseline comparisons"""
    print("\n[Phase 3] Fair Baseline Comparisons")
    try:
        # Check LLMLingua baseline script
        script_path = Path(__file__).parent.parent / "scripts" / "run_llmlingua_baseline.sh"
        if script_path.exists():
            print("  ✓ run_llmlingua_baseline.sh exists")

            # Check if llmlingua_baseline.py exists
            py_path = Path(__file__).parent.parent / "latentwire" / "llmlingua_baseline.py"
            if py_path.exists():
                print("  ✓ llmlingua_baseline.py exists")
                return True
            else:
                print("  ✗ llmlingua_baseline.py not found")
                return False
        else:
            print("  ✗ run_llmlingua_baseline.sh not found")
            return False
    except Exception as e:
        print("  ✗ Error: {}".format(e))
        return False

def test_phase4_imports():
    """Test Phase 4: Efficiency measurements"""
    print("\n[Phase 4] Efficiency Measurements")
    try:
        # Check if benchmark script exists
        script_path = Path(__file__).parent.parent / "scripts" / "benchmark_efficiency.py"
        if script_path.exists():
            print("  ✓ benchmark_efficiency.py exists")
            return True
        else:
            # Alternative: check if efficiency measurement is part of eval.py
            eval_path = Path(__file__).parent.parent / "latentwire" / "eval.py"
            if eval_path.exists():
                print("  ⚠ benchmark_efficiency.py not found, but eval.py exists (may contain efficiency metrics)")
                return True
            else:
                print("  ✗ No efficiency measurement scripts found")
                return False
    except Exception as e:
        print("  ✗ Error: {}".format(e))
        return False

def test_core_modules():
    """Test core LatentWire modules"""
    print("\n[Core Modules]")

    modules_to_check = [
        ("latentwire.data", "Data loading"),
        ("latentwire.train", "Training"),
        ("latentwire.eval", "Evaluation"),
        ("latentwire.models", "Model definitions"),
        ("latentwire.losses", "Loss functions"),
    ]

    all_available = True
    for module_name, description in modules_to_check:
        try:
            __import__(module_name)
            print("  ✓ {}: {}".format(description, module_name))
        except ImportError as e:
            print("  ✗ {}: {} - {}".format(description, module_name, str(e)[:50]))
            all_available = False

    return all_available

def main():
    print("="*60)
    print("DIRECT PHASE FUNCTIONALITY TEST")
    print("="*60)

    results = {
        "core": test_core_modules(),
        "phase1": test_phase1_imports(),
        "phase2": test_phase2_imports(),
        "phase3": test_phase3_imports(),
        "phase4": test_phase4_imports(),
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = all(results.values())

    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print("  {} {}".format(status, name.replace("phase", "Phase ").title() if "phase" in name else name.title()))

    print("\n" + "="*60)
    if all_passed:
        print("✅ All phase components are available")
    else:
        print("❌ Some phase components are missing")
        print("\nNote: Missing PyTorch is expected on this Mac.")
        print("The actual experiments will run on HPC with proper GPU support.")
    print("="*60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())