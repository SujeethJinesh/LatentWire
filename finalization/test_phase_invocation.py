#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify all 4 experiment phases can be invoked correctly.
This ensures the consolidated system is working properly.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_command(cmd, capture=True, check=False):
    """Run a shell command and return output."""
    print("\n[CMD] {}".format(cmd))
    try:
        if capture:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=check
            )
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True, check=check)
            return result.returncode, "", ""
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout if hasattr(e, 'stdout') else "", e.stderr if hasattr(e, 'stderr') else str(e)

def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)

    modules_to_test = [
        ("Core modules:", [
            "torch",
            "transformers",
            "numpy",
            "pandas",
            "scipy",
        ]),
        ("LatentWire modules:", [
            "latentwire.train",
            "latentwire.eval",
            "latentwire.data",
            "latentwire.models",
            "latentwire.losses",
        ]),
        ("Baseline modules:", [
            "latentwire.linear_probe_baseline",
            "latentwire.llmlingua_baseline",
        ]),
        ("Evaluation modules:", [
            "latentwire.eval_sst2",
            "latentwire.eval_agnews",
        ]),
    ]

    all_passed = True
    for category, modules in modules_to_test:
        print("\n{}".format(category))
        for module in modules:
            try:
                __import__(module)
                print("  ✓ {}".format(module))
            except ImportError as e:
                print("  ✗ {}: {}".format(module, e))
                all_passed = False

    return all_passed

def test_phase_invocation():
    """Test that each phase can be invoked via RUN_ALL.sh."""
    print("\n" + "="*60)
    print("TESTING PHASE INVOCATION")
    print("="*60)

    phases = [
        (1, "Statistical Rigor", "--phase 1"),
        (2, "Linear Probe Baseline", "--phase 2"),
        (3, "Fair Baseline Comparisons", "--phase 3"),
        (4, "Efficiency Measurements", "--phase 4"),
    ]

    results = {}
    for phase_num, phase_name, phase_flag in phases:
        print("\n[Phase {}] {}".format(phase_num, phase_name))

        # Test with dry run to check if command structure is valid
        cmd = "bash RUN_ALL.sh experiment {} --dry-run --skip-train --local 2>&1".format(phase_flag)
        returncode, stdout, stderr = run_command(cmd)

        # Check if phase was recognized
        success = False
        if returncode == 0:
            # Look for phase-specific output
            phase_indicators = [
                "PHASE {}".format(phase_num),
                "Phase {}".format(phase_num),
                phase_name.upper(),
            ]
            for indicator in phase_indicators:
                if indicator in stdout or indicator in stderr:
                    success = True
                    break

        results["phase_{}".format(phase_num)] = {
            "name": phase_name,
            "success": success,
            "returncode": returncode,
            "command": cmd,
        }

        if success:
            print("  ✓ Phase {} can be invoked".format(phase_num))
        else:
            print("  ✗ Phase {} invocation failed (return code: {})".format(phase_num, returncode))
            if stderr:
                print("    Error: {}".format(stderr[:200]))

    return results

def test_script_availability():
    """Test that all required scripts are available."""
    print("\n" + "="*60)
    print("TESTING SCRIPT AVAILABILITY")
    print("="*60)

    scripts_to_check = [
        ("Main runner", "RUN_ALL.sh"),
        ("Main experiment", "MAIN_EXPERIMENT.py"),
        ("Statistical testing", "../scripts/statistical_testing.py"),
        ("LLMLingua baseline", "../scripts/run_llmlingua_baseline.sh"),
        ("Linear probe", "../latentwire/linear_probe_baseline.py"),
        ("Training", "../latentwire/train.py"),
        ("Evaluation", "../latentwire/eval.py"),
        ("SST2 eval", "../latentwire/eval_sst2.py"),
        ("AG News eval", "../latentwire/eval_agnews.py"),
    ]

    all_found = True
    for desc, script_path in scripts_to_check:
        full_path = Path(__file__).parent / script_path
        exists = full_path.exists()

        if exists:
            # Check if executable (for .sh files)
            if script_path.endswith('.sh'):
                is_exec = os.access(full_path, os.X_OK)
                if is_exec:
                    print("  ✓ {}: {} (executable)".format(desc, script_path))
                else:
                    print("  ⚠ {}: {} (not executable)".format(desc, script_path))
                    # Try to make it executable
                    os.chmod(full_path, 0o755)
                    print("    → Made executable")
            else:
                print("  ✓ {}: {}".format(desc, script_path))
        else:
            print("  ✗ {}: {} (NOT FOUND)".format(desc, script_path))
            all_found = False

    return all_found

def test_help_command():
    """Test that help command works."""
    print("\n" + "="*60)
    print("TESTING HELP COMMAND")
    print("="*60)

    cmd = "bash RUN_ALL.sh help 2>&1"
    returncode, stdout, stderr = run_command(cmd)

    # Check for expected help content
    expected_keywords = ["Commands:", "Options:", "experiment", "train", "eval", "phase"]
    found_keywords = sum(1 for kw in expected_keywords if kw in stdout or kw in stderr)

    success = returncode == 0 and found_keywords >= 3

    if success:
        print("  ✓ Help command works correctly")
        print("    Found {}/{} expected keywords".format(found_keywords, len(expected_keywords)))
    else:
        print("  ✗ Help command failed (return code: {})".format(returncode))

    return success

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LATENTWIRE PHASE INVOCATION TEST")
    print("Time: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("="*60)

    # Change to script directory
    os.chdir(Path(__file__).parent)

    # Run tests
    test_results = {
        "imports": test_imports(),
        "scripts": test_script_availability(),
        "help": test_help_command(),
        "phases": test_phase_invocation(),
    }

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True

    print("\n[Import Tests]")
    if test_results["imports"]:
        print("  ✓ All required modules can be imported")
    else:
        print("  ✗ Some modules failed to import")
        all_passed = False

    print("\n[Script Availability]")
    if test_results["scripts"]:
        print("  ✓ All required scripts are available")
    else:
        print("  ✗ Some scripts are missing")
        all_passed = False

    print("\n[Help Command]")
    if test_results["help"]:
        print("  ✓ Help command works")
    else:
        print("  ✗ Help command failed")
        all_passed = False

    print("\n[Phase Invocation]")
    if isinstance(test_results["phases"], dict):
        for phase_key, phase_data in test_results["phases"].items():
            if phase_data["success"]:
                print("  ✓ {} can be invoked".format(phase_data['name']))
            else:
                print("  ✗ {} cannot be invoked".format(phase_data['name']))
                all_passed = False

    # Final verdict
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - System is ready for experiments!")
    else:
        print("❌ SOME TESTS FAILED - Please fix issues before running experiments")
    print("="*60)

    # Save results
    results_file = Path("test_results.json")
    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    print("\nResults saved to: {}".format(results_file))

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())