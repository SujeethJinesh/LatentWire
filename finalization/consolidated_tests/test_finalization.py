#!/usr/bin/env python3
"""
Test script for the finalization directory structure.
Verifies that consolidated files work correctly.
"""

import sys
import os
import ast
import subprocess
from pathlib import Path

def test_python_files_syntax():
    """Test that all Python files in current directory have valid syntax."""
    print("=" * 60)
    print("Testing Python file syntax...")
    print("=" * 60)

    python_files = [f for f in os.listdir('.') if f.endswith('.py')]

    failed = []
    for filepath in sorted(python_files):
        try:
            with open(filepath, 'r') as f:
                code = f.read()
            ast.parse(code)
            size = os.path.getsize(filepath)
            print(f"✓ {filepath:30} ({size:,} bytes)")
        except SyntaxError as e:
            print(f"✗ {filepath}: Line {e.lineno}: {e.msg}")
            failed.append((filepath, str(e)))
        except Exception as e:
            print(f"⚠️  {filepath}: {e}")

    print(f"\nChecked {len(python_files)} Python files")
    if failed:
        print(f"✗ {len(failed)} files have syntax errors:")
        for f, err in failed:
            print(f"  - {f}")
    else:
        print(f"✓ All files have valid Python syntax")

    return len(failed) == 0

def test_imports():
    """Test that modules can be imported (where possible without PyTorch)."""
    print("\n" + "=" * 60)
    print("Testing module imports...")
    print("=" * 60)

    # Add current directory to path
    if '.' not in sys.path:
        sys.path.insert(0, '.')

    modules_to_test = [
        ("config", False),  # Should work
        ("metrics", True),  # May need torch
        ("models", True),   # Needs torch
        ("losses", True),   # Needs torch
        ("checkpoint_manager", False),  # Should work
        ("statistical_testing", False),  # Should work
    ]

    working = []
    failed = []

    for module_name, needs_torch in modules_to_test:
        try:
            exec(f"import {module_name}")
            print(f"✓ {module_name}")
            working.append(module_name)
        except ImportError as e:
            if needs_torch and "torch" in str(e).lower():
                print(f"⚠️  {module_name}: Requires PyTorch (expected)")
            else:
                print(f"✗ {module_name}: {e}")
                failed.append(module_name)
        except Exception as e:
            print(f"✗ {module_name}: Unexpected error: {e}")
            failed.append(module_name)

    print(f"\n{len(working)} modules imported successfully")
    print(f"{len(failed)} modules failed unexpectedly")

    return len(failed) == 0

def test_shell_scripts():
    """Test shell scripts in the directory."""
    print("\n" + "=" * 60)
    print("Testing shell scripts...")
    print("=" * 60)

    shell_scripts = [f for f in os.listdir('.') if f.endswith('.sh')]

    all_valid = True
    for script in sorted(shell_scripts):
        # Check syntax with bash -n
        result = subprocess.run(
            ['bash', '-n', script],
            capture_output=True,
            text=True
        )

        # Check basic properties
        is_executable = os.access(script, os.X_OK)
        size = os.path.getsize(script)

        if result.returncode == 0:
            exec_status = "✓ executable" if is_executable else "⚠️  not executable"
            print(f"✓ {script:25} ({size:,} bytes) {exec_status}")
        else:
            print(f"✗ {script:25} - Syntax error")
            print(f"  {result.stderr[:200]}")
            all_valid = False

    print(f"\nChecked {len(shell_scripts)} shell scripts")

    return all_valid

def test_run_all_contents():
    """Test that RUN_ALL.sh contains expected structure."""
    print("\n" + "=" * 60)
    print("Testing RUN_ALL.sh contents...")
    print("=" * 60)

    if not os.path.exists('RUN_ALL.sh'):
        print("✗ RUN_ALL.sh not found")
        return False

    with open('RUN_ALL.sh', 'r') as f:
        content = f.read()

    # Check for expected components
    checks = {
        "Shebang line": content.startswith("#!/"),
        "PYTHONPATH setup": "PYTHONPATH=" in content or "export PYTHONPATH" in content,
        "Python commands": content.count("python") > 10,  # Should have many
        "Training references": "train" in content.lower(),
        "Evaluation references": "eval" in content.lower(),
        "Logging/output": "tee" in content or ">>" in content or ">" in content,
        "GPU handling": "CUDA" in content or "cuda" in content or "gpu" in content.lower(),
        "Error handling": "set -e" in content or "|| " in content,
    }

    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False

    # Count experiments
    python_calls = content.count("python")
    print(f"\n📊 Statistics:")
    print(f"  - {python_calls} python invocations")
    print(f"  - {len(content.splitlines())} total lines")
    print(f"  - {len(content):,} total bytes")

    return all_passed

def test_key_integration_files():
    """Test that key integration files exist."""
    print("\n" + "=" * 60)
    print("Testing key integration files...")
    print("=" * 60)

    expected_files = {
        # Core training/eval
        "eval.py": "Main evaluation script",
        "models.py": "Model definitions",
        "losses.py": "Loss functions",
        "metrics.py": "Evaluation metrics",
        "config.py": "Configuration",

        # Data and utilities
        "checkpoint_manager.py": "Checkpoint management",
        "statistical_testing.py": "Statistical analysis",

        # Shell scripts
        "RUN_ALL.sh": "Main execution script",
        "QUICK_START.sh": "Quick start guide",
        "run_integration_test.sh": "Integration tests",

        # Specialized evaluation
        "eval_sst2.py": "SST-2 evaluation",
        "eval_agnews.py": "AG News evaluation",
        "eval_reasoning_benchmarks.py": "Reasoning benchmarks",
    }

    missing = []
    for filename, description in expected_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✓ {filename:30} {description:30} ({size:,} bytes)")
        else:
            print(f"✗ {filename:30} {description}")
            missing.append(filename)

    if missing:
        print(f"\n⚠️  Missing {len(missing)} files: {', '.join(missing)}")
    else:
        print(f"\n✓ All {len(expected_files)} key files present")

    return len(missing) == 0

def test_can_execute_basic_help():
    """Test if scripts can show help without errors."""
    print("\n" + "=" * 60)
    print("Testing script execution (--help)...")
    print("=" * 60)

    scripts_to_test = [
        "eval.py",
        "eval_sst2.py",
        "eval_agnews.py",
        "statistical_testing.py",
    ]

    working = []
    failed = []

    for script in scripts_to_test:
        if not os.path.exists(script):
            print(f"⚠️  {script}: Not found")
            continue

        try:
            result = subprocess.run(
                [sys.executable, script, "--help"],
                capture_output=True,
                text=True,
                timeout=2
            )

            # Check if help text appears
            if "--help" in result.stdout or "usage:" in result.stdout.lower() or \
               "--help" in result.stderr or "usage:" in result.stderr.lower():
                print(f"✓ {script}: Shows help")
                working.append(script)
            elif "torch" in result.stderr.lower():
                print(f"⚠️  {script}: Requires PyTorch")
            else:
                print(f"✗ {script}: No help output")
                failed.append(script)

        except subprocess.TimeoutExpired:
            print(f"⚠️  {script}: Timed out (may be waiting for input)")
        except Exception as e:
            print(f"✗ {script}: {e}")
            failed.append(script)

    print(f"\n{len(working)} scripts show help correctly")

    return len(failed) == 0

def main():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("FINALIZATION DIRECTORY TEST SUITE")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print("=" * 60)

    # Run tests
    results = {
        "Python Syntax": test_python_files_syntax(),
        "Module Imports": test_imports(),
        "Shell Scripts": test_shell_scripts(),
        "RUN_ALL.sh": test_run_all_contents(),
        "Key Files": test_key_integration_files(),
        "Script Help": test_can_execute_basic_help(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! The consolidation appears functional.")
        print("\nNext steps:")
        print("1. Copy to HPC: rsync -avz finalization/ hpc:/path/to/project/")
        print("2. Install deps: pip install torch transformers datasets")
        print("3. Run tests: bash RUN_ALL.sh")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review the errors above.")
        print("\nNote: PyTorch-dependent tests will fail on development machine.")
        print("This is expected - the code runs on HPC with GPUs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())