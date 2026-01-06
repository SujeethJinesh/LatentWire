#!/usr/bin/env python3
"""
Basic structure test that works without PyTorch installed.
Tests code organization, syntax, and shell scripts.
"""

import ast
import os
import sys
from pathlib import Path
import subprocess

def test_python_syntax():
    """Test that all Python files have valid syntax."""
    print("=" * 60)
    print("Testing Python syntax...")
    print("=" * 60)

    python_files = []
    for root, dirs, files in os.walk("latentwire"):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    # Also check scripts directory
    for root, dirs, files in os.walk("scripts"):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    failed = []
    for filepath in python_files:
        try:
            with open(filepath, 'r') as f:
                code = f.read()
            ast.parse(code)
            print(f"✓ {filepath}")
        except SyntaxError as e:
            print(f"✗ {filepath}: {e}")
            failed.append((filepath, str(e)))
        except Exception as e:
            print(f"⚠️  {filepath}: {e}")

    print(f"\nChecked {len(python_files)} Python files")
    if failed:
        print(f"✗ {len(failed)} files have syntax errors")
    else:
        print(f"✓ All files have valid Python syntax")

    return len(failed) == 0

def test_imports_without_torch():
    """Test imports that don't require PyTorch."""
    print("\n" + "=" * 60)
    print("Testing non-torch imports...")
    print("=" * 60)

    modules = [
        "latentwire",
        "latentwire.data",
        "latentwire.common",
        "latentwire.checkpointing",
        "latentwire.config",
        "latentwire.cli",
    ]

    passed = []
    failed = []
    for module_name in modules:
        try:
            exec(f"import {module_name}")
            print(f"✓ {module_name}")
            passed.append(module_name)
        except ImportError as e:
            if "torch" in str(e).lower():
                print(f"⚠️  {module_name}: Requires PyTorch (expected)")
            else:
                print(f"✗ {module_name}: {e}")
                failed.append(module_name)
        except Exception as e:
            print(f"✗ {module_name}: {e}")
            failed.append(module_name)

    print(f"\n{len(passed)} modules imported successfully")
    return len(failed) == 0

def test_shell_scripts():
    """Test shell script syntax and structure."""
    print("\n" + "=" * 60)
    print("Testing shell scripts...")
    print("=" * 60)

    shell_scripts = []
    # Root level scripts
    for file in Path(".").glob("*.sh"):
        shell_scripts.append(file)
    # Scripts directory
    for file in Path("scripts").glob("*.sh"):
        shell_scripts.append(file)

    issues = []
    for script_path in shell_scripts:
        with open(script_path, 'r') as f:
            content = f.read()

        # Check shebang
        if not content.startswith("#!/"):
            issues.append(f"{script_path}: Missing shebang")

        # Check for basic structure
        checks = {
            "PYTHONPATH set": "PYTHONPATH=" in content or "export PYTHONPATH" in content,
            "Has python/bash calls": "python" in content.lower() or "bash" in content,
            "Executable": os.access(script_path, os.X_OK),
        }

        passed_checks = sum(checks.values())
        if passed_checks == len(checks):
            print(f"✓ {script_path.name}: All checks passed")
        else:
            failed = [k for k, v in checks.items() if not v]
            print(f"⚠️  {script_path.name}: Failed {failed}")

    print(f"\nChecked {len(shell_scripts)} shell scripts")
    return len(issues) == 0

def test_directory_structure():
    """Test that expected directories exist."""
    print("\n" + "=" * 60)
    print("Testing directory structure...")
    print("=" * 60)

    expected_dirs = [
        "latentwire",
        "latentwire/features",
        "latentwire/cli",
        "scripts",
        "telepathy",
        "experimental",
        "experimental/learning",
    ]

    all_exist = True
    for dir_path in expected_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ not found")
            all_exist = False

    return all_exist

def test_key_files():
    """Test that key files exist."""
    print("\n" + "=" * 60)
    print("Testing key files...")
    print("=" * 60)

    key_files = [
        ("latentwire/train.py", "Training script"),
        ("latentwire/eval.py", "Evaluation script"),
        ("latentwire/models.py", "Model definitions"),
        ("latentwire/data.py", "Data loading"),
        ("latentwire/losses.py", "Loss functions"),
        ("latentwire/metrics.py", "Metrics"),
        ("latentwire/config.py", "Configuration"),
        ("RUN_ALL.sh", "Main run script"),
        ("QUICK_START.sh", "Quick start script"),
        ("LOG.md", "Development log"),
        ("CLAUDE.md", "Claude instructions"),
    ]

    all_exist = True
    for filepath, description in key_files:
        if Path(filepath).exists():
            size = Path(filepath).stat().st_size
            print(f"✓ {filepath}: {description} ({size:,} bytes)")
        else:
            print(f"✗ {filepath}: {description} - NOT FOUND")
            all_exist = False

    return all_exist

def test_run_all_script():
    """Test that RUN_ALL.sh can be parsed."""
    print("\n" + "=" * 60)
    print("Testing RUN_ALL.sh structure...")
    print("=" * 60)

    if not Path("RUN_ALL.sh").exists():
        print("✗ RUN_ALL.sh not found")
        return False

    with open("RUN_ALL.sh", 'r') as f:
        content = f.read()

    # Check for key sections
    checks = {
        "Has shebang": content.startswith("#!/"),
        "Sets PYTHONPATH": "PYTHONPATH=" in content,
        "Has training commands": "train.py" in content,
        "Has eval commands": "eval.py" in content,
        "Has logging": "tee" in content or ">" in content,
        "Is executable": os.access("RUN_ALL.sh", os.X_OK),
    }

    all_passed = True
    for check_name, passed in checks.items():
        if passed:
            print(f"✓ {check_name}")
        else:
            print(f"✗ {check_name}")
            all_passed = False

    # Count number of experiments
    experiment_count = content.count("python") + content.count("python3")
    print(f"\nFound approximately {experiment_count} Python command invocations")

    return all_passed

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("BASIC STRUCTURE TEST (No PyTorch Required)")
    print("=" * 60)

    results = {
        "Python Syntax": test_python_syntax(),
        "Basic Imports": test_imports_without_torch(),
        "Shell Scripts": test_shell_scripts(),
        "Directories": test_directory_structure(),
        "Key Files": test_key_files(),
        "RUN_ALL.sh": test_run_all_script(),
    }

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
        print("\n✓ All basic structure tests passed!")
        print("\nThe codebase structure is intact. To test functionality:")
        print("1. Install PyTorch: pip install torch transformers datasets")
        print("2. Run on HPC: bash RUN_ALL.sh")
        return 0
    else:
        print("\n⚠️  Some structure tests failed.")
        print("Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())