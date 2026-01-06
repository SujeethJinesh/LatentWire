#!/usr/bin/env python3
"""
Validate that the integration test environment is properly configured.

This script checks:
1. Required Python packages are available
2. Required files and directories exist
3. GPU/compute resources are available
4. Basic imports work

Usage:
    python validate_integration_test.py
"""

import os
import sys
from pathlib import Path
import importlib.util


def check_python_version():
    """Check Python version is suitable."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ✗ Python 3.8+ required")
        return False

    print("  ✓ Python version OK")
    return True


def check_required_packages():
    """Check that required packages are installed."""
    required_packages = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
        "numpy": "NumPy",
        "tqdm": "tqdm",
        "datasets": "Hugging Face Datasets"
    }

    all_ok = True
    print("\nRequired packages:")

    for package, name in required_packages.items():
        spec = importlib.util.find_spec(package)
        if spec is None:
            print(f"  ✗ {name} ({package}) - NOT INSTALLED")
            all_ok = False
        else:
            print(f"  ✓ {name} ({package})")

            # Special check for PyTorch GPU support
            if package == "torch":
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_count = torch.cuda.device_count()
                        print(f"    - CUDA available: {gpu_count} GPU(s)")
                        for i in range(gpu_count):
                            print(f"      GPU {i}: {torch.cuda.get_device_name(i)}")
                    elif torch.backends.mps.is_available():
                        print(f"    - MPS (Apple Silicon) available")
                    else:
                        print(f"    - CPU only mode")
                except Exception as e:
                    print(f"    - Could not check compute backend: {e}")

    return all_ok


def check_project_structure():
    """Check that expected project files exist."""
    base_dir = Path(__file__).parent
    required_files = [
        "latentwire/train.py",
        "latentwire/eval.py",
        "latentwire/models.py",
        "latentwire/losses.py",
        "latentwire/data.py",
        "latentwire/core_utils.py",
        "test_integration.py",
        "run_integration_test.sh"
    ]

    print("\nProject structure:")
    all_ok = True

    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - NOT FOUND")
            all_ok = False

    return all_ok


def check_environment_variables():
    """Check environment variables."""
    print("\nEnvironment variables:")

    recommended_vars = {
        "PYTHONPATH": "Should include project root",
        "PYTORCH_ENABLE_MPS_FALLBACK": "Recommended for Mac: 1",
        "TOKENIZERS_PARALLELISM": "Recommended: false"
    }

    for var, description in recommended_vars.items():
        value = os.environ.get(var, "NOT SET")
        if value == "NOT SET":
            print(f"  ⚠ {var}: {value} ({description})")
        else:
            print(f"  ✓ {var}: {value}")

    return True


def test_basic_imports():
    """Test that basic imports work."""
    print("\nTesting imports:")

    test_imports = [
        "from latentwire.models import InterlinguaInterlinguaEncoder",
        "from latentwire.losses import k_token_ce_from_prefix",
        "from latentwire.data import load_examples_loader",
        "from latentwire.core_utils import calibrate_to_embed_rms"
    ]

    all_ok = True

    for import_stmt in test_imports:
        try:
            exec(import_stmt)
            print(f"  ✓ {import_stmt}")
        except Exception as e:
            print(f"  ✗ {import_stmt}")
            print(f"    Error: {e}")
            all_ok = False

    return all_ok


def check_disk_space():
    """Check available disk space."""
    print("\nDisk space:")

    try:
        import shutil
        usage = shutil.disk_usage(".")
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        used_percent = (usage.used / usage.total) * 100

        print(f"  Free: {free_gb:.2f} GB")
        print(f"  Total: {total_gb:.2f} GB")
        print(f"  Used: {used_percent:.1f}%")

        if free_gb < 10:
            print("  ⚠ Warning: Less than 10GB free space")
            return False
        else:
            print("  ✓ Sufficient disk space")

    except Exception as e:
        print(f"  Could not check disk space: {e}")

    return True


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("LatentWire Integration Test Environment Validation")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Project Structure", check_project_structure),
        ("Environment Variables", check_environment_variables),
        ("Basic Imports", test_basic_imports),
        ("Disk Space", check_disk_space)
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ {name} check failed with error: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    for name, result in results.items():
        status = "✓" if result else "✗"
        print(f"{status} {name}")

    print(f"\nChecks passed: {passed}/{len(results)}")

    if failed == 0:
        print("\n✓ Environment is ready for integration testing!")
        print("\nRun the integration test with:")
        print("  bash run_integration_test.sh")
        sys.exit(0)
    else:
        print(f"\n✗ {failed} checks failed. Please fix issues before running integration test.")
        sys.exit(1)


if __name__ == "__main__":
    main()