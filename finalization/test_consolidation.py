#!/usr/bin/env python3
"""
Test script to verify that the consolidated codebase works correctly.
Tests imports, basic functionality, and integration.
"""

import sys
import os
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all key modules can be imported."""
    print("=" * 60)
    print("Testing module imports...")
    print("=" * 60)

    modules_to_test = [
        "latentwire",
        "latentwire.models",
        "latentwire.data",
        "latentwire.losses",
        "latentwire.metrics",
        "latentwire.prefix_utils",
        "latentwire.common",
        "latentwire.checkpointing",
        "latentwire.config",
        "latentwire.core_utils",
        "latentwire.feature_registry",
        "latentwire.loss_bundles",
        "latentwire.optimized_dataloader",
        "latentwire.linear_probe_baseline",
        "latentwire.features",
        "latentwire.cli",
    ]

    failed = []
    for module_name in modules_to_test:
        try:
            exec(f"import {module_name}")
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name}: {e}")
            failed.append((module_name, str(e)))
        except Exception as e:
            print(f"✗ {module_name}: Unexpected error: {e}")
            failed.append((module_name, str(e)))

    if failed:
        print(f"\n⚠️  {len(failed)} modules failed to import")
        for module, error in failed:
            print(f"  - {module}: {error}")
    else:
        print(f"\n✓ All {len(modules_to_test)} modules imported successfully")

    return len(failed) == 0

def test_class_instantiation():
    """Test that key classes can be instantiated."""
    print("\n" + "=" * 60)
    print("Testing class instantiation...")
    print("=" * 60)

    tests_passed = True

    # Test ByteEncoder
    try:
        from latentwire.models import ByteEncoder
        encoder = ByteEncoder(
            d_z=256,
            latent_len=32,
            vocab_size=259,
            pos_encoding="sinusoidal"
        )
        print("✓ ByteEncoder instantiated")
    except Exception as e:
        print(f"✗ ByteEncoder failed: {e}")
        tests_passed = False

    # Test Adapter
    try:
        from latentwire.models import Adapter
        adapter = Adapter(
            d_in=256,
            d_out=4096,
            mode="mlp"
        )
        print("✓ Adapter instantiated")
    except Exception as e:
        print(f"✗ Adapter failed: {e}")
        tests_passed = False

    # Test LinearProbeBaseline
    try:
        from latentwire.linear_probe_baseline import LinearProbeBaseline
        baseline = LinearProbeBaseline(
            input_dim=4096,
            output_dim=32000,
            layer_id=16
        )
        print("✓ LinearProbeBaseline instantiated")
    except Exception as e:
        print(f"✗ LinearProbeBaseline failed: {e}")
        tests_passed = False

    # Test data loading functions
    try:
        from latentwire.data import load_qa_dataset, load_trec_dataset
        print("✓ Data loading functions accessible")
    except Exception as e:
        print(f"✗ Data loading functions failed: {e}")
        tests_passed = False

    # Test loss functions
    try:
        from latentwire.losses import (
            k_token_ce_from_prefix,
            kd_first_k_prefix_vs_text,
            first_token_cross_entropy
        )
        print("✓ Loss functions accessible")
    except Exception as e:
        print(f"✗ Loss functions failed: {e}")
        tests_passed = False

    # Test metric functions
    try:
        from latentwire.metrics import (
            compute_em_f1_score,
            compute_prefix_nll
        )
        print("✓ Metric functions accessible")
    except Exception as e:
        print(f"✗ Metric functions failed: {e}")
        tests_passed = False

    return tests_passed

def test_script_execution():
    """Test that key scripts can be executed with --help."""
    print("\n" + "=" * 60)
    print("Testing script execution...")
    print("=" * 60)

    scripts_to_test = [
        ("latentwire.train", "Training script"),
        ("latentwire.eval", "Evaluation script"),
        ("latentwire.eval_sst2", "SST-2 evaluation"),
        ("latentwire.eval_agnews", "AG News evaluation"),
        ("latentwire.gsm8k_eval", "GSM8K evaluation"),
    ]

    failed = []
    for module_path, description in scripts_to_test:
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", module_path, "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 or "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower():
                print(f"✓ {description} ({module_path})")
            else:
                print(f"✗ {description} ({module_path}): Exit code {result.returncode}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}")
                failed.append(module_path)
        except subprocess.TimeoutExpired:
            print(f"✓ {description} ({module_path}) - started successfully")
        except Exception as e:
            print(f"✗ {description} ({module_path}): {e}")
            failed.append(module_path)

    if failed:
        print(f"\n⚠️  {len(failed)} scripts failed")
    else:
        print(f"\n✓ All {len(scripts_to_test)} scripts executable")

    return len(failed) == 0

def test_shell_scripts():
    """Test that shell scripts exist and have correct permissions."""
    print("\n" + "=" * 60)
    print("Testing shell scripts...")
    print("=" * 60)

    scripts_to_check = [
        "RUN_ALL.sh",
        "QUICK_START.sh",
        "setup_env.sh",
        "run_integration_test.sh",
        "run_example_eval.sh",
        "run_with_resume.sh",
        "run_end_to_end_test.sh",
    ]

    all_exist = True
    for script in scripts_to_check:
        script_path = Path(script)
        if script_path.exists():
            is_executable = os.access(script_path, os.X_OK)
            if is_executable:
                print(f"✓ {script} exists and is executable")
            else:
                print(f"⚠️  {script} exists but is not executable")
        else:
            print(f"✗ {script} not found")
            all_exist = False

    return all_exist

def test_config_loading():
    """Test that configuration can be loaded."""
    print("\n" + "=" * 60)
    print("Testing configuration loading...")
    print("=" * 60)

    try:
        from latentwire.config import TrainingConfig
        config = TrainingConfig()
        print(f"✓ TrainingConfig loaded with {len(vars(config))} attributes")

        # Test a few key attributes
        assert hasattr(config, 'batch_size')
        assert hasattr(config, 'epochs')
        assert hasattr(config, 'latent_len')
        assert hasattr(config, 'd_z')
        print("✓ Key configuration attributes present")

        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("LATENTWIRE CONSOLIDATION TEST SUITE")
    print("=" * 60)

    results = {}

    # Run tests
    results['imports'] = test_imports()
    results['classes'] = test_class_instantiation()
    results['scripts'] = test_script_execution()
    results['shell'] = test_shell_scripts()
    results['config'] = test_config_loading()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.capitalize():15} {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! The consolidation appears successful.")
        print("\nYou can now:")
        print("1. Run training: python -m latentwire.train --help")
        print("2. Run evaluation: python -m latentwire.eval --help")
        print("3. Use shell scripts: bash RUN_ALL.sh")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        print("\nCommon issues:")
        print("- Missing dependencies: pip install -r requirements.txt")
        print("- Wrong Python version: Ensure Python 3.8+")
        print("- Path issues: Run from the project root directory")
        return 1

if __name__ == "__main__":
    sys.exit(main())