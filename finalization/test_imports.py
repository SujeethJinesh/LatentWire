#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive import test for LatentWire project.

This script tests that all Python modules can be imported correctly.
It helps identify missing dependencies, circular imports, and other import issues.

Usage:
    python test_imports.py                    # Test all imports (requires dependencies)
    python test_imports.py --check-structure  # Only check file structure
"""

import sys
import os
import traceback
from pathlib import Path
from typing import List, Tuple, Optional

# Parse arguments
check_structure_only = "--check-structure" in sys.argv

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
os.environ['PYTHONPATH'] = str(project_root)

# Track results
successful_imports = []
failed_imports = []
warnings = []


def test_import(module_path: str, description: str = "") -> bool:
    """Test importing a module.

    Args:
        module_path: Python import path (e.g., 'latentwire.train')
        description: Optional description of the module

    Returns:
        True if import successful, False otherwise
    """
    try:
        print(f"Testing import: {module_path}...", end=" ")
        exec(f"import {module_path}")
        print("✓")
        successful_imports.append((module_path, description))
        return True
    except ImportError as e:
        print(f"✗ ImportError: {e}")
        failed_imports.append((module_path, str(e), description))
        return False
    except Exception as e:
        print(f"✗ {type(e).__name__}: {e}")
        failed_imports.append((module_path, str(e), description))
        return False


def test_from_import(module_path: str, items: List[str], description: str = "") -> bool:
    """Test importing specific items from a module.

    Args:
        module_path: Python module path
        items: List of items to import
        description: Optional description

    Returns:
        True if all imports successful, False otherwise
    """
    all_success = True
    for item in items:
        try:
            print(f"Testing: from {module_path} import {item}...", end=" ")
            exec(f"from {module_path} import {item}")
            print("✓")
            successful_imports.append((f"{module_path}.{item}", description))
        except ImportError as e:
            print(f"✗ ImportError: {e}")
            failed_imports.append((f"{module_path}.{item}", str(e), description))
            all_success = False
        except Exception as e:
            print(f"✗ {type(e).__name__}: {e}")
            failed_imports.append((f"{module_path}.{item}", str(e), description))
            all_success = False
    return all_success


def check_pytorch() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("MPS (Apple Silicon) available")
        else:
            print("CPU-only PyTorch")
        return True
    except ImportError:
        print("WARNING: PyTorch not installed!")
        warnings.append("PyTorch not installed - many modules will fail")
        return False


def check_transformers() -> bool:
    """Check if Transformers library is available."""
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        return True
    except ImportError:
        print("WARNING: Transformers not installed!")
        warnings.append("Transformers not installed - model loading will fail")
        return False


def check_file_structure():
    """Check that all expected files exist."""
    print("Checking file structure:")
    print("-" * 40)

    expected_files = [
        # Core modules
        "latentwire/__init__.py",
        "latentwire/models.py",
        "latentwire/losses.py",
        "latentwire/metrics.py",
        "latentwire/common.py",
        "latentwire/prefix_utils.py",
        "latentwire/core_utils.py",
        "latentwire/data.py",
        "latentwire/train.py",
        "latentwire/eval.py",
        "latentwire/error_handling.py",

        # Feature modules
        "latentwire/features/__init__.py",
        "latentwire/features/coproc.py",
        "latentwire/features/latent_adapters.py",
        "latentwire/features/deep_prefix.py",

        # CLI modules
        "latentwire/cli/__init__.py",
        "latentwire/cli/train.py",
        "latentwire/cli/eval.py",
        "latentwire/cli/utils.py",

        # Telepathy modules
        "telepathy/__init__.py",
        "telepathy/latent_bridge.py",

        # Scripts
        "scripts/run_pipeline.sh",
        "scripts/run_llmlingua_baseline.sh",
        "scripts/statistical_testing.py",
    ]

    missing_files = []
    for filepath in expected_files:
        full_path = project_root / filepath
        if full_path.exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} - MISSING")
            missing_files.append(filepath)

    return missing_files


def main():
    """Run all import tests."""
    print("=" * 70)
    print("LatentWire Import Test Suite")
    print("=" * 70)
    print()

    if check_structure_only:
        # Just check file structure
        missing_files = check_file_structure()
        print()
        print("=" * 70)
        print("File Structure Check Results")
        print("=" * 70)

        if missing_files:
            print(f"Missing {len(missing_files)} files:")
            for f in missing_files:
                print(f"  - {f}")
            print("\nPlease ensure all required files exist.")
            sys.exit(1)
        else:
            print("✅ All expected files exist!")
            sys.exit(0)
        return

    # Check critical dependencies
    print("Checking critical dependencies:")
    print("-" * 40)
    pytorch_available = check_pytorch()
    transformers_available = check_transformers()
    print()

    # Core latentwire modules
    print("Testing core latentwire modules:")
    print("-" * 40)

    core_modules = [
        ("latentwire", "Main package"),
        ("latentwire.models", "Model definitions"),
        ("latentwire.losses", "Loss functions"),
        ("latentwire.metrics", "Evaluation metrics"),
        ("latentwire.common", "Common utilities"),
        ("latentwire.prefix_utils", "Prefix handling"),
        ("latentwire.core_utils", "Core utilities"),
        ("latentwire.data", "Data loading"),
        ("latentwire.data_pipeline", "Data pipeline"),
        ("latentwire.checkpointing", "Checkpoint management"),
        ("latentwire.config", "Configuration"),
        ("latentwire.error_handling", "Error handling"),
        ("latentwire.feature_registry", "Feature registry"),
        ("latentwire.loss_bundles", "Loss bundles"),
        ("latentwire.optimized_dataloader", "Optimized dataloader"),
        ("latentwire.rouge_xsum_metrics", "ROUGE metrics"),
        ("latentwire.linear_probe_baseline", "Linear probe baseline"),
        ("latentwire.llmlingua_baseline", "LLMLingua baseline"),
    ]

    for module, desc in core_modules:
        test_import(module, desc)

    print()

    # Main training and evaluation scripts
    print("Testing main scripts:")
    print("-" * 40)

    main_scripts = [
        ("latentwire.train", "Training script"),
        ("latentwire.eval", "Evaluation script"),
        ("latentwire.eval_sst2", "SST-2 evaluation"),
        ("latentwire.eval_agnews", "AG News evaluation"),
        ("latentwire.gsm8k_eval", "GSM8K evaluation"),
        ("latentwire.embed_experiments", "Embedding experiments"),
    ]

    for module, desc in main_scripts:
        test_import(module, desc)

    print()

    # Feature modules
    print("Testing feature modules:")
    print("-" * 40)

    feature_modules = [
        ("latentwire.features", "Features package"),
        ("latentwire.features.coproc", "Co-processing features"),
        ("latentwire.features.latent_adapters", "Latent adapters"),
        ("latentwire.features.deep_prefix", "Deep prefix"),
    ]

    for module, desc in feature_modules:
        test_import(module, desc)

    print()

    # CLI modules
    print("Testing CLI modules:")
    print("-" * 40)

    cli_modules = [
        ("latentwire.cli", "CLI package"),
        ("latentwire.cli.utils", "CLI utilities"),
        ("latentwire.cli.train", "CLI training"),
        ("latentwire.cli.eval", "CLI evaluation"),
        ("latentwire.cli.run_ablation", "Ablation runner"),
    ]

    for module, desc in cli_modules:
        test_import(module, desc)

    print()

    # Telepathy modules
    print("Testing telepathy modules:")
    print("-" * 40)

    telepathy_modules = [
        ("telepathy", "Telepathy package"),
        ("telepathy.latent_bridge", "Latent bridge"),
        ("telepathy.latent_bridge_v7", "Latent bridge v7"),
        ("telepathy.latent_bridge_v8", "Latent bridge v8"),
        ("telepathy.latent_bridge_v9", "Latent bridge v9"),
        ("telepathy.latent_bridge_v12", "Latent bridge v12"),
        ("telepathy.latent_bridge_v13", "Latent bridge v13"),
        ("telepathy.training_signals", "Training signals"),
        ("telepathy.latency_utils", "Latency utilities"),
        ("telepathy.gpu_monitor", "GPU monitoring"),
        ("telepathy.dynamic_batch_size", "Dynamic batching"),
        ("telepathy.phase1_calibration", "Phase 1 calibration"),
    ]

    for module, desc in telepathy_modules:
        test_import(module, desc)

    print()

    # Test specific imports from key modules
    print("Testing specific imports from key modules:")
    print("-" * 40)

    # Test models imports
    test_from_import("latentwire.models", [
        "ByteLatentEncoder",
        "BPELatentEncoder",
        "CharLatentEncoder",
        "CausalLMWrapper",
        "AdapterLayer",
    ], "Model classes")

    # Test loss imports
    test_from_import("latentwire.losses", [
        "compute_kl_divergence",
        "k_token_ce_from_prefix",
        "compute_first_token_ce",
    ], "Loss functions")

    # Test error handling imports
    test_from_import("latentwire.error_handling", [
        "ErrorTracker",
        "retry_on_oom",
        "RobustCheckpointer",
        "MemoryMonitor",
    ], "Error handling utilities")

    print()

    # Test scripts in root directory
    print("Testing root-level scripts:")
    print("-" * 40)

    root_scripts = [
        ("eval", "Root evaluation script"),
        ("eval_sst2", "Root SST-2 evaluation"),
        ("eval_agnews", "Root AG News evaluation"),
        ("eval_reasoning_benchmarks", "Reasoning benchmarks"),
        ("statistical_testing", "Statistical testing"),
        ("analyze_all_results", "Results analysis"),
        ("benchmark_dataloader", "Dataloader benchmark"),
        ("benchmark_latency", "Latency benchmark"),
        ("benchmark_batched_latency", "Batched latency benchmark"),
        ("benchmark_memory", "Memory benchmark"),
    ]

    for module, desc in root_scripts:
        test_import(module, desc)

    print()
    print("=" * 70)
    print("Import Test Results")
    print("=" * 70)
    print()

    # Summary
    total_tests = len(successful_imports) + len(failed_imports)
    print(f"Total tests: {total_tests}")
    print(f"Successful: {len(successful_imports)} ✓")
    print(f"Failed: {len(failed_imports)} ✗")

    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print(f"  ⚠ {warning}")

    if failed_imports:
        print()
        print("Failed imports:")
        print("-" * 40)
        for module, error, desc in failed_imports:
            print(f"  ✗ {module}")
            if desc:
                print(f"    Description: {desc}")
            print(f"    Error: {error}")
            print()

    # Provide recommendations
    if failed_imports:
        print("Recommendations to fix import errors:")
        print("-" * 40)

        # Check for common issues
        missing_torch = any("torch" in error for _, error, _ in failed_imports)
        missing_transformers = any("transformers" in error for _, error, _ in failed_imports)

        if missing_torch:
            print("  • Install PyTorch: pip install torch")
        if missing_transformers:
            print("  • Install Transformers: pip install transformers")

        # Check for missing local modules
        missing_local = [
            module for module, error, _ in failed_imports
            if "No module named" in error and ("latentwire" in error or "telepathy" in error)
        ]

        if missing_local:
            print("  • Ensure PYTHONPATH includes project root:")
            print(f"    export PYTHONPATH={project_root}")
            print("  • Check that all __init__.py files exist")

        print()

    # Return status
    if failed_imports:
        print("❌ Import test FAILED - fix errors above before running training/evaluation")
        sys.exit(1)
    else:
        print("✅ All imports successful! Project is ready to run.")
        sys.exit(0)


if __name__ == "__main__":
    main()