#!/usr/bin/env python3
"""
GPU Detection and Batch Size Adjustment Test Script

This script verifies that:
1. GPU detection works correctly with different CUDA_VISIBLE_DEVICES settings
2. Batch sizes adjust based on available GPU count
3. The system handles multi-GPU and CPU-only scenarios correctly

Usage:
    # Test with no GPUs visible
    CUDA_VISIBLE_DEVICES="" python test_gpu_detection.py

    # Test with single GPU
    CUDA_VISIBLE_DEVICES="0" python test_gpu_detection.py

    # Test with multiple GPUs
    CUDA_VISIBLE_DEVICES="0,1,2,3" python test_gpu_detection.py

    # Test with specific GPU subset
    CUDA_VISIBLE_DEVICES="2,3" python test_gpu_detection.py
"""

import os
import sys
import json
from typing import Dict, List, Optional, Tuple
import warnings

# Try importing PyTorch
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. Some tests will be skipped.")


# =============================================================================
# GPU Detection Utilities
# =============================================================================

def get_gpu_info() -> Dict:
    """
    Get comprehensive GPU information.

    Returns:
        Dictionary containing:
        - cuda_available: Whether CUDA is available
        - device_count: Number of visible GPUs
        - visible_devices: CUDA_VISIBLE_DEVICES value
        - device_names: List of GPU names
        - device_memory_gb: List of GPU memory in GB
        - cuda_version: CUDA runtime version
        - pytorch_version: PyTorch version
    """
    info = {
        "cuda_available": False,
        "device_count": 0,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        "device_names": [],
        "device_memory_gb": [],
        "cuda_version": None,
        "pytorch_version": None,
    }

    if not PYTORCH_AVAILABLE:
        info["error"] = "PyTorch not installed"
        return info

    info["pytorch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()

    if info["cuda_available"]:
        info["device_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda

        for i in range(info["device_count"]):
            # Get device name
            device_name = torch.cuda.get_device_name(i)
            info["device_names"].append(device_name)

            # Get device memory
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            info["device_memory_gb"].append(round(memory_gb, 2))

    return info


def calculate_batch_size(
    base_batch_size: int = 8,
    n_gpus: int = 0,
    strategy: str = "linear",
    min_batch_size: int = 1,
    max_batch_size: int = 256,
) -> int:
    """
    Calculate recommended batch size based on GPU count.

    Args:
        base_batch_size: Base batch size for single GPU
        n_gpus: Number of available GPUs
        strategy: Scaling strategy ("linear", "sqrt", "fixed")
        min_batch_size: Minimum allowed batch size
        max_batch_size: Maximum allowed batch size

    Returns:
        Recommended batch size
    """
    if n_gpus == 0:
        # CPU-only: use smaller batch size
        batch_size = max(min_batch_size, base_batch_size // 4)
    elif strategy == "linear":
        # Linear scaling with GPU count
        batch_size = base_batch_size * n_gpus
    elif strategy == "sqrt":
        # Square root scaling (more conservative)
        import math
        batch_size = int(base_batch_size * math.sqrt(n_gpus))
    elif strategy == "fixed":
        # Fixed batch size regardless of GPU count
        batch_size = base_batch_size
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Apply bounds
    batch_size = max(min_batch_size, min(max_batch_size, batch_size))

    return batch_size


def test_memory_capacity(device: str = "cuda:0", dtype: torch.dtype = None) -> Dict:
    """
    Test GPU memory capacity with different tensor sizes.

    Args:
        device: Device to test
        dtype: Data type to use (default: torch.float16)

    Returns:
        Dictionary with memory test results
    """
    if not PYTORCH_AVAILABLE or not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    if dtype is None:
        dtype = torch.float16

    results = {
        "device": device,
        "dtype": str(dtype),
        "tests": []
    }

    # Test different tensor sizes
    test_sizes = [
        (1024, 1024),        # ~2MB in fp16
        (4096, 4096),        # ~32MB in fp16
        (8192, 8192),        # ~128MB in fp16
        (16384, 16384),      # ~512MB in fp16
        (32768, 32768),      # ~2GB in fp16
    ]

    for size in test_sizes:
        torch.cuda.empty_cache()

        try:
            # Try allocating tensor
            tensor = torch.zeros(size, dtype=dtype, device=device)
            memory_mb = (tensor.numel() * tensor.element_size()) / (1024**2)

            results["tests"].append({
                "size": size,
                "memory_mb": round(memory_mb, 2),
                "success": True
            })

            del tensor

        except torch.cuda.OutOfMemoryError as e:
            results["tests"].append({
                "size": size,
                "success": False,
                "error": "OOM"
            })
            break
        except Exception as e:
            results["tests"].append({
                "size": size,
                "success": False,
                "error": str(e)
            })
            break

    torch.cuda.empty_cache()
    return results


def simulate_data_parallel(n_gpus: int, batch_size: int) -> Dict:
    """
    Simulate data parallel training configuration.

    Args:
        n_gpus: Number of GPUs
        batch_size: Total batch size

    Returns:
        Configuration dictionary
    """
    if n_gpus == 0:
        return {
            "mode": "CPU",
            "total_batch_size": batch_size,
            "per_device_batch_size": batch_size,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": batch_size,
        }

    # Calculate per-device batch size
    per_device_batch_size = batch_size // n_gpus

    # Handle case where batch_size < n_gpus
    if per_device_batch_size == 0:
        per_device_batch_size = 1
        gradient_accumulation_steps = batch_size
    else:
        gradient_accumulation_steps = 1

    return {
        "mode": "DataParallel" if n_gpus > 1 else "Single GPU",
        "n_gpus": n_gpus,
        "total_batch_size": batch_size,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": per_device_batch_size * n_gpus * gradient_accumulation_steps,
    }


# =============================================================================
# Test Suite
# =============================================================================

def run_comprehensive_tests() -> Dict:
    """
    Run comprehensive GPU detection and configuration tests.

    Returns:
        Dictionary with all test results
    """
    results = {
        "timestamp": None,
        "environment": {},
        "gpu_info": {},
        "batch_size_tests": [],
        "memory_tests": [],
        "data_parallel_configs": [],
        "recommendations": {},
    }

    # Get timestamp
    from datetime import datetime
    results["timestamp"] = datetime.now().isoformat()

    # Environment info
    results["environment"] = {
        "python_version": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        "pytorch_available": PYTORCH_AVAILABLE,
    }

    # GPU information
    print("=" * 80)
    print("GPU DETECTION TEST")
    print("=" * 80)

    gpu_info = get_gpu_info()
    results["gpu_info"] = gpu_info

    print(f"CUDA Available: {gpu_info['cuda_available']}")
    print(f"Device Count: {gpu_info['device_count']}")
    print(f"CUDA_VISIBLE_DEVICES: {gpu_info['visible_devices']}")

    if gpu_info["cuda_available"]:
        print(f"CUDA Version: {gpu_info['cuda_version']}")
        print(f"PyTorch Version: {gpu_info['pytorch_version']}")
        print("\nVisible GPUs:")
        for i, (name, mem) in enumerate(zip(gpu_info["device_names"], gpu_info["device_memory_gb"])):
            print(f"  GPU {i}: {name} ({mem} GB)")
    else:
        print("No GPUs available - will use CPU")

    print()

    # Batch size calculations
    print("=" * 80)
    print("BATCH SIZE CALCULATION TESTS")
    print("=" * 80)

    n_gpus = gpu_info["device_count"]
    base_batch_sizes = [4, 8, 16, 32]
    strategies = ["linear", "sqrt", "fixed"]

    for base_bs in base_batch_sizes:
        for strategy in strategies:
            calculated_bs = calculate_batch_size(
                base_batch_size=base_bs,
                n_gpus=n_gpus,
                strategy=strategy,
            )

            test_result = {
                "base_batch_size": base_bs,
                "n_gpus": n_gpus,
                "strategy": strategy,
                "calculated_batch_size": calculated_bs,
                "scaling_factor": calculated_bs / base_bs if base_bs > 0 else 0,
            }
            results["batch_size_tests"].append(test_result)

            print(f"Base BS={base_bs:3d}, Strategy={strategy:6s} -> BS={calculated_bs:3d} (scale={test_result['scaling_factor']:.2f}x)")

    print()

    # Memory capacity tests (if GPU available)
    if gpu_info["cuda_available"] and PYTORCH_AVAILABLE:
        print("=" * 80)
        print("MEMORY CAPACITY TESTS")
        print("=" * 80)

        for i in range(min(2, n_gpus)):  # Test first 2 GPUs only
            device = f"cuda:{i}"
            print(f"\nTesting {device}...")

            memory_test = test_memory_capacity(device=device)
            results["memory_tests"].append(memory_test)

            for test in memory_test.get("tests", []):
                if test["success"]:
                    print(f"  ✓ Allocated {test['size']} tensor ({test['memory_mb']} MB)")
                else:
                    print(f"  ✗ Failed at {test['size']}: {test.get('error', 'Unknown error')}")
                    break

    # Data parallel configurations
    print("\n" + "=" * 80)
    print("DATA PARALLEL CONFIGURATIONS")
    print("=" * 80)

    test_batch_sizes = [8, 16, 32, 64, 128]

    for batch_size in test_batch_sizes:
        config = simulate_data_parallel(n_gpus, batch_size)
        results["data_parallel_configs"].append(config)

        print(f"\nBatch size {batch_size}:")
        print(f"  Mode: {config['mode']}")
        if n_gpus > 0:
            print(f"  Per-device batch size: {config['per_device_batch_size']}")
            print(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
            print(f"  Effective batch size: {config['effective_batch_size']}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if n_gpus == 0:
        recommended_config = {
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "mixed_precision": False,
            "notes": "CPU-only mode: Use small batch sizes with gradient accumulation"
        }
    elif n_gpus == 1:
        recommended_config = {
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "mixed_precision": True,
            "notes": "Single GPU: Moderate batch size with fp16 for efficiency"
        }
    else:
        recommended_config = {
            "batch_size": 8 * n_gpus,
            "gradient_accumulation_steps": 1,
            "mixed_precision": True,
            "data_parallel": True,
            "notes": f"Multi-GPU ({n_gpus}): Use DataParallel or DistributedDataParallel"
        }

    results["recommendations"] = recommended_config

    print(f"Recommended configuration for {n_gpus} GPU(s):")
    for key, value in recommended_config.items():
        print(f"  {key}: {value}")

    return results


def verify_script_compatibility(script_paths: List[str]) -> Dict:
    """
    Verify that scripts properly handle different GPU configurations.

    Args:
        script_paths: List of script paths to check

    Returns:
        Compatibility report
    """
    report = {
        "scripts": {},
        "issues": [],
        "recommendations": []
    }

    patterns_to_check = [
        ("cuda_device_count", r"torch\.cuda\.device_count\(\)"),
        ("cuda_available", r"torch\.cuda\.is_available\(\)"),
        ("batch_size_adjustment", r"batch_size.*device_count|batch_size.*n_gpu"),
        ("data_parallel", r"DataParallel|DistributedDataParallel"),
        ("device_selection", r"cuda:\d+|device.*cuda"),
    ]

    for script_path in script_paths:
        if not os.path.exists(script_path):
            report["scripts"][script_path] = {"error": "File not found"}
            continue

        with open(script_path, 'r') as f:
            content = f.read()

        script_report = {
            "checks": {},
            "has_gpu_handling": False,
            "has_batch_adjustment": False,
        }

        # Check for patterns
        import re
        for pattern_name, pattern in patterns_to_check:
            matches = re.findall(pattern, content)
            script_report["checks"][pattern_name] = len(matches) > 0

            if pattern_name == "cuda_device_count" and len(matches) > 0:
                script_report["has_gpu_handling"] = True
            if pattern_name == "batch_size_adjustment" and len(matches) > 0:
                script_report["has_batch_adjustment"] = True

        report["scripts"][script_path] = script_report

        # Check for issues
        if not script_report["has_gpu_handling"]:
            report["issues"].append(f"{script_path}: No GPU detection logic found")
        if not script_report["has_batch_adjustment"]:
            report["issues"].append(f"{script_path}: No batch size adjustment for GPU count")

    # Add recommendations
    if report["issues"]:
        report["recommendations"].append("Add GPU detection: torch.cuda.device_count()")
        report["recommendations"].append("Adjust batch sizes: batch_size = base_batch_size * n_gpus")
        report["recommendations"].append("Handle CPU fallback: device = 'cuda' if torch.cuda.is_available() else 'cpu'")

    return report


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main test execution."""

    # Run comprehensive tests
    results = run_comprehensive_tests()

    # Save results to JSON
    output_file = "gpu_detection_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("TEST RESULTS SAVED")
    print("=" * 80)
    print(f"Results saved to: {output_file}")

    # Check specific scripts if provided
    if len(sys.argv) > 1:
        print("\n" + "=" * 80)
        print("SCRIPT COMPATIBILITY CHECK")
        print("=" * 80)

        scripts = sys.argv[1:]
        compatibility_report = verify_script_compatibility(scripts)

        for script, report in compatibility_report["scripts"].items():
            print(f"\n{script}:")
            if "error" in report:
                print(f"  Error: {report['error']}")
            else:
                print(f"  GPU handling: {'✓' if report['has_gpu_handling'] else '✗'}")
                print(f"  Batch adjustment: {'✓' if report['has_batch_adjustment'] else '✗'}")

        if compatibility_report["issues"]:
            print("\nIssues found:")
            for issue in compatibility_report["issues"]:
                print(f"  - {issue}")

        if compatibility_report["recommendations"]:
            print("\nRecommendations:")
            for rec in compatibility_report["recommendations"]:
                print(f"  - {rec}")

    # Test with different CUDA_VISIBLE_DEVICES values
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT CUDA_VISIBLE_DEVICES CONFIGURATIONS")
    print("=" * 80)
    print("\nTo test different configurations, run:")
    print('  CUDA_VISIBLE_DEVICES="" python test_gpu_detection.py      # No GPUs')
    print('  CUDA_VISIBLE_DEVICES="0" python test_gpu_detection.py     # Single GPU')
    print('  CUDA_VISIBLE_DEVICES="0,1" python test_gpu_detection.py   # Two GPUs')
    print('  CUDA_VISIBLE_DEVICES="0,1,2,3" python test_gpu_detection.py # Four GPUs')

    return results


if __name__ == "__main__":
    main()