#!/usr/bin/env python3
"""
Test script to verify multi-GPU setup works correctly.
Tests CUDA_VISIBLE_DEVICES simulation and batch size scaling.

Usage:
    # Test single GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/test_multi_gpu_setup.py

    # Test dual GPU
    CUDA_VISIBLE_DEVICES=0,1 python scripts/test_multi_gpu_setup.py

    # Test all GPUs
    python scripts/test_multi_gpu_setup.py

    # Test with DDP
    python scripts/test_multi_gpu_setup.py --test-ddp
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch not available: {e}")
    sys.exit(1)


class DummyModel(nn.Module):
    """Simple model for testing multi-GPU setup."""
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.layers(x)


def test_cuda_detection() -> Dict:
    """Test CUDA device detection and CUDA_VISIBLE_DEVICES handling."""
    results = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "devices": []
    }

    if results["cuda_available"] and results["device_count"] > 0:
        for i in range(results["device_count"]):
            try:
                props = torch.cuda.get_device_properties(i)
                results["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count
                })
            except Exception as e:
                results["devices"].append({
                    "index": i,
                    "error": str(e)
                })

    return results


def test_memory_info() -> Dict:
    """Test GPU memory reporting."""
    results = {"gpus": {}}

    if not torch.cuda.is_available():
        results["error"] = "CUDA not available"
        return results

    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_device(i)
            results["gpus"][i] = {
                "allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
                "max_allocated_gb": torch.cuda.max_memory_allocated(i) / (1024**3),
                "max_reserved_gb": torch.cuda.max_memory_reserved(i) / (1024**3),
            }
            # Get total memory
            props = torch.cuda.get_device_properties(i)
            results["gpus"][i]["total_gb"] = props.total_memory / (1024**3)
            results["gpus"][i]["available_gb"] = (
                results["gpus"][i]["total_gb"] - results["gpus"][i]["allocated_gb"]
            )
        except Exception as e:
            results["gpus"][i] = {"error": str(e)}

    return results


def test_batch_size_scaling(base_batch_size: int = 32) -> Dict:
    """Test batch size scaling with GPU count."""
    results = {
        "base_batch_size": base_batch_size,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "recommendations": {}
    }

    if results["gpu_count"] == 0:
        results["error"] = "No GPUs available"
        return results

    # Get memory per GPU
    min_memory_gb = float('inf')
    for i in range(results["gpu_count"]):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        min_memory_gb = min(min_memory_gb, memory_gb)

    # Calculate recommended batch sizes
    results["min_gpu_memory_gb"] = min_memory_gb

    # Conservative scaling (linear with GPU count)
    results["recommendations"]["conservative"] = {
        "per_gpu": base_batch_size,
        "total_effective": base_batch_size * results["gpu_count"],
        "gradient_accumulation": 1
    }

    # Aggressive scaling (assumes perfect parallelization)
    aggressive_per_gpu = base_batch_size * 2 if min_memory_gb > 40 else base_batch_size
    results["recommendations"]["aggressive"] = {
        "per_gpu": aggressive_per_gpu,
        "total_effective": aggressive_per_gpu * results["gpu_count"],
        "gradient_accumulation": 1
    }

    # Memory-aware scaling
    if min_memory_gb >= 80:  # H100 80GB
        memory_multiplier = 4
    elif min_memory_gb >= 40:  # A100 40GB
        memory_multiplier = 2
    elif min_memory_gb >= 24:  # RTX 3090/4090
        memory_multiplier = 1.5
    else:
        memory_multiplier = 1

    memory_aware_per_gpu = int(base_batch_size * memory_multiplier)
    results["recommendations"]["memory_aware"] = {
        "per_gpu": memory_aware_per_gpu,
        "total_effective": memory_aware_per_gpu * results["gpu_count"],
        "gradient_accumulation": 1,
        "memory_multiplier": memory_multiplier
    }

    # Gradient accumulation strategy (for large effective batch sizes)
    target_effective = 512  # Common target for transformer training
    if target_effective > base_batch_size * results["gpu_count"]:
        grad_accum = target_effective // (base_batch_size * results["gpu_count"])
        results["recommendations"]["gradient_accumulation"] = {
            "per_gpu": base_batch_size,
            "total_effective": target_effective,
            "gradient_accumulation": grad_accum,
            "actual_per_step": base_batch_size * results["gpu_count"]
        }

    return results


def test_data_parallel(model_size_mb: int = 100) -> Dict:
    """Test DataParallel wrapper functionality."""
    results = {
        "test": "DataParallel",
        "model_size_mb": model_size_mb,
        "success": False
    }

    if not torch.cuda.is_available():
        results["error"] = "CUDA not available"
        return results

    device_count = torch.cuda.device_count()
    results["device_count"] = device_count

    if device_count < 2:
        results["warning"] = f"DataParallel requires 2+ GPUs, found {device_count}"

    try:
        # Create model
        hidden_dim = int((model_size_mb * 1024 * 1024) / (4 * 3 * 1024))  # Rough estimate
        model = DummyModel(hidden_dim=max(hidden_dim, 256))

        # Move to GPU
        model = model.cuda()
        initial_device = next(model.parameters()).device
        results["initial_device"] = str(initial_device)

        # Wrap with DataParallel
        if device_count > 1:
            model = DataParallel(model)
            results["wrapped"] = True
            results["device_ids"] = model.device_ids
            results["output_device"] = model.output_device
        else:
            results["wrapped"] = False
            results["reason"] = "Single GPU, DataParallel not applied"

        # Test forward pass
        batch_size = 16
        input_tensor = torch.randn(batch_size, max(hidden_dim, 256)).cuda()
        output = model(input_tensor)

        results["forward_pass"] = {
            "input_shape": list(input_tensor.shape),
            "output_shape": list(output.shape),
            "output_device": str(output.device)
        }

        # Test backward pass
        loss = output.sum()
        loss.backward()

        results["backward_pass"] = True
        results["success"] = True

        # Memory usage after model creation
        memory_info = {}
        for i in range(device_count):
            memory_info[f"gpu_{i}"] = {
                "allocated_mb": torch.cuda.memory_allocated(i) / (1024**2),
                "reserved_mb": torch.cuda.memory_reserved(i) / (1024**2)
            }
        results["memory_usage"] = memory_info

    except Exception as e:
        results["error"] = str(e)
        results["success"] = False

    return results


def test_distributed_setup() -> Dict:
    """Test distributed training setup (single node, multi-GPU)."""
    results = {
        "test": "DistributedDataParallel",
        "success": False
    }

    if not torch.cuda.is_available():
        results["error"] = "CUDA not available"
        return results

    device_count = torch.cuda.device_count()
    results["device_count"] = device_count

    if device_count < 2:
        results["warning"] = f"DDP works best with 2+ GPUs, found {device_count}"

    # Check if we're already in a distributed environment
    results["dist_initialized"] = dist.is_initialized()

    if not dist.is_initialized():
        # Try to initialize for testing (won't actually work without proper launch)
        results["init_backend"] = "nccl"
        results["init_method"] = "env://"
        results["note"] = "DDP requires proper launch with torchrun or torch.distributed.launch"

        # Check environment variables
        env_vars = {
            "RANK": os.environ.get("RANK"),
            "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
            "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
            "MASTER_PORT": os.environ.get("MASTER_PORT")
        }
        results["dist_env_vars"] = env_vars

        # Provide launch command example
        results["launch_example"] = (
            f"torchrun --nproc_per_node={device_count} "
            f"--master_port=29500 {sys.argv[0]} --test-ddp-worker"
        )
    else:
        results["rank"] = dist.get_rank()
        results["world_size"] = dist.get_world_size()
        results["backend"] = dist.get_backend()
        results["success"] = True

    return results


def test_ddp_worker():
    """Worker function for testing DDP (called by torchrun)."""
    # Initialize distributed process group
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {rank}] Initialized on device {device}")

    # Create model and wrap with DDP
    model = DummyModel(hidden_dim=512).to(device)
    model = DDP(model, device_ids=[local_rank])

    # Test forward/backward
    input_tensor = torch.randn(16, 512).to(device)
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Synchronize
    dist.barrier()

    if rank == 0:
        print(f"DDP test successful with {world_size} processes")

    # Cleanup
    dist.destroy_process_group()


def test_cuda_visible_devices_simulation() -> Dict:
    """Test different CUDA_VISIBLE_DEVICES configurations."""
    results = {
        "current_setting": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "actual_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "simulations": []
    }

    if not torch.cuda.is_available():
        results["error"] = "CUDA not available for simulation"
        return results

    # Save original
    original_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")

    # Test configurations
    test_configs = [
        ("single", "0"),
        ("dual", "0,1"),
        ("triple", "0,1,2"),
        ("quad", "0,1,2,3"),
        ("reverse", "3,2,1,0"),
        ("skip", "0,2"),  # Skip GPU 1
        ("last_two", "2,3")
    ]

    for config_name, cvd_value in test_configs:
        # We can't actually change CUDA_VISIBLE_DEVICES after import
        # This would require subprocess, so we'll simulate
        expected_count = len(cvd_value.split(","))
        results["simulations"].append({
            "name": config_name,
            "cuda_visible_devices": cvd_value,
            "expected_count": expected_count,
            "note": "Would need subprocess to actually test"
        })

    # Show subprocess command for actual testing
    results["subprocess_test"] = (
        "To actually test: "
        "CUDA_VISIBLE_DEVICES=0,1 python scripts/test_multi_gpu_setup.py"
    )

    return results


def run_subprocess_gpu_test(cvd_setting: str) -> Dict:
    """Run GPU test in subprocess with specific CUDA_VISIBLE_DEVICES."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cvd_setting

    test_script = """
import torch
import json
print(json.dumps({
    'cuda_visible_devices': '%s',
    'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
}))
""" % cvd_setting

    try:
        result = subprocess.run(
            ["python", "-c", test_script],
            env=env,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {"error": result.stderr}
    except Exception as e:
        return {"error": str(e)}


def test_elastic_gpu_config() -> Dict:
    """Test ElasticGPUConfig from train.py."""
    results = {
        "test": "ElasticGPUConfig",
        "success": False
    }

    try:
        # Import ElasticGPUConfig from train.py
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from latentwire.train import ElasticGPUConfig

        # Test with current GPU setup
        config = ElasticGPUConfig(base_batch_size=64, model_size_gb=14.0)

        results["gpu_count"] = config.gpu_count
        results["gpu_specs"] = config.gpu_specs
        results["config"] = config.config
        results["success"] = True

    except ImportError as e:
        results["error"] = f"Could not import ElasticGPUConfig: {e}"
    except Exception as e:
        results["error"] = str(e)

    return results


def main():
    parser = argparse.ArgumentParser(description="Test multi-GPU setup")
    parser.add_argument("--test-ddp-worker", action="store_true",
                       help="Run as DDP worker (internal use)")
    parser.add_argument("--subprocess-tests", action="store_true",
                       help="Run subprocess tests with different CUDA_VISIBLE_DEVICES")
    parser.add_argument("--output-json", type=str,
                       help="Save results to JSON file")
    args = parser.parse_args()

    # If running as DDP worker
    if args.test_ddp_worker:
        test_ddp_worker()
        return

    # Run all tests
    print("=" * 80)
    print("Multi-GPU Setup Test Suite")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print()

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "tests": {}
    }

    # Test 1: CUDA Detection
    print("Test 1: CUDA Detection")
    print("-" * 40)
    cuda_results = test_cuda_detection()
    all_results["tests"]["cuda_detection"] = cuda_results
    print(f"CUDA Available: {cuda_results['cuda_available']}")
    print(f"Device Count: {cuda_results['device_count']}")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_results['cuda_visible_devices']}")
    if cuda_results["devices"]:
        for dev in cuda_results["devices"]:
            if "error" not in dev:
                print(f"  GPU {dev['index']}: {dev['name']} "
                      f"({dev['total_memory_gb']:.1f} GB)")
    print()

    # Test 2: Memory Info
    print("Test 2: GPU Memory Info")
    print("-" * 40)
    memory_results = test_memory_info()
    all_results["tests"]["memory_info"] = memory_results
    if "error" not in memory_results:
        for gpu_id, info in memory_results["gpus"].items():
            if "error" not in info:
                print(f"GPU {gpu_id}:")
                print(f"  Total: {info['total_gb']:.1f} GB")
                print(f"  Available: {info['available_gb']:.1f} GB")
                print(f"  Allocated: {info['allocated_gb']:.3f} GB")
    else:
        print(f"Error: {memory_results['error']}")
    print()

    # Test 3: Batch Size Scaling
    print("Test 3: Batch Size Scaling")
    print("-" * 40)
    batch_results = test_batch_size_scaling(base_batch_size=64)
    all_results["tests"]["batch_scaling"] = batch_results
    if "error" not in batch_results:
        print(f"GPU Count: {batch_results['gpu_count']}")
        print(f"Min GPU Memory: {batch_results['min_gpu_memory_gb']:.1f} GB")
        print("Recommended batch sizes:")
        for strategy, config in batch_results["recommendations"].items():
            print(f"  {strategy}:")
            print(f"    Per-GPU: {config['per_gpu']}")
            print(f"    Total Effective: {config['total_effective']}")
            if config['gradient_accumulation'] > 1:
                print(f"    Gradient Accumulation: {config['gradient_accumulation']}")
    else:
        print(f"Error: {batch_results['error']}")
    print()

    # Test 4: DataParallel
    print("Test 4: DataParallel")
    print("-" * 40)
    dp_results = test_data_parallel(model_size_mb=100)
    all_results["tests"]["data_parallel"] = dp_results
    print(f"Success: {dp_results['success']}")
    if dp_results.get("wrapped"):
        print(f"Device IDs: {dp_results['device_ids']}")
        print(f"Output Device: {dp_results['output_device']}")
    if "warning" in dp_results:
        print(f"Warning: {dp_results['warning']}")
    if "error" in dp_results:
        print(f"Error: {dp_results['error']}")
    print()

    # Test 5: Distributed Setup
    print("Test 5: Distributed DataParallel (DDP)")
    print("-" * 40)
    ddp_results = test_distributed_setup()
    all_results["tests"]["distributed"] = ddp_results
    if ddp_results["dist_initialized"]:
        print(f"Already initialized: Rank {ddp_results['rank']}/{ddp_results['world_size']}")
    else:
        print("DDP not initialized (expected in single-process mode)")
        print(f"To test DDP, run: {ddp_results['launch_example']}")
    print()

    # Test 6: CUDA_VISIBLE_DEVICES Simulation
    print("Test 6: CUDA_VISIBLE_DEVICES Simulation")
    print("-" * 40)
    cvd_results = test_cuda_visible_devices_simulation()
    all_results["tests"]["cuda_visible_devices"] = cvd_results
    print(f"Current Setting: {cvd_results['current_setting']}")
    print(f"Actual Devices: {cvd_results['actual_devices']}")
    print("Simulated configurations:")
    for sim in cvd_results["simulations"]:
        print(f"  {sim['name']}: CUDA_VISIBLE_DEVICES={sim['cuda_visible_devices']} "
              f"→ {sim['expected_count']} GPU(s)")
    print()

    # Test 7: Subprocess tests (optional)
    if args.subprocess_tests and torch.cuda.is_available():
        print("Test 7: Subprocess GPU Tests")
        print("-" * 40)
        subprocess_results = []
        test_settings = ["0", "0,1", "1", "0,1,2,3"]
        for setting in test_settings:
            result = run_subprocess_gpu_test(setting)
            subprocess_results.append(result)
            if "error" not in result:
                print(f"CUDA_VISIBLE_DEVICES={setting}: {result['device_count']} GPU(s)")
            else:
                print(f"CUDA_VISIBLE_DEVICES={setting}: Error - {result['error']}")
        all_results["tests"]["subprocess_tests"] = subprocess_results
        print()

    # Test 8: ElasticGPUConfig
    print("Test 8: ElasticGPUConfig Integration")
    print("-" * 40)
    elastic_results = test_elastic_gpu_config()
    all_results["tests"]["elastic_gpu"] = elastic_results
    if elastic_results["success"]:
        print(f"GPU Count: {elastic_results['gpu_count']}")
        if elastic_results.get("config"):
            print(f"Configuration: {elastic_results['config']}")
    else:
        print(f"Error: {elastic_results.get('error', 'Unknown')}")
    print()

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)

    if cuda_results["cuda_available"]:
        print(f"✅ CUDA is available with {cuda_results['device_count']} GPU(s)")

        if cuda_results["device_count"] >= 2:
            print("✅ Multi-GPU setup ready")
            print("   - DataParallel: Available")
            print("   - DistributedDataParallel: Available (requires torchrun)")
        else:
            print("⚠️  Single GPU detected - multi-GPU parallelism limited")

        # Recommendations
        print("\nRecommendations:")
        print(f"1. For {cuda_results['device_count']} GPU(s), use batch size: "
              f"{batch_results.get('recommendations', {}).get('memory_aware', {}).get('total_effective', 'N/A')}")
        print("2. For distributed training, launch with:")
        print(f"   torchrun --nproc_per_node={cuda_results['device_count']} your_script.py")
        print("3. For testing specific GPU configs, use:")
        print("   CUDA_VISIBLE_DEVICES=0,1 python your_script.py")
    else:
        print("❌ CUDA is not available - CPU mode only")

    # Save results if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output_json}")

    print("\n" + "=" * 80)
    print("Test complete!")


if __name__ == "__main__":
    main()