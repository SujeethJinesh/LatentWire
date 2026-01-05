#!/usr/bin/env python3
"""Test script for elastic GPU configuration.

This script demonstrates how the ElasticGPUConfig adapts to different GPU counts
and provides optimal configurations for LatentWire training.

Usage:
    python scripts/test_elastic_gpu.py

    # Simulate different GPU counts:
    CUDA_VISIBLE_DEVICES=0 python scripts/test_elastic_gpu.py       # 1 GPU
    CUDA_VISIBLE_DEVICES=0,1 python scripts/test_elastic_gpu.py     # 2 GPUs
    CUDA_VISIBLE_DEVICES=0,1,2 python scripts/test_elastic_gpu.py   # 3 GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/test_elastic_gpu.py # 4 GPUs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Import the ElasticGPUConfig from train.py
from latentwire.train import ElasticGPUConfig


def test_configurations():
    """Test elastic GPU configuration with current hardware."""

    print("="*80)
    print("ELASTIC GPU CONFIGURATION TEST")
    print("="*80)
    print()

    # Check available GPUs
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"PyTorch CUDA available: Yes")
        print(f"Number of GPUs detected: {gpu_count}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        print("PyTorch CUDA available: No")
        print("Running in CPU-only mode")

    print()
    print("-"*80)

    # Test different model configurations
    configs = [
        ("Llama-8B only", 14.0, "llama"),
        ("Qwen-7B only", 13.0, "qwen"),
        ("Both models", 27.0, "llama,qwen"),
    ]

    for name, model_size, models in configs:
        print(f"\n### Configuration: {name}")
        print(f"    Model size: {model_size:.1f} GB")
        print(f"    Models: {models}")
        print()

        # Create elastic config
        elastic = ElasticGPUConfig(
            base_batch_size=64,
            model_size_gb=model_size,
            target_util=0.75
        )

        # Get optimal configuration
        optimal = elastic.get_optimal_config(
            dataset_size=10000,
            target_steps=100
        )

        # Print key settings
        print(f"    Strategy: {optimal['strategy']}")
        print(f"    Batch size: {optimal['batch_size']}")
        print(f"    Gradient accumulation: {optimal['grad_accum_steps']}")
        print(f"    Effective batch size: {optimal['effective_batch_size']}")

        if optimal.get('llama_devices'):
            print(f"    Llama devices: {optimal['llama_devices']}")
        if optimal.get('qwen_devices'):
            print(f"    Qwen devices: {optimal['qwen_devices']}")

        print(f"    Notes: {optimal['notes']}")

    print()
    print("-"*80)
    print()

    # Show command-line arguments
    print("### Example command-line arguments for training:")
    print()

    elastic = ElasticGPUConfig(base_batch_size=64, model_size_gb=27.0)
    args = elastic.to_args()

    print(f"python latentwire/train.py \\")
    print(f"    --elastic_gpu \\")
    for arg in args.split():
        if arg.startswith("--"):
            print(f"    {arg} \\")
        else:
            print(f"{arg} \\")
    print(f"    --samples 10000 --epochs 3")

    print()
    print("="*80)


def simulate_different_gpu_counts():
    """Show what would happen with different GPU counts."""

    print()
    print("="*80)
    print("SIMULATED CONFIGURATIONS FOR DIFFERENT GPU COUNTS")
    print("="*80)
    print()

    # Save original CUDA state
    original_cuda = torch.cuda.is_available()

    # We can't actually change GPU count, but we can show the logic
    gpu_scenarios = [
        (0, "CPU only"),
        (1, "Single GPU"),
        (2, "Dual GPU"),
        (3, "Three GPUs"),
        (4, "Four GPUs (HPC cluster)"),
        (8, "Eight GPUs (large cluster)"),
    ]

    for gpu_count, description in gpu_scenarios:
        print(f"\n### {description} ({gpu_count} GPUs)")
        print("-"*40)

        # Mock the GPU count for demonstration
        # In real usage, this would be detected automatically
        if gpu_count == 0:
            config = {
                'batch_size': 1,
                'effective_batch_size': 1,
                'grad_accum_steps': 1,
                'device': 'cpu',
                'strategy': 'single_device',
                'notes': 'CPU-only mode (very slow)',
            }
        elif gpu_count == 1:
            # Assuming 40GB GPU (A100-40GB)
            config = {
                'batch_size': 32,
                'effective_batch_size': 64,
                'grad_accum_steps': 2,
                'device': 'cuda:0',
                'strategy': 'single_gpu',
                'llama_devices': '0',
                'qwen_devices': '0',
                'notes': 'Single GPU mode with gradient accumulation (2 steps)',
            }
        elif gpu_count == 2:
            config = {
                'batch_size': 48,
                'effective_batch_size': 48,
                'grad_accum_steps': 1,
                'device': 'cuda',
                'strategy': 'model_split',
                'llama_devices': '0',
                'qwen_devices': '1',
                'notes': 'Model splitting: Llama on GPU0, Qwen on GPU1',
            }
        elif gpu_count == 3:
            config = {
                'batch_size': 64,
                'effective_batch_size': 64,
                'grad_accum_steps': 1,
                'device': 'cuda',
                'strategy': 'hybrid_3gpu',
                'llama_devices': '0,1',
                'qwen_devices': '2',
                'notes': 'Hybrid: Llama on GPU0-1, Qwen on GPU2',
            }
        elif gpu_count == 4:
            config = {
                'batch_size': 256,
                'effective_batch_size': 256,
                'grad_accum_steps': 1,
                'device': 'cuda',
                'strategy': 'ddp_4gpu',
                'llama_devices': '0,1,2,3',
                'qwen_devices': '0,1,2,3',
                'notes': 'Full DDP on 4x H100 GPUs (maximum throughput)',
            }
        else:
            config = {
                'batch_size': 64 * gpu_count,
                'effective_batch_size': 64 * gpu_count,
                'grad_accum_steps': 1,
                'device': 'cuda',
                'strategy': f'ddp_{gpu_count}gpu',
                'notes': f'Scaled DDP on {gpu_count} GPUs',
            }

        print(f"  Strategy: {config['strategy']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Gradient accumulation: {config['grad_accum_steps']}")
        print(f"  Effective batch size: {config['effective_batch_size']}")

        if 'llama_devices' in config:
            print(f"  Llama devices: {config['llama_devices']}")
        if 'qwen_devices' in config:
            print(f"  Qwen devices: {config['qwen_devices']}")

        print(f"  Notes: {config['notes']}")

    print()
    print("="*80)


def main():
    """Run all tests."""
    test_configurations()
    simulate_different_gpu_counts()

    print("\nTest complete!")


if __name__ == "__main__":
    main()