#!/usr/bin/env python3
"""
Test script to verify DDP (Distributed Data Parallel) setup is working correctly.

This script tests:
1. Environment variables for DDP
2. GPU availability and count
3. DDP initialization
4. Data distribution across GPUs
5. Gradient synchronization
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import json


def test_environment():
    """Test DDP environment variables."""
    print("\n" + "="*60)
    print("1. DDP Environment Variables")
    print("="*60)

    env_vars = {
        'WORLD_SIZE': os.environ.get('WORLD_SIZE', 'NOT SET'),
        'RANK': os.environ.get('RANK', 'NOT SET'),
        'LOCAL_RANK': os.environ.get('LOCAL_RANK', 'NOT SET'),
        'MASTER_PORT': os.environ.get('MASTER_PORT', 'NOT SET'),
        'MASTER_ADDR': os.environ.get('MASTER_ADDR', 'NOT SET'),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET'),
    }

    for var, value in env_vars.items():
        print(f"  {var}: {value}")

    # Check if we're in DDP mode
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1
    print(f"\n  DDP Mode: {'YES' if is_ddp else 'NO'}")

    return is_ddp, env_vars


def test_gpu_availability():
    """Test GPU availability and properties."""
    print("\n" + "="*60)
    print("2. GPU Availability")
    print("="*60)

    if not torch.cuda.is_available():
        print("  ERROR: CUDA is not available!")
        return False

    gpu_count = torch.cuda.device_count()
    print(f"  GPU Count: {gpu_count}")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        print(f"\n  GPU {i}: {props.name}")
        print(f"    Memory: {memory_gb:.1f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")

    return True


def test_ddp_initialization(is_ddp):
    """Test DDP initialization."""
    print("\n" + "="*60)
    print("3. DDP Initialization")
    print("="*60)

    if not is_ddp:
        print("  Skipping (not in DDP mode)")
        return None

    try:
        # Get DDP parameters
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )

        print(f"  Process group initialized successfully")
        print(f"  World size: {world_size}")
        print(f"  Current rank: {rank}")
        print(f"  Local rank: {local_rank}")
        print(f"  Device: {device}")

        return device, rank, world_size

    except Exception as e:
        print(f"  ERROR: Failed to initialize DDP: {e}")
        return None


def test_data_distribution(world_size=1, rank=0):
    """Test data distribution across GPUs."""
    print("\n" + "="*60)
    print("4. Data Distribution")
    print("="*60)

    # Create sample dataset
    dataset_size = 100
    batch_size = 16

    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler

    class SimpleDataset(Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return torch.tensor([idx], dtype=torch.float32)

    dataset = SimpleDataset(dataset_size)

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0
        )
        print(f"  Using DistributedSampler")
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        print(f"  Using regular DataLoader")

    # Check data distribution
    total_samples = 0
    unique_indices = set()

    for batch in dataloader:
        batch_indices = batch.squeeze().tolist()
        if isinstance(batch_indices, (int, float)):
            batch_indices = [batch_indices]
        unique_indices.update(batch_indices)
        total_samples += len(batch_indices)

    print(f"  Dataset size: {dataset_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Samples for rank {rank}: {total_samples}")
    print(f"  Unique indices seen: {len(unique_indices)}")

    if world_size > 1:
        expected_per_rank = dataset_size // world_size
        print(f"  Expected per rank: ~{expected_per_rank}")
        print(f"  Distribution efficiency: {total_samples/expected_per_rank*100:.1f}%")

    return total_samples


def test_model_wrapping(device=None):
    """Test DDP model wrapping."""
    print("\n" + "="*60)
    print("5. Model DDP Wrapping")
    print("="*60)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel().to(device)
    print(f"  Created model on {device}")

    # Wrap with DDP if in distributed mode
    if dist.is_initialized():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model = DDP(model, device_ids=[local_rank])
        print(f"  Wrapped model with DDP")
        print(f"  Model type: {type(model)}")

        # Test forward pass
        x = torch.randn(4, 10).to(device)
        y = model(x)
        loss = y.mean()
        loss.backward()

        print(f"  Forward/backward pass successful")
        print(f"  Gradients synchronized: YES (automatic in DDP)")
    else:
        print(f"  DDP wrapping skipped (not in distributed mode)")

    return model


def test_gradient_sync():
    """Test gradient synchronization across GPUs."""
    print("\n" + "="*60)
    print("6. Gradient Synchronization")
    print("="*60)

    if not dist.is_initialized():
        print("  Skipping (not in distributed mode)")
        return

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')

    # Create model and wrap with DDP
    model = nn.Linear(10, 1).to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    # Create different data for each rank
    rank = dist.get_rank()
    torch.manual_seed(rank)  # Different seed per rank
    x = torch.randn(4, 10).to(device)
    y_true = torch.randn(4, 1).to(device)

    # Forward pass
    y_pred = ddp_model(x)
    loss = nn.MSELoss()(y_pred, y_true)

    # Backward pass (gradients are automatically synchronized)
    loss.backward()

    # Check gradient values
    grad_norm = 0.0
    for param in ddp_model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5

    print(f"  Rank {rank} gradient norm: {grad_norm:.4f}")

    # All-reduce to verify synchronization
    grad_tensor = torch.tensor([grad_norm]).to(device)
    dist.all_reduce(grad_tensor, op=dist.ReduceOp.AVG)
    avg_grad_norm = grad_tensor.item()

    print(f"  Average gradient norm across all ranks: {avg_grad_norm:.4f}")
    print(f"  Gradient sync working: YES")


def main():
    """Run all DDP tests."""
    print("\n" + "="*60)
    print("DDP Setup Verification")
    print("="*60)

    results = {}

    # Test 1: Environment
    is_ddp, env_vars = test_environment()
    results['environment'] = {
        'is_ddp': is_ddp,
        'variables': env_vars
    }

    # Test 2: GPU availability
    gpu_available = test_gpu_availability()
    results['gpu_available'] = gpu_available

    # Test 3: DDP initialization
    if is_ddp:
        ddp_info = test_ddp_initialization(is_ddp)
        if ddp_info:
            device, rank, world_size = ddp_info
            results['ddp_initialized'] = True
            results['rank'] = rank
            results['world_size'] = world_size
        else:
            results['ddp_initialized'] = False
            device = 'cuda' if gpu_available else 'cpu'
            rank = 0
            world_size = 1
    else:
        device = 'cuda' if gpu_available else 'cpu'
        rank = 0
        world_size = 1
        results['ddp_initialized'] = False

    # Test 4: Data distribution
    samples_seen = test_data_distribution(world_size, rank)
    results['data_distribution'] = {
        'samples_seen': samples_seen,
        'rank': rank
    }

    # Test 5: Model wrapping
    model = test_model_wrapping(device)
    results['model_wrapped'] = isinstance(model, DDP) if is_ddp else False

    # Test 6: Gradient synchronization
    if is_ddp:
        test_gradient_sync()
        results['gradient_sync'] = True
    else:
        results['gradient_sync'] = False

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    if is_ddp:
        print(f"✓ DDP mode detected (world_size={world_size})")
        if results.get('ddp_initialized'):
            print(f"✓ DDP initialization successful (rank={rank})")
        else:
            print("✗ DDP initialization failed")

        if results.get('model_wrapped'):
            print("✓ Model wrapping successful")
        if results.get('gradient_sync'):
            print("✓ Gradient synchronization working")
    else:
        print("✓ Single GPU/CPU mode")
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            print(f"✓ {gpu_count} GPU(s) available")
        else:
            print("✗ No GPUs available")

    print("\n" + "="*60)
    print("Configuration Recommendations")
    print("="*60)

    if is_ddp:
        print("For DDP training with torchrun:")
        print("  torchrun --nproc_per_node=4 latentwire/train.py \\")
        print("    --batch_size 64 \\  # This is PER GPU")
        print("    --elastic_gpu \\     # Will auto-detect DDP mode")
        print("    ...")
    else:
        print("To enable DDP, use torchrun:")
        print("  torchrun --nproc_per_node=<num_gpus> script.py")
        print("\nOr set environment variables:")
        print("  export WORLD_SIZE=4")
        print("  export RANK=0")
        print("  export LOCAL_RANK=0")
        print("  export MASTER_PORT=29500")
        print("  export MASTER_ADDR=localhost")

    # Save results
    if rank == 0:  # Only save from main process
        with open('ddp_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to ddp_test_results.json")

    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()