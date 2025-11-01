"""Utility functions for distributed training and platform detection."""

import os
import torch
import torch.distributed as dist
from datetime import timedelta


def get_device_and_config():
    """Auto-detect platform and return appropriate device and config."""
    config = {}

    # Check environment variables
    use_mps = os.environ.get('USE_MPS', '0') == '1'
    use_cuda = os.environ.get('USE_CUDA', '0') == '1'
    disable_flash = os.environ.get('DISABLE_FLASH_ATTENTION', '0') == '1'

    # Detect device
    if use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
        platform = 'mac'
        print("==> Using MPS (Metal Performance Shaders) on Mac")
    elif use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        platform = 'hpc'
        print(f"==> Using CUDA on HPC ({torch.cuda.device_count()} GPUs available)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        platform = 'mac'
        print("==> Auto-detected MPS on Mac")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        platform = 'hpc'
        print("==> Auto-detected CUDA")
    else:
        device = torch.device('cpu')
        platform = 'cpu'
        print("==> Using CPU (no GPU available)")

    # Platform-specific config
    if platform == 'mac':
        config['batch_size'] = int(os.environ.get('MAC_BATCH_SIZE', '4'))
        config['num_samples'] = int(os.environ.get('MAC_SAMPLES', '1000'))
        config['epochs'] = int(os.environ.get('MAC_EPOCHS', '2'))
        config['use_bf16'] = False  # BF16 not well supported on MPS
        config['use_flash_attention'] = False
        config['grad_accum_steps'] = 8  # More accumulation for smaller batches
        print(f"  - Batch size: {config['batch_size']} (Mac memory constraints)")
        print(f"  - Samples: {config['num_samples']} (reduced for testing)")
        print(f"  - Epochs: {config['epochs']} (reduced for testing)")
    else:
        # HPC configuration - batch size optimized for multi-GPU with DDP
        num_gpus = torch.cuda.device_count() if platform == 'hpc' else 1
        # DDP: Each process gets batch_size samples
        config['batch_size'] = 10  # Per-process batch size
        config['num_samples'] = 10000  # Conservative for preemptible cluster
        config['epochs'] = 5  # Reduced based on convergence analysis
        config['use_bf16'] = torch.cuda.is_bf16_supported() if platform == 'hpc' else False
        config['use_flash_attention'] = not disable_flash and platform == 'hpc'
        config['grad_accum_steps'] = 8  # DDP is more efficient
        if platform == 'hpc':
            print(f"  - Batch size per GPU: {config['batch_size']}")
            print(f"  - Global batch size: {config['batch_size'] * num_gpus}")
            print(f"  - Effective batch (with grad accum): {config['batch_size'] * num_gpus * config['grad_accum_steps']}")
            print(f"  - Samples: {config['num_samples']}")
            print(f"  - Epochs: {config['epochs']}")
            print(f"  - BF16: {config['use_bf16']}")
            print(f"  - Flash Attention: {config['use_flash_attention']}")

    return device, platform, config


def setup_ddp():
    """
    Initialize DDP for multi-GPU training.

    Returns:
        tuple: (rank, world_size, device)
            - rank: Process rank (0 to world_size-1)
            - world_size: Total number of processes
            - device: torch.device for this process
    """
    if not dist.is_available():
        device, _, _ = get_device_and_config()
        return 0, 1, device

    if not dist.is_initialized():
        # Initialize process group with extended timeout
        # Inference experiments on rank 0 can take >10 min, so increase timeout to 60 min
        timeout_seconds = 3600  # 60 minutes
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(seconds=timeout_seconds)
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device, _, _ = get_device_and_config()

    return rank, world_size, device


def cleanup_ddp():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Return True if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """Get current process rank, returns 0 if not using DDP."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get world size, returns 1 if not using DDP."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1
