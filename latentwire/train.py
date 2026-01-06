# latentwire/train.py
import os
import re
import time
import json
import math
import argparse
import random
import ast
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union, Any, Sequence
from contextlib import contextmanager, nullcontext

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    PYTORCH_AVAILABLE = True
except ImportError as e:
    PYTORCH_AVAILABLE = False
    PYTORCH_IMPORT_ERROR = str(e)
from latentwire.core_utils import (
    capture_env_snapshot,
    patch_dataloader_defaults,
    apply_anchor_normalization,
    collate_bytes,
    calibrate_to_embed_rms,
    bos_policy,
    first_non_bos,
    anchor_token_ids,
    tensor_rms,
    assistant_header_anchor,
    SYSTEM_PROMPT,
    split_user_and_anchor,
)

from latentwire.models import (
    InterlinguaEncoder,
    Adapter,
    LatentRefiner,
    GistReconstructionHead,
    LMWrapper,
    LMConfig,
    ByteTokenizer,
    SimpleEncoder,
    STQueryEncoder,
    DeepPrefixGenerator,
    LatentCoprocessor,
    apply_lora_if_requested,
    apply_prefix_if_requested,
)
from latentwire.checkpointing import save_latest_checkpoint
from latentwire.data_pipeline import prepare_training_data
from latentwire.feature_registry import FeatureRegistry
from latentwire.loss_bundles import (
    loss_with_text_prompt_chunked,
    alignment_mse,
    manifold_stat_loss,
    scale_penalty,
    rms_raw_penalty,
)


class ElasticGPUConfig:
    """Elastic GPU configuration that adapts to available hardware.

    Automatically detects GPU count and configures optimal settings for:
    - Batch size (per-GPU and effective)
    - Gradient accumulation steps
    - Model parallelism strategy
    - Memory allocation
    """

    def __init__(self, base_batch_size=64, model_size_gb=14.0, target_util=0.75):
        """Initialize elastic GPU configuration.

        Args:
            base_batch_size: Base batch size for single GPU
            model_size_gb: Estimated model size in GB (Llama-8B ≈ 14GB, Qwen-7B ≈ 13GB)
            target_util: Target GPU memory utilization (0.75 = conservative)
        """
        self.base_batch_size = base_batch_size
        self.model_size_gb = model_size_gb
        self.target_util = target_util
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Detect GPU specifications
        self.gpu_specs = self._detect_gpu_specs()

        # Configure based on GPU count
        self.config = self._configure_for_gpus()

    def _detect_gpu_specs(self):
        """Detect GPU specifications (memory, type, etc.)."""
        if self.gpu_count == 0:
            return None

        specs = []
        for i in range(self.gpu_count):
            props = torch.cuda.get_device_properties(i)
            specs.append({
                'id': i,
                'name': props.name,
                'memory_gb': props.total_memory / 1e9,
                'compute_capability': (props.major, props.minor),
                'is_h100': 'H100' in props.name or 'h100' in props.name.lower(),
                'is_a100': 'A100' in props.name or 'a100' in props.name.lower(),
            })
        return specs

    def _configure_for_gpus(self):
        """Configure optimal settings based on GPU count and specs."""
        if self.gpu_count == 0:
            # CPU-only fallback
            return {
                'batch_size': 1,
                'effective_batch_size': 1,
                'grad_accum_steps': 1,
                'device': 'cpu',
                'device_map': None,
                'strategy': 'single_device',
                'ddp': False,
                'model_parallel': False,
                'notes': 'CPU-only mode (very slow)',
            }

        # Get total and per-GPU memory
        total_memory = sum(g['memory_gb'] for g in self.gpu_specs)
        min_gpu_memory = min(g['memory_gb'] for g in self.gpu_specs)

        # Detect GPU type for optimizations
        is_h100_cluster = all(g.get('is_h100', False) for g in self.gpu_specs)
        is_a100_cluster = all(g.get('is_a100', False) for g in self.gpu_specs)

        # Memory calculations
        # Model needs ~14GB for Llama-8B, ~13GB for Qwen-7B
        # Activations need ~0.5-1GB per batch item for seq_len=256
        activation_gb_per_item = 0.75  # Conservative estimate
        available_per_gpu = min_gpu_memory * self.target_util
        model_per_gpu = self.model_size_gb / max(1, self.gpu_count // 2)  # Assume model splitting if >2 GPUs

        # Configurations based on GPU count
        configs = {
            1: self._config_single_gpu(available_per_gpu, activation_gb_per_item),
            2: self._config_dual_gpu(available_per_gpu, activation_gb_per_item, is_h100_cluster),
            3: self._config_three_gpu(available_per_gpu, activation_gb_per_item, is_h100_cluster),
            4: self._config_four_gpu(available_per_gpu, activation_gb_per_item, is_h100_cluster),
        }

        config = configs.get(self.gpu_count, self._config_multi_gpu(
            self.gpu_count, available_per_gpu, activation_gb_per_item, is_h100_cluster
        ))

        # Add GPU info to config
        config['gpu_count'] = self.gpu_count
        config['gpu_specs'] = self.gpu_specs
        config['total_memory_gb'] = total_memory

        return config

    def _config_single_gpu(self, available_gb, activation_gb_per_item):
        """Configuration for single GPU."""
        # Need to fit model + activations + optimizer states
        usable_for_activations = max(1, available_gb - self.model_size_gb - 4)  # 4GB for optimizer
        batch_size = min(self.base_batch_size, int(usable_for_activations / activation_gb_per_item))
        batch_size = max(1, batch_size)

        # Use gradient accumulation to reach effective batch size
        target_effective = self.base_batch_size
        grad_accum = max(1, target_effective // batch_size)

        return {
            'batch_size': batch_size,
            'effective_batch_size': batch_size * grad_accum,
            'grad_accum_steps': grad_accum,
            'device': 'cuda:0',
            'device_map': None,
            'strategy': 'single_gpu',
            'ddp': False,
            'model_parallel': False,
            'llama_devices': '0',
            'qwen_devices': '0',
            'notes': f'Single GPU mode with gradient accumulation ({grad_accum} steps)',
        }

    def _config_dual_gpu(self, available_gb, activation_gb_per_item, is_h100):
        """Configuration for 2 GPUs."""
        if is_h100:
            # H100s have 80GB - can fit both models easily
            batch_per_gpu = self.base_batch_size
            return {
                'batch_size': batch_per_gpu * 2,  # DDP with 2 GPUs
                'effective_batch_size': batch_per_gpu * 2,
                'grad_accum_steps': 1,
                'device': 'cuda',
                'device_map': 'auto',
                'strategy': 'ddp',
                'ddp': True,
                'model_parallel': False,
                'llama_devices': '0,1',
                'qwen_devices': '0,1',
                'notes': 'DDP on 2 H100 GPUs (data parallel)',
            }
        else:
            # Split models across GPUs for memory efficiency
            batch_size = min(self.base_batch_size, int((available_gb - self.model_size_gb) / activation_gb_per_item))
            return {
                'batch_size': batch_size,
                'effective_batch_size': batch_size,
                'grad_accum_steps': 1,
                'device': 'cuda',
                'device_map': 'auto',
                'strategy': 'model_split',
                'ddp': False,
                'model_parallel': True,
                'llama_devices': '0',
                'qwen_devices': '1',
                'notes': 'Model splitting: Llama on GPU0, Qwen on GPU1',
            }

    def _config_three_gpu(self, available_gb, activation_gb_per_item, is_h100):
        """Configuration for 3 GPUs."""
        # Use 2 for models, 1 for encoder/adapters
        batch_per_gpu = min(self.base_batch_size, int((available_gb - self.model_size_gb/2) / activation_gb_per_item))

        return {
            'batch_size': batch_per_gpu,
            'effective_batch_size': batch_per_gpu,
            'grad_accum_steps': 1,
            'device': 'cuda',
            'device_map': 'auto',
            'strategy': 'hybrid_3gpu',
            'ddp': False,
            'model_parallel': True,
            'llama_devices': '0,1',  # Llama split across 2 GPUs
            'qwen_devices': '2',      # Qwen on dedicated GPU
            'encoder_device': 'cuda:0',  # Encoder shares with Llama
            'notes': 'Hybrid: Llama on GPU0-1, Qwen on GPU2',
        }

    def _config_four_gpu(self, available_gb, activation_gb_per_item, is_h100):
        """Configuration for 4 GPUs (common HPC setup)."""
        if is_h100:
            # 4x H100 (80GB each) - maximum parallelism
            batch_per_gpu = self.base_batch_size

            # Check if we're running under torchrun (DDP mode)
            is_torchrun = 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1

            if is_torchrun:
                # When using torchrun, batch_size is PER GPU
                # torchrun will handle the distribution
                return {
                    'batch_size': batch_per_gpu,  # Per-GPU batch size
                    'effective_batch_size': batch_per_gpu * 4,
                    'grad_accum_steps': 1,
                    'device': 'cuda',
                    'device_map': 'auto',
                    'strategy': 'ddp_torchrun_4gpu',
                    'ddp': True,
                    'model_parallel': False,
                    'llama_devices': '0,1,2,3',
                    'qwen_devices': '0,1,2,3',
                    'notes': 'Full DDP on 4x H100 GPUs via torchrun (per-GPU batch size)',
                }
            else:
                return {
                    'batch_size': batch_per_gpu * 4,  # Total batch size (legacy mode)
                    'effective_batch_size': batch_per_gpu * 4,
                    'grad_accum_steps': 1,
                    'device': 'cuda',
                    'device_map': 'auto',
                    'strategy': 'ddp_4gpu',
                    'ddp': True,
                    'model_parallel': False,
                    'llama_devices': '0,1,2,3',
                    'qwen_devices': '0,1,2,3',
                    'notes': 'Full DDP on 4x H100 GPUs (maximum throughput)',
                }
        else:
            # 4x A100 or similar - balanced model/data parallelism
            batch_per_gpu = min(self.base_batch_size, int((available_gb - self.model_size_gb/2) / activation_gb_per_item))
            return {
                'batch_size': batch_per_gpu * 2,  # 2-way data parallel
                'effective_batch_size': batch_per_gpu * 2,
                'grad_accum_steps': 1,
                'device': 'cuda',
                'device_map': 'auto',
                'strategy': 'hybrid_4gpu',
                'ddp': True,
                'model_parallel': True,
                'llama_devices': '0,1',  # 2 GPUs for Llama
                'qwen_devices': '2,3',    # 2 GPUs for Qwen
                'notes': 'Hybrid: 2-way model parallel + 2-way data parallel',
            }

    def _config_multi_gpu(self, n_gpus, available_gb, activation_gb_per_item, is_h100):
        """Configuration for >4 GPUs."""
        # Scale up data parallelism
        batch_per_gpu = min(self.base_batch_size, int((available_gb - self.model_size_gb/4) / activation_gb_per_item))

        return {
            'batch_size': batch_per_gpu * n_gpus,
            'effective_batch_size': batch_per_gpu * n_gpus,
            'grad_accum_steps': 1,
            'device': 'cuda',
            'device_map': 'auto',
            'strategy': f'ddp_{n_gpus}gpu',
            'ddp': True,
            'model_parallel': n_gpus > 8,  # Use model parallelism for very large clusters
            'notes': f'Scaled DDP on {n_gpus} GPUs',
        }

    def get_optimal_config(self, dataset_size=None, target_steps=None):
        """Get optimal configuration with optional dataset-aware adjustments.

        Args:
            dataset_size: Size of training dataset
            target_steps: Target number of optimization steps per epoch

        Returns:
            Configuration dict with all settings
        """
        config = self.config.copy()

        # Adjust for dataset size if provided
        if dataset_size and target_steps:
            min_batch = max(1, dataset_size // target_steps)
            if config['effective_batch_size'] < min_batch:
                # Need more gradient accumulation
                extra_accum = min_batch // config['effective_batch_size']
                config['grad_accum_steps'] *= extra_accum
                config['effective_batch_size'] *= extra_accum
                config['notes'] += f' (adjusted for {target_steps} steps/epoch)'

        return config

    def print_config(self):
        """Print the current configuration in a readable format."""
        print("=" * 70)
        print("ELASTIC GPU CONFIGURATION")
        print("=" * 70)

        if self.gpu_count == 0:
            print("WARNING: No GPUs detected, using CPU (training will be very slow)")
        else:
            print(f"Detected {self.gpu_count} GPU(s):")
            for gpu in self.gpu_specs:
                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")

        print("\nOptimal Configuration:")
        print(f"  Strategy: {self.config['strategy']}")
        print(f"  Batch size per step: {self.config['batch_size']}")
        print(f"  Gradient accumulation: {self.config['grad_accum_steps']}")
        print(f"  Effective batch size: {self.config['effective_batch_size']}")

        if self.config.get('ddp'):
            print("  Data Parallel: Yes (DDP)")
        if self.config.get('model_parallel'):
            print("  Model Parallel: Yes")

        if self.config.get('llama_devices'):
            print(f"  Llama GPUs: {self.config.get('llama_devices')}")
        if self.config.get('qwen_devices'):
            print(f"  Qwen GPUs: {self.config.get('qwen_devices')}")

        print(f"\nNotes: {self.config['notes']}")
        print("=" * 70)

    def to_args(self):
        """Convert configuration to command-line arguments."""
        args = [
            f"--batch_size {self.config['batch_size']}",
            f"--grad_accum_steps {self.config['grad_accum_steps']}",
        ]

        if self.config.get('llama_devices'):
            args.append(f"--llama_devices {self.config['llama_devices']}")
        if self.config.get('qwen_devices'):
            args.append(f"--qwen_devices {self.config['qwen_devices']}")

        return ' '.join(args)


def get_gpu_memory_stats():
    """Get current GPU memory usage across all visible GPUs.

    Returns dict with per-GPU stats and summary:
    {
        'gpus': [{'id': 0, 'allocated_gb': 12.5, 'reserved_gb': 14.2, 'free_gb': 65.8, 'total_gb': 80.0}, ...],
        'total_allocated_gb': 25.0,
        'total_reserved_gb': 28.4,
        'total_free_gb': 131.6,
        'peak_allocated_gb': 30.2
    }
    """
    if not torch.cuda.is_available():
        return {}

    stats = {'gpus': []}
    total_allocated = 0.0
    total_reserved = 0.0
    total_free = 0.0
    peak_allocated = 0.0

    for device_id in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(device_id) / 1e9
        reserved = torch.cuda.memory_reserved(device_id) / 1e9
        total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
        free = total - reserved
        peak = torch.cuda.max_memory_allocated(device_id) / 1e9

        stats['gpus'].append({
            'id': device_id,
            'allocated_gb': round(allocated, 2),
            'reserved_gb': round(reserved, 2),
            'free_gb': round(free, 2),
            'total_gb': round(total, 2),
            'utilization_pct': round(100 * reserved / total, 1),
        })

        total_allocated += allocated
        total_reserved += reserved
        total_free += free
        peak_allocated += peak

    stats['total_allocated_gb'] = round(total_allocated, 2)
    stats['total_reserved_gb'] = round(total_reserved, 2)
    stats['total_free_gb'] = round(total_free, 2)
    stats['peak_allocated_gb'] = round(peak_allocated, 2)

    return stats


def log_gpu_memory(prefix="", reset_peak=False):
    """Log GPU memory usage with optional prefix. Optionally reset peak stats."""
    stats = get_gpu_memory_stats()
    if not stats:
        return

    # Per-GPU summary
    gpu_summary = ", ".join([
        f"GPU{g['id']}:{g['allocated_gb']:.1f}GB({g['utilization_pct']:.0f}%)"
        for g in stats['gpus']
    ])

    # Overall summary
    total_msg = (
        f"{prefix}[GPU Memory] {gpu_summary} | "
        f"Total: {stats['total_allocated_gb']:.1f}GB allocated, "
        f"{stats['total_reserved_gb']:.1f}GB reserved, "
        f"{stats['total_free_gb']:.1f}GB free, "
        f"Peak: {stats['peak_allocated_gb']:.1f}GB"
    )
    print(total_msg)

    if reset_peak:
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_id)

    return stats


class DDPManager:
    """Manages Distributed Data Parallel training setup and cleanup."""

    def __init__(self):
        self.initialized = False
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = None
        self.is_main_process = True

    def initialize(self, backend='nccl', timeout_mins=30):
        """Initialize distributed training environment.

        Args:
            backend: Communication backend ('nccl' for GPUs, 'gloo' for CPUs)
            timeout_mins: Timeout for collective operations in minutes

        Returns:
            True if successfully initialized, False otherwise
        """
        # Check if we're in a distributed environment
        if 'WORLD_SIZE' not in os.environ:
            print("DDP: Not in distributed environment (WORLD_SIZE not set)")
            return False

        if not torch.cuda.is_available() and backend == 'nccl':
            print("DDP: CUDA not available, falling back to gloo backend")
            backend = 'gloo'

        try:
            # Get distributed training parameters
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            self.rank = int(os.environ.get('RANK', 0))
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

            # Set device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f'cuda:{self.local_rank}')
            else:
                self.device = torch.device('cpu')

            # Initialize process group
            if self.world_size > 1:
                import datetime
                timeout = datetime.timedelta(minutes=timeout_mins)
                dist.init_process_group(
                    backend=backend,
                    init_method='env://',
                    world_size=self.world_size,
                    rank=self.rank,
                    timeout=timeout
                )
                self.initialized = True
                self.is_main_process = (self.rank == 0)

                if self.is_main_process:
                    print(f"DDP: Initialized with {self.world_size} processes")
                    print(f"DDP: Current process - Rank: {self.rank}, Local Rank: {self.local_rank}")
                    print(f"DDP: Device: {self.device}")

                # Synchronize all processes
                self.barrier()
                return True
            else:
                print("DDP: Single process mode (WORLD_SIZE=1)")
                return False

        except Exception as e:
            print(f"DDP: Initialization failed: {e}")
            return False

    def wrap_model(self, model, device_ids=None, output_device=None, find_unused_parameters=False):
        """Wrap a model with DistributedDataParallel.

        Args:
            model: The model to wrap
            device_ids: CUDA devices for this process (default: [local_rank])
            output_device: Device for output (default: local_rank)
            find_unused_parameters: Whether to find unused parameters in backward pass

        Returns:
            DDP-wrapped model or original model if not distributed
        """
        if not self.initialized:
            return model

        if device_ids is None and torch.cuda.is_available():
            device_ids = [self.local_rank]
        if output_device is None and torch.cuda.is_available():
            output_device = self.local_rank

        # Move model to device first
        model = model.to(self.device)

        # Wrap with DDP
        model = DDP(
            model,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=find_unused_parameters
        )

        if self.is_main_process:
            print(f"DDP: Model wrapped for distributed training")

        return model

    def get_dataloader(self, dataset, batch_size, shuffle=True, num_workers=4, **kwargs):
        """Create a DataLoader with DistributedSampler if in distributed mode.

        Args:
            dataset: The dataset to load
            batch_size: Batch size PER GPU
            shuffle: Whether to shuffle (only used if not distributed)
            num_workers: Number of data loading workers
            **kwargs: Additional arguments for DataLoader

        Returns:
            DataLoader configured for distributed training
        """
        if self.initialized:
            # Use DistributedSampler for DDP
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
            # Don't shuffle in DataLoader when using DistributedSampler
            shuffle = False
        else:
            sampler = None

        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            **kwargs
        )

        return dataloader

    def set_epoch(self, epoch):
        """Set epoch for DistributedSampler to ensure different shuffling each epoch.

        Args:
            epoch: Current epoch number
        """
        if self.initialized and hasattr(self, 'dataloader') and hasattr(self.dataloader.sampler, 'set_epoch'):
            self.dataloader.sampler.set_epoch(epoch)

    def all_reduce(self, tensor, op=dist.ReduceOp.SUM):
        """All-reduce a tensor across all processes.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation (SUM, PRODUCT, MIN, MAX)

        Returns:
            Reduced tensor
        """
        if self.initialized:
            dist.all_reduce(tensor, op=op)
        return tensor

    def all_gather(self, tensor_list, tensor):
        """All-gather tensors from all processes.

        Args:
            tensor_list: List to store gathered tensors
            tensor: Tensor to gather

        Returns:
            List of gathered tensors
        """
        if self.initialized:
            dist.all_gather(tensor_list, tensor)
        else:
            tensor_list[0] = tensor
        return tensor_list

    def barrier(self):
        """Synchronize all processes."""
        if self.initialized:
            dist.barrier()

    def cleanup(self):
        """Clean up distributed training."""
        if self.initialized:
            dist.destroy_process_group()
            self.initialized = False

    @property
    def should_save(self):
        """Whether this process should save checkpoints (only main process)."""
        return self.is_main_process

    @property
    def should_log(self):
        """Whether this process should log to console (only main process)."""
        return self.is_main_process

    def get_world_size(self):
        """Get total number of processes."""
        return self.world_size

    def get_rank(self):
        """Get rank of current process."""
        return self.rank

    def scale_loss(self, loss):
        """Scale loss by world size for correct gradient averaging.

        Args:
            loss: The loss tensor

        Returns:
            Scaled loss
        """
        if self.initialized and self.world_size > 1:
            return loss / self.world_size
        return loss

    def print(self, *args, **kwargs):
        """Print only from main process."""
        if self.should_log:
            print(*args, **kwargs)


def initialize_ddp_from_elastic_config(elastic_config):
    """Initialize DDP based on ElasticGPUConfig settings.

    Args:
        elastic_config: ElasticGPUConfig instance

    Returns:
        DDPManager instance (initialized if DDP is enabled)
    """
    ddp_manager = DDPManager()

    config = elastic_config.config
    if config.get('ddp', False) and config.get('gpu_count', 0) > 1:
        # Set up environment variables if not already set
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = str(config['gpu_count'])
            os.environ['RANK'] = os.environ.get('RANK', '0')
            os.environ['LOCAL_RANK'] = os.environ.get('LOCAL_RANK', '0')

        # Initialize DDP
        if ddp_manager.initialize():
            print(f"DDP successfully initialized with {ddp_manager.world_size} GPUs")
        else:
            print("DDP initialization failed, falling back to single GPU")

    return ddp_manager


def suggest_batch_size_adjustment(current_batch_size, peak_allocated_gb, after_forward_pass=False, target_utilization=0.75):
    """Suggest batch size adjustment based on GPU memory utilization AFTER forward passes.

    IMPORTANT: Only call this AFTER at least one forward+backward pass to get accurate
    activation memory estimates. Pre-forward memory is misleading (model memory != activation memory)!

    Args:
        current_batch_size: Current batch size
        peak_allocated_gb: Peak allocated memory from get_gpu_memory_stats()
        after_forward_pass: Whether we've run at least one forward+backward pass
        target_utilization: Target GPU utilization (0.0-1.0), default 0.75 (75% - conservative)

    Returns:
        Tuple of (suggested_batch_size, reason_message)
    """
    stats = get_gpu_memory_stats()
    if not stats or not stats['gpus']:
        return current_batch_size, "No GPU stats available"

    # Never suggest changes before we've seen actual activation memory
    if not after_forward_pass:
        return current_batch_size, "Waiting for forward pass to measure activation memory"

    # Use peak allocated (not reserved) to account for actual usage
    total_capacity_gb = sum(g['total_gb'] for g in stats['gpus'])
    peak_util = peak_allocated_gb / total_capacity_gb if total_capacity_gb > 0 else 0

    # Very conservative safety margins (activation memory can be 3-5x model memory!)
    if peak_util > 0.90:
        # Dangerous territory - reduce significantly
        suggested = max(1, int(current_batch_size * 0.5))
        return suggested, f"CRITICAL: Peak memory {peak_util:.1%}, reduce batch size immediately"
    elif peak_util > 0.80:
        # High utilization - reduce conservatively
        suggested = max(1, int(current_batch_size * 0.75))
        return suggested, f"High peak memory ({peak_util:.1%}), reducing batch size for safety"
    elif peak_util < target_utilization - 0.15 and peak_allocated_gb < total_capacity_gb * 0.6:
        # Only suggest increase if we're well below target AND have significant headroom
        suggested = int(current_batch_size * 1.2)  # Conservative 20% increase
        return suggested, f"Low peak memory ({peak_util:.1%}), can cautiously increase batch size"
    else:
        return current_batch_size, f"Current peak memory ({peak_util:.1%}) is in safe range"


def _align_optimizer_state_to_param_devices(optimizer):
    """Ensure optimizer state tensors live on the same device as their params (multi-GPU safety)."""
    try:
        for param, st in optimizer.state.items():
            if not isinstance(st, dict):
                continue
            pdev = getattr(param, "device", None)
            if pdev is None:
                continue
            for k, v in list(st.items()):
                try:
                    if torch.is_tensor(v) and v.device != pdev:
                        st[k] = v.to(pdev, non_blocking=True)
                except Exception:
                    pass
    except Exception:
        pass
from latentwire.losses import (
    k_token_ce_from_prefix,
    kd_first_k_prefix_vs_text,
    kd_hidden_states_first_k,
)

DEFAULT_SEED = 42
DEFAULT_ANSWER_PREFIX = "Answer: "

@contextmanager
def _temp_padding_side(tokenizer, side: str):
    old = getattr(tokenizer, "padding_side", "right")
    try:
        tokenizer.padding_side = side
        yield
    finally:
        try:
            tokenizer.padding_side = old
        except Exception:
            pass

# ---------------------------
# Checkpoint helpers
# ---------------------------

def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    if not save_dir:
        return None
    if os.path.isfile(save_dir):
        return save_dir
    if not os.path.isdir(save_dir):
        return None

    for name in ["state.pt", "last.pt"]:
        p = os.path.join(save_dir, name)
        if os.path.isfile(p):
            return p

    candidates = []
    for fn in os.listdir(save_dir):
        m = re.match(r"state_step(\d+)\.pt$", fn)
        if m:
            candidates.append((int(m.group(1)), os.path.join(save_dir, fn)))
    if candidates:
        candidates.sort()
        return candidates[-1][1]
    return None


def _safe_load(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _maybe_to_device_optimizer_state(optimizer: optim.Optimizer, device: str):
    for p in optimizer.state.values():
        for k, v in p.items():
            if torch.is_tensor(v):
                p[k] = v.to(device)


def _debug_print_optimizer_state_devices(optimizer: optim.Optimizer, limit: int = 8) -> None:
    if getattr(_debug_print_optimizer_state_devices, "_printed", False):
        return
    try:
        lines = []
        for idx, (param, state) in enumerate(optimizer.state.items()):
            param_dev = str(getattr(param, "device", "None"))
            state_devs = {}
            if isinstance(state, dict):
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state_devs[key] = str(value.device)
            lines.append(f"param_dev={param_dev} state_devs={state_devs}")
            if idx + 1 >= limit:
                break
        if lines:
            print("[DEBUG] Optimizer state devices:\n  " + "\n  ".join(lines), flush=True)
    except Exception:
        pass
    _debug_print_optimizer_state_devices._printed = True


def _merge_kv_caches(
    primary: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
    secondary: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
    if primary is None or len(primary) == 0:
        return list(secondary) if secondary else None
    if secondary is None or len(secondary) == 0:
        return list(primary)
    if len(primary) != len(secondary):
        raise ValueError("KV cache length mismatch when merging prefix sources")
    merged: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for (pk, pv), (sk, sv) in zip(primary, secondary):
        merged.append((torch.cat([pk, sk], dim=2), torch.cat([pv, sv], dim=2)))
    return merged


def load_checkpoint(
    path: str,
    encoder: InterlinguaEncoder,
    adapters: Dict[str, Adapter],
    refiner: Optional[LatentRefiner] = None,
    deep_prefix_generators: Optional[Dict[str, DeepPrefixGenerator]] = None,
    coprocessors: Optional[Dict[str, LatentCoprocessor]] = None,
    gist_heads: Optional[Dict[str, GistReconstructionHead]] = None,
    optimizer: Optional[optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    strict: bool = True,
    device: str = "cpu",
    wrappers: Optional[Dict[str, LMWrapper]] = None,
) -> Tuple[int, int]:
    if path and os.path.isfile(path):
        state = _safe_load(path, map_location="cpu")
        ckpt_dir = os.path.dirname(path)
    elif path and os.path.isdir(path):
        state = {}
        ckpt_dir = path
    else:
        state = {}
        ckpt_dir = None

    enc_loaded = False
    refiner_loaded = refiner is None
    adapters_loaded: Dict[str, bool] = {name: False for name in adapters.keys()}
    deep_prefix_loaded: Dict[str, bool] = (
        {name: False for name in (deep_prefix_generators or {}).keys()}
        if deep_prefix_generators else {}
    )
    coprocessor_loaded: Dict[str, bool] = (
        {name: False for name in (coprocessors or {}).keys()}
        if coprocessors else {}
    )
    gist_loaded: Dict[str, bool] = (
        {name: False for name in (gist_heads or {}).keys()}
        if gist_heads else {}
    )
    latent_adapters_loaded: Dict[str, bool] = (
        {name: False for name, wrapper in (wrappers or {}).items() if wrapper.use_latent_adapters}
        if wrappers else {}
    )
    if isinstance(state, dict) and "encoder" in state:
        try:
            encoder.load_state_dict(state["encoder"], strict=strict)
            enc_loaded = True
        except Exception as exc:
            print(f"   -> failed to load encoder weights from state.pt ({exc}); will try .pt files")
            enc_loaded = False

        if enc_loaded:
            for name, adapter in adapters.items():
                key = f"adp_{name}"
                alt_key = {
                    "llama": "adp_llama",
                    "qwen": "adp_qwen",
                }.get(name, key)
                state_key = key if key in state else alt_key
                if state_key in state:
                    try:
                        adapter.load_state_dict(state[state_key], strict=strict)
                        adapters_loaded[name] = True
                    except Exception as exc:
                        print(f"   -> adapter '{name}' from state.pt failed ({exc}); will retry from disk")
                        adapters_loaded[name] = False
                else:
                    print(f"   -> adapter '{name}' missing in state.pt; will retry from disk")
            if deep_prefix_generators:
                for name, generator in deep_prefix_generators.items():
                    key = f"deep_prefix_{name}"
                    if key in state:
                        try:
                            generator.load_state_dict(state[key], strict=strict)
                            deep_prefix_loaded[name] = True
                        except Exception as exc:
                            print(f"   -> deep prefix '{name}' from state.pt failed ({exc}); will retry from disk")
                            deep_prefix_loaded[name] = False
                    else:
                        print(f"   -> deep prefix '{name}' missing in state.pt; will retry from disk")
            if coprocessors:
                for name, module in coprocessors.items():
                    key = f"coprocessor_{name}"
                    if key in state:
                        try:
                            module.load_state_dict(state[key], strict=strict)
                            coprocessor_loaded[name] = True
                        except Exception as exc:
                            print(f"   -> coprocessor '{name}' from state.pt failed ({exc}); will retry from disk")
                            coprocessor_loaded[name] = False
                    else:
                        print(f"   -> coprocessor '{name}' missing in state.pt; will retry from disk")
            if gist_heads:
                for name, head in gist_heads.items():
                    key = f"gist_{name}"
                    if key in state:
                        try:
                            head.load_state_dict(state[key], strict=strict)
                            gist_loaded[name] = True
                        except Exception as exc:
                            print(f"   -> gist head '{name}' from state.pt failed ({exc}); will retry from disk")
                            gist_loaded[name] = False
                    else:
                        print(f"   -> gist head '{name}' missing in state.pt; will retry from disk")
            if wrappers:
                for name, wrapper in wrappers.items():
                    if wrapper.use_latent_adapters:
                        key = f"latent_adapters_{name}"
                        if key in state:
                            try:
                                wrapper.latent_adapters.load_state_dict(state[key], strict=strict)
                                latent_adapters_loaded[name] = True
                            except Exception as exc:
                                print(f"   -> latent adapters '{name}' from state.pt failed ({exc}); will retry from disk")
                                latent_adapters_loaded[name] = False
                        else:
                            print(f"   -> latent adapters '{name}' missing in state.pt; will retry from disk")
        if refiner is not None and "refiner" in state:
            try:
                refiner.load_state_dict(state["refiner"], strict=strict)
                refiner_loaded = True
            except Exception as exc:
                print(f"   -> refiner from state.pt failed ({exc}); will retry from disk")
                refiner_loaded = False
        else:
            refiner_loaded = refiner is None
    deep_prefix_ok = True if not deep_prefix_generators else all(deep_prefix_loaded.values())
    coprocessor_ok = True if not coprocessors else all(coprocessor_loaded.values())
    gist_ok = True if not gist_heads else all(gist_loaded.values())
    latent_adapters_ok = True if not latent_adapters_loaded else all(latent_adapters_loaded.values())
    if enc_loaded and all(adapters_loaded.values()) and refiner_loaded and deep_prefix_ok and coprocessor_ok and gist_ok and latent_adapters_ok:
        suffix = "encoder/adapters"
        if deep_prefix_generators:
            suffix += "/deep_prefix"
        if coprocessors:
            suffix += "/coprocessor"
        if refiner is not None:
            suffix += "/refiner"
        if gist_heads:
            suffix += "/gist"
        if latent_adapters_loaded:
            suffix += "/latent_adapters"
        print(f"   -> loaded {suffix} FROM state.pt")

    if (not enc_loaded or not all(adapters_loaded.values()) or not refiner_loaded or not coprocessor_ok) and ckpt_dir:
        enc_path = os.path.join(ckpt_dir, "encoder.pt")
        missing: List[str] = []
        if not enc_loaded:
            if os.path.isfile(enc_path):
                encoder.load_state_dict(_safe_load(enc_path, map_location=device), strict=strict)
                enc_loaded = True
            else:
                missing.append(enc_path)

        for name, adapter in adapters.items():
            if adapters_loaded.get(name):
                continue
            adapter_path = os.path.join(ckpt_dir, f"adapter_{name}.pt")
            legacy_path = os.path.join(ckpt_dir, f"adapter_{name if name in ('llama','qwen') else name}.pt")
            path_to_use = adapter_path if os.path.isfile(adapter_path) else legacy_path
            if os.path.isfile(path_to_use):
                adapter.load_state_dict(_safe_load(path_to_use, map_location=device), strict=strict)
                adapters_loaded[name] = True
            else:
                missing.append(adapter_path)

        if deep_prefix_generators:
            for name, generator in deep_prefix_generators.items():
                if deep_prefix_loaded.get(name):
                    continue
                prefix_path = os.path.join(ckpt_dir, f"deep_prefix_{name}.pt")
                if os.path.isfile(prefix_path):
                    generator.load_state_dict(_safe_load(prefix_path, map_location=device), strict=strict)
                    deep_prefix_loaded[name] = True
                else:
                    missing.append(prefix_path)
        if coprocessors:
            for name, module in coprocessors.items():
                if coprocessor_loaded.get(name):
                    continue
                coproc_path = os.path.join(ckpt_dir, f"coprocessor_{name}.pt")
                if os.path.isfile(coproc_path):
                    module.load_state_dict(_safe_load(coproc_path, map_location=device), strict=strict)
                    coprocessor_loaded[name] = True
                else:
                    missing.append(coproc_path)

        if refiner is not None and not refiner_loaded:
            refiner_path = os.path.join(ckpt_dir, "refiner.pt")
            if os.path.isfile(refiner_path):
                refiner.load_state_dict(_safe_load(refiner_path, map_location=device), strict=strict)
                refiner_loaded = True
            else:
                missing.append(refiner_path)
        if gist_heads:
            for name, head in gist_heads.items():
                if gist_loaded.get(name):
                    continue
                gist_path = os.path.join(ckpt_dir, f"gist_{name}.pt")
                if os.path.isfile(gist_path):
                    head.load_state_dict(_safe_load(gist_path, map_location=device), strict=strict)
                    gist_loaded[name] = True
                else:
                    missing.append(gist_path)

        if wrappers:
            for name, wrapper in wrappers.items():
                if wrapper.use_latent_adapters and not latent_adapters_loaded.get(name, True):
                    adapters_path = os.path.join(ckpt_dir, f"latent_adapters_{name}.pt")
                    if os.path.isfile(adapters_path):
                        wrapper.latent_adapters.load_state_dict(_safe_load(adapters_path, map_location=device), strict=strict)
                        latent_adapters_loaded[name] = True
                    else:
                        missing.append(adapters_path)

        if missing:
            raise FileNotFoundError(
                "Missing checkpoint artifacts: " + ", ".join(missing)
            )
        else:
            suffix = "encoder/adapters"
            if deep_prefix_generators:
                suffix += "/deep_prefix"
            if refiner is not None:
                suffix += "/refiner"
            if gist_heads:
                suffix += "/gist"
            print(f"   -> loaded {suffix} FROM encoder.pt + adapter_*.pt")

    if optimizer is not None and isinstance(state, dict):
        opt_state = state.get("optimizer", None) or state.get("optim", None)
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
            except ValueError as exc:
                print(f"[WARN] Optimizer state incompatible; continuing with fresh optimizer ({exc})")
            # Important: keep optimizer state tensors on the same device as *their* params.
            # Do NOT mass-move to a single device, because adapters live on different GPUs.
            _align_optimizer_state_to_param_devices(optimizer)
            _debug_print_optimizer_state_devices(optimizer)
            print("   -> restored optimizer state")

    if lr_scheduler is not None and isinstance(state, dict):
        scheduler_state = state.get("lr_scheduler", None)
        if scheduler_state is not None:
            try:
                lr_scheduler.load_state_dict(scheduler_state)
                print("   -> restored lr_scheduler state")
            except Exception as exc:
                print(f"[WARN] LR scheduler state incompatible; continuing with fresh scheduler ({exc})")

    if isinstance(state, dict) and "rng" in state:
        try:
            rng = state["rng"]
            if "torch" in rng and isinstance(rng["torch"], torch.ByteTensor):
                torch.set_rng_state(rng["torch"])
            elif "torch" in rng:
                torch.set_rng_state(torch.tensor(rng["torch"], dtype=torch.uint8))
            if torch.cuda.is_available() and rng.get("cuda"):
                torch.cuda.set_rng_state_all(rng["cuda"])
            print("   -> restored RNG state")
        except Exception as e:
            print(f"   -> RNG restore skipped ({e})")

    epoch = int(state.get("epoch", 0)) if isinstance(state, dict) else 0
    global_step = int(state.get("global_step", 0)) if isinstance(state, dict) else 0
    return epoch, global_step


# ---------------------------
# Small helpers
# ---------------------------

def _to_float(value: Union[torch.Tensor, float, int]) -> float:
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def _parse_device_map(spec: Optional[str]):
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None
    if s.lower() == "auto":
        return "auto"
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, int):
            return {"": int(parsed)}
        if isinstance(parsed, str) and parsed.isdigit():
            return {"": int(parsed)}
        return parsed
    except Exception:
        pass
    if s.isdigit():
        return {"": int(s)}
    return s


@dataclass
class ModelTrainContext:
    name: str
    wrapper: LMWrapper
    adapter: Adapter
    token_ids: torch.Tensor
    first_token_ids: torch.Tensor
    anchor_ids: List[int]
    bos_flag: Optional[bool]
    answer_lengths: torch.Tensor
    anchor_text: str
    anchor_mode: str


def _primary_device(wrapper: LMWrapper) -> torch.device:
    return next(wrapper.model.parameters()).device


def _assert_t0_alignment(tokenizer, answer_prefix: str = "Answer: ", skip_if_chat: bool = False):
    """Sanity check: the first gold token should appear immediately after the anchor."""
    if skip_if_chat:
        return
    try:
        q = "Q: Capital of France?"
        c = "C: Paris is the capital of France."
        g = "Paris"
        prompt = f"{c}\n\n{q}\n{answer_prefix}"

        ids_all = tokenizer.encode(prompt + g, add_special_tokens=False)
        ids_pref = tokenizer.encode(prompt, add_special_tokens=False)
        ids_gold = tokenizer.encode(g, add_special_tokens=False)

        if len(ids_all) == len(ids_pref):  # tokenizer swallowed the answer prefix boundary
            prompt_sp = prompt + " "
            ids_all = tokenizer.encode(prompt_sp + g, add_special_tokens=False)
            ids_pref = tokenizer.encode(prompt_sp, add_special_tokens=False)

        assert ids_gold, "gold tokenized to empty"
        assert len(ids_all) > len(ids_pref), "concatenation produced no new tokens"
        got = ids_all[len(ids_pref)]
        expect = ids_gold[0]
        assert got == expect, f"t=0 mismatch: got {got}, expected {expect}"
        print(f"[OK] t=0 alignment for {getattr(tokenizer, 'name_or_path', 'tokenizer')}")
    except Exception as exc:
        print(f"[WARN] t=0 alignment failed: {exc}")


def _render_chat_prompt(tokenizer, user_text: str, system_prompt: Optional[str]) -> str:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if rendered:
            return rendered
    except Exception:
        pass
    system_block = f"System: {system_prompt}\n" if system_prompt else ""
    return f"{system_block}User: {user_text}\nAssistant:"


def _answer_lengths(token_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        lengths = torch.full((token_ids.size(0),), token_ids.size(1), device=token_ids.device)
    else:
        lengths = token_ids.ne(int(pad_id)).sum(dim=1)
    return lengths


def main():
    # Check PyTorch availability first
    if not PYTORCH_AVAILABLE:
        print("\n" + "="*60)
        print("ERROR: PyTorch is not properly installed or configured.")
        print("="*60)
        print(f"\nOriginal error: {PYTORCH_IMPORT_ERROR}")
        print("\nPlease install PyTorch by running:")
        print("  pip install torch torchvision torchaudio")
        print("\nOr visit https://pytorch.org for platform-specific instructions.")
        print("="*60 + "\n")
        import sys
        sys.exit(1)

    ap = argparse.ArgumentParser()
    # Models & data
    ap.add_argument("--llama_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--qwen_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--llama_device_map", type=str, default=None,
                    help=("Device map for the Llama wrapper (e.g., 0, 'auto', or JSON dict). "
                          "Pins the model to a subset of GPUs when running multi-model training."))
    ap.add_argument("--qwen_device_map", type=str, default=None,
                    help="Device map for the Qwen wrapper (e.g., 1, 'auto', or JSON dict).")
    ap.add_argument("--require_cuda", type=str, default="yes", choices=["yes", "no"],
                    help="Abort immediately if CUDA is not available (default: yes).")
    ap.add_argument("--dataset", type=str, default="hotpot", choices=["hotpot", "squad", "squad_v2"])
    ap.add_argument("--models", type=str, default="llama,qwen",
                    help="Comma-separated subset of models to train (subset of llama,qwen).")
    ap.add_argument("--hotpot_config", type=str, default="fullwiki")
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=1,
                    help="Number of micro-batches to accumulate before an optimizer step.")
    ap.add_argument("--elastic_gpu", action="store_true",
                    help="Enable elastic GPU configuration that adapts to available hardware.")
    ap.add_argument("--elastic_base_batch", type=int, default=64,
                    help="Base batch size for elastic GPU mode (default: 64).")
    ap.add_argument("--elastic_target_util", type=float, default=0.75,
                    help="Target GPU memory utilization for elastic mode (default: 0.75).")

    # Repro / randomness
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global seed for RNGs & epoch permutations.")
    ap.add_argument("--data_seed", type=int, default=DEFAULT_SEED, help="Seed for picking dataset subset.")

    # Optimized DataLoader settings
    ap.add_argument("--use_optimized_dataloader", action="store_true",
                    help="Use optimized multi-worker DataLoader with caching and prefetching.")
    ap.add_argument("--num_dataloader_workers", type=int, default=4,
                    help="Number of worker processes for DataLoader (0 for main thread, -1 for auto).")
    ap.add_argument("--dataloader_prefetch_factor", type=int, default=2,
                    help="Number of batches to prefetch per worker.")
    ap.add_argument("--dataloader_cache_tokenization", action="store_true",
                    help="Cache tokenized samples to disk for faster subsequent epochs.")
    ap.add_argument("--dataloader_pin_memory", action="store_true",
                    help="Use pinned memory for faster GPU transfers.")

    # Interlingua / encoder
    ap.add_argument("--latent_len", type=int, default=8)
    ap.add_argument("--latent_shared_len", type=int, default=None,
                    help="Optional explicit shared latent length; overrides derived value.")
    ap.add_argument("--latent_private_len", type=int, default=0,
                    help="Per-model private latent length (combined with shared length).")
    ap.add_argument("--d_z", type=int, default=256)
    ap.add_argument("--max_bytes", type=int, default=512)
    ap.add_argument("--encoder_type", type=str, default="byte", choices=["byte", "simple-st", "stq"])
    ap.add_argument("--encoder_use_chat_template", action="store_true",
                    help="Wrap encoder input with a neutral chat-style header (SimpleEncoder only).")
    ap.add_argument("--encoder_backbone", type=str, default=None,
                    help="Optional SentenceTransformer backbone when --encoder_type=simple-st")
    ap.add_argument("--hf_encoder_id", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="HF encoder id for --encoder_type=stq (frozen).")
    ap.add_argument("--max_enc_tokens", type=int, default=1024,
                    help="Max source tokens for the HF encoder when --encoder_type=stq.")
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="Freeze encoder parameters (e.g., during Stage B prefix-tuning).")
    ap.add_argument("--use_chat_template", action="store_true",
                    help="Apply tokenizer.apply_chat_template when constructing teacher scaffolds and anchors.")

    # Training & stability
    ap.add_argument("--max_answer_tokens", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scale_l2", type=float, default=0.05,
                    help="L2 penalty weight to keep adapter.scale near 1.0; set 0 to disable.")
    ap.add_argument("--adapter_rms_l2", type=float, default=0.0,
                    help="Optional penalty to pull RAW adapter RMS toward input embedding RMS.")
    ap.add_argument("--max_grad_norm", type=float, default=1.0,
                    help="Clip grad norm to this value (set <=0 to disable).")
    ap.add_argument("--grad_diag_interval", type=int, default=0,
                    help="If >0, compute gradient-norm diagnostics every N global steps (0 disables).")
    ap.add_argument("--grad_diag_components", type=str,
                    default="tf,first,kce,kd,align,latent_align,latent_prefix_align",
                    help="Comma-separated loss names to include in gradient diagnostics (e.g., 'tf,first,kd').")
    ap.add_argument("--diagnostic_log", type=str, default="",
                    help="Optional JSONL path to append diagnostic summaries each log interval.")
    ap.add_argument("--adapter_freeze_scale", action="store_true",
                    help="If set, fix adapter.scale at 1.0 (no learning).")
    ap.add_argument("--first_token_ce_weight", type=float, default=0.5,
                    help="Weight for the first-token CE objective; 0 disables.")
    ap.add_argument("--first_token_ce_schedule", type=str, default="none", choices=["none", "cosine", "warmup"],
                    help="Optional schedule for first-token CE weights (default: none; warmup=linear 0→weight over 200 steps).")
    ap.add_argument("--first_token_entropy_weight", type=float, default=0.0,
                    help="If >0, adds a negative-entropy penalty on latent first-token logits to discourage mode collapse.")
    ap.add_argument("--first_token_ce_peak", type=float, default=None,
                    help="Peak first-token CE weight during warmup when using a schedule.")
    ap.add_argument("--first_token_ce_warmup_frac", type=float, default=0.4,
                    help="Fraction of total steps to hold the peak first-token CE weight before cosine decay.")
    ap.add_argument("--first_token_autoscale", type=str, default="yes", choices=["yes", "no"],
                    help="If 'yes', dynamically boost first-token CE weight when latent first-token loss stays larger than the teacher-forced loss.")
    ap.add_argument("--train_append_bos_after_prefix", type=str, default="no",
                    choices=["auto","yes","no"],
                    help="Controls BOS appending when computing first-token CE (train).")
    ap.add_argument("--adapter_hidden_mult", type=int, default=2,
                    help="Hidden width multiplier for the adapter MLP.")
    ap.add_argument("--adapter_dropout", type=float, default=0.0,
                    help="Dropout probability for adapter MLP hidden states (0 disables).")
    ap.add_argument("--adapter_colorize", action="store_true",
                    help="If set, add per-dim colorizer to align adapter outputs with LM embeddings.")
    ap.add_argument("--no_adapter_metadata", action="store_false", dest="adapter_metadata",
                    help="Disable positional/answer-length metadata injection in the adapter.")
    ap.set_defaults(adapter_metadata=True)
    ap.add_argument("--manifold_stat_weight", type=float, default=0.0,
                    help="Optional weight for μ/σ matching loss (1e-3..5e-3 recommended).")
    ap.add_argument("--state_kd_weight", type=float, default=0.0,
                    help="Weight for hidden-state KD on first K steps (0 disables).")
    ap.add_argument("--state_kd_layers", type=str, default="0,1,2",
                    help="Comma-separated transformer layer indices for hidden-state KD.")
    ap.add_argument("--use_gist_head", action="store_true",
                    help="Enable gist reconstruction head that rebuilds teacher prompts from the latent wire.")
    ap.add_argument("--gist_target_len", type=int, default=48,
                    help="Number of tokens to reconstruct with the gist head.")
    ap.add_argument("--gist_hidden", type=int, default=512,
                    help="Hidden dimension inside the gist head MLP.")
    ap.add_argument("--gist_layers", type=int, default=2,
                    help="Number of residual MLP blocks in the gist head.")
    ap.add_argument("--gist_dropout", type=float, default=0.1,
                    help="Dropout applied inside the gist head.")
    ap.add_argument("--gist_weight", type=float, default=0.0,
                    help="Weight for the gist reconstruction loss (0 disables).")
    ap.add_argument("--gist_mask_prob", type=float, default=0.15,
                    help="Probability of masking each gist target token when computing the loss (simulates gist masking).")
    ap.add_argument("--use_coprocessor", action="store_true",
                    help="Enable latent coprocessor that injects KV deltas derived from the latent wire.")
    ap.add_argument("--coprocessor_len", type=int, default=1,
                    help="Number of KV positions per layer produced by the coprocessor (default: 1).")
    ap.add_argument("--coprocessor_width", type=int, default=256,
                    help="Hidden width of the coprocessor MLP.")
    ap.add_argument("--coprocessor_dropout", type=float, default=0.1,
                    help="Dropout probability inside the coprocessor.")
    ap.add_argument("--coprocessor_kv_scale", type=float, default=0.8,
                    help="Scale factor applied to coprocessor-generated KV vectors.")
    ap.add_argument("--coprocessor_pool", type=str, default="mean",
                    help="Pooling strategy for latent inputs before the coprocessor MLP (mean|first|max).")
    ap.add_argument("--coprocessor_heads", type=str, default="",
                    help="Optional comma-separated override for KV head counts per layer (blank = use model defaults).")
    # PEFT toggles
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_firstN", type=int, default=None,
                    help="If set, only the first N transformer layers keep LoRA weights trainable.")
    ap.add_argument("--lora_target_modules", type=str, default="auto",
                    help="Comma-separated module list or presets (auto, attn_mlp_firstN:12, ...).")
    ap.add_argument("--use_prefix", action="store_true")
    ap.add_argument("--prefix_tokens", type=int, default=16)
    ap.add_argument("--prefix_projection", action="store_true")
    ap.add_argument("--peft_prefix_all_layers", type=str, default="yes",
                    help="yes/no toggle to apply prefix adapters across every transformer layer.")
    # Multi-depth latent adapters (IAA-style)
    ap.add_argument("--use_latent_adapters", action="store_true",
                    help="Enable multi-depth latent adapters (IAA-style) that inject latent at multiple layers.")
    ap.add_argument("--latent_adapter_layers", type=str, default="8,16,24",
                    help="Comma-separated list of layer indices for latent adapters (default: 8,16,24 for even spacing in 32-layer models, matching IAA paper).")
    ap.add_argument("--latent_adapter_heads", type=int, default=8,
                    help="Number of attention heads in latent adapter cross-attention.")
    ap.add_argument("--latent_adapter_dropout", type=float, default=0.1,
                    help="Dropout rate for latent adapters.")
    ap.add_argument("--use_deep_prefix", action="store_true",
                    help="Enable learned per-layer prefixes derived from the latent interlingua.")
    ap.add_argument("--deep_prefix_len", type=int, default=None,
                    help="Number of latent slots used to seed the deep prefixes (defaults to shared latent length).")
    ap.add_argument("--deep_prefix_dropout", type=float, default=0.1,
                    help="Dropout probability applied inside the deep prefix generator.")
    # K-token supervision + KD
    ap.add_argument("--K", type=int, default=4, help="Number of early tokens to supervise (A1/A2).")
    ap.add_argument("--adaptive_k_start", type=int, default=None,
                    help="Optional starting K for curriculum (defaults to --K).")
    ap.add_argument("--adaptive_k_end", type=int, default=None,
                    help="Optional final K for curriculum (defaults to --K).")
    ap.add_argument("--latent_keep_start", type=float, default=1.0,
                    help="Starting keep probability for latent dropout curriculum (1.0 = keep all).")
    ap.add_argument("--latent_keep_end", type=float, default=1.0,
                    help="Final keep probability for latent dropout curriculum.")
    ap.add_argument("--latent_keep_power", type=float, default=1.0,
                    help="Exponent controlling schedule shape (1.0 linear, >1 later drop).")
    ap.add_argument("--warmup_text_latent_steps", type=int, default=0,
                    help="Number of initial optimizer steps to alternate text vs latent teacher forcing (0 disables).")
    ap.add_argument("--warmup_text_latent_epochs", type=float, default=0.0,
                    help="Alternative way to specify warm-up length via epochs (e.g., 1.0 for first epoch).")
    ap.add_argument("--warmup_align_tokens", type=int, default=1,
                    help="During warm-up text steps, align this many leading answer tokens (0 disables alignment).")
    ap.add_argument("--warmup_align_weight", type=float, default=1.0,
                    help="Weight for the warm-up embedding alignment loss (text-mode steps only).")
    ap.add_argument("--warmup_text_teacher_weight", type=float, default=1.0,
                    help="Weight for the teacher-forced text loss during warm-up text steps.")
    ap.add_argument("--warmup_text_latent_weight", type=float, default=0.2,
                    help="Multiplier applied to latent losses on warm-up text batches (0 disables latent CE on those steps).")
    ap.add_argument("--warmup_text_latent_weight_end", type=float, default=1.0,
                    help="Target latent-loss multiplier once warm-up completes (tail text batches use this value).")
    ap.add_argument("--warmup_tail_prob", type=float, default=0.0,
                    help="After the warm-up window, continue sampling text batches with this probability (0 disables).")
    ap.add_argument("--latent_align_weight", type=float, default=0.0,
                    help="Weight for matching latent prefix embeddings to the teacher's first token embedding during latent batches.")
    ap.add_argument("--latent_prefix_align_weight", type=float, default=0.0,
                    help="Weight for aligning the entire latent prefix to the teacher's token embeddings (first slots).")
    ap.add_argument("--latent_align_metric", type=str, default="cosine",
                    choices=["mse", "cosine", "both"],
                    help="Distance metric for latent alignment losses (default cosine).")
    ap.add_argument("--k_ce_weight", type=float, default=0.5,
                    help="Aux weight for K-token CE on first K steps.")
    ap.add_argument("--kd_first_k_weight", type=float, default=1.0,
                    help="Weight for prefix KD vs text teacher (first K steps).")
    ap.add_argument("--kd_tau", type=float, default=2.0,
                    help="Temperature for KD (recommend 1.5–2.0).")
    ap.add_argument("--teacher_llama_id", type=str, default=None,
                    help="Optional hub ID for Llama KD teacher (defaults to --llama_id).")
    ap.add_argument("--teacher_qwen_id", type=str, default=None,
                    help="Optional hub ID for Qwen KD teacher (defaults to --qwen_id).")
    ap.add_argument("--kd_skip_text", action="store_true",
                    help="Skip KD loss entirely whenever training operates in text mode (warm-up/tail batches).")
    ap.add_argument("--latent_refiner_layers", type=int, default=0,
                    help="If >0, use a Transformer refiner with this many layers on latent slots before adapters.")
    ap.add_argument("--latent_refiner_heads", type=int, default=4,
                    help="Number of attention heads for the latent refiner (when enabled).")
    ap.add_argument("--use_latent_refiner", action="store_true",
                    help="Enable latent refiner before adapters (requires latent_refiner_layers > 0).")

    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--sequential_models", action="store_true")
    ap.add_argument("--llama_devices", type=str, default=None,
                    help="Comma-separated CUDA device ids reserved for Llama (e.g., '0,1').")
    ap.add_argument("--qwen_devices", type=str, default=None,
                    help="Comma-separated CUDA device ids reserved for Qwen (e.g., '2,3').")
    ap.add_argument("--gpu_mem_gib", type=float, default=78.0,
                    help="Per-GPU memory budget (GiB) when constraining auto device maps.")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--fp16_mps", action="store_true")

    # Mixed Precision Training
    ap.add_argument("--mixed_precision", type=str, default="no",
                    choices=["no", "fp16", "bf16", "fp8"],
                    help="Enable mixed precision training. bf16 recommended for H100 (better range, no overflow).")
    ap.add_argument("--grad_scaler_init", type=float, default=2**16,
                    help="Initial scale factor for GradScaler (fp16 only). Lower if seeing infs.")
    ap.add_argument("--grad_scaler_growth_interval", type=int, default=2000,
                    help="Number of iterations between scale factor increases.")
    ap.add_argument("--grad_scaler_backoff", type=float, default=0.5,
                    help="Scale factor multiplier when overflow detected (fp16 only).")
    ap.add_argument("--amp_opt_level", type=str, default="O1",
                    choices=["O0", "O1", "O2"],
                    help="AMP optimization level. O1=mixed, O2=almost everything fp16/bf16.")

    ap.add_argument("--warm_anchor_text", type=str, default="",
                    help="Optional anchor tokens AFTER latent prefix during training (text mode).")
    ap.add_argument(
        "--warm_anchor_mode",
        type=str,
        default="auto",
        choices=["auto", "text", "chat", "none"],
        help="How to choose the training anchor: 'text' uses --warm_anchor_text, 'chat' injects the"
             " tokenizer's assistant header, 'none' disables anchors, 'auto' matches legacy behaviour.",
    )
    ap.add_argument(
        "--max_anchor_tokens",
        type=int,
        default=32,
        help="Upper bound on the number of discrete anchor tokens to inject (prevents chat headers"
             " from ballooning the prefix). Set <=0 to disable truncation.",
    )
    ap.add_argument("--debug", action="store_true")

    # Checkpointing
    ap.add_argument("--save_dir", type=str, default="./ckpt")
    ap.add_argument("--save_every", type=int, default=0, help="If >0, save the latest checkpoint every N steps and prune old files.")
    ap.add_argument("--resume_from", type=str, default="", help="Path to state.pt or directory containing checkpoints. Empty string disables resume.")
    ap.add_argument("--auto_resume", action="store_true")
    ap.add_argument("--no_load_optimizer", action="store_true")
    ap.add_argument("--no_load_lr_scheduler", action="store_true")
    ap.add_argument("--reset_epoch", action="store_true", help="When resuming, ignore stored epoch/step counters and restart from zero.")

    # Training stats (for eval-time calibration)
    ap.add_argument("--save_training_stats", action="store_true", help="Record running mean of prefix RMS per model and save to training_stats.json")

    # Milestone 0: Baseline Verification Mode
    ap.add_argument("--baseline_verification", action="store_true",
                    help="[Milestone 0] Disable all advanced features (deep prefix, latent adapters, KD, gist) "
                         "for baseline verification. Keeps only encoder + LoRA trainable.")

    args = ap.parse_args()

    # Apply baseline verification overrides if requested
    if args.baseline_verification:
        print("\n" + "="*80)
        print("BASELINE VERIFICATION MODE (Milestone 0)")
        print("="*80)
        print("Disabling advanced features:")
        print("  - Deep prefix: OFF")
        print("  - Latent adapters: OFF")
        print("  - Knowledge distillation: OFF (all KD weights → 0)")
        print("  - Gist reconstruction head: OFF")
        print("  - State KD: OFF")
        print("Keeping enabled:")
        print("  - Encoder: TRAINABLE")
        print("  - LoRA: ENABLED (if --use_lora is set)")
        print("  - Basic adapters: TRAINABLE")
        print("="*80 + "\n")

        # Disable advanced features
        args.use_deep_prefix = False
        args.use_latent_adapters = False
        args.use_gist_head = False

        # Zero out all KD weights
        args.kd_first_k_weight = 0.0
        args.state_kd_weight = 0.0
        args.gist_weight = 0.0

        # Ensure encoder stays trainable
        args.freeze_encoder = False
    # global runtime patches
    patch_dataloader_defaults()
    apply_anchor_normalization(args)

    grad_diag_interval = max(0, int(getattr(args, "grad_diag_interval", 0)))
    grad_diag_components = [
        token.strip().lower()
        for token in (getattr(args, "grad_diag_components", "") or "").split(",")
        if token.strip()
    ]
    grad_diag_components = list(dict.fromkeys(grad_diag_components))  # preserve order, dedupe
    grad_diag_component_set = set(grad_diag_components)
    diagnostic_log_path = (getattr(args, "diagnostic_log", "") or "").strip()
    if diagnostic_log_path:
        diag_dir = os.path.dirname(diagnostic_log_path)
        if diag_dir:
            os.makedirs(diag_dir, exist_ok=True)
        else:
            os.makedirs(".", exist_ok=True)
        try:
            with open(diagnostic_log_path, "a") as _f:
                pass
        except Exception as exc:
            print(f"[WARN] Unable to open diagnostic log '{diagnostic_log_path}': {exc}")
            diagnostic_log_path = ""

    # Device + dtype
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if args.require_cuda.lower() != "no":
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            print("[FATAL] CUDA unavailable. torch.cuda.is_available()=", torch.cuda.is_available(),
                  "count=", torch.cuda.device_count())
            print("  CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
            try:
                subprocess.run(["nvidia-smi"], check=False)
            except Exception as exc:
                print("  nvidia-smi not runnable:", exc)
            raise SystemExit(2)

    # Elastic GPU configuration
    if args.elastic_gpu and device == "cuda":
        print("\n" + "="*70)
        print("ELASTIC GPU MODE ENABLED")
        print("="*70)

        # Detect model size based on which models we're using
        model_size_gb = 14.0  # Default for Llama-8B
        if "qwen" in args.enabled_models.lower() and "llama" not in args.enabled_models.lower():
            model_size_gb = 13.0  # Qwen-7B only
        elif "qwen" in args.enabled_models.lower() and "llama" in args.enabled_models.lower():
            model_size_gb = 27.0  # Both models

        # Initialize elastic configuration
        elastic_config = ElasticGPUConfig(
            base_batch_size=args.elastic_base_batch,
            model_size_gb=model_size_gb,
            target_util=args.elastic_target_util
        )

        # Print configuration
        elastic_config.print_config()

        # Get optimal settings
        optimal = elastic_config.get_optimal_config(
            dataset_size=args.samples,
            target_steps=100  # Target ~100 steps per epoch for good convergence
        )

        # Override args with elastic settings
        print("\nApplying elastic configuration...")
        original_batch = args.batch_size
        original_accum = args.grad_accum_steps

        args.batch_size = optimal['batch_size']
        args.grad_accum_steps = optimal['grad_accum_steps']

        if optimal.get('llama_devices') and not args.llama_devices:
            args.llama_devices = optimal['llama_devices']
        if optimal.get('qwen_devices') and not args.qwen_devices:
            args.qwen_devices = optimal['qwen_devices']

        print(f"  Batch size: {original_batch} → {args.batch_size}")
        print(f"  Grad accum: {original_accum} → {args.grad_accum_steps}")
        print(f"  Effective batch: {args.batch_size * args.grad_accum_steps}")

        # Initialize DDP if suggested
        ddp_manager = None
        if optimal.get('ddp'):
            print("\nInitializing DDP support...")
            ddp_manager = initialize_ddp_from_elastic_config(elastic_config)
            if ddp_manager.initialized:
                print(f"  DDP enabled with {ddp_manager.world_size} processes")
                # Adjust batch size for DDP (batch_size is per-GPU)
                args.batch_size = args.batch_size // ddp_manager.world_size
                print(f"  Adjusted batch size per GPU: {args.batch_size}")
            else:
                print("  DDP initialization failed, using single GPU mode")

        print("="*70 + "\n")
    else:
        # No elastic GPU mode, but still check for manual DDP setup
        ddp_manager = DDPManager()
        if 'WORLD_SIZE' in os.environ and device == "cuda":
            if ddp_manager.initialize():
                print(f"DDP manually initialized with {ddp_manager.world_size} processes")
                device = ddp_manager.device
            else:
                print("Manual DDP initialization failed, using single GPU mode")

    env_dtype = os.environ.get("TORCH_DTYPE")
    if env_dtype:
        dtype_lookup = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        dtype = dtype_lookup.get(env_dtype.lower(), torch.bfloat16 if device == "cuda" else torch.float32)
    elif device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16 if args.fp16_mps else torch.float32
    else:
        dtype = torch.float32

    # Enable kernel optimizations for better GPU utilization
    if device == "cuda":
        # Enable cuDNN autotuner to select best algorithms for your hardware
        torch.backends.cudnn.benchmark = True
        # Enable FlashAttention-2 and memory-efficient attention kernels
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)  # Disable slower fallback
            print("[Optimization] Enabled FlashAttention-2 and memory-efficient kernels")
        except AttributeError:
            print("[Optimization] FlashAttention APIs not available in this PyTorch version")
        # Enable TF32 for matrix multiplications on Ampere+ GPUs (H100 benefits)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[Optimization] Enabled TF32 for matmul and cuDNN")

    supported_models = ["llama", "qwen"]
    requested_models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    if not requested_models:
        requested_models = supported_models
    for name in requested_models:
        if name not in supported_models:
            raise ValueError(f"Unknown model '{name}'. Choose from {supported_models}.")
    model_keys = requested_models

    grad_accum_steps = max(1, int(args.grad_accum_steps))
    adaptive_k_start = int(args.adaptive_k_start) if args.adaptive_k_start is not None else args.K
    adaptive_k_end = int(args.adaptive_k_end) if args.adaptive_k_end is not None else args.K
    latent_keep_start = float(args.latent_keep_start)
    latent_keep_end = float(args.latent_keep_end)
    latent_keep_power = max(1e-6, float(args.latent_keep_power))
    if args.latent_shared_len is not None:
        latent_shared_len = int(args.latent_shared_len)
        latent_private_len = max(0, int(args.latent_private_len))
        total_latent_len = latent_shared_len + latent_private_len * len(model_keys)
    else:
        latent_private_len = max(0, int(args.latent_private_len))
        total_latent_len = int(args.latent_len)
        latent_shared_len = max(total_latent_len - latent_private_len * len(model_keys), 0)
    total_latent_len = max(total_latent_len, 0)

    # ===== Repro =====
    random.seed(args.seed)
    try:
        import numpy as _np
        _np.random.seed(args.seed)
    except Exception:
        pass
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ===== Data =====
    texts, answers = prepare_training_data(
        dataset=args.dataset,
        samples=args.samples,
        data_seed=args.data_seed,
        hotpot_config=args.hotpot_config,
    )

    # ===== Models =====
    def _build_max_memory(devices_csv: Optional[str], budget_gib: float):
        if devices_csv is None or not torch.cuda.is_available():
            return None
        devs: List[int] = []
        for item in devices_csv.split(","):
            name = item.strip()
            if not name:
                continue
            try:
                dev_id = int(name)
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device id '{name}' in '{devices_csv}'") from exc
            devs.append(dev_id)
        if not devs:
            return None
        max_mem = {}
        for idx in range(torch.cuda.device_count()):
            max_mem[idx] = f"{int(budget_gib)}GiB" if idx in devs else "0GiB"
        return max_mem

    llama_device_map = _parse_device_map(args.llama_device_map)
    qwen_device_map = _parse_device_map(args.qwen_device_map)

    llama_max_memory = _build_max_memory(args.llama_devices, args.gpu_mem_gib)
    qwen_max_memory = _build_max_memory(args.qwen_devices, args.gpu_mem_gib)

    if llama_device_map is None and llama_max_memory is not None and device == "cuda":
        llama_device_map = "auto"
    if qwen_device_map is None and qwen_max_memory is not None and device == "cuda":
        qwen_device_map = "auto"

    # Parse latent adapter layers
    latent_adapter_layers_tuple = tuple(int(x.strip()) for x in args.latent_adapter_layers.split(",") if x.strip())

    llama = None
    if "llama" in model_keys:
        llama = LMWrapper(LMConfig(
            model_id=args.llama_id,
            device=device,
            dtype=dtype,
            load_4bit=args.load_4bit,
            device_map=llama_device_map,
            max_memory=llama_max_memory,
            # Multi-depth latent adapters
            use_latent_adapters=args.use_latent_adapters,
            latent_adapter_layers=latent_adapter_layers_tuple,
            latent_d_z=args.d_z,
            latent_adapter_heads=args.latent_adapter_heads,
            latent_adapter_dropout=args.latent_adapter_dropout,
        ))
    qwen = None
    if "qwen" in model_keys:
        qwen = LMWrapper(LMConfig(
            model_id=args.qwen_id,
            device=device,
            dtype=dtype,
            load_4bit=args.load_4bit,
            device_map=qwen_device_map,
            max_memory=qwen_max_memory,
            # Multi-depth latent adapters
            use_latent_adapters=args.use_latent_adapters,
            latent_adapter_layers=latent_adapter_layers_tuple,
            latent_d_z=args.d_z,
            latent_adapter_heads=args.latent_adapter_heads,
            latent_adapter_dropout=args.latent_adapter_dropout,
        ))

    wrappers_in_use: List[LMWrapper] = [w for w in (llama, qwen) if w is not None]
    for wrapper in wrappers_in_use:
        try:
            if hasattr(wrapper.model.config, "use_cache"):
                wrapper.model.config.use_cache = False
        except Exception:
            pass

    def _collect_trainable(module: nn.Module) -> List[nn.Parameter]:
        return [p for p in module.parameters() if p.requires_grad]

    def _compute_lora_weight_norms(model: nn.Module) -> dict[str, float]:
        """Compute L2 norms of LoRA weights for diagnostic tracking"""
        lora_norms = {}
        try:
            from peft import PeftModel
            if isinstance(model, PeftModel):
                for name, param in model.named_parameters():
                    if 'lora_' in name and param.requires_grad:
                        norm = param.data.norm(2).item()
                        # Simplify name: "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight" -> "L0_q_A"
                        parts = name.split('.')
                        layer_idx = next((p.replace('layers', 'L') for p in parts if 'layers' in p and parts[parts.index(p)+1].isdigit()), '')
                        if layer_idx:
                            layer_idx = layer_idx + parts[parts.index('layers')+1]
                        module_name = next((p for p in parts if p in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']), '')
                        module_name = module_name.replace('_proj', '')
                        lora_type = next((p for p in parts if p in ['lora_A', 'lora_B']), '')
                        lora_type = lora_type.replace('lora_', '')
                        short_name = f"{layer_idx}_{module_name}_{lora_type}" if all([layer_idx, module_name, lora_type]) else name[-20:]
                        lora_norms[short_name] = norm
        except Exception:
            pass
        return lora_norms

    extra_llama_params: List[nn.Parameter] = []
    extra_qwen_params: List[nn.Parameter] = []

    feature_registry = FeatureRegistry(args)
    feature_registry.set_extra("latent_shared_len", latent_shared_len)
    feature_registry.set_extra("latent_private_len", latent_private_len)
    feature_wrappers: Dict[str, LMWrapper] = {}
    if llama is not None:
        feature_wrappers["llama"] = llama
    if qwen is not None:
        feature_wrappers["qwen"] = qwen

    feature_extra_params = feature_registry.apply_post_model_build(feature_wrappers)
    extra_llama_params.extend(feature_extra_params.get("llama", []))
    extra_qwen_params.extend(feature_extra_params.get("qwen", []))
    deep_prefix_generators = feature_registry.state.get("deep_prefix_generators", {})
    latent_adapter_param_map = feature_registry.state.get("latent_adapter_params", {})
    latent_adapter_summaries = feature_registry.state.get("latent_adapter_summaries", {})
    coprocessors = feature_registry.state.get("coprocessors", {})
    coprocessor_summaries = feature_registry.state.get("coprocessor_summaries", {})
    coprocessor_param_bank = feature_registry.state.get("coprocessor_params", {})

    if args.use_prefix:
        prefix_cfg = {
            "tokens": args.prefix_tokens,
            "projection": args.prefix_projection,
            "all_layers": str(args.peft_prefix_all_layers).lower() != "no",
        }
        if llama is not None:
            llama.model = apply_prefix_if_requested(llama.model, prefix_cfg, llama.tokenizer)
            extra_llama_params = _collect_trainable(llama.model)
            llama.model.train()
            llama.input_embed = llama.model.get_input_embeddings()
        if qwen is not None:
            qwen.model = apply_prefix_if_requested(qwen.model, prefix_cfg, qwen.tokenizer)
            extra_qwen_params = _collect_trainable(qwen.model)
            qwen.model.train()
            qwen.input_embed = qwen.model.get_input_embeddings()

    if (feature_registry.has("lora") or args.use_prefix) and not (extra_llama_params or extra_qwen_params):
        raise RuntimeError("No trainable PEFT parameters detected – check LoRA/Prefix flags")

    if llama is not None and qwen is not None:
        print(f"Llama hidden size: {llama.d_model}, Qwen hidden size: {qwen.d_model}")
    elif llama is not None:
        print(f"Llama hidden size: {llama.d_model}")
    elif qwen is not None:
        print(f"Qwen hidden size: {qwen.d_model}")
    try:
        if llama is not None:
            print("[DeviceMap] Llama:", getattr(llama.model, "hf_device_map", None))
        if qwen is not None:
            print("[DeviceMap] Qwen :", getattr(qwen.model, "hf_device_map", None))
    except Exception:
        pass

    # Log GPU memory after model loading
    log_gpu_memory(prefix="[After Model Loading] ")

    strip_anchor_literal = args.warm_anchor_text if args.warm_anchor_text else DEFAULT_ANSWER_PREFIX
    if strip_anchor_literal and not strip_anchor_literal.endswith(" "):
        strip_anchor_literal = strip_anchor_literal + " "

    embed_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    if llama is not None:
        embed_stats["llama"] = llama.embedding_stats()
    if qwen is not None:
        embed_stats["qwen"] = qwen.embedding_stats()

    def _anchor_text_for(wrapper, fallback: str) -> Tuple[str, str]:
        requested = (getattr(args, "warm_anchor_mode", "auto") or "auto").lower()
        mode = requested

        def _ensure_trailing_space(txt: str) -> str:
            if txt and not txt.endswith(" "):
                return txt + " "
            return txt

        if mode == "none":
            return "", "none"
        if mode == "chat":
            header = assistant_header_anchor(wrapper.tokenizer) or ""
            # FIX: Don't append "Answer: " literal in chat mode - breaks first-token contract
            return header, "chat"
        if mode == "text":
            base = fallback or ""
            if not base and args.use_chat_template:
                base = "Answer: "
            return _ensure_trailing_space(base), "text"

        # auto (legacy behaviour): prefer explicit text, otherwise chat header when templates are used
        base = fallback or ""
        if base:
            return _ensure_trailing_space(base), "text"
        if args.use_chat_template:
            anchor = assistant_header_anchor(wrapper.tokenizer) or "Answer: "
            return anchor, "chat"
        return "", "none"

    def _truncate_anchor(wrapper, text: str) -> str:
        max_tok = int(getattr(args, "max_anchor_tokens", 0) or 0)
        if max_tok <= 0 or not text:
            return text
        try:
            ids = wrapper._encode_anchor_text(text)
        except Exception:
            return text
        if len(ids) <= max_tok:
            return text
        kept = ids[:max_tok]
        truncated = wrapper.tokenizer.decode(kept, skip_special_tokens=False)
        if text.endswith(" ") and not truncated.endswith(" "):
            truncated += " "
        print(f"[WARN] Anchor trimmed from {len(ids)} to {len(kept)} tokens for {wrapper.cfg.model_id}")
        return truncated

    anchor_texts: Dict[str, str] = {}
    anchor_modes: Dict[str, str] = {}
    anchor_token_lists: Dict[str, List[int]] = {}
    bos_flags: Dict[str, Optional[bool]] = {}

    for name, wrapper in (('llama', llama), ('qwen', qwen)):
        if wrapper is None:
            continue
        text, mode = _anchor_text_for(wrapper, args.warm_anchor_text)
        if mode == "text":
            text = _truncate_anchor(wrapper, text)
        # In chat mode, text is already the header from _anchor_text_for - don't override
        anchor_texts[name] = text
        anchor_modes[name] = mode

        try:
            skip_chat = bool(args.use_chat_template)
            _assert_t0_alignment(wrapper.tokenizer, text or "Answer: ", skip_if_chat=skip_chat)
        except Exception as exc:
            print(f"[WARN] A0 sanity check skipped/failed for {name}: {exc}")

        # Use the actual anchor text for both text and chat modes
        anchor_tokens_source = text
        anchor_ids = anchor_token_ids(wrapper, anchor_tokens_source) if anchor_tokens_source else []
        anchor_token_lists[name] = anchor_ids
        if anchor_ids:
            print(f"[INFO] {name} anchor tokens: {len(anchor_ids)}")
        flag = bos_policy(args.train_append_bos_after_prefix, anchor_ids)
        if mode == "chat":
            flag = False
        bos_flags[name] = flag

    if 'llama' in anchor_texts and 'qwen' in anchor_texts:
        if anchor_texts['llama'] != anchor_texts['qwen']:
            print("[WARN] Anchor strings differ between models; using Llama variant for shared config.")

    # Update strip_anchor_literal to match the actual anchor in chat mode
    if anchor_modes.get('llama') == 'chat' or anchor_modes.get('qwen') == 'chat':
        # Use llama's header if available, otherwise qwen's
        chat_header = anchor_texts.get('llama') or anchor_texts.get('qwen') or ""
        strip_anchor_literal = chat_header
        print(f"[INFO] Chat mode: updated strip_anchor_literal to header: {repr(strip_anchor_literal)}")

    if args.grad_ckpt:
        if llama is not None:
            llama.enable_gradient_checkpointing()
            try:
                llama.model.config.use_cache = False
            except Exception:
                pass
        if qwen is not None:
            qwen.enable_gradient_checkpointing()
            try:
                qwen.model.config.use_cache = False
            except Exception:
                pass

    # ===== Encoder =====
    def _structure_latents(raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = raw[:, :latent_shared_len] if latent_shared_len > 0 else raw.new_zeros(raw.size(0), 0, raw.size(-1))
        private = {}
        start = latent_shared_len
        for key in model_keys:
            if latent_private_len > 0:
                private[key] = raw[:, start:start + latent_private_len]
            else:
                private[key] = raw.new_zeros(raw.size(0), 0, raw.size(-1))
            start += latent_private_len
        return {"shared": shared, "private": private}

    def _neutral_chat_wrap(s: str) -> str:
        system = "You are a concise QA assistant. Use the context to answer with a short phrase only."
        return f"System: {system}\nUser: {s}\nAssistant:"

    def _maybe_chat_texts(batch_texts: List[str]) -> List[str]:
        if not args.encoder_use_chat_template:
            return batch_texts
        return [_neutral_chat_wrap(t) for t in batch_texts]

    if args.encoder_type == "byte":
        encoder = InterlinguaEncoder(
            d_z=args.d_z,
            latent_shared_len=latent_shared_len,
            latent_private_len=latent_private_len,
            model_keys=tuple(model_keys),
        ).to(device)
        byte_tok = ByteTokenizer(max_bytes=args.max_bytes)

        def encode_fn(batch_texts):
            texts = _maybe_chat_texts(batch_texts)
            z_bytes = collate_bytes(texts, byte_tok, device)
            return encoder(z_bytes, return_components=True)

    elif args.encoder_type == "stq":
        encoder = STQueryEncoder(
            d_z=args.d_z,
            latent_len=total_latent_len,
            hf_encoder_id=(args.hf_encoder_id or "sentence-transformers/all-MiniLM-L6-v2"),
            max_tokens=args.max_enc_tokens,
        ).to(device)

        def encode_fn(batch_texts):
            texts = _maybe_chat_texts(batch_texts)
            raw = encoder(texts)
            return _structure_latents(raw)

    else:
        encoder = SimpleEncoder(
            d_z=args.d_z,
            latent_len=total_latent_len,
            backbone=(args.encoder_backbone or "sentence-transformers/all-MiniLM-L6-v2"),
        ).to(device)

        def encode_fn(batch_texts):
            texts = _maybe_chat_texts(batch_texts)
            raw = encoder(texts)
            return _structure_latents(raw)

    latent_refiner = None
    if getattr(args, "use_latent_refiner", False) and int(args.latent_refiner_layers) > 0:
        latent_refiner = LatentRefiner(
            d_z=args.d_z,
            num_layers=int(args.latent_refiner_layers),
            num_heads=int(max(args.latent_refiner_heads, 1)),
        ).to(device)
        latent_refiner.train()
    elif getattr(args, "use_latent_refiner", False) and int(args.latent_refiner_layers) <= 0:
        print("[WARN] --use_latent_refiner specified but --latent_refiner_layers <= 0; refiner disabled.")

    if args.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad_(False)
        print("[INFO] Encoder frozen (--freeze_encoder)")

    # ===== Adapters =====
    adapters: Dict[str, Adapter] = {}
    gist_heads: Dict[str, GistReconstructionHead] = {}
    adp_llama: Optional[Adapter] = None
    adp_qwen: Optional[Adapter] = None

    if llama is not None:
        adp_llama = Adapter(
            d_z=args.d_z,
            d_model=llama.d_model,
            latent_length=latent_shared_len + latent_private_len,
            enable_metadata=bool(args.adapter_metadata),
            length_norm=float(args.max_answer_tokens),
            hidden_mult=args.adapter_hidden_mult,
            colorize=bool(args.adapter_colorize),
            dropout=float(args.adapter_dropout),
        ).to(_primary_device(llama))
        adapters["llama"] = adp_llama

    if qwen is not None:
        adp_qwen = Adapter(
            d_z=args.d_z,
            d_model=qwen.d_model,
            latent_length=latent_shared_len + latent_private_len,
            enable_metadata=bool(args.adapter_metadata),
            length_norm=float(args.max_answer_tokens),
            hidden_mult=args.adapter_hidden_mult,
            colorize=bool(args.adapter_colorize),
            dropout=float(args.adapter_dropout),
        ).to(_primary_device(qwen))
        adapters["qwen"] = adp_qwen

    if args.adapter_colorize:
        for name, adapter in adapters.items():
            wrapper = llama if name == "llama" else qwen
            if wrapper is None:
                continue
            try:
                adapter.install_color_from_wrapper(wrapper)
                print(f"Initialized adapter colorizer for {name} from LM embedding stats.")
            except Exception as exc:
                print(f"[WARN] Adapter colorizer initialization skipped for {name}: {exc}")

    if args.adapter_freeze_scale:
        for adapter in adapters.values():
            if adapter.scale is None:
                continue
            adapter.scale.requires_grad_(False)
            with torch.no_grad():
                adapter.scale.fill_(1.0)

    if args.use_gist_head and args.gist_weight > 0.0:
        gist_len = max(1, int(args.gist_target_len))
        for name, wrapper in (('llama', llama), ('qwen', qwen)):
            if wrapper is None:
                continue
            head = GistReconstructionHead(
                d_latent=args.d_z,
                d_model=wrapper.d_model,
                target_len=gist_len,
                hidden=int(max(args.gist_hidden, args.d_z)),
                num_layers=int(max(args.gist_layers, 0)),
                dropout=float(max(args.gist_dropout, 0.0)),
            ).to(_primary_device(wrapper))
            gist_heads[name] = head

    # ===== DDP Model Wrapping =====
    if 'ddp_manager' in locals() and ddp_manager is not None and ddp_manager.initialized:
        print("\n" + "="*60)
        print("Wrapping models with DistributedDataParallel")
        print("="*60)

        # Move and wrap encoder
        encoder = encoder.to(ddp_manager.device)
        encoder = ddp_manager.wrap_model(encoder, find_unused_parameters=True)
        print("  ✓ Encoder wrapped with DDP")

        # Wrap adapters
        for name, adapter in adapters.items():
            adapters[name] = adapter.to(ddp_manager.device)
            adapters[name] = ddp_manager.wrap_model(adapters[name], find_unused_parameters=True)
            print(f"  ✓ Adapter '{name}' wrapped with DDP")

        # Update references to wrapped adapters
        if 'adp_llama' in locals() and adp_llama is not None:
            adp_llama = adapters.get('llama', adp_llama)
        if 'adp_qwen' in locals() and adp_qwen is not None:
            adp_qwen = adapters.get('qwen', adp_qwen)

        # Wrap deep prefix generators
        for name, gen in deep_prefix_generators.items():
            deep_prefix_generators[name] = gen.to(ddp_manager.device)
            deep_prefix_generators[name] = ddp_manager.wrap_model(deep_prefix_generators[name], find_unused_parameters=True)
        if deep_prefix_generators:
            print(f"  ✓ {len(deep_prefix_generators)} deep prefix generators wrapped with DDP")

        # Wrap latent refiner if present
        if latent_refiner is not None:
            latent_refiner = latent_refiner.to(ddp_manager.device)
            latent_refiner = ddp_manager.wrap_model(latent_refiner, find_unused_parameters=True)
            print("  ✓ Latent refiner wrapped with DDP")

        # Wrap gist heads
        for name, head in gist_heads.items():
            gist_heads[name] = head.to(ddp_manager.device)
            gist_heads[name] = ddp_manager.wrap_model(gist_heads[name], find_unused_parameters=True)
        if gist_heads:
            print(f"  ✓ {len(gist_heads)} gist heads wrapped with DDP")

        # Wrap coprocessors if present
        for name, coprocessor in coprocessors.items():
            coprocessors[name] = coprocessor.to(ddp_manager.device)
            coprocessors[name] = ddp_manager.wrap_model(coprocessors[name], find_unused_parameters=True)
        if coprocessors:
            print(f"  ✓ {len(coprocessors)} coprocessors wrapped with DDP")

        # Move LLMs to proper device (but don't wrap - they're frozen)
        if llama is not None:
            # For frozen models, just ensure they're on the right device
            if not hasattr(llama.model, 'hf_device_map') or llama.model.hf_device_map is None:
                llama.model = llama.model.to(ddp_manager.device)
                print(f"  ✓ Llama model moved to {ddp_manager.device}")
        if qwen is not None:
            if not hasattr(qwen.model, 'hf_device_map') or qwen.model.hf_device_map is None:
                qwen.model = qwen.model.to(ddp_manager.device)
                print(f"  ✓ Qwen model moved to {ddp_manager.device}")

        print("="*60 + "\n")

    # ===== torch.compile Optimization =====
    # DISABLED: torch.compile causes symbolic_shapes warnings and slow first steps
    # Compile encoder and adapters for ~20-30% speedup on forward/backward
    # if device == "cuda":
    #     try:
    #         # Compile encoder (biggest win - runs every batch)
    #         encoder = torch.compile(encoder, mode="reduce-overhead")
    #         print("[Optimization] Compiled encoder with torch.compile")
    #
    #         # Compile adapters (moderate win - runs every batch)
    #         for name, adapter in adapters.items():
    #             adapters[name] = torch.compile(adapter, mode="reduce-overhead")
    #         print(f"[Optimization] Compiled {len(adapters)} adapters with torch.compile")
    #
    #         # Compile deep prefix generators if enabled
    #         if deep_prefix_generators:
    #             for name, gen in deep_prefix_generators.items():
    #                 deep_prefix_generators[name] = torch.compile(gen, mode="reduce-overhead")
    #             print(f"[Optimization] Compiled {len(deep_prefix_generators)} deep prefix generators")
    #
    #         # Compile latent refiner if enabled
    #         if latent_refiner is not None:
    #             latent_refiner = torch.compile(latent_refiner, mode="reduce-overhead")
    #             print("[Optimization] Compiled latent refiner")
    #
    #     except Exception as e:
    #         print(f"[Optimization] torch.compile not available or failed: {e}")

    # ===== Optimizer =====
    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    llama_params = [p for p in adp_llama.parameters() if p.requires_grad] if adp_llama is not None else []
    qwen_params = [p for p in adp_qwen.parameters() if p.requires_grad] if adp_qwen is not None else []
    refiner_params = [p for p in latent_refiner.parameters() if p.requires_grad] if latent_refiner is not None else []
    gist_params: List[torch.nn.Parameter] = []
    for head in gist_heads.values():
        gist_params.extend([p for p in head.parameters() if p.requires_grad])

    # Collect latent adapter params via feature registry (fallbacks if missing)
    latent_adapter_params: List[torch.nn.Parameter] = []
    if latent_adapter_param_map:
        print("[Optimizer] Gathering latent adapter parameters from feature registry...")
        for name, params in latent_adapter_param_map.items():
            latent_adapter_params.extend(params)
            summary = latent_adapter_summaries.get(name, {})
            num_params = summary.get("num_params", sum(p.numel() for p in params))
            num_tensors = summary.get("num_tensors", len(params))
            layers = summary.get("layers", [])
            layer_str = ",".join(str(x) for x in layers) if layers else "-"
            print(f"[Optimizer]   {name}: {num_params:,} params in {num_tensors} tensors (layers={layer_str})")
    else:
        print("[Optimizer] Gathering latent adapter parameters (direct wrappers fallback)...")
        for wrapper in wrappers_in_use:
            wrapper_name = getattr(wrapper.cfg, "model_id", "unknown").split("/")[-1]
            if wrapper.use_latent_adapters:
                adapter_params_list = [p for p in wrapper.latent_adapters.parameters() if p.requires_grad]
                num_params = sum(p.numel() for p in adapter_params_list)
                num_tensors = len(adapter_params_list)
                latent_adapter_params.extend(adapter_params_list)
                print(f"[Optimizer]   {wrapper_name}: {num_params:,} params in {num_tensors} tensors (use_latent_adapters=True)")
                if num_params == 0:
                    print(f"[Optimizer]   ⚠️  WARNING: {wrapper_name} has use_latent_adapters=True but contributed 0 parameters!")
                    print(f"[Optimizer]       Adapter layers configured: {wrapper.latent_adapter_layers}")
                    print(f"[Optimizer]       Check that adapters were initialized before optimizer creation")
            else:
                print(f"[Optimizer]   {wrapper_name}: skipped (use_latent_adapters=False)")

    # Log total adapter params collected
    total_adapter_params = sum(p.numel() for p in latent_adapter_params)
    total_tensors = len(latent_adapter_params)
    print(f"[Optimizer] Latent adapter summary: {total_adapter_params:,} params in {total_tensors} tensors")

    if not latent_adapter_params:
        any_enabled = any(w.use_latent_adapters for w in wrappers_in_use)
        if any_enabled:
            print(f"[Optimizer] ⚠️  ERROR: Adapters enabled but 0 parameters collected - optimizer will not train them!")
        else:
            print(f"[Optimizer] No latent adapters enabled (expected)")

    optim_groups = []
    if enc_params:
        optim_groups.append({"params": enc_params, "lr": args.lr})
    if llama_params:
        optim_groups.append({"params": llama_params, "lr": args.lr})
    if qwen_params:
        optim_groups.append({"params": qwen_params, "lr": args.lr})
    if extra_llama_params:
        optim_groups.append({"params": extra_llama_params, "lr": args.lr})
    if extra_qwen_params:
        optim_groups.append({"params": extra_qwen_params, "lr": args.lr})
    if refiner_params:
        optim_groups.append({"params": refiner_params, "lr": args.lr})
    if gist_params:
        optim_groups.append({"params": gist_params, "lr": args.lr})
    # Latent adapter groups are injected via feature registry (optimizer_param_groups)

    # Allow features to contribute additional parameter groups.
    optim_groups.extend(feature_registry.optimizer_param_groups())

    # Enable fused AdamW for better performance on CUDA (15-20% faster optimizer step)
    use_fused = device == "cuda" and torch.cuda.is_available()
    optimizer = optim.AdamW(optim_groups, lr=args.lr, fused=use_fused, foreach=False)
    if use_fused:
        print("[Optimization] Using fused AdamW optimizer")

    # ===== Mixed Precision Training Setup =====
    grad_scaler = None
    amp_dtype = None
    amp_enabled = False

    if args.mixed_precision != "no" and device == "cuda":
        if args.mixed_precision == "bf16":
            amp_dtype = torch.bfloat16
            amp_enabled = True
            print("[AMP] Using BF16 mixed precision (recommended for H100)")
            print("      - Better numerical stability than FP16")
            print("      - No GradScaler needed (native hardware support)")
            print(f"      - Optimization level: {args.amp_opt_level}")
        elif args.mixed_precision == "fp16":
            amp_dtype = torch.float16
            amp_enabled = True
            # GradScaler is only needed for FP16, not BF16
            from torch.cuda.amp import GradScaler
            grad_scaler = GradScaler(
                init_scale=args.grad_scaler_init,
                growth_interval=args.grad_scaler_growth_interval,
                backoff_factor=args.grad_scaler_backoff,
                growth_factor=2.0,
                enabled=True
            )
            print("[AMP] Using FP16 mixed precision")
            print(f"      - GradScaler initialized (init_scale={args.grad_scaler_init})")
            print(f"      - Growth interval: {args.grad_scaler_growth_interval} steps")
            print(f"      - Optimization level: {args.amp_opt_level}")
        elif args.mixed_precision == "fp8":
            # FP8 support for H100 (experimental, requires transformer_engine)
            try:
                import transformer_engine.pytorch as te
                amp_dtype = torch.float8_e4m3fn  # H100 native FP8
                amp_enabled = True
                print("[AMP] Using FP8 mixed precision (H100 native)")
                print("      - Maximum performance on H100 Tensor Cores")
                print("      - Requires transformer_engine library")
            except ImportError:
                print("[WARNING] FP8 requested but transformer_engine not available")
                print("         Falling back to BF16")
                amp_dtype = torch.bfloat16
                amp_enabled = True
    elif args.mixed_precision != "no":
        print(f"[WARNING] Mixed precision requested but device is {device}, not cuda. Disabled.")

    # Log memory and performance expectations
    if amp_enabled:
        expected_speedup = 2.0 if args.mixed_precision == "fp16" else 2.5
        expected_memory_savings = 0.5 if args.amp_opt_level == "O2" else 0.3
        print(f"[AMP] Expected benefits:")
        print(f"      - Training speedup: ~{expected_speedup:.1f}x")
        print(f"      - Memory savings: ~{expected_memory_savings*100:.0f}%")
        print(f"      - Larger effective batch size possible")

    # Log optimizer groups for debugging
    print(f"[Optimizer] Created {len(optim_groups)} parameter groups:")
    group_names = []
    if enc_params: group_names.append(f"encoder({len(enc_params)} tensors)")
    if llama_params: group_names.append(f"llama_adapter({len(llama_params)} tensors)")
    if qwen_params: group_names.append(f"qwen_adapter({len(qwen_params)} tensors)")
    if extra_llama_params: group_names.append(f"llama_extra({len(extra_llama_params)} tensors)")
    if extra_qwen_params: group_names.append(f"qwen_extra({len(extra_qwen_params)} tensors)")
    if refiner_params: group_names.append(f"refiner({len(refiner_params)} tensors)")
    if feature_registry.has("deep_prefix") and deep_prefix_generators:
        tensor_count = sum(
            1
            for gen in deep_prefix_generators.values()
            for p in gen.parameters()
            if p.requires_grad
        )
        group_names.append(f"deep_prefix({tensor_count} tensors)")
    if gist_params: group_names.append(f"gist({len(gist_params)} tensors)")
    if latent_adapter_params: group_names.append(f"latent_adapters({len(latent_adapter_params)} tensors)")
    for i, name in enumerate(group_names):
        print(f"  [{i+1}] {name}")

    # ===== Learning rate scheduler (cosine annealing for stability) =====
    from torch.optim.lr_scheduler import CosineAnnealingLR

    total_steps = (args.samples // args.batch_size) * args.epochs
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * 0.02  # Decay to 2% of initial LR (5e-5 → 1e-6)
    )
    print(f"[INFO] LR scheduler: CosineAnnealingLR (T_max={total_steps}, eta_min={args.lr * 0.02:.2e})")

    # ===== Tokenize answers (teacher forcing) =====
    token_ids_map: Dict[str, torch.Tensor] = {}
    answer_lengths_map: Dict[str, torch.Tensor] = {}
    first_token_ids_map: Dict[str, torch.Tensor] = {}

    for name, wrapper in (('llama', llama), ('qwen', qwen)):
        if wrapper is None:
            continue
        with _temp_padding_side(wrapper.tokenizer, "right"):
            tok = wrapper.tokenizer(
                answers,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_answer_tokens,
                add_special_tokens=True,
            )
        ids = tok["input_ids"].to(device)
        token_ids_map[name] = ids
        answer_lengths_map[name] = _answer_lengths(ids, wrapper.tokenizer)
        first_token_ids_map[name] = first_non_bos(wrapper.tokenizer, ids)

    N = len(texts)
    steps_per_epoch = (N + args.batch_size - 1) // args.batch_size

    # ===== Resume (optional) =====
    start_epoch = 0
    global_step = 0
    os.makedirs(args.save_dir, exist_ok=True)
    try:
        os.makedirs(args.save_dir, exist_ok=True)
        capture_env_snapshot(args.save_dir, extras={"phase":"train"})
    except Exception:
        pass

    ckpt_path = None
    if args.resume_from:
        if os.path.isdir(args.resume_from):
            ckpt_path = find_latest_checkpoint(args.resume_from)
        else:
            ckpt_path = args.resume_from
    elif args.auto_resume:
        ckpt_path = find_latest_checkpoint(args.save_dir)

    if ckpt_path:
        print(f"⏪ Resuming from: {ckpt_path}", flush=True)
        # Build wrappers dict for latent adapter loading
        wrappers_dict = {}
        if llama is not None:
            wrappers_dict["llama"] = llama
        if qwen is not None:
            wrappers_dict["qwen"] = qwen
        epoch_loaded, global_loaded = load_checkpoint(
            ckpt_path,
            encoder,
            adapters,
            refiner=latent_refiner,
            deep_prefix_generators=deep_prefix_generators,
            coprocessors=coprocessors,
            gist_heads=gist_heads,
            optimizer=None if args.no_load_optimizer else optimizer,
            lr_scheduler=None if args.no_load_lr_scheduler else lr_scheduler,
            strict=True,
            device=device,
            wrappers=wrappers_dict,
        )
        start_epoch = epoch_loaded
        global_step = global_loaded
        if args.reset_epoch:
            start_epoch = 0
            global_step = 0
            print("   -> reset epoch/global_step to zero as requested", flush=True)
        print(f"   -> start_epoch={start_epoch}, global_step={global_step}", flush=True)
        if args.adapter_colorize:
            try:
                for name, adapter in adapters.items():
                    wrapper = llama if name == "llama" else qwen
                    if wrapper is None:
                        continue
                    adapter.install_color_from_wrapper(wrapper)
            except Exception as exc:
                print(f"[WARN] Adapter colorizer re-install skipped after resume: {exc}")
    else:
        print("⚠️  No valid checkpoint found to resume; starting fresh.", flush=True)

    # ===== Training stats trackers =====
    class _RunningMean:
        def __init__(self):
            self.n = 0
            self.sum = 0.0
        def update(self, value: float):
            self.n += 1
            self.sum += float(value)
        @property
        def mean(self):
            return (self.sum / self.n) if self.n > 0 else 0.0

    stats_trackers: Dict[str, Dict[str, Any]] = {}
    if llama is not None:
        stats_trackers["llama"] = {
            "rms_raw": _RunningMean(),
            "rms_cal": _RunningMean(),
            "embed_rms": llama.input_embedding_rms(),
        }
    if qwen is not None:
        stats_trackers["qwen"] = {
            "rms_raw": _RunningMean(),
            "rms_cal": _RunningMean(),
            "embed_rms": qwen.input_embedding_rms(),
        }

    model_contexts: List[ModelTrainContext] = []

    if llama is not None:
        model_contexts.append(
            ModelTrainContext(
                name="llama",
                wrapper=llama,
                adapter=adp_llama,
                token_ids=token_ids_map["llama"],
                first_token_ids=first_token_ids_map["llama"],
                anchor_ids=anchor_token_lists["llama"],
                bos_flag=bos_flags.get("llama"),
                answer_lengths=answer_lengths_map["llama"],
                anchor_text=anchor_texts.get("llama", ""),
                anchor_mode=anchor_modes.get("llama", "none"),
            )
        )

    if qwen is not None:
        model_contexts.append(
            ModelTrainContext(
                name="qwen",
                wrapper=qwen,
                adapter=adp_qwen,
                token_ids=token_ids_map["qwen"],
                first_token_ids=first_token_ids_map["qwen"],
                anchor_ids=anchor_token_lists["qwen"],
                bos_flag=bos_flags.get("qwen"),
                answer_lengths=answer_lengths_map["qwen"],
                anchor_text=anchor_texts.get("qwen", ""),
                anchor_mode=anchor_modes.get("qwen", "none"),
            )
        )

    if not model_contexts:
        raise RuntimeError("No models selected for training. Use --models to include at least one backend.")

    # ===== Train =====
    ema_step_time = None

    def _parse_layers_arg(value: str) -> Tuple[int, ...]:
        try:
            items = [int(v) for v in re.split(r"[\s,]+", value.strip()) if v != ""]
            return tuple(items) if items else (0, 1, 2)
        except Exception:
            return (0, 1, 2)

    state_kd_layers = _parse_layers_arg(args.state_kd_layers)

    params_for_clip = enc_params + llama_params + qwen_params

    def _grad_norm(params) -> float:
        norms = []
        for p in params:
            grad = getattr(p, "grad", None)
            if grad is None:
                continue
            g = grad.detach()
            if not torch.isfinite(g).all():
                return float("nan")
            norms.append(g.float().norm(2).cpu())
        if not norms:
            return 0.0
        stacked = torch.stack(norms)
        return float(torch.norm(stacked, 2).item())

    optimizer.zero_grad(set_to_none=True)

    total_batches = steps_per_epoch * args.epochs
    warmup_steps_from_epochs = int(round(max(float(args.warmup_text_latent_epochs), 0.0) * steps_per_epoch))
    warmup_total_steps = max(int(args.warmup_text_latent_steps), warmup_steps_from_epochs)
    warmup_total_steps = max(0, min(warmup_total_steps, total_batches))
    if warmup_total_steps > 0:
        print(f"[warmup] alternating text/latent for first {warmup_total_steps} steps")

    first_ce_schedule = str(getattr(args, "first_token_ce_schedule", "none")).lower()
    peak_override = args.first_token_ce_peak
    autoscale_first = str(getattr(args, "first_token_autoscale", "yes")).lower() != "no"

    def _first_token_weight_for_step(step_idx: int) -> float:
        base = max(float(args.first_token_ce_weight), 0.0)
        if first_ce_schedule == "none" or total_batches <= 0:
            return base

        # "warmup" schedule: linear ramp 0 → base over first 200 steps, then hold
        if first_ce_schedule == "warmup":
            warmup_steps = 200
            if step_idx < warmup_steps:
                return base * (step_idx / warmup_steps)
            return base

        # "cosine" schedule: hold at peak, then cosine decay to base
        total = max(int(total_batches), 1)
        peak = peak_override if (peak_override is not None and peak_override > 0.0) else max(8.0, 2.0 * max(base, 1e-6))
        warm_frac = min(max(float(args.first_token_ce_warmup_frac), 0.0), 1.0)
        warm_steps = int(round(total * warm_frac))
        warm_steps = min(warm_steps, total)
        if step_idx < warm_steps:
            return peak
        if step_idx >= total or total == warm_steps:
            return base
        t = (step_idx - warm_steps) / max(1, total - warm_steps)
        t = min(max(t, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return base + (peak - base) * cosine

    # Peak checkpointing: track best first_acc for latent mode
    # Use exponential moving average (EMA) to smooth out batch-level noise
    best_first_acc = -1.0
    best_checkpoint_step = -1
    first_acc_ema = 0.0  # Exponential moving average of first_acc
    ema_alpha = 0.1      # Smoothing factor (0.1 = 10% current, 90% history)

    # Setup optimized dataloader if requested
    optimized_dataloader = None
    distributed_sampler = None  # Track sampler for epoch setting
    if args.use_optimized_dataloader:
        print("Setting up optimized DataLoader...")
        from latentwire.optimized_dataloader import create_optimized_dataloader
        from torch.utils.data import Dataset, DataLoader
        from torch.utils.data.distributed import DistributedSampler

        # Create a simple dataset wrapper for DDP compatibility
        class SimpleDataset(Dataset):
            def __init__(self, texts, answers):
                self.texts = texts
                self.answers = answers

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                return self.texts[idx], self.answers[idx]

        # Check if we're using DDP
        if 'ddp_manager' in locals() and ddp_manager is not None and ddp_manager.initialized:
            # Create dataset and distributed sampler
            dataset = SimpleDataset(texts, answers)
            distributed_sampler = DistributedSampler(
                dataset,
                num_replicas=ddp_manager.world_size,
                rank=ddp_manager.rank,
                shuffle=True,
                drop_last=False
            )

            # Note: When using DistributedSampler, shuffle must be False in DataLoader
            # and we need to use the sampler instead
            print(f"  Using DistributedSampler (rank {ddp_manager.rank}/{ddp_manager.world_size})")

            # For now, we'll still use the manual approach but prepare for future improvement
            optimized_dataloader = None  # Disabled for DDP until properly integrated
        else:
            optimized_dataloader = create_optimized_dataloader(
                texts=texts,
                answers=answers,
                model_contexts=model_contexts,
                batch_size=args.batch_size,
                num_workers=args.num_dataloader_workers,
                use_chat_template=args.use_chat_template,
                strip_anchor_literal=strip_anchor_literal,
                device=device,
                shuffle=False,  # We'll shuffle per epoch
                pin_memory=args.dataloader_pin_memory,
                prefetch_factor=args.dataloader_prefetch_factor,
                persistent_workers=(args.num_dataloader_workers > 0),
                cache_tokenization=args.dataloader_cache_tokenization,
                use_prefetcher=torch.cuda.is_available(),
            )
            print(f"  Workers: {args.num_dataloader_workers}")
            print(f"  Prefetch factor: {args.dataloader_prefetch_factor}")
            print(f"  Cache tokenization: {args.dataloader_cache_tokenization}")
            print(f"  Pin memory: {args.dataloader_pin_memory}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # Initialize epoch loss tracking
        epoch_losses = []

        # Set epoch for distributed sampler if using DDP
        if 'ddp_manager' in locals() and ddp_manager is not None and ddp_manager.initialized:
            ddp_manager.set_epoch(epoch)
            # Also set epoch for our distributed sampler if it exists
            if distributed_sampler is not None:
                distributed_sampler.set_epoch(epoch)
            if ddp_manager.should_log:
                print(f"Epoch {epoch+1}/{args.epochs}", flush=True)
                # Log GPU memory at epoch start, reset peak stats
                log_gpu_memory(prefix=f"[Epoch {epoch+1} Start] ", reset_peak=True)
        else:
            print(f"Epoch {epoch+1}/{args.epochs}", flush=True)
            # Log GPU memory at epoch start, reset peak stats
            log_gpu_memory(prefix=f"[Epoch {epoch+1} Start] ", reset_peak=True)

        # For DDP, we need to ensure each process gets different data
        if 'ddp_manager' in locals() and ddp_manager is not None and ddp_manager.initialized:
            # Create a distributed sampler for the epoch
            g = torch.Generator(device="cpu")
            g.manual_seed(int(args.seed) + int(epoch) + ddp_manager.rank * 1000)

            # Calculate which samples this rank should process
            samples_per_rank = N // ddp_manager.world_size
            extra_samples = N % ddp_manager.world_size

            # Distribute extra samples to first few ranks
            if ddp_manager.rank < extra_samples:
                rank_start = ddp_manager.rank * (samples_per_rank + 1)
                rank_end = rank_start + samples_per_rank + 1
            else:
                rank_start = ddp_manager.rank * samples_per_rank + extra_samples
                rank_end = rank_start + samples_per_rank

            # Create permutation for this rank's samples
            all_perm = torch.randperm(N, generator=g)
            rank_indices = all_perm[rank_start:rank_end]
            perm = rank_indices[torch.randperm(len(rank_indices), generator=g)]

            # Adjust steps per epoch for this rank
            rank_steps_per_epoch = len(perm) // args.batch_size
        else:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(args.seed) + int(epoch))
            perm = torch.randperm(N, generator=g)
            rank_steps_per_epoch = steps_per_epoch

        for step in range(rank_steps_per_epoch):
            # Synchronize before timing for accurate GPU measurements
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            idx = perm[step*args.batch_size : (step+1)*args.batch_size]
            batch_texts = [texts[i] for i in idx.tolist()]
            if args.use_chat_template:
                batch_user_texts = [split_user_and_anchor(raw, strip_anchor_literal)[0] for raw in batch_texts]
            else:
                batch_user_texts = batch_texts

            global_batch_idx = epoch * steps_per_epoch + step
            if total_batches > 1:
                progress = min(max(global_batch_idx / (total_batches - 1), 0.0), 1.0)
            else:
                progress = 1.0
            progress_pow = progress ** latent_keep_power
            keep_prob = latent_keep_start + (latent_keep_end - latent_keep_start) * progress_pow
            keep_prob = float(min(max(keep_prob, 0.0), 1.0))
            current_K = int(round(adaptive_k_start + (adaptive_k_end - adaptive_k_start) * progress_pow))
            current_K = max(1, min(current_K, args.max_answer_tokens))

            current_first_weight = _first_token_weight_for_step(global_step)
            enable_first_token_loss = current_first_weight > 0.0

            scaffolds = {}
            bad_latent_sources: List[str] = []
            for ctx in model_contexts:
                if args.use_chat_template:
                    pad_token = getattr(ctx.wrapper.tokenizer, "pad_token_id", None)
                    if pad_token is None:
                        pad_token = int(getattr(ctx.wrapper.tokenizer, "eos_token_id", 0))
                    ids_list = []
                    for user_text in batch_user_texts:
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_text},
                        ]
                        rendered = ctx.wrapper.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        assistant_prefill = ctx.anchor_text if ctx.anchor_mode == "text" else strip_anchor_literal
                        if assistant_prefill:
                            rendered = rendered + assistant_prefill
                        toks = ctx.wrapper.tokenizer(
                            rendered,
                            return_tensors="pt",
                            padding=False,
                            truncation=False,
                            add_special_tokens=False,
                        )
                        ids = toks["input_ids"][0].to(device)
                        ids_list.append(ids)
                    scaffolds[ctx.name] = torch.nn.utils.rnn.pad_sequence(
                        ids_list, batch_first=True, padding_value=int(pad_token)
                    )
                else:
                    anchor_suffix = ctx.anchor_text if ctx.anchor_mode == "text" else strip_anchor_literal
                    texts_with_anchor = [f"{text}{anchor_suffix}" for text in batch_texts]
                    tok = ctx.wrapper.tokenizer(
                        texts_with_anchor,
                        return_tensors="pt",
                        padding=True,
                        truncation=False,
                        add_special_tokens=False,
                    )
                    scaffolds[ctx.name] = tok["input_ids"].to(device)

            effective_texts = batch_user_texts if args.use_chat_template else batch_texts
            batch_index = epoch * steps_per_epoch + step
            warmup_active = warmup_total_steps > 0 and batch_index < warmup_total_steps
            training_mode = "latent"
            if warmup_active:
                training_mode = "text" if (batch_index % 2 == 0) else "latent"
            elif args.warmup_tail_prob > 0.0 and random.random() < float(args.warmup_tail_prob):
                training_mode = "text"

            if warmup_active:
                if training_mode == "text" and batch_index < 10:
                    print(f"[warmup] step={batch_index} mode=text (warm-up)")
                elif training_mode == "latent" and batch_index < 10:
                    print(f"[warmup] step={batch_index} mode=latent (warm-up)")
            elif training_mode == "text" and batch_index < warmup_total_steps + 50:
                print(f"[warmup] step={batch_index} mode=text (tail)")

            per_model_losses: Dict[str, Dict[str, torch.Tensor]] = {}
            total_model_loss = torch.zeros((), device=device)
            loss_device = total_model_loss.device  # Explicit device for loss aggregation in multi-GPU
            penalty = torch.zeros((), device=device)
            rms_pen = torch.zeros((), device=device)
            feature_grad_norms: Dict[str, float] = {}

            # Start autocast context for mixed precision forward pass
            # The entire forward pass should be in autocast scope for maximum benefit
            autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype) if amp_enabled else nullcontext()

            with autocast_ctx:
                encoded_latents = encode_fn(effective_texts)
                shared_latents = encoded_latents["shared"]
                private_latents = encoded_latents["private"]

                # Move memory profiling inside autocast for consistency
                if batch_index < 3:
                    mem_stats = get_gpu_memory_stats()
                    if mem_stats:
                        print(f"    [Memory after encoder] {mem_stats['total_allocated_gb']:.1f}GB allocated")

                dropout_keep = keep_prob if training_mode == "latent" else 1.0
                if shared_latents.size(1) > 0 and dropout_keep < 1.0:
                    mask = (torch.rand(shared_latents.shape[:2], device=shared_latents.device) < dropout_keep).float()
                    need_fix = mask.sum(dim=1) == 0
                    if need_fix.any():
                        mask[need_fix, 0] = 1.0
                    mask = mask.unsqueeze(-1)
                    shared_latents = shared_latents * mask / max(dropout_keep, 1e-3)
            model_latents = {
                name: torch.cat([shared_latents, private_latents[name]], dim=1)
                for name in model_keys
            }
            if latent_refiner is not None:
                for name in model_keys:
                    model_latents[name] = latent_refiner(model_latents[name])

            # Flag to skip batch if NaN detected early
            skip_batch_due_to_nan = False

            for ctx in model_contexts:
                target_device = _primary_device(ctx.wrapper)
                targets = ctx.token_ids[idx].to(target_device, non_blocking=True)
                scaffold = scaffolds[ctx.name].to(target_device, non_blocking=True)

                losses_record: Dict[str, torch.Tensor] = {}
                latents_for_adapter = model_latents[ctx.name].to(target_device, non_blocking=True)
                answer_lengths = ctx.answer_lengths[idx].to(target_device, non_blocking=True)

                prefix_raw = ctx.adapter(latents_for_adapter, answer_lengths=answer_lengths)
                prefix = calibrate_to_embed_rms(prefix_raw, ctx.wrapper)

                # Check for NaN/Inf in prefix after calibration
                if not torch.isfinite(prefix).all():
                    print(f"⚠️  NaN/Inf detected in prefix for {ctx.name} at step {step+1}")
                    print(f"    prefix_raw finite: {torch.isfinite(prefix_raw).all()}")
                    print(f"    latents finite: {torch.isfinite(latents_for_adapter).all()}")
                    # Set flag and break out of model loop
                    skip_batch_due_to_nan = True
                    bad_latent_sources.append(ctx.name)
                    break

                deep_prefix_cache: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None
                if ctx.name in deep_prefix_generators:
                    deep_prefix_cache = deep_prefix_generators[ctx.name](prefix)
                coprocessor_cache: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None
                if ctx.name in coprocessors:
                    coprocessor_cache = coprocessors[ctx.name](latents_for_adapter)
                prefix_kv_cache = _merge_kv_caches(deep_prefix_cache, coprocessor_cache)
                if args.debug and epoch == start_epoch and step == 0:
                    print(
                        f"[DEBUG:{ctx.name}] prefix_len={prefix.shape[1]} anchor_ids={len(ctx.anchor_ids)} tf_len={targets.size(1)}",
                        flush=True,
                    )
                    print(
                        f"[DEBUG:{ctx.name}] scaffold_len={scaffold.size(1)} anchor_mode={ctx.anchor_mode}",
                        flush=True,
                    )
                # Pass latent for multi-depth adapters
                latent_for_adapters_tf = latents_for_adapter if ctx.wrapper.use_latent_adapters else None
                loss_tf_latent = ctx.wrapper.forward_with_prefix_loss(
                    prefix,
                    targets,
                    anchor_token_ids=ctx.anchor_ids,
                    deep_prefix_past=prefix_kv_cache,
                    latent=latent_for_adapters_tf,
                )
                # Move loss to device for aggregation (use non_blocking to avoid hang in multi-GPU)
                loss_tf_latent = loss_tf_latent.to(loss_device)

                first_anchor_text = ctx.anchor_text if ctx.anchor_mode == "text" else strip_anchor_literal
                entropy_bonus = torch.zeros((), device=target_device)
                first_entropy = torch.zeros((), device=target_device)
                if enable_first_token_loss:
                    # Pass raw latent for multi-depth adapters
                    latent_for_adapters = latents_for_adapter if ctx.wrapper.use_latent_adapters else None
                    logits_first = ctx.wrapper.first_token_logits_from_prefix(
                        prefix,
                        anchor_token_text=first_anchor_text,
                        append_bos_after_prefix=ctx.bos_flag,
                        deep_prefix_past=prefix_kv_cache,
                        latent=latent_for_adapters,
                    )
                    first_targets = ctx.first_token_ids[idx].to(logits_first.device, non_blocking=True)
                    loss_first_raw = nn.functional.cross_entropy(logits_first.float(), first_targets)
                    # Move loss to device for aggregation (avoid device mismatch in multi-GPU)
                    loss_first_raw = loss_first_raw.to(loss_device)
                    if training_mode == "latent" and args.first_token_entropy_weight > 0.0:
                        probs = torch.softmax(logits_first.float(), dim=-1)
                        entropy_vals = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
                        first_entropy = entropy_vals.mean()
                        entropy_bonus = -first_entropy * float(args.first_token_entropy_weight)
                        # Move entropy_bonus to device for aggregation
                        entropy_bonus = entropy_bonus.to(loss_device)
                    with torch.no_grad():
                        first_pred = logits_first.argmax(dim=-1)
                        first_acc_raw = (first_pred == first_targets).float().mean()

                        # Enhanced logging: top-k probabilities, margins, and top-5 accuracy
                        if training_mode == "latent":
                            probs_for_stats = torch.softmax(logits_first.float(), dim=-1)
                            top5_probs, top5_idx = torch.topk(probs_for_stats, k=min(5, probs_for_stats.size(-1)), dim=-1)

                            # Top-k statistics
                            max_prob = top5_probs[:, 0].mean().item()
                            second_prob = top5_probs[:, 1].mean().item() if top5_probs.size(-1) > 1 else 0.0
                            margin = (top5_probs[:, 0] - top5_probs[:, 1]).mean().item() if top5_probs.size(-1) > 1 else 0.0

                            # Top-5 entropy (entropy over just the top 5 tokens)
                            top5_entropy = -(top5_probs * torch.log(top5_probs.clamp_min(1e-8))).sum(-1).mean().item()

                            # Top-5 accuracy: does gold appear in top-5 predictions?
                            first_targets_expanded = first_targets.unsqueeze(-1).expand_as(top5_idx)
                            top5_match = (top5_idx == first_targets_expanded).any(dim=-1)
                            first_acc_top5 = top5_match.float().mean().item()

                            # Store enhanced stats
                            losses_record["first_token_logit_stats"] = {
                                "max_prob": max_prob,
                                "second_prob": second_prob,
                                "margin": margin,
                                "top5_entropy": top5_entropy,
                            }
                            losses_record["first_acc_top5"] = first_acc_top5

                            # Prediction histogram: collect token strings
                            try:
                                pred_tokens = [ctx.wrapper.tokenizer.decode([p.item()], skip_special_tokens=False)
                                             for p in first_pred[:24]]  # Limit to 24 for logging
                                losses_record["pred_tokens"] = pred_tokens
                            except Exception:
                                losses_record["pred_tokens"] = []

                        # Store predictions for logging (will be accessible via per_model_losses later)
                        losses_record["first_pred"] = first_pred
                        losses_record["first_targets"] = first_targets
                else:
                    loss_first_raw = torch.zeros((), device=target_device)
                    first_acc_raw = torch.zeros((), device=target_device)

                if args.k_ce_weight and args.k_ce_weight > 0.0:
                    # Pass latent for multi-depth adapters
                    latent_for_adapters_kce = latents_for_adapter if ctx.wrapper.use_latent_adapters else None
                    loss_kce_raw = k_token_ce_from_prefix(
                        ctx.wrapper,
                        prefix,
                        targets,
                        K=current_K,
                        anchor_ids=ctx.anchor_ids,
                        append_bos_after_prefix=ctx.bos_flag,
                        deep_prefix_past=prefix_kv_cache,
                        latent=latent_for_adapters_kce,
                    )
                else:
                    loss_kce_raw = torch.zeros((), device=target_device)

                if training_mode == "latent" and args.kd_first_k_weight and args.kd_first_k_weight > 0.0:
                    teacher_model = ctx.wrapper.model
                    disable_fn = getattr(teacher_model, "disable_adapter", None)
                    teacher_device = next(teacher_model.parameters()).device
                    if disable_fn is not None:
                        with teacher_model.disable_adapter():
                            loss_kd_raw = kd_first_k_prefix_vs_text(
                                ctx.wrapper,
                                ctx.wrapper,
                                prefix,
                                scaffold.to(teacher_device, non_blocking=True),
                                targets,
                                K=current_K,
                            tau=args.kd_tau,
                            anchor_ids=ctx.anchor_ids,
                            append_bos_after_prefix=ctx.bos_flag,
                            deep_prefix_past=prefix_kv_cache,
                        )
                    else:
                        loss_kd_raw = kd_first_k_prefix_vs_text(
                            ctx.wrapper,
                            ctx.wrapper,
                            prefix,
                            scaffold.to(teacher_device, non_blocking=True),
                            targets,
                            K=current_K,
                            tau=args.kd_tau,
                            anchor_ids=ctx.anchor_ids,
                            append_bos_after_prefix=ctx.bos_flag,
                            deep_prefix_past=prefix_kv_cache,
                        )
                else:
                    loss_kd_raw = torch.zeros((), device=target_device)

                if args.state_kd_weight and args.state_kd_weight > 0.0:
                    loss_state_raw = kd_hidden_states_first_k(
                        ctx.wrapper,
                        prefix,
                        scaffold,
                        targets,
                        K=current_K,
                        layers=state_kd_layers,
                        append_bos_after_prefix=ctx.bos_flag,
                        anchor_ids=ctx.anchor_ids,
                        deep_prefix_past=prefix_kv_cache,
                    )
                else:
                    loss_state_raw = torch.zeros((), device=target_device)

                gist_loss_raw = torch.zeros((), device=target_device)
                if args.use_gist_head and args.gist_weight > 0.0:
                    head = gist_heads.get(ctx.name)
                    if head is not None:
                        gist_len = head.target_len
                        scaffold_slice = scaffold[:, :gist_len]
                        gist_pred = head(latents_for_adapter)
                        pad_id = getattr(ctx.wrapper.tokenizer, "pad_token_id", None)
                        valid = torch.ones_like(scaffold_slice, dtype=torch.bool, device=target_device)
                        if pad_id is not None:
                            valid = valid & scaffold_slice.ne(int(pad_id))
                        if args.gist_mask_prob > 0.0:
                            mask_rand = torch.rand_like(scaffold_slice.float(), device=target_device)
                            valid = valid & (mask_rand >= float(args.gist_mask_prob))
                        gist_targets = ctx.wrapper.input_embed(scaffold_slice)
                        mask = valid.unsqueeze(-1).float()
                        denom = mask.sum().clamp_min(1.0)
                        diff = (gist_pred - gist_targets) * mask
                        diff_sq = diff.pow(2).sum(dim=-1)
                        gist_loss_raw = diff_sq.sum() / (denom * gist_targets.size(-1))
                        # Move gist_loss_raw to device for aggregation
                        gist_loss_raw = gist_loss_raw.to(loss_device)

                align_loss = torch.zeros((), device=device)
                latent_align_loss = torch.zeros((), device=device)
                latent_prefix_align_loss = torch.zeros((), device=device)
                if training_mode == "text" and args.warmup_align_tokens > 0 and args.warmup_align_weight > 0.0:
                    max_align = min(int(args.warmup_align_tokens), prefix.shape[1])
                    pad_id = getattr(ctx.wrapper.tokenizer, "pad_token_id", None)
                    bos_id = getattr(ctx.wrapper.tokenizer, "bos_token_id", None)
                    if max_align > 0 and prefix.shape[1] > 0:
                        teacher_ids = ctx.token_ids[idx].to(target_device, non_blocking=True)
                        start = 0
                        if bos_id is not None and teacher_ids.size(1) > 0 and (teacher_ids[:, 0] == int(bos_id)).all():
                            start = 1
                        stop = min(start + max_align, teacher_ids.size(1))
                        token_slice = teacher_ids[:, start:stop]
                        if token_slice.numel() > 0:
                            mask = None
                            if pad_id is not None:
                                mask = token_slice.ne(int(pad_id))
                            teacher_embeds = ctx.wrapper.input_embed(token_slice)
                            prefix_slice = prefix[:, : token_slice.size(1), :]
                            align_loss = alignment_mse(prefix_slice, teacher_embeds, mask)
                            align_loss = align_loss * float(max(args.warmup_align_weight, 0.0))
                            # Move align_loss to device for aggregation
                            align_loss = align_loss.to(loss_device)
                if training_mode == "latent" and args.latent_align_weight > 0.0 and prefix.shape[1] > 0:
                    teacher_first_ids = ctx.first_token_ids[idx].to(target_device, non_blocking=True)
                    teacher_first_ids = teacher_first_ids.view(-1, 1)
                    teacher_emb = ctx.wrapper.input_embed(teacher_first_ids).squeeze(1).to(prefix.dtype)
                    latent_embed = prefix[:, 0, :]
                    if args.latent_align_metric in ("cosine", "both"):
                        cos = 1.0 - nn.functional.cosine_similarity(latent_embed, teacher_emb, dim=-1)
                        latent_align_loss = latent_align_loss + cos.mean()
                    if args.latent_align_metric in ("mse", "both"):
                        latent_align_loss = latent_align_loss + nn.functional.mse_loss(latent_embed, teacher_emb)
                    latent_align_loss = latent_align_loss * float(max(args.latent_align_weight, 0.0))
                    # Move latent_align_loss to device for aggregation
                    latent_align_loss = latent_align_loss.to(loss_device)
                if training_mode == "latent" and args.latent_prefix_align_weight > 0.0 and prefix.shape[1] > 0:
                    prefix_len = prefix.shape[1]
                    teacher_prefix_ids = ctx.token_ids[idx].to(target_device, non_blocking=True)
                    teacher_prefix_emb = ctx.wrapper.input_embed(teacher_prefix_ids).to(prefix.dtype)
                    teacher_prefix_emb = teacher_prefix_emb[:, :prefix_len]
                    overlap = min(prefix_len, teacher_prefix_emb.size(1))
                    if overlap > 0:
                        latent_prefix_align_loss = torch.zeros((), device=device)
                        if args.latent_align_metric in ("cosine", "both"):
                            cos = 1.0 - nn.functional.cosine_similarity(
                                prefix[:, :overlap, :], teacher_prefix_emb[:, :overlap, :], dim=-1
                            )
                            latent_prefix_align_loss = latent_prefix_align_loss + cos.mean()
                        if args.latent_align_metric in ("mse", "both"):
                            latent_prefix_align_loss = latent_prefix_align_loss + nn.functional.mse_loss(
                                prefix[:, :overlap, :], teacher_prefix_emb[:, :overlap, :]
                            )
                        latent_prefix_align_loss = latent_prefix_align_loss * float(max(args.latent_prefix_align_weight, 0.0))
                        # Move latent_prefix_align_loss to device for aggregation
                        latent_prefix_align_loss = latent_prefix_align_loss.to(loss_device)

                grad_diag_values: Dict[str, torch.Tensor] = {}
                if (
                    grad_diag_interval > 0
                    and grad_diag_component_set
                    and prefix_raw.requires_grad
                    and (global_step % grad_diag_interval == 0)
                ):
                    diag_sources: Dict[str, torch.Tensor] = {
                        "tf": loss_tf_latent,
                        "first": loss_first_raw,
                        "kce": loss_kce_raw,
                        "kd": loss_kd_raw,
                        "state": loss_state_raw,
                        "align": align_loss,
                        "latent_align": latent_align_loss,
                        "latent_prefix_align": latent_prefix_align_loss,
                        "gist": gist_loss_raw,
                    }
                    for name in grad_diag_components:
                        term = diag_sources.get(name)
                        if term is None:
                            continue
                        try:
                            grads = torch.autograd.grad(
                                term,
                                prefix_raw,
                                retain_graph=True,
                                allow_unused=True,
                            )
                        except RuntimeError as exc:
                            if args.debug:
                                print(f"[WARN] grad_diag({ctx.name}:{name}) failed: {exc}")
                            continue
                        grad_tensor = grads[0] if grads else None
                        if grad_tensor is None:
                            value = torch.zeros((), device=target_device)
                        else:
                            value = grad_tensor.float().pow(2).sum().sqrt()
                        grad_diag_values[f"grad_{name}"] = value.detach()

                latent_scale = 1.0
                if training_mode == "text":
                    start_scale = float(max(args.warmup_text_latent_weight, 0.0))
                    end_scale = float(max(args.warmup_text_latent_weight_end, 0.0))
                    if warmup_active and warmup_total_steps > 0:
                        frac = min(1.0, max(0.0, batch_index / float(max(1, warmup_total_steps))))
                        latent_scale = start_scale + (end_scale - start_scale) * frac
                    else:
                        latent_scale = end_scale

                effective_first_weight = float(current_first_weight)
                loss_tf = latent_scale * loss_tf_latent
                loss_first = latent_scale * loss_first_raw
                loss_kce = latent_scale * loss_kce_raw
                loss_kd = latent_scale * loss_kd_raw
                loss_state = latent_scale * loss_state_raw
                gist_weight = float(max(args.gist_weight, 0.0)) if args.use_gist_head else 0.0
                gist_loss = gist_weight * gist_loss_raw

                if (
                    enable_first_token_loss
                    and autoscale_first
                    and training_mode == "latent"
                    and latent_scale > 0.0
                ):
                    denom_val = float(loss_tf_latent.detach().abs().clamp_min(1e-4).item())
                    numer_val = float(loss_first_raw.detach().abs().item())
                    if denom_val > 0.0:
                        ratio = numer_val / denom_val
                        if ratio > 1.0:
                            effective_first_weight *= max(min(ratio, 8.0), 1.0)

                manifold_loss = manifold_stat_loss(
                    prefix,
                    embed_stats[ctx.name],
                    args.manifold_stat_weight,
                )
                # Move manifold_loss to device for aggregation (avoid device mismatch in multi-GPU)
                manifold_loss = manifold_loss.to(loss_device)

                text_teacher_loss = torch.zeros((), device=device)
                align_loss = torch.zeros((), device=device)
                latent_align_loss = torch.zeros((), device=device)
                latent_prefix_align_loss = torch.zeros((), device=device)
                if training_mode == "text" and args.warmup_align_tokens > 0 and args.warmup_align_weight > 0.0:
                    max_align = min(int(args.warmup_align_tokens), prefix.shape[1])
                    pad_id = getattr(ctx.wrapper.tokenizer, "pad_token_id", None)
                    bos_id = getattr(ctx.wrapper.tokenizer, "bos_token_id", None)
                    if max_align > 0 and prefix.shape[1] > 0:
                        teacher_ids = ctx.token_ids[idx].to(target_device, non_blocking=True)
                        start = 0
                        if bos_id is not None and teacher_ids.size(1) > 0 and (teacher_ids[:, 0] == int(bos_id)).all():
                            start = 1
                        stop = min(start + max_align, teacher_ids.size(1))
                        token_slice = teacher_ids[:, start:stop]
                        if token_slice.numel() > 0:
                            mask = None
                            if pad_id is not None:
                                mask = token_slice.ne(int(pad_id))
                            teacher_embeds = ctx.wrapper.input_embed(token_slice)
                            prefix_slice = prefix[:, : token_slice.size(1), :]
                            align_loss = alignment_mse(prefix_slice, teacher_embeds, mask)
                            align_loss = align_loss * float(max(args.warmup_align_weight, 0.0))
                            # Move align_loss to device for aggregation
                            align_loss = align_loss.to(loss_device)
                if training_mode == "latent" and args.latent_align_weight > 0.0 and prefix.shape[1] > 0:
                    teacher_first_ids = ctx.first_token_ids[idx].to(target_device, non_blocking=True)
                    teacher_first_ids = teacher_first_ids.view(-1, 1)
                    teacher_emb = ctx.wrapper.input_embed(teacher_first_ids).squeeze(1).to(prefix.dtype)
                    latent_embed = prefix[:, 0, :]
                    if args.latent_align_metric in ("cosine", "both"):
                        cos = 1.0 - nn.functional.cosine_similarity(latent_embed, teacher_emb, dim=-1)
                        latent_align_loss = latent_align_loss + cos.mean()
                    if args.latent_align_metric in ("mse", "both"):
                        latent_align_loss = latent_align_loss + nn.functional.mse_loss(latent_embed, teacher_emb)
                    latent_align_loss = latent_align_loss * float(max(args.latent_align_weight, 0.0))
                    # Move latent_align_loss to device for aggregation
                    latent_align_loss = latent_align_loss.to(loss_device)
                if training_mode == "latent" and args.latent_prefix_align_weight > 0.0 and prefix.shape[1] > 0:
                    prefix_len = prefix.shape[1]
                    teacher_prefix_ids = ctx.token_ids[idx].to(target_device, non_blocking=True)
                    teacher_prefix_emb = ctx.wrapper.input_embed(teacher_prefix_ids).to(prefix.dtype)
                    teacher_prefix_emb = teacher_prefix_emb[:, :prefix_len]
                    overlap = min(prefix_len, teacher_prefix_emb.size(1))
                    if overlap > 0:
                        latent_prefix_align_loss = torch.zeros((), device=device)
                        if args.latent_align_metric in ("cosine", "both"):
                            cos = 1.0 - nn.functional.cosine_similarity(
                                prefix[:, :overlap, :], teacher_prefix_emb[:, :overlap, :], dim=-1
                            )
                            latent_prefix_align_loss = latent_prefix_align_loss + cos.mean()
                        if args.latent_align_metric in ("mse", "both"):
                            latent_prefix_align_loss = latent_prefix_align_loss + nn.functional.mse_loss(
                                prefix[:, :overlap, :], teacher_prefix_emb[:, :overlap, :]
                            )
                        latent_prefix_align_loss = latent_prefix_align_loss * float(max(args.latent_prefix_align_weight, 0.0))
                        # Move latent_prefix_align_loss to device for aggregation
                        latent_prefix_align_loss = latent_prefix_align_loss.to(loss_device)
                if training_mode == "text" and float(max(args.warmup_text_teacher_weight, 0.0)) > 0.0:
                    text_teacher_loss, _, _ = loss_with_text_prompt_chunked(ctx.wrapper, scaffold, targets)
                    # Move loss to device for aggregation (avoid device mismatch in multi-GPU)
                    text_teacher_loss = text_teacher_loss.to(loss_device)
                else:
                    text_teacher_loss = torch.zeros((), device=device)

                model_loss = (
                    loss_tf
                    + effective_first_weight * loss_first
                    + args.k_ce_weight * loss_kce
                    + args.kd_first_k_weight * loss_kd
                    + args.state_kd_weight * loss_state
                    + args.manifold_stat_weight * manifold_loss
                    + align_loss
                    + latent_align_loss
                    + latent_prefix_align_loss
                    + float(max(args.warmup_text_teacher_weight, 0.0)) * text_teacher_loss
                    + entropy_bonus
                    + gist_loss
                ).to(device)
                total_model_loss = total_model_loss + model_loss

                penalty = penalty + scale_penalty(ctx.adapter, args.scale_l2, device).to(device)
                rms_pen = rms_pen + rms_raw_penalty(prefix_raw, ctx.wrapper, args.adapter_rms_l2).to(device)

                rms_raw_val = tensor_rms(prefix_raw)
                rms_cal_val = tensor_rms(prefix)
                stats_trackers[ctx.name]["rms_raw"].update(rms_raw_val)
                stats_trackers[ctx.name]["rms_cal"].update(rms_cal_val)

                losses_record.update({
                    "tf": loss_tf,
                    "first": loss_first,
                    "first_weight": torch.tensor(float(effective_first_weight), device=target_device),
                    "first_acc": first_acc_raw,
                    "first_entropy": first_entropy,
                    "entropy_loss": entropy_bonus,
                    "kce": loss_kce,
                    "kd": loss_kd,
                    "state": loss_state,
                    "manifold": manifold_loss,
                    "rms_raw": rms_raw_val,
                    "rms_cal": rms_cal_val,
                    "align": align_loss,
                    "latent_align": latent_align_loss,
                    "latent_prefix_align": latent_prefix_align_loss,
                    "text_tf": text_teacher_loss,
                    "latent_scale": torch.tensor(float(latent_scale), device=target_device),
                    "gist": gist_loss_raw,
                })
                for key, value in grad_diag_values.items():
                    losses_record[key] = value
                per_model_losses[ctx.name] = losses_record
                per_model_losses[ctx.name]["mode"] = "latent"
                if training_mode == "text":
                    per_model_losses[ctx.name]["mode"] = "text"

            # End of autocast context - loss aggregation happens outside
            # This ensures proper dtype handling for backward pass

            # Skip batch if NaN detected early in prefix
            if skip_batch_due_to_nan:
                if bad_latent_sources:
                    print(f"[WARN] Skipping batch due to non-finite prefix for {', '.join(bad_latent_sources)}")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            if training_mode == "text":
                parts_text = [
                    f"  step  {step+1}/{steps_per_epoch}",
                    "(warm-up text)" if warmup_active else "(tail text)",
                    f"align={_to_float(align_loss):.4f}",
                    f"text_tf={_to_float(text_teacher_loss):.4f}",
                    f"latent_scale={latent_scale:.2f}",
                ]
                print(" | ".join(parts_text))

            loss = (
                total_model_loss / float(len(model_contexts))
                + args.scale_l2 * penalty
                + args.adapter_rms_l2 * rms_pen
            )

            # Synchronize loss across DDP processes for logging
            if 'ddp_manager' in locals() and ddp_manager is not None and ddp_manager.initialized:
                # All-reduce the loss for consistent logging across processes
                loss_tensor = loss.clone().detach()
                ddp_manager.all_reduce(loss_tensor, op=dist.ReduceOp.AVG if 'dist' in dir() else None)
                loss_for_logging = loss_tensor.item()
            else:
                loss_for_logging = loss.item()

            # Track loss for epoch averaging
            if torch.isfinite(loss):
                epoch_losses.append(loss_for_logging)

            if not torch.isfinite(loss):
                if 'ddp_manager' not in locals() or ddp_manager is None or ddp_manager.should_log:
                    print("NaN/Inf loss; skipping step")
                # Clean up gradients and memory before skipping
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            loss_backward = loss / float(grad_accum_steps)

            # Handle backward pass with mixed precision
            if grad_scaler is not None:
                # FP16 requires GradScaler
                grad_scaler.scale(loss_backward).backward()
            else:
                # BF16 or no AMP - direct backward
                loss_backward.backward()

            # Log mixed precision statistics
            if grad_scaler is not None and batch_index % 100 == 0:
                scale = grad_scaler.get_scale()
                growth_tracker = grad_scaler._get_growth_tracker()
                print(f"[AMP Stats] Scale: {scale:.1f}, Growth tracker: {growth_tracker}")

            # On first step, verify latent adapters are receiving gradients
            if global_step == 0 and latent_adapter_params:
                adapter_grads_exist = sum(1 for p in latent_adapter_params if p.grad is not None)
                adapter_grad_norms = [p.grad.norm().item() for p in latent_adapter_params if p.grad is not None]
                print(f"[Gradient Check] Latent adapters: {adapter_grads_exist}/{len(latent_adapter_params)} have gradients")
                if adapter_grad_norms:
                    avg_grad = sum(adapter_grad_norms) / len(adapter_grad_norms)
                    max_grad = max(adapter_grad_norms)
                    print(f"[Gradient Check] Adapter gradient norms: avg={avg_grad:.6f}, max={max_grad:.6f}")
                    if avg_grad < 1e-8:
                        print(f"[Gradient Check] ⚠️  WARNING: Adapter gradients are near-zero (not learning!)")
                else:
                    print(f"[Gradient Check] ⚠️  WARNING: No adapter gradients found (adapters not training!)")

            # Detailed memory profiling for first few steps
            if batch_index < 3:
                mem_stats = get_gpu_memory_stats()
                if mem_stats:
                    print(f"    [Memory after backward] {mem_stats['total_allocated_gb']:.1f}GB allocated, peak {mem_stats['peak_allocated_gb']:.1f}GB")

            def _safe_grad_norm(params: Sequence[torch.nn.Parameter]) -> float:
                return float(_grad_norm(params)) if params else 0.0

            feature_grad_norms: Dict[str, float] = {}
            if enc_params:
                feature_grad_norms["encoder"] = _safe_grad_norm(enc_params)
            if llama_params:
                feature_grad_norms["adapter_llama"] = _safe_grad_norm(llama_params)
            if qwen_params:
                feature_grad_norms["adapter_qwen"] = _safe_grad_norm(qwen_params)
            if extra_llama_params:
                feature_grad_norms["extra_llama"] = _safe_grad_norm(extra_llama_params)
            if extra_qwen_params:
                feature_grad_norms["extra_qwen"] = _safe_grad_norm(extra_qwen_params)
            if latent_adapter_params:
                feature_grad_norms["latent_adapter"] = _safe_grad_norm(latent_adapter_params)
            if refiner_params:
                feature_grad_norms["latent_refiner"] = _safe_grad_norm(refiner_params)
            if gist_params:
                feature_grad_norms["gist"] = _safe_grad_norm(gist_params)
            if deep_prefix_generators:
                dp_params = [p for gen in deep_prefix_generators.values() for p in gen.parameters() if p.requires_grad]
                if dp_params:
                    feature_grad_norms["deep_prefix"] = _safe_grad_norm(dp_params)
            for name, plist in coprocessor_param_bank.items():
                if plist:
                    feature_grad_norms[f"coprocessor_{name}"] = _safe_grad_norm(plist)

            grad_norm_val = _grad_norm(params_for_clip)
            if not math.isfinite(grad_norm_val):
                print("⚠️  Non-finite gradient detected; skipping optimizer step for this batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Determine if we should take an optimizer step
            should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == rank_steps_per_epoch)

            # In DDP mode with gradient accumulation, we need to control gradient sync
            # Only sync gradients on the last accumulation step to avoid unnecessary communication
            if 'ddp_manager' in locals() and ddp_manager is not None and ddp_manager.initialized and grad_accum_steps > 1:
                # This is an advanced DDP optimization - disable gradient sync except on step
                for model in [encoder] + list(adapters.values()):
                    if hasattr(model, 'require_backward_grad_sync'):
                        model.require_backward_grad_sync = should_step

            if should_step:
                if grad_scaler is not None:
                    # FP16 path with GradScaler
                    # Unscale gradients before clipping
                    grad_scaler.unscale_(optimizer)

                    if args.max_grad_norm and args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(args.max_grad_norm))

                    _align_optimizer_state_to_param_devices(optimizer)

                    # Step with scaled gradients
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    lr_scheduler.step()  # Update learning rate after optimizer step
                else:
                    # BF16 or no AMP - direct optimizer step
                    if args.max_grad_norm and args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=float(args.max_grad_norm))
                    _align_optimizer_state_to_param_devices(optimizer)
                    optimizer.step()
                    lr_scheduler.step()  # Update learning rate after optimizer step

                optimizer.zero_grad(set_to_none=True)

                # Detailed memory profiling for first few steps
                if batch_index < 3:
                    mem_stats = get_gpu_memory_stats()
                    if mem_stats:
                        print(f"    [Memory after optimizer] {mem_stats['total_allocated_gb']:.1f}GB allocated")

            total_norm = grad_norm_val

            # Grad norm (monitor) – encoder only as a proxy
            global_step += 1
            # Synchronize after operations for accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.time() - t0
            ema_step_time = dt if (locals().get("ema_step_time", None) is None) else (0.9 * locals()["ema_step_time"] + 0.1 * dt)

            # Synchronize all processes at the end of each step if using DDP
            if 'ddp_manager' in locals() and ddp_manager is not None and ddp_manager.initialized:
                ddp_manager.barrier()

            if (step+1) % 10 == 0 or (step+1) == rank_steps_per_epoch:
                # Only print from main process in DDP mode
                should_print = ('ddp_manager' not in locals() or ddp_manager is None or
                               ddp_manager.should_log)
                if should_print:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    parts = [
                        f"  step  {step+1}/{rank_steps_per_epoch}",
                        f"grad_norm={total_norm:.2f}",
                        f"sec/step~{dt:.2f}",
                        f"lr={current_lr:.2e}",
                        f"keep={keep_prob:.2f}",
                        f"K={current_K}",
                    ]
                if first_ce_schedule != "none":
                    parts.append(f"first_w={current_first_weight:.2f}")
                for ctx in model_contexts:
                    metrics = per_model_losses[ctx.name]
                    mode_tag = metrics.get("mode", "latent")
                    mode_label = "T" if mode_tag == "text" else "L"
                    msg_ctx = (
                        f"{ctx.name}({mode_label}): tf={_to_float(metrics['tf']):.4f}"
                        f" first={_to_float(metrics['first']):.4f}"
                        f" kCE={_to_float(metrics['kce']):.4f}"
                        f" KD={_to_float(metrics['kd']):.4f}"
                    )
                    msg_ctx += f" acc={_to_float(metrics.get('first_acc', 0.0)):.3f}"

                    # Log predictions when accuracy > 0 (for debugging learning)
                    if training_mode == "latent" and _to_float(metrics.get('first_acc', 0.0)) > 0.0:
                        if "first_pred" in metrics and "first_targets" in metrics:
                            preds = metrics["first_pred"]
                            targets = metrics["first_targets"]
                            # Find matches for debugging
                            matches = (preds == targets).nonzero(as_tuple=True)[0]
                            if len(matches) > 0:
                                # Show first correct prediction
                                idx = matches[0].item()
                                pred_tok = ctx.wrapper.tokenizer.decode([preds[idx].item()], skip_special_tokens=False)
                                msg_ctx += f" [✓'{pred_tok}']"

                    if args.state_kd_weight > 0.0:
                        msg_ctx += f" state={_to_float(metrics['state']):.4f}"
                    if args.first_token_entropy_weight > 0.0 and "first_entropy" in metrics:
                        msg_ctx += f" ent={_to_float(metrics['first_entropy']):.3f}"
                    if args.manifold_stat_weight > 0.0:
                        msg_ctx += f" man={_to_float(metrics['manifold']):.4f}"
                    if args.warmup_align_weight > 0.0:
                        msg_ctx += f" align={_to_float(metrics['align']):.4f}"
                    if args.latent_align_weight > 0.0:
                        msg_ctx += f" latA={_to_float(metrics['latent_align']):.4f}"
                    if args.latent_prefix_align_weight > 0.0:
                        msg_ctx += f" latP={_to_float(metrics['latent_prefix_align']):.4f}"
                    if args.use_gist_head and args.gist_weight > 0.0:
                        msg_ctx += f" gist={_to_float(metrics['gist']):.4f}"
                    if grad_diag_components:
                        for diag_name in grad_diag_components:
                            key = f"grad_{diag_name}"
                            if key in metrics:
                                msg_ctx += f" {key}={_to_float(metrics[key]):.3e}"
                    parts.append(msg_ctx)
                    if args.scale_l2 > 0.0:
                        parts.append(
                            f"scale_pen({ctx.name})={scale_penalty(ctx.adapter, args.scale_l2, device).item():.4e}"
                        )
                if feature_grad_norms:
                    grad_bits = ", ".join(f"{k}={v:.3e}" for k, v in feature_grad_norms.items())
                    parts.append(f"feature_grads[{grad_bits}]")
                parts.append(f"K={current_K} tau={args.kd_tau:.2f}")
                if args.save_training_stats:
                    stats_msgs = []
                    for ctx in model_contexts:
                        tracker = stats_trackers[ctx.name]
                        stats_msgs.append(
                            f"{ctx.name}: rms_raw~{tracker['rms_raw'].mean:.4f}"
                            f" rms_cal~{tracker['rms_cal'].mean:.4f}"
                            f" embed_rms~{tracker['embed_rms']:.5f}"
                        )
                    parts.append("stats=[" + "; ".join(stats_msgs) + "]")
                print(" | ".join(parts))

            # Log GPU memory every 10 steps
            gpu_stats = log_gpu_memory(prefix=f"  [Step {step+1}] ")

            # Check batch size after step 5 (once we have real activation memory data)
            if (step + 1) == 5 and epoch == start_epoch and gpu_stats:
                peak_gb = gpu_stats.get('peak_allocated_gb', 0)
                suggested_batch, reason = suggest_batch_size_adjustment(
                    args.batch_size,
                    peak_gb,
                    after_forward_pass=True
                )
                if suggested_batch != args.batch_size:
                    print(f"  [Batch Size Suggestion after {step+1} steps] {reason}")
                    print(f"    Current: {args.batch_size}, Suggested: {suggested_batch}")
                    print(f"    To apply: set BATCH_SIZE_STAGEA/B={suggested_batch} in run script")

            if diagnostic_log_path and ((step + 1) % 10 == 0 or (step + 1) == steps_per_epoch):
                    diag_entry: Dict[str, Any] = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "epoch_step": step + 1,
                        "mode": training_mode,
                        "keep_prob": keep_prob,
                        "K": current_K,
                        "first_weight": float(current_first_weight),
                        "grad_norm": float(total_norm),
                        "sec_per_step": float(dt),
                        "latent_scale": float(latent_scale),
                        "gpu_memory": gpu_stats if gpu_stats else {},
                        "models": {},
                    }
                    for ctx in model_contexts:
                        metrics = per_model_losses[ctx.name]
                        diag_entry["models"][ctx.name] = {
                            "mode": metrics.get("mode", "latent"),
                            "tf": _to_float(metrics.get("tf", 0.0)),
                            "first": _to_float(metrics.get("first", 0.0)),
                            "first_acc": _to_float(metrics.get("first_acc", 0.0)),
                            "first_entropy": _to_float(metrics.get("first_entropy", 0.0)),
                            "entropy_loss": _to_float(metrics.get("entropy_loss", 0.0)),
                            "kce": _to_float(metrics.get("kce", 0.0)),
                            "kd": _to_float(metrics.get("kd", 0.0)),
                            "state": _to_float(metrics.get("state", 0.0)),
                            "align": _to_float(metrics.get("align", 0.0)),
                            "latent_align": _to_float(metrics.get("latent_align", 0.0)),
                            "latent_prefix_align": _to_float(metrics.get("latent_prefix_align", 0.0)),
                            "gist": _to_float(metrics.get("gist", 0.0)),
                        }
                        # Add enhanced first-token statistics
                        if "first_token_logit_stats" in metrics:
                            diag_entry["models"][ctx.name]["first_token_logit_stats"] = metrics["first_token_logit_stats"]
                        if "first_acc_top5" in metrics:
                            diag_entry["models"][ctx.name]["first_acc_top5"] = _to_float(metrics.get("first_acc_top5", 0.0))
                        # Add prediction histogram (token counts)
                        if "pred_tokens" in metrics and metrics["pred_tokens"]:
                            from collections import Counter
                            token_counts = Counter(metrics["pred_tokens"])
                            # Store top 10 most frequent predictions
                            diag_entry["models"][ctx.name]["prediction_histogram"] = dict(token_counts.most_common(10))
                        # Add LoRA weight norms if available
                        if args.use_lora and ctx.wrapper.model is not None:
                            lora_norms = _compute_lora_weight_norms(ctx.wrapper.model)
                            if lora_norms:
                                # Compute average norm across all LoRA weights for summary
                                avg_lora_norm = sum(lora_norms.values()) / len(lora_norms) if lora_norms else 0.0
                                diag_entry["models"][ctx.name]["lora_avg_norm"] = float(avg_lora_norm)
                                # Store first few layer norms as examples
                                example_norms = {k: v for k, v in list(lora_norms.items())[:3]}
                                diag_entry["models"][ctx.name]["lora_examples"] = example_norms
                        for diag_name in grad_diag_components:
                            key = f"grad_{diag_name}"
                            if key in metrics:
                                diag_entry["models"][ctx.name][key] = _to_float(metrics[key])
                    try:
                        with open(diagnostic_log_path, "a") as diag_f:
                            if feature_grad_norms:
                                diag_entry["feature_grads"] = {k: float(v) for k, v in feature_grad_norms.items()}
                            json.dump(diag_entry, diag_f)
                            diag_f.write("\n")
                    except Exception as exc:
                        if args.debug:
                            print(f"[WARN] Failed to append diagnostic log: {exc}")

            # ---- Peak checkpointing: save when first_acc improves in latent mode
            if training_mode == "latent" and model_contexts:
                # Get first_acc from first model (typically llama)
                current_first_acc_raw = float(_to_float(per_model_losses[model_contexts[0].name].get("first_acc", 0.0)))

                # Update exponential moving average to smooth out batch-level noise
                # EMA reduces false peaks from lucky batches (e.g., 9/36 correct = 25% but not sustained)
                first_acc_ema = ema_alpha * current_first_acc_raw + (1.0 - ema_alpha) * first_acc_ema

                # Dual-trigger peak detection:
                # (a) EMA improves and >= 1%, OR
                # (b) Raw batch >= 8% (hero threshold, catches spikes before EMA responds)
                should_save_peak = (
                    (first_acc_ema > best_first_acc and first_acc_ema >= 0.01) or  # EMA trigger (lowered from 5%)
                    (current_first_acc_raw >= 0.08)  # Raw batch trigger (catches transient spikes)
                )

                if should_save_peak:
                    # New peak detected via dual trigger (EMA >= 1% OR raw batch >= 8%)
                    # Lowered EMA threshold from 5% → 1% (previous runs showed EMA ~1.7% at peaks)
                    # Added raw batch fallback to catch spikes before EMA responds (e.g., 12.5% raw at step 263)
                    best_first_acc = max(best_first_acc, first_acc_ema)  # Track best EMA seen
                    best_checkpoint_step = global_step

                    # Save "best" checkpoint
                    best_save_dir = os.path.join(os.path.dirname(args.save_dir), f"{os.path.basename(args.save_dir)}_best")
                    os.makedirs(best_save_dir, exist_ok=True)

                    cfg = {
                        "d_z": args.d_z,
                        "latent_len": total_latent_len,
                        "latent_shared_len": latent_shared_len,
                        "latent_private_len": latent_private_len,
                        "byte_max": args.max_bytes,
                        "llama_id": args.llama_id,
                        "qwen_id": args.qwen_id,
                        "encoder_type": args.encoder_type,
                        "encoder_use_chat_template": bool(args.encoder_use_chat_template),
                        "hf_encoder_id": (args.hf_encoder_id if hasattr(args, "hf_encoder_id") else ""),
                        "max_enc_tokens": (args.max_enc_tokens if hasattr(args, "max_enc_tokens") else 1024),
                        "encoder_backbone": (args.encoder_backbone or ""),
                        "freeze_encoder": bool(args.freeze_encoder),
                        "use_chat_template": bool(args.use_chat_template),
                        "warm_anchor_mode": args.warm_anchor_mode,
                        "strip_anchor_text": strip_anchor_literal,
                        "max_anchor_tokens": args.max_anchor_tokens,
                        "train_append_bos_after_prefix": args.train_append_bos_after_prefix,
                        "first_token_ce_weight": args.first_token_ce_weight,
                        "first_token_ce_schedule": args.first_token_ce_schedule,
                        "first_token_ce_peak": args.first_token_ce_peak,
                        "first_token_ce_warmup_frac": args.first_token_ce_warmup_frac,
                        "first_token_entropy_weight": args.first_token_entropy_weight,
                        "adapter_hidden_mult": args.adapter_hidden_mult,
                        "adapter_dropout": args.adapter_dropout,
                        "adapter_colorize": bool(args.adapter_colorize),
                        "adapter_enable_metadata": bool(args.adapter_metadata),
                        "llama_device_map": args.llama_device_map,
                        "qwen_device_map": args.qwen_device_map,
                        "llama_devices": args.llama_devices,
                        "qwen_devices": args.qwen_devices,
                        "gpu_mem_gib": args.gpu_mem_gib,
                        "use_lora": bool(args.use_lora),
                        "lora_r": args.lora_r,
                        "lora_alpha": args.lora_alpha,
                        "lora_dropout": args.lora_dropout,
                        "lora_target_modules": args.lora_target_modules,
                        "lora_firstN": args.lora_firstN,
                        "use_latent_adapters": bool(args.use_latent_adapters),
                        "latent_adapter_layers": args.latent_adapter_layers,
                        "latent_adapter_heads": args.latent_adapter_heads,
                        "latent_adapter_dropout": args.latent_adapter_dropout,
                    }
                    dp_len_cfg = int(
                        args.deep_prefix_len
                        if (feature_registry.has("deep_prefix") and args.deep_prefix_len is not None)
                        else (latent_shared_len + latent_private_len)
                    ) if feature_registry.has("deep_prefix") else 0
                    cfg["deep_prefix"] = {
                        "enabled": feature_registry.has("deep_prefix"),
                        "len": dp_len_cfg,
                        "dropout": float(args.deep_prefix_dropout),
                    }
                    cfg["coprocessor"] = {
                        "enabled": feature_registry.has("coprocessor"),
                        "len": int(args.coprocessor_len) if feature_registry.has("coprocessor") else 0,
                        "width": int(args.coprocessor_width),
                        "dropout": float(args.coprocessor_dropout),
                        "kv_scale": float(args.coprocessor_kv_scale),
                        "pool": args.coprocessor_pool,
                        "summaries": coprocessor_summaries,
                    }
                    cfg["warm_anchor_text"] = anchor_texts.get("llama", "")
                    cfg["warm_anchor_texts"] = anchor_texts
                    cfg["warm_anchor_modes"] = anchor_modes
                    cfg["best_first_acc"] = float(best_first_acc)
                    cfg["best_step"] = int(best_checkpoint_step)

                    state_blob = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "args": vars(args),
                        "rng": {
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        },
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "adapter_scale": {
                            name: float(adapter.scale.detach().cpu().item())
                            for name, adapter in adapters.items()
                            if getattr(adapter, "scale", None) is not None
                        },
                        "encoder": encoder.state_dict(),
                        "best_first_acc": float(best_first_acc),
                        "best_step": int(best_checkpoint_step),
                    }
                    for name, adapter in adapters.items():
                        state_blob[f"adp_{name}"] = adapter.state_dict()
                    for name, gen in deep_prefix_generators.items():
                        state_blob[f"deep_prefix_{name}"] = gen.state_dict()
                    for name, module in coprocessors.items():
                        state_blob[f"coprocessor_{name}"] = module.state_dict()
                    if latent_refiner is not None:
                        state_blob["refiner"] = latent_refiner.state_dict()
                    # Save latent adapters
                    for wrapper in wrappers_in_use:
                        if wrapper.use_latent_adapters:
                            # Find wrapper name (llama or qwen)
                            wrapper_name = "llama" if wrapper is llama else "qwen"
                            state_blob[f"latent_adapters_{wrapper_name}"] = wrapper.latent_adapters.state_dict()
                    artifacts = {
                        "encoder.pt": encoder.state_dict(),
                        "state.pt": state_blob,
                        "config.json": cfg,
                    }
                    for name, adapter in adapters.items():
                        artifacts[f"adapter_{name}.pt"] = adapter.state_dict()
                    for name, gen in deep_prefix_generators.items():
                        artifacts[f"deep_prefix_{name}.pt"] = gen.state_dict()
                    for name, module in coprocessors.items():
                        artifacts[f"coprocessor_{name}.pt"] = module.state_dict()
                    if latent_refiner is not None:
                        artifacts["refiner.pt"] = latent_refiner.state_dict()
                    # Save latent adapters as separate .pt files
                    for wrapper in wrappers_in_use:
                        if wrapper.use_latent_adapters:
                            wrapper_name = "llama" if wrapper is llama else "qwen"
                            artifacts[f"latent_adapters_{wrapper_name}.pt"] = wrapper.latent_adapters.state_dict()
                    # Only save from main process in DDP
                    if 'ddp_manager' not in locals() or ddp_manager is None or ddp_manager.should_save:
                        save_latest_checkpoint(best_save_dir, artifacts, pre_prune=False, post_prune=False, verbose=False)

                    # Save LoRA weights (critical for proper evaluation)
                    try:
                        from peft import PeftModel  # type: ignore
                        if isinstance(getattr(llama, "model", None), PeftModel):
                            if args.use_lora:
                                llama.model.save_pretrained(os.path.join(best_save_dir, "lora_llama"))
                        if isinstance(getattr(qwen, "model", None), PeftModel):
                            if args.use_lora:
                                qwen.model.save_pretrained(os.path.join(best_save_dir, "lora_qwen"))
                    except ImportError:
                        pass

                    print(f"  🌟 NEW PEAK: first_acc_ema={best_first_acc:.1%} (raw_batch={current_first_acc_raw:.1%}) at step {global_step} → saved to {best_save_dir}", flush=True)

                    # Log sample predictions to verify quality (not just lucky guesses)
                    ctx_name = model_contexts[0].name
                    if "first_pred" in per_model_losses[ctx_name] and "first_targets" in per_model_losses[ctx_name]:
                        preds = per_model_losses[ctx_name]["first_pred"]
                        targets = per_model_losses[ctx_name]["first_targets"]
                        tokenizer = model_contexts[0].wrapper.tokenizer

                        # Show first 5 predictions vs targets
                        print(f"      Sample predictions (first 5):")
                        for i in range(min(5, len(preds))):
                            pred_tok = tokenizer.decode([preds[i].item()], skip_special_tokens=False)
                            targ_tok = tokenizer.decode([targets[i].item()], skip_special_tokens=False)
                            match = "✓" if preds[i] == targets[i] else "✗"
                            print(f"        {match} pred='{pred_tok}' | gold='{targ_tok}'")

                        # Check diversity: unique predictions
                        unique_preds = len(torch.unique(preds))
                        print(f"      Prediction diversity: {unique_preds}/{len(preds)} unique tokens")

                        # Top-3 most frequent predictions
                        pred_counts = {}
                        for p in preds.cpu().tolist():
                            pred_counts[p] = pred_counts.get(p, 0) + 1
                        top_preds = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                        print(f"      Top-3 predictions:", end=" ")
                        for tok_id, count in top_preds:
                            tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
                            print(f"'{tok_str}'({count})", end=" ")
                        print()

            # ---- Periodic checkpoint: save + prune
            if args.save_every and (global_step % args.save_every == 0):
                os.makedirs(args.save_dir, exist_ok=True)
                cfg = {
                    "d_z": args.d_z,
                    "latent_len": total_latent_len,
                    "latent_shared_len": latent_shared_len,
                    "latent_private_len": latent_private_len,
                    "byte_max": args.max_bytes,
                    "llama_id": args.llama_id,
                    "qwen_id": args.qwen_id,
                    "encoder_type": args.encoder_type,
                    "encoder_use_chat_template": bool(args.encoder_use_chat_template),
                    "hf_encoder_id": (args.hf_encoder_id if hasattr(args, "hf_encoder_id") else ""),
                    "max_enc_tokens": (args.max_enc_tokens if hasattr(args, "max_enc_tokens") else 1024),
                    "encoder_backbone": (args.encoder_backbone or ""),
                    "freeze_encoder": bool(args.freeze_encoder),
                    "use_chat_template": bool(args.use_chat_template),
                    "warm_anchor_mode": args.warm_anchor_mode,
                    "strip_anchor_text": strip_anchor_literal,
                    "max_anchor_tokens": args.max_anchor_tokens,
                    "train_append_bos_after_prefix": args.train_append_bos_after_prefix,
                    "first_token_ce_weight": args.first_token_ce_weight,
                    "first_token_ce_schedule": args.first_token_ce_schedule,
                    "first_token_ce_peak": args.first_token_ce_peak,
                    "first_token_ce_warmup_frac": args.first_token_ce_warmup_frac,
                    "first_token_entropy_weight": args.first_token_entropy_weight,
                    "adapter_hidden_mult": args.adapter_hidden_mult,
                    "adapter_dropout": args.adapter_dropout,
                    "adapter_colorize": bool(args.adapter_colorize),
                    "adapter_enable_metadata": bool(args.adapter_metadata),
                    "llama_device_map": args.llama_device_map,
                    "qwen_device_map": args.qwen_device_map,
                    "llama_devices": args.llama_devices,
                    "qwen_devices": args.qwen_devices,
                    "gpu_mem_gib": args.gpu_mem_gib,
                    "use_lora": bool(args.use_lora),
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "lora_target_modules": args.lora_target_modules,
                    "lora_firstN": args.lora_firstN,
                    "use_prefix": bool(args.use_prefix),
                    "prefix_tokens": args.prefix_tokens,
                    "prefix_projection": bool(args.prefix_projection),
                    "peft_prefix_all_layers": str(getattr(args, "peft_prefix_all_layers", "yes")),
                    "manifold_stat_weight": args.manifold_stat_weight,
                    "state_kd_weight": args.state_kd_weight,
                    "state_kd_layers": args.state_kd_layers,
                    "K": args.K,
                    "k_ce_weight": args.k_ce_weight,
                    "kd_first_k_weight": args.kd_first_k_weight,
                    "kd_tau": args.kd_tau,
                    "max_answer_tokens": args.max_answer_tokens,
                    "grad_accum_steps": grad_accum_steps,
                    "seed": args.seed,
                    "data_seed": args.data_seed,
                    "models": model_keys,
                    "warmup_text_latent_steps": warmup_total_steps,
                    "warmup_text_latent_weight": args.warmup_text_latent_weight,
                    "warmup_text_latent_weight_end": args.warmup_text_latent_weight_end,
                    "warmup_text_teacher_weight": args.warmup_text_teacher_weight,
                    "warmup_tail_prob": args.warmup_tail_prob,
                    "warmup_align_tokens": args.warmup_align_tokens,
                    "warmup_align_weight": args.warmup_align_weight,
                    "grad_diag_interval": grad_diag_interval,
                    "grad_diag_components": args.grad_diag_components,
                    "gist_head": {
                        "enabled": bool(args.use_gist_head),
                        "target_len": int(args.gist_target_len),
                        "hidden": int(args.gist_hidden),
                        "layers": int(args.gist_layers),
                        "dropout": float(args.gist_dropout),
                        "weight": float(args.gist_weight),
                        "mask_prob": float(args.gist_mask_prob),
                    },
                }
                dp_len_cfg = int(
                    args.deep_prefix_len
                    if (feature_registry.has("deep_prefix") and args.deep_prefix_len is not None)
                    else (latent_shared_len + latent_private_len)
                ) if feature_registry.has("deep_prefix") else 0
                cfg["deep_prefix"] = {
                    "enabled": feature_registry.has("deep_prefix"),
                    "len": dp_len_cfg,
                    "dropout": float(args.deep_prefix_dropout),
                }
                cfg["coprocessor"] = {
                    "enabled": feature_registry.has("coprocessor"),
                    "len": int(args.coprocessor_len) if feature_registry.has("coprocessor") else 0,
                    "width": int(args.coprocessor_width),
                    "dropout": float(args.coprocessor_dropout),
                    "kv_scale": float(args.coprocessor_kv_scale),
                    "pool": args.coprocessor_pool,
                    "summaries": coprocessor_summaries,
                }
                cfg["warm_anchor_text"] = anchor_texts.get("llama", "")
                cfg["warm_anchor_texts"] = anchor_texts
                cfg["warm_anchor_modes"] = anchor_modes

                state_blob = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "args": vars(args),
                    "rng": {
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    },
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "adapter_scale": {
                        name: float(adapter.scale.detach().cpu().item())
                        for name, adapter in adapters.items()
                        if getattr(adapter, "scale", None) is not None
                    },
                    "encoder": encoder.state_dict(),
                }
                for name, adapter in adapters.items():
                    state_blob[f"adp_{name}"] = adapter.state_dict()
                for name, gen in deep_prefix_generators.items():
                    state_blob[f"deep_prefix_{name}"] = gen.state_dict()
                for name, module in coprocessors.items():
                    state_blob[f"coprocessor_{name}"] = module.state_dict()
                if latent_refiner is not None:
                    state_blob["refiner"] = latent_refiner.state_dict()
                artifacts = {
                    "encoder.pt": encoder.state_dict(),
                    "state.pt": state_blob,
                    "config.json": cfg,
                }
                for name, adapter in adapters.items():
                    artifacts[f"adapter_{name}.pt"] = adapter.state_dict()
                for name, gen in deep_prefix_generators.items():
                    artifacts[f"deep_prefix_{name}.pt"] = gen.state_dict()
                for name, module in coprocessors.items():
                    artifacts[f"coprocessor_{name}.pt"] = module.state_dict()
                if latent_refiner is not None:
                    artifacts["refiner.pt"] = latent_refiner.state_dict()
                # Only save from main process in DDP
                if 'ddp_manager' not in locals() or ddp_manager is None or ddp_manager.should_save:
                    save_latest_checkpoint(args.save_dir, artifacts, pre_prune=True, post_prune=True, verbose=True)
                    print(f"  ✅ Saved (and pruned to) latest at step {global_step}", flush=True)

        # ===== End-of-epoch evaluation and metrics saving =====
        # Compute and save comprehensive metrics after each epoch
        if 'ddp_manager' not in locals() or ddp_manager is None or ddp_manager.should_log:
            epoch_metrics = {}
            epoch_metrics['epoch'] = epoch + 1
            epoch_metrics['global_step'] = global_step
            epoch_metrics['timestamp'] = time.time()

            # Calculate average training loss for this epoch
            if 'epoch_losses' in locals() and epoch_losses:
                avg_train_loss = sum(epoch_losses) / len(epoch_losses)
                epoch_metrics['train_loss'] = float(avg_train_loss)

            # Calculate perplexity from cross-entropy loss
            if 'avg_train_loss' in locals():
                epoch_metrics['train_perplexity'] = float(math.exp(min(avg_train_loss, 100)))  # Cap to avoid overflow

            # Collect model-specific metrics if available
            for model_name in ['llama', 'qwen']:
                if model_name in stats_trackers:
                    tracker = stats_trackers[model_name]
                    model_metrics = {}

                    # RMS statistics
                    if "rms_raw" in tracker and hasattr(tracker["rms_raw"], 'mean'):
                        model_metrics['rms_raw_mean'] = float(tracker["rms_raw"].mean)
                        if hasattr(tracker["rms_raw"], 'std'):
                            model_metrics['rms_raw_std'] = float(tracker["rms_raw"].std)

                    if "rms_cal" in tracker and hasattr(tracker["rms_cal"], 'mean'):
                        model_metrics['rms_cal_mean'] = float(tracker["rms_cal"].mean)
                        if hasattr(tracker["rms_cal"], 'std'):
                            model_metrics['rms_cal_std'] = float(tracker["rms_cal"].std)

                    # First token accuracy (important metric for evaluation)
                    if "first_acc" in tracker and hasattr(tracker["first_acc"], 'mean'):
                        model_metrics['first_token_accuracy'] = float(tracker["first_acc"].mean)
                        if hasattr(tracker["first_acc"], 'max'):
                            model_metrics['first_token_accuracy_max'] = float(tracker["first_acc"].max)

                    # Add any other tracked metrics
                    for metric_name in ['loss_tf', 'loss_first', 'loss_kce', 'loss_kd']:
                        if metric_name in tracker and hasattr(tracker[metric_name], 'mean'):
                            model_metrics[metric_name] = float(tracker[metric_name].mean)

                    epoch_metrics[f'{model_name}_metrics'] = model_metrics

            # Track best first token accuracy
            if 'first_acc_ema' in locals():
                epoch_metrics['first_acc_ema'] = float(first_acc_ema)
            if 'best_first_acc' in locals():
                epoch_metrics['best_first_acc'] = float(best_first_acc)

            # Validation metrics (if validation data available)
            # Note: Full evaluation would require running eval.py logic
            # For production use, integrate with eval.py functions

            # Compute simple validation metrics from training data
            # These serve as proxies until full evaluation is implemented
            if epoch_losses:
                # Use last 10% of epoch as validation proxy
                val_split = max(1, len(epoch_losses) // 10)
                val_losses = epoch_losses[-val_split:]
                epoch_metrics['validation_loss_proxy'] = float(sum(val_losses) / len(val_losses))
                epoch_metrics['validation_perplexity_proxy'] = float(math.exp(min(epoch_metrics['validation_loss_proxy'], 100)))

            # Placeholder for full evaluation metrics
            # To enable: run eval.py functions here with validation dataset
            epoch_metrics['f1_score'] = None  # Requires eval.py integration
            epoch_metrics['exact_match'] = None  # Requires eval.py integration
            epoch_metrics['bleu_score'] = None  # Requires eval.py integration

            # Add configuration info for reproducibility
            epoch_metrics['config'] = {
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'latent_len': args.latent_len,
                'd_z': args.d_z,
                'dataset': args.dataset if hasattr(args, 'dataset') else 'unknown'
            }

            # Save metrics to JSON file
            metrics_filename = os.path.join(args.save_dir, f'results_epoch_{epoch + 1}.json')
            os.makedirs(args.save_dir, exist_ok=True)
            with open(metrics_filename, 'w') as f:
                json.dump(epoch_metrics, f, indent=2)
            print(f"  📊 Saved epoch {epoch + 1} metrics to {metrics_filename}")

            # Also append to a consolidated metrics file for easy graphing
            all_metrics_file = os.path.join(args.save_dir, 'training_metrics.jsonl')
            with open(all_metrics_file, 'a') as f:
                f.write(json.dumps(epoch_metrics) + '\n')

            # Print summary metrics
            print(f"\n=== Epoch {epoch + 1} Summary ===")
            if 'avg_train_loss' in locals():
                print(f"  Average training loss: {avg_train_loss:.4f}")
                print(f"  Training perplexity: {epoch_metrics.get('train_perplexity', 'N/A'):.2f}")
            if 'first_acc_ema' in locals():
                print(f"  First token accuracy (EMA): {first_acc_ema:.4f}")
            print(f"================================\n")

    # ===== Final save =====
    os.makedirs(args.save_dir, exist_ok=True)
    cfg = {
        "d_z": args.d_z,
        "latent_len": total_latent_len,
        "latent_shared_len": latent_shared_len,
        "latent_private_len": latent_private_len,
        "byte_max": args.max_bytes,
        "llama_id": args.llama_id,
        "qwen_id": args.qwen_id,
        "encoder_type": args.encoder_type,
        "encoder_use_chat_template": bool(args.encoder_use_chat_template),
        "hf_encoder_id": (args.hf_encoder_id if hasattr(args, "hf_encoder_id") else ""),
        "max_enc_tokens": (args.max_enc_tokens if hasattr(args, "max_enc_tokens") else 1024),
        "encoder_backbone": (args.encoder_backbone or ""),
        "freeze_encoder": bool(args.freeze_encoder),
        "use_chat_template": bool(args.use_chat_template),
        "warm_anchor_text": anchor_texts.get("llama", ""),
        "warm_anchor_texts": anchor_texts,
        "warm_anchor_modes": anchor_modes,
        "strip_anchor_text": strip_anchor_literal,
        "max_anchor_tokens": args.max_anchor_tokens,
        "train_append_bos_after_prefix": args.train_append_bos_after_prefix,
        "first_token_ce_weight": args.first_token_ce_weight,
        "first_token_ce_schedule": args.first_token_ce_schedule,
        "first_token_ce_peak": args.first_token_ce_peak,
        "first_token_ce_warmup_frac": args.first_token_ce_warmup_frac,
        "first_token_entropy_weight": args.first_token_entropy_weight,
        "adapter_hidden_mult": args.adapter_hidden_mult,
        "adapter_dropout": args.adapter_dropout,
        "adapter_colorize": bool(args.adapter_colorize),
        "adapter_enable_metadata": bool(args.adapter_metadata),
        "llama_device_map": args.llama_device_map,
        "qwen_device_map": args.qwen_device_map,
        "llama_devices": args.llama_devices,
        "qwen_devices": args.qwen_devices,
        "gpu_mem_gib": args.gpu_mem_gib,
        "use_lora": bool(args.use_lora),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "lora_firstN": args.lora_firstN,
        "use_prefix": bool(args.use_prefix),
        "prefix_tokens": args.prefix_tokens,
        "prefix_projection": bool(args.prefix_projection),
        "peft_prefix_all_layers": str(getattr(args, "peft_prefix_all_layers", "yes")),
        "manifold_stat_weight": args.manifold_stat_weight,
        "state_kd_weight": args.state_kd_weight,
        "state_kd_layers": args.state_kd_layers,
        "K": args.K,
        "k_ce_weight": args.k_ce_weight,
        "kd_first_k_weight": args.kd_first_k_weight,
        "kd_tau": args.kd_tau,
        "max_answer_tokens": args.max_answer_tokens,
        "grad_accum_steps": grad_accum_steps,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "models": model_keys,
        "warmup_text_latent_steps": warmup_total_steps,
        "warmup_text_latent_weight": args.warmup_text_latent_weight,
        "warmup_text_latent_weight_end": args.warmup_text_latent_weight_end,
        "warmup_text_teacher_weight": args.warmup_text_teacher_weight,
        "warmup_tail_prob": args.warmup_tail_prob,
        "warmup_align_tokens": args.warmup_align_tokens,
        "warmup_align_weight": args.warmup_align_weight,
        "grad_diag_interval": grad_diag_interval,
        "grad_diag_components": args.grad_diag_components,
        "gist_head": {
            "enabled": bool(args.use_gist_head),
            "target_len": int(args.gist_target_len),
            "hidden": int(args.gist_hidden),
            "layers": int(args.gist_layers),
            "dropout": float(args.gist_dropout),
            "weight": float(args.gist_weight),
            "mask_prob": float(args.gist_mask_prob),
        },
    }
    dp_len_cfg = int(
        args.deep_prefix_len
        if (feature_registry.has("deep_prefix") and args.deep_prefix_len is not None)
        else (latent_shared_len + latent_private_len)
    ) if feature_registry.has("deep_prefix") else 0
    cfg["deep_prefix"] = {
        "enabled": feature_registry.has("deep_prefix"),
        "len": dp_len_cfg,
        "dropout": float(args.deep_prefix_dropout),
    }
    cfg["coprocessor"] = {
        "enabled": feature_registry.has("coprocessor"),
        "len": int(args.coprocessor_len) if feature_registry.has("coprocessor") else 0,
        "width": int(args.coprocessor_width),
        "dropout": float(args.coprocessor_dropout),
        "kv_scale": float(args.coprocessor_kv_scale),
        "pool": args.coprocessor_pool,
        "summaries": coprocessor_summaries,
    }
    state_blob = {
        "epoch": epoch + 1 if 'epoch' in locals() else None,
        "global_step": global_step if 'global_step' in locals() else None,
        "args": vars(args),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "adapter_scale": {
            name: float(adapter.scale.detach().cpu().item())
            for name, adapter in adapters.items()
            if getattr(adapter, "scale", None) is not None
        },
        "encoder": encoder.state_dict(),
    }
    for name, adapter in adapters.items():
        state_blob[f"adp_{name}"] = adapter.state_dict()
    for name, gen in deep_prefix_generators.items():
        state_blob[f"deep_prefix_{name}"] = gen.state_dict()
    for name, module in coprocessors.items():
        state_blob[f"coprocessor_{name}"] = module.state_dict()
    if latent_refiner is not None:
        state_blob["refiner"] = latent_refiner.state_dict()
    for name, head in gist_heads.items():
        state_blob[f"gist_{name}"] = head.state_dict()

    artifacts = {
        "encoder.pt": encoder.state_dict(),
        "state.pt": state_blob,
        "config.json": cfg,
    }
    for name, adapter in adapters.items():
        artifacts[f"adapter_{name}.pt"] = adapter.state_dict()
    for name, gen in deep_prefix_generators.items():
        artifacts[f"deep_prefix_{name}.pt"] = gen.state_dict()
    for name, module in coprocessors.items():
        artifacts[f"coprocessor_{name}.pt"] = module.state_dict()
    if latent_refiner is not None:
        artifacts["refiner.pt"] = latent_refiner.state_dict()
    for name, head in gist_heads.items():
        artifacts[f"gist_{name}.pt"] = head.state_dict()
    # Only save from main process in DDP
    if 'ddp_manager' not in locals() or ddp_manager is None or ddp_manager.should_save:
        save_latest_checkpoint(args.save_dir, artifacts, pre_prune=True, post_prune=True, verbose=True)
        print(f"✅ Saved latest checkpoint to {args.save_dir}", flush=True)

    # Persist PEFT adapters (LoRA) if present
    try:
        from peft import PeftModel  # type: ignore

        def _save_peft(model: nn.Module, path: str) -> None:
            os.makedirs(path, exist_ok=True)
            model.save_pretrained(path)

        if isinstance(getattr(llama, "model", None), PeftModel):
            if args.use_lora:
                _save_peft(llama.model, os.path.join(args.save_dir, "lora_llama"))
                print("📝 Saved LoRA adapters for Llama")
        if isinstance(getattr(qwen, "model", None), PeftModel):
            if args.use_lora:
                _save_peft(qwen.model, os.path.join(args.save_dir, "lora_qwen"))
                print("📝 Saved LoRA adapters for Qwen")
    except Exception as exc:
        print(f"[WARN] Skipped PEFT adapter save: {exc}")

    if args.save_training_stats:
        # Only save from main process in DDP
        if 'ddp_manager' not in locals() or ddp_manager is None or ddp_manager.should_save:
            stats = {
                name: {
                    "rms_mean_raw": tracker["rms_raw"].mean,
                    "rms_mean_cal": tracker["rms_cal"].mean,
                    "embed_rms": tracker["embed_rms"],
                    "count": tracker["rms_cal"].n,
                }
                for name, tracker in stats_trackers.items()
            }
            with open(os.path.join(args.save_dir, "training_stats.json"), "w") as f:
                json.dump(stats, f, indent=2)
            print(f"📝 Saved training_stats.json: {stats}")

    # Clean up DDP if initialized
    if 'ddp_manager' in locals() and ddp_manager is not None and ddp_manager.initialized:
        ddp_manager.print("\nCleaning up DDP...")
        ddp_manager.cleanup()
        ddp_manager.print("DDP cleanup complete.")

    # Final success message (only from main process if using DDP)
    if 'ddp_manager' not in locals() or ddp_manager is None or ddp_manager.should_log:
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)


if __name__ == "__main__":
    main()
