#!/usr/bin/env python3
"""
Main entry point for cross-model activation communication experiments.

Reproduces "Communicating Activations Between Language Model Agents" (Ramesh & Li, ICML 2025).
"""

import os
import sys
import warnings
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path to import latentwire
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import local modules
from cross_model.utils import get_device_and_config, setup_ddp, cleanup_ddp, is_main_process
from cross_model.models import LearnedProjection
from cross_model.training import train_learned_projection
from cross_model.checkpointing import load_projection, save_projection, get_projection_path
from cross_model import experiments

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
os.environ['HF_HOME'] = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))


# Model identifiers
LLAMA_8B = "meta-llama/Llama-3.1-8B"
LLAMA_3B = "meta-llama/Llama-3.2-3B"
RAMESH_LI_LAYER = 26  # Paper's recommended layer


def main():
    """Run cross-model activation communication experiments."""

    # Initialize DDP if running under torchrun
    if 'RANK' in os.environ and not torch.distributed.is_initialized():
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Initialize process group with extended timeout
        from datetime import timedelta
        timeout_seconds = 3600  # 60 minutes
        torch.distributed.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            timeout=timedelta(seconds=timeout_seconds)
        )

        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        if rank == 0:
            print(f"\n✓ Initialized DDP: {world_size} processes (rank {rank})")
            print(f"  Backend: {'nccl' if torch.cuda.is_available() else 'gloo'}")
            print(f"  Device for rank {rank}: cuda:{local_rank}")

    # Only print header on main process
    if is_main_process():
        print("=" * 80)
        print("CROSS-MODEL ACTIVATION COMMUNICATION EXPERIMENTS")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 80)

    # Get device and platform config
    device, platform, config = get_device_and_config()
    use_bf16 = config.get('use_bf16', False)
    use_flash_attention = config.get('use_flash_attention', False)

    if is_main_process():
        print(f"\nPlatform: {platform}")
        print(f"Device: {device}")
        if platform == 'hpc':
            print(f"Available CUDA GPUs: {torch.cuda.device_count()}")

    # Model identifiers
    model_a_id = LLAMA_8B
    model_b_id = LLAMA_3B

    if is_main_process():
        print(f"\nModel A (source): {model_a_id}")
        print(f"Model B (target): {model_b_id}")
        print(f"Target layer: {RAMESH_LI_LAYER}")

    # Load models
    if is_main_process():
        print(f"\nLoading models...")

    if platform == 'mac':
        dtype = torch.float32
        if is_main_process():
            print("Using float32 for MPS")
    elif use_bf16:
        dtype = torch.bfloat16
        if is_main_process():
            print("Using bfloat16 for H100")
    else:
        dtype = torch.float16
        if is_main_process():
            print("Using float16")

    # Model loading kwargs
    model_kwargs = {
        'torch_dtype': dtype,
        'device_map': "auto" if platform == 'mac' else None,
    }

    if use_flash_attention and platform == 'hpc':
        model_kwargs['attn_implementation'] = "flash_attention_2"
        if is_main_process():
            print("Using Flash Attention 2")

    model_a = AutoModelForCausalLM.from_pretrained(model_a_id, **model_kwargs).eval()
    model_b = AutoModelForCausalLM.from_pretrained(model_b_id, **model_kwargs).eval()

    # Freeze parameters
    for param in model_a.parameters():
        param.requires_grad = False
    for param in model_b.parameters():
        param.requires_grad = False

    # Move to device if needed
    if platform == 'hpc':
        model_a = model_a.to(device)
        model_b = model_b.to(device)

    tokenizer_a = AutoTokenizer.from_pretrained(model_a_id)
    tokenizer_b = AutoTokenizer.from_pretrained(model_b_id)

    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
    if tokenizer_b.pad_token is None:
        tokenizer_b.pad_token = tokenizer_b.eos_token

    # Get model dimensions
    dim_a = model_a.config.hidden_size
    dim_b = model_b.config.hidden_size

    if is_main_process():
        print(f"\nModel A: {dim_a} hidden_dim")
        print(f"Model B: {dim_b} hidden_dim")

    # Handle dimension mismatch with learned projection
    learned_projection = None
    if dim_a != dim_b:
        if is_main_process():
            print(f"\nDimension mismatch detected ({dim_a} → {dim_b})")

        # Try to load existing projection
        learned_projection = load_projection(dim_a, dim_b, device=device, dtype=dtype)

        # Train if not found
        if learned_projection is None and is_main_process():
            print("  Training new projection on C4 dataset...")
            learned_projection = train_learned_projection(
                model_a=model_a,
                model_b=model_b,
                tokenizer_a=tokenizer_a,
                tokenizer_b=tokenizer_b,
                dim_a=dim_a,
                dim_b=dim_b,
                layer_idx=RAMESH_LI_LAYER,
                num_samples=3072,
                learning_rate=1e-3,
                num_epochs=10,
                batch_size=96,
                device=device,
                seed=42
            )

            # Save trained projection
            save_projection(learned_projection, dim_a, dim_b)

    # Import and run the activation communication experiment
    # This uses the original implementation from experimental/learning
    sys.path.insert(0, str(Path(__file__).parent.parent / "experimental" / "learning"))
    from unified_cross_model_experiments import run_activation_communication_experiment

    # Run activation communication experiment
    # This includes: text similarity, SQuAD evaluation, and GSM8K evaluation
    if is_main_process():
        print("\n" + "=" * 80)
        print("RUNNING ACTIVATION COMMUNICATION EXPERIMENTS")
        print("=" * 80)

    # Run using the models and projection we set up
    try:
        run_activation_communication_experiment(model_a_id=model_a_id, model_b_id=model_b_id)
    except Exception as e:
        if is_main_process():
            print(f"\n✗ Experiment failed: {e}")
            import traceback
            traceback.print_exc()

    # Cleanup
    cleanup_ddp()

    if is_main_process():
        print("\n" + "=" * 80)
        print("EXPERIMENTS COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    main()
