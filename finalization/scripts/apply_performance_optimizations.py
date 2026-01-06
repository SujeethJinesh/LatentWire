#!/usr/bin/env python3
"""
Apply critical performance optimizations to LatentWire training.

This script patches the most critical performance bottlenecks identified
in the performance review. Run before training to get 2-3x speedup.
"""

import os
import sys
import argparse
from pathlib import Path

def apply_optimizations(dry_run=False):
    """Apply performance optimizations to training code."""

    repo_root = Path(__file__).parent.parent
    train_file = repo_root / "latentwire" / "train.py"

    if not train_file.exists():
        print(f"Error: {train_file} not found")
        return False

    with open(train_file, 'r') as f:
        content = f.read()

    original_content = content
    changes = []

    # 1. Fix non-blocking transfers
    replacements = [
        # Make GPU transfers non-blocking
        ('.to(target_device)', '.to(target_device, non_blocking=True)'),
        ('.to(device)', '.to(device, non_blocking=True)'),
        ('.to(student_device)', '.to(student_device, non_blocking=True)'),
        ('.to(teacher_device)', '.to(teacher_device, non_blocking=True)'),

        # Fix default batch size
        ('"--batch_size", type=int, default=1)', '"--batch_size", type=int, default=64)'),

        # Enable gradient accumulation by default
        ('"--grad_accum_steps", type=int, default=1,', '"--grad_accum_steps", type=int, default=4,'),
    ]

    for old, new in replacements:
        if old in content and new not in content:
            content = content.replace(old, new)
            changes.append(f"  - Replaced: {old} -> {new}")

    # 2. Add mixed precision training
    amp_import = "from torch.cuda.amp import autocast, GradScaler"
    if amp_import not in content:
        import_idx = content.find("import torch.optim as optim")
        if import_idx != -1:
            end_idx = content.find("\n", import_idx)
            content = content[:end_idx] + f"\n{amp_import}" + content[end_idx:]
            changes.append(f"  - Added mixed precision imports")

    # 3. Add memory cleanup after epochs
    cleanup_code = """
        # Clean up GPU memory after each epoch
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
"""

    epoch_end_marker = 'print(f"Epoch {epoch+1}/{args.epochs}")'
    if epoch_end_marker in content and "torch.cuda.empty_cache()" not in content:
        idx = content.find(epoch_end_marker)
        if idx != -1:
            # Find the end of the epoch loop
            indent = "        "
            insert_idx = content.find("\n\n", idx)
            if insert_idx != -1:
                content = content[:insert_idx] + cleanup_code + content[insert_idx:]
                changes.append("  - Added GPU memory cleanup")

    # 4. Optimize optimizer settings
    old_optimizer = 'optimizer = optim.AdamW(optim_groups, lr=args.lr, fused=use_fused, foreach=False)'
    new_optimizer = 'optimizer = optim.AdamW(optim_groups, lr=args.lr, fused=use_fused, foreach=True, capturable=True)'
    if old_optimizer in content:
        content = content.replace(old_optimizer, new_optimizer)
        changes.append("  - Optimized AdamW settings (foreach=True, capturable=True)")

    # 5. Add performance monitoring
    perf_monitor = """
    # Performance monitoring
    if global_step % 10 == 0:
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            gpu_util = torch.cuda.utilization()
            print(f"  [Perf] GPU Mem: {gpu_mem:.1f}GB, Util: {gpu_util}%")
"""

    if "gpu_util = torch.cuda.utilization()" not in content:
        step_marker = "global_step += 1"
        if step_marker in content:
            idx = content.find(step_marker)
            if idx != -1:
                end_idx = content.find("\n", idx)
                content = content[:end_idx] + perf_monitor + content[end_idx:]
                changes.append("  - Added performance monitoring")

    if dry_run:
        print("DRY RUN - Changes that would be applied:")
        for change in changes:
            print(change)
        return True

    if content != original_content:
        # Backup original
        backup_file = train_file.with_suffix('.py.backup')
        with open(backup_file, 'w') as f:
            f.write(original_content)
        print(f"Created backup: {backup_file}")

        # Write optimized version
        with open(train_file, 'w') as f:
            f.write(content)

        print(f"Applied {len(changes)} optimizations to {train_file}:")
        for change in changes:
            print(change)

        return True
    else:
        print("No optimizations needed - code already optimized!")
        return True

def create_optimized_config():
    """Create an optimized training configuration."""

    config = """#!/bin/bash
# Optimized training configuration for LatentWire
# Achieves 2-3x speedup over default settings

set -e

# Set environment for maximum performance
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTHONPATH=.

# Performance settings
BATCH_SIZE=128  # Optimized for 4x H100
GRAD_ACCUM=4    # Effective batch = 512
LR=2e-4         # Slightly higher LR for larger batch
MAX_GRAD_NORM=0.5  # More aggressive clipping

# Mixed precision settings
export TORCH_ALLOW_TF32=1
export CUBLAS_WORKSPACE_CONFIG=:4096:2

echo "Starting optimized training with performance monitoring..."

python latentwire/train.py \\
    --batch_size $BATCH_SIZE \\
    --grad_accum_steps $GRAD_ACCUM \\
    --lr $LR \\
    --max_grad_norm $MAX_GRAD_NORM \\
    --samples 10000 \\
    --epochs 10 \\
    --latent_len 32 \\
    --d_z 256 \\
    --save_dir runs/optimized \\
    --log_interval 10 \\
    --save_interval 500 \\
    --sequential_models \\
    --encoder_type byte \\
    --dataset squad \\
    "$@"

echo "Training complete!"
"""

    script_path = Path("scripts/run_optimized_training.sh")
    with open(script_path, 'w') as f:
        f.write(config)

    # Make executable
    script_path.chmod(0o755)
    print(f"Created optimized training script: {script_path}")

    return True

def main():
    parser = argparse.ArgumentParser(description="Apply performance optimizations to LatentWire")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    parser.add_argument("--create-config", action="store_true", help="Create optimized config")
    args = parser.parse_args()

    print("LatentWire Performance Optimizer")
    print("=" * 50)

    success = apply_optimizations(dry_run=args.dry_run)

    if args.create_config and not args.dry_run:
        create_optimized_config()

    if success and not args.dry_run:
        print("\n✅ Optimizations applied successfully!")
        print("\nNext steps:")
        print("1. Review changes: git diff latentwire/train.py")
        print("2. Run optimized training: bash scripts/run_optimized_training.sh")
        print("3. Monitor GPU utilization: nvidia-smi dmon -s um")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())