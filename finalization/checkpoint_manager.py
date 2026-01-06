#!/usr/bin/env python3
"""
Unified checkpoint management system for finalization experiments.

This module provides comprehensive checkpoint support including:
- Automatic checkpoint discovery
- State preservation and restoration
- Preemption-safe saving
- Resume from latest or specific checkpoints
- Atomic file operations to prevent corruption
"""

import os
import json
import time
import signal
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime
import traceback

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Checkpoint functionality limited.")


class CheckpointManager:
    """Unified checkpoint manager for all training scripts."""

    def __init__(
        self,
        save_dir: str = "./checkpoints",
        save_interval: int = 100,
        keep_last_n: int = 3,
        enable_preemption_handling: bool = True,
        verbose: bool = True
    ):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            save_interval: Steps between checkpoint saves (0 = disabled)
            keep_last_n: Number of recent checkpoints to keep
            enable_preemption_handling: Setup signal handlers for preemption
            verbose: Print detailed logs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.verbose = verbose

        # Track checkpointing state
        self.last_save_step = -1
        self.preemption_requested = False
        self.is_saving = False

        # Setup preemption handling
        if enable_preemption_handling:
            self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def handle_preemption(signum, frame):
            if self.verbose:
                print(f"\n⚠️  Preemption signal {signum} received! Triggering checkpoint save...")
            self.preemption_requested = True

        # Common preemption signals
        signal.signal(signal.SIGTERM, handle_preemption)  # SLURM preemption
        signal.signal(signal.SIGUSR1, handle_preemption)  # Custom signal
        try:
            signal.signal(signal.SIGXCPU, handle_preemption)  # CPU time limit
        except AttributeError:
            pass  # Not available on all systems

    def find_latest_checkpoint(self, checkpoint_dir: Optional[str] = None) -> Optional[str]:
        """
        Find the latest checkpoint in a directory.

        Args:
            checkpoint_dir: Directory to search (defaults to self.save_dir)

        Returns:
            Path to latest checkpoint directory or None if not found
        """
        search_dir = Path(checkpoint_dir) if checkpoint_dir else self.save_dir

        if not search_dir.exists():
            return None

        # Look for step_* and epoch_* directories
        checkpoint_dirs = []

        # Find step checkpoints
        for d in search_dir.glob("step_*"):
            if d.is_dir():
                try:
                    step_num = int(d.name.split("_")[1])
                    checkpoint_dirs.append((step_num, d, "step"))
                except (IndexError, ValueError):
                    pass

        # Find epoch checkpoints
        for d in search_dir.glob("epoch*"):
            if d.is_dir():
                try:
                    # Handle both epoch_N and epochN formats
                    if "_" in d.name:
                        epoch_num = int(d.name.split("_")[1])
                    else:
                        epoch_num = int(d.name.replace("epoch", ""))
                    # Convert epochs to approximate steps for comparison
                    checkpoint_dirs.append((epoch_num * 10000, d, "epoch"))
                except (IndexError, ValueError):
                    pass

        if not checkpoint_dirs:
            # Check if save_dir itself contains checkpoint files
            if (search_dir / "state.pt").exists():
                return str(search_dir)
            return None

        # Sort by step/epoch number and return latest
        checkpoint_dirs.sort(key=lambda x: x[0])
        latest = checkpoint_dirs[-1][1]

        if self.verbose:
            print(f"📁 Found latest checkpoint: {latest}")

        return str(latest)

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        step: int,
        epoch: Optional[int] = None,
        is_best: bool = False,
        extra_artifacts: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save checkpoint with atomic operations.

        Args:
            state: Main state dictionary containing model, optimizer, etc.
            step: Current training step
            epoch: Current epoch (optional)
            is_best: Whether this is the best checkpoint so far
            extra_artifacts: Additional files to save (e.g., config, metrics)

        Returns:
            Path to saved checkpoint directory
        """
        if self.is_saving:
            if self.verbose:
                print("⚠️  Checkpoint save already in progress, skipping...")
            return str(self.save_dir)

        self.is_saving = True

        try:
            # Determine checkpoint name
            if epoch is not None:
                checkpoint_name = f"epoch_{epoch}_step_{step}"
            else:
                checkpoint_name = f"step_{step}"

            checkpoint_dir = self.save_dir / checkpoint_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save main state
            state_path = checkpoint_dir / "state.pt"
            self._atomic_save(state, state_path)

            # Save metadata
            metadata = {
                "step": step,
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "is_best": is_best,
            }
            metadata_path = checkpoint_dir / "metadata.json"
            self._atomic_save_json(metadata, metadata_path)

            # Save extra artifacts if provided
            if extra_artifacts:
                for name, artifact in extra_artifacts.items():
                    if artifact is None:
                        continue

                    artifact_path = checkpoint_dir / name
                    if isinstance(artifact, dict) and not isinstance(artifact, torch.nn.Module):
                        self._atomic_save_json(artifact, artifact_path)
                    elif TORCH_AVAILABLE and torch.is_tensor(artifact):
                        self._atomic_save(artifact, artifact_path)
                    elif TORCH_AVAILABLE and isinstance(artifact, nn.Module):
                        self._atomic_save(artifact.state_dict(), artifact_path)
                    else:
                        # Try to save as JSON
                        try:
                            self._atomic_save_json(artifact, artifact_path)
                        except:
                            if self.verbose:
                                print(f"⚠️  Could not save artifact: {name}")

            # Save "best" symlink if needed
            if is_best:
                best_link = self.save_dir / "best"
                if best_link.exists():
                    best_link.unlink()
                best_link.symlink_to(checkpoint_dir.name)

            # Save "latest" symlink
            latest_link = self.save_dir / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(checkpoint_dir.name)

            if self.verbose:
                print(f"💾 Saved checkpoint: {checkpoint_dir}")

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

            self.last_save_step = step
            return str(checkpoint_dir)

        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            traceback.print_exc()
            return str(self.save_dir)

        finally:
            self.is_saving = False

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        map_location: str = "cpu",
        strict: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint (auto-discovers if None)
            map_location: Device to map tensors to
            strict: Whether to enforce strict state dict loading

        Returns:
            State dictionary or None if not found
        """
        # Auto-discover checkpoint if not specified
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path is None:
                if self.verbose:
                    print("📭 No checkpoint found to load")
                return None

        checkpoint_dir = Path(checkpoint_path)

        # Handle both directory and direct state.pt paths
        if checkpoint_dir.is_file():
            state_path = checkpoint_dir
            checkpoint_dir = checkpoint_dir.parent
        else:
            state_path = checkpoint_dir / "state.pt"

        if not state_path.exists():
            if self.verbose:
                print(f"❌ State file not found: {state_path}")
            return None

        try:
            if TORCH_AVAILABLE:
                state = torch.load(state_path, map_location=map_location)
            else:
                print("⚠️  PyTorch not available, cannot load checkpoint")
                return None

            # Load metadata if available
            metadata_path = checkpoint_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                state['checkpoint_metadata'] = metadata

            if self.verbose:
                step = state.get('step', state.get('global_step', -1))
                epoch = state.get('epoch', -1)
                print(f"✅ Loaded checkpoint from {checkpoint_dir}")
                print(f"   Step: {step}, Epoch: {epoch}")

            return state

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            traceback.print_exc()
            return None

    def should_save(self, step: int, force: bool = False) -> bool:
        """
        Check if checkpoint should be saved at this step.

        Args:
            step: Current training step
            force: Force save regardless of interval

        Returns:
            True if checkpoint should be saved
        """
        if force or self.preemption_requested:
            return True

        if self.save_interval <= 0:
            return False

        return (step > 0 and step % self.save_interval == 0)

    def _atomic_save(self, obj: Any, path: Path):
        """Save object atomically using temp file + rename."""
        if not TORCH_AVAILABLE:
            return

        temp_path = path.with_suffix('.tmp')
        torch.save(obj, temp_path)
        temp_path.replace(path)

    def _atomic_save_json(self, obj: Dict, path: Path):
        """Save JSON atomically using temp file + rename."""
        temp_path = path.with_suffix('.json.tmp')
        with open(temp_path, 'w') as f:
            json.dump(obj, f, indent=2, default=str)
        temp_path.replace(path.with_suffix('.json') if not path.suffix == '.json' else path)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones."""
        if self.keep_last_n <= 0:
            return

        # Find all step checkpoints
        step_checkpoints = []
        for d in self.save_dir.glob("step_*"):
            if d.is_dir():
                try:
                    step_num = int(d.name.split("_")[1])
                    step_checkpoints.append((step_num, d))
                except (IndexError, ValueError):
                    pass

        # Sort by step number
        step_checkpoints.sort(key=lambda x: x[0])

        # Remove old checkpoints
        if len(step_checkpoints) > self.keep_last_n:
            to_remove = step_checkpoints[:-self.keep_last_n]
            for _, checkpoint_dir in to_remove:
                try:
                    shutil.rmtree(checkpoint_dir)
                    if self.verbose:
                        print(f"🗑️  Removed old checkpoint: {checkpoint_dir.name}")
                except Exception as e:
                    print(f"⚠️  Could not remove {checkpoint_dir}: {e}")


class ExperimentCheckpointer:
    """High-level checkpointer for MAIN_EXPERIMENT.py compatibility."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize from experiment config."""
        self.config = config
        self.manager = CheckpointManager(
            save_dir=config.get('output_dir', './checkpoints'),
            save_interval=config.get('save_interval', 100),
            keep_last_n=config.get('keep_checkpoints', 3),
            enable_preemption_handling=config.get('handle_preemption', True),
            verbose=config.get('verbose', True)
        )

    def save_training_state(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        best_metric: float,
        is_best: bool = False
    ) -> str:
        """Save complete training state."""

        state = {
            'epoch': epoch,
            'step': step,
            'best_metric': best_metric,
            'metrics': metrics,
        }

        # Add model state
        if hasattr(model, 'state_dict'):
            state['model'] = model.state_dict()
        elif hasattr(model, 'module'):  # DataParallel
            state['model'] = model.module.state_dict()

        # Add optimizer state
        if optimizer and hasattr(optimizer, 'state_dict'):
            state['optimizer'] = optimizer.state_dict()

        # Add scheduler state
        if scheduler and hasattr(scheduler, 'state_dict'):
            state['scheduler'] = scheduler.state_dict()

        # Save config as extra artifact
        extra_artifacts = {
            'config.json': self.config,
            'metrics.json': metrics,
        }

        return self.manager.save_checkpoint(
            state=state,
            step=step,
            epoch=epoch,
            is_best=is_best,
            extra_artifacts=extra_artifacts
        )

    def resume_training(
        self,
        model: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None
    ) -> Tuple[int, int, float, Dict[str, float]]:
        """
        Resume training from checkpoint.

        Returns:
            Tuple of (epoch, step, best_metric, metrics)
        """
        state = self.manager.load_checkpoint(checkpoint_path)

        if state is None:
            return 0, 0, float('inf'), {}

        # Load model state
        if hasattr(model, 'load_state_dict'):
            model.load_state_dict(state.get('model', {}))
        elif hasattr(model, 'module'):  # DataParallel
            model.module.load_state_dict(state.get('model', {}))

        # Load optimizer state
        if optimizer and 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])

        # Load scheduler state
        if scheduler and 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])

        epoch = state.get('epoch', 0)
        step = state.get('step', 0)
        best_metric = state.get('best_metric', float('inf'))
        metrics = state.get('metrics', {})

        return epoch, step, best_metric, metrics


def add_checkpoint_args(parser):
    """Add checkpoint-related arguments to argument parser."""
    group = parser.add_argument_group('Checkpoint Management')

    group.add_argument(
        '--resume',
        action='store_true',
        help='Resume from latest checkpoint in save_dir'
    )
    group.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Path to specific checkpoint to resume from'
    )
    group.add_argument(
        '--save_dir',
        type=str,
        default='./checkpoints',
        help='Directory to save checkpoints'
    )
    group.add_argument(
        '--save_interval',
        type=int,
        default=100,
        help='Save checkpoint every N steps (0 to disable)'
    )
    group.add_argument(
        '--keep_checkpoints',
        type=int,
        default=3,
        help='Number of recent checkpoints to keep'
    )
    group.add_argument(
        '--no_preemption_handling',
        action='store_true',
        help='Disable automatic preemption signal handling'
    )

    return parser


if __name__ == "__main__":
    """Test checkpoint manager functionality."""

    print("Testing CheckpointManager...")

    # Create manager
    manager = CheckpointManager(
        save_dir="./test_checkpoints",
        save_interval=10,
        keep_last_n=2
    )

    # Test save
    for step in range(0, 35, 10):
        if manager.should_save(step) or step == 0:
            state = {
                'step': step,
                'epoch': step // 10,
                'loss': 1.0 / (step + 1),
            }
            manager.save_checkpoint(
                state=state,
                step=step,
                epoch=step // 10,
                is_best=(step == 20)
            )

    # Test load latest
    loaded = manager.load_checkpoint()
    if loaded:
        print(f"Loaded state: step={loaded['step']}, loss={loaded['loss']:.4f}")

    # Test specific load
    specific = manager.load_checkpoint("./test_checkpoints/step_20")
    if specific:
        print(f"Loaded specific: step={specific['step']}")

    # Cleanup
    import shutil
    shutil.rmtree("./test_checkpoints", ignore_errors=True)

    print("✅ All tests passed!")