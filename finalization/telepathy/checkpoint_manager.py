"""
Robust checkpoint management system for preemptible jobs.

Features:
- Automatic checkpointing every 5 minutes
- Keep last 3 checkpoints with automatic cleanup
- Atomic writes to prevent corruption
- Full state preservation (model, optimizer, scheduler, RNG states)
- Fast save/load with pickle protocol 4
- Checkpoint validation and corruption detection
- Background saving (non-blocking)
- Auto-discovery of latest valid checkpoint
"""

import os
import json
import time
import hashlib
import shutil
import threading
import pickle
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from collections import deque

import torch
import numpy as np


class CheckpointManager:
    """
    Manages checkpoints for preemptible training jobs with automatic
    saving, rotation, and recovery.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
        save_interval_minutes: float = 5.0,
        enable_background_save: bool = True,
        validate_on_save: bool = True,
        validate_on_load: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_interval_minutes: Auto-save interval in minutes
            enable_background_save: Whether to save in background thread
            validate_on_save: Validate checkpoint after saving
            validate_on_load: Validate checkpoint before loading
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_interval_seconds = save_interval_minutes * 60
        self.enable_background_save = enable_background_save
        self.validate_on_save = validate_on_save
        self.validate_on_load = validate_on_load

        self.last_save_time = 0
        self.save_thread = None
        self.checkpoint_history = deque(maxlen=max_checkpoints)

        # Discover existing checkpoints
        self._discover_checkpoints()

    def _discover_checkpoints(self) -> None:
        """Discover and validate existing checkpoints."""
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime
        )

        valid_checkpoints = []
        for ckpt_path in checkpoint_files:
            meta_path = ckpt_path.with_suffix(".json")
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        metadata = json.load(f)

                    # Validate if requested
                    if self.validate_on_load:
                        if self._validate_checkpoint(ckpt_path, metadata):
                            valid_checkpoints.append((ckpt_path, metadata))
                        else:
                            print(f"Warning: Corrupted checkpoint {ckpt_path}, skipping")
                    else:
                        valid_checkpoints.append((ckpt_path, metadata))
                except Exception as e:
                    print(f"Warning: Failed to read metadata for {ckpt_path}: {e}")

        # Keep track of valid checkpoints
        self.checkpoint_history.clear()
        for ckpt_path, metadata in valid_checkpoints[-self.max_checkpoints:]:
            self.checkpoint_history.append(ckpt_path)

        if self.checkpoint_history:
            print(f"Found {len(self.checkpoint_history)} existing checkpoints")

    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA256 checksum of data."""
        return hashlib.sha256(data).hexdigest()

    def _validate_checkpoint(self, checkpoint_path: Path, metadata: Dict) -> bool:
        """
        Validate checkpoint integrity.

        Args:
            checkpoint_path: Path to checkpoint file
            metadata: Checkpoint metadata

        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            # Check file exists and has non-zero size
            if not checkpoint_path.exists():
                return False

            file_size = checkpoint_path.stat().st_size
            if file_size == 0:
                return False

            # Verify checksum if present
            if "checksum" in metadata:
                with open(checkpoint_path, "rb") as f:
                    data = f.read()
                computed_checksum = self._compute_checksum(data)
                if computed_checksum != metadata["checksum"]:
                    return False

            # Try to load checkpoint header to verify it's valid PyTorch file
            try:
                # Use weights_only=False for full checkpoint with optimizer state
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                    pickle_module=pickle
                )

                # Verify expected keys are present
                required_keys = ["model_state_dict", "optimizer_state_dict", "epoch"]
                for key in required_keys:
                    if key not in checkpoint:
                        return False

                del checkpoint  # Free memory
                return True

            except Exception:
                return False

        except Exception as e:
            print(f"Validation error for {checkpoint_path}: {e}")
            return False

    def should_save(self) -> bool:
        """Check if it's time to save a checkpoint."""
        return time.time() - self.last_save_time >= self.save_interval_seconds

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        batch_idx: int = 0,
        samples_seen: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[Path]:
        """
        Save checkpoint with full state preservation.

        Args:
            model: Model to checkpoint
            optimizer: Optimizer state
            scheduler: Optional LR scheduler
            epoch: Current epoch
            global_step: Global training step
            batch_idx: Current batch index
            samples_seen: Total samples processed
            metrics: Training metrics to save
            extra_state: Additional state to save
            force: Force save regardless of interval

        Returns:
            Path to saved checkpoint or None if skipped
        """
        # Check if we should save
        if not force and not self.should_save():
            return None

        # Generate checkpoint name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timestamp}_{global_step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        temp_path = checkpoint_path.with_suffix(".tmp")

        # Prepare checkpoint data
        checkpoint_data = {
            # Model and optimizer
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),

            # Training state
            "epoch": epoch,
            "global_step": global_step,
            "batch_idx": batch_idx,
            "samples_seen": samples_seen,

            # Random number generator states
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },

            # Training metrics
            "metrics": metrics or {},

            # Timestamp
            "timestamp": timestamp,
            "save_time": time.time(),
        }

        # Add CUDA RNG state if available
        if torch.cuda.is_available():
            checkpoint_data["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()

        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

        # Add extra state if provided
        if extra_state:
            checkpoint_data["extra_state"] = extra_state

        # Save function (can run in background)
        def _save():
            try:
                # Save to temporary file first (atomic write)
                torch.save(
                    checkpoint_data,
                    temp_path,
                    pickle_protocol=4,  # Fast protocol
                    _use_new_zipfile_serialization=True
                )

                # Compute checksum if validation enabled
                checksum = None
                if self.validate_on_save:
                    with open(temp_path, "rb") as f:
                        data = f.read()
                    checksum = self._compute_checksum(data)

                # Save metadata
                metadata = {
                    "timestamp": timestamp,
                    "epoch": epoch,
                    "global_step": global_step,
                    "samples_seen": samples_seen,
                    "file_size": temp_path.stat().st_size,
                    "checksum": checksum,
                    "metrics": metrics or {},
                }

                meta_path = checkpoint_path.with_suffix(".json")
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                # Atomic rename
                temp_path.rename(checkpoint_path)

                # Add to history and cleanup old checkpoints
                self.checkpoint_history.append(checkpoint_path)
                self._cleanup_old_checkpoints()

                # Update last save time
                self.last_save_time = time.time()

                print(f"Saved checkpoint: {checkpoint_path.name} "
                      f"(epoch={epoch}, step={global_step}, size={metadata['file_size']/1e6:.1f}MB)")

            except Exception as e:
                print(f"Error saving checkpoint: {e}")
                if temp_path.exists():
                    temp_path.unlink()

        # Save in background or foreground
        if self.enable_background_save and not force:
            # Wait for previous save to complete
            if self.save_thread and self.save_thread.is_alive():
                self.save_thread.join()

            self.save_thread = threading.Thread(target=_save)
            self.save_thread.start()
        else:
            _save()
            # Wait for completion if forced
            if self.save_thread and self.save_thread.is_alive():
                self.save_thread.join()

        return checkpoint_path

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        while len(self.checkpoint_history) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_history.popleft()

            # Remove checkpoint and metadata
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                print(f"Removed old checkpoint: {old_checkpoint.name}")

            meta_path = old_checkpoint.with_suffix(".json")
            if meta_path.exists():
                meta_path.unlink()

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[Path] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint and restore full state.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Optional scheduler to load state into
            checkpoint_path: Specific checkpoint to load (or auto-discover latest)
            map_location: Device to map tensors to
            strict: Whether to strictly enforce state dict matching

        Returns:
            Dictionary with checkpoint info or None if no checkpoint found
        """
        # Auto-discover latest checkpoint if not specified
        if checkpoint_path is None:
            if not self.checkpoint_history:
                print("No checkpoints found")
                return None
            checkpoint_path = self.checkpoint_history[-1]
        else:
            checkpoint_path = Path(checkpoint_path)

        # Validate checkpoint if requested
        if self.validate_on_load:
            meta_path = checkpoint_path.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    metadata = json.load(f)

                if not self._validate_checkpoint(checkpoint_path, metadata):
                    print(f"Error: Checkpoint {checkpoint_path} is corrupted")

                    # Try previous checkpoint
                    if len(self.checkpoint_history) > 1:
                        print("Trying previous checkpoint...")
                        self.checkpoint_history.pop()
                        return self.load_checkpoint(
                            model, optimizer, scheduler,
                            map_location=map_location, strict=strict
                        )
                    return None

        try:
            print(f"Loading checkpoint: {checkpoint_path.name}")

            # Load checkpoint
            checkpoint = torch.load(
                checkpoint_path,
                map_location=map_location or "cpu",
                weights_only=False,
                pickle_module=pickle
            )

            # Restore model state
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

            # Restore optimizer state
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Restore scheduler state if provided
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Restore RNG states
            if "rng_state" in checkpoint:
                rng_state = checkpoint["rng_state"]

                # Python RNG
                if "python" in rng_state:
                    random.setstate(rng_state["python"])

                # NumPy RNG
                if "numpy" in rng_state:
                    np.random.set_state(rng_state["numpy"])

                # PyTorch RNG
                if "torch" in rng_state:
                    torch.set_rng_state(rng_state["torch"])

                # CUDA RNG
                if "cuda" in rng_state and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng_state["cuda"])

            # Return checkpoint info
            info = {
                "epoch": checkpoint.get("epoch", 0),
                "global_step": checkpoint.get("global_step", 0),
                "batch_idx": checkpoint.get("batch_idx", 0),
                "samples_seen": checkpoint.get("samples_seen", 0),
                "metrics": checkpoint.get("metrics", {}),
                "extra_state": checkpoint.get("extra_state", {}),
                "checkpoint_path": str(checkpoint_path),
            }

            print(f"Resumed from epoch {info['epoch']}, "
                  f"step {info['global_step']}, "
                  f"samples {info['samples_seen']}")

            return info

        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")

            # Try previous checkpoint
            if len(self.checkpoint_history) > 1:
                print("Trying previous checkpoint...")
                self.checkpoint_history.pop()
                return self.load_checkpoint(
                    model, optimizer, scheduler,
                    map_location=map_location, strict=strict
                )

            return None

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest valid checkpoint."""
        if not self.checkpoint_history:
            return None
        return self.checkpoint_history[-1]

    def list_checkpoints(self) -> List[Tuple[Path, Dict[str, Any]]]:
        """
        List all available checkpoints with metadata.

        Returns:
            List of (checkpoint_path, metadata) tuples
        """
        checkpoints = []

        for ckpt_path in self.checkpoint_history:
            meta_path = ckpt_path.with_suffix(".json")
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        metadata = json.load(f)
                    checkpoints.append((ckpt_path, metadata))
                except Exception as e:
                    print(f"Warning: Failed to read metadata for {ckpt_path}: {e}")

        return checkpoints

    def wait_for_save(self) -> None:
        """Wait for any background save operation to complete."""
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join()

    def cleanup(self) -> None:
        """Clean up resources and ensure all saves are complete."""
        self.wait_for_save()


# Convenience functions for integration with training loops

def create_checkpoint_manager(args: Any) -> CheckpointManager:
    """
    Create checkpoint manager from training arguments.

    Args:
        args: Training arguments with checkpoint_dir, etc.

    Returns:
        Configured CheckpointManager instance
    """
    checkpoint_dir = getattr(args, "checkpoint_dir", "checkpoints")
    max_checkpoints = getattr(args, "max_checkpoints", 3)
    save_interval = getattr(args, "checkpoint_interval_minutes", 5.0)
    background_save = getattr(args, "background_checkpoints", True)

    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=max_checkpoints,
        save_interval_minutes=save_interval,
        enable_background_save=background_save,
        validate_on_save=True,
        validate_on_load=True,
    )


def auto_save_checkpoint(
    manager: CheckpointManager,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    global_step: int,
    batch_idx: int,
    samples_seen: int,
    metrics: Dict[str, Any],
    force: bool = False,
) -> bool:
    """
    Auto-save checkpoint if interval has passed.

    Returns:
        True if checkpoint was saved, False otherwise
    """
    checkpoint_path = manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        batch_idx=batch_idx,
        samples_seen=samples_seen,
        metrics=metrics,
        force=force,
    )

    return checkpoint_path is not None


def resume_from_checkpoint(
    manager: CheckpointManager,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resume training from latest checkpoint if available.

    Returns:
        Checkpoint info dict or None if no checkpoint
    """
    return manager.load_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=device,
    )


# Example usage in training loop:
"""
# Initialize
manager = create_checkpoint_manager(args)

# Resume if checkpoint exists
checkpoint_info = resume_from_checkpoint(manager, model, optimizer, scheduler)
if checkpoint_info:
    start_epoch = checkpoint_info["epoch"]
    global_step = checkpoint_info["global_step"]
    samples_seen = checkpoint_info["samples_seen"]
else:
    start_epoch = 0
    global_step = 0
    samples_seen = 0

# Training loop
for epoch in range(start_epoch, num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # Training step
        loss = train_step(batch)

        # Auto-save checkpoint
        auto_save_checkpoint(
            manager, model, optimizer, scheduler,
            epoch, global_step, batch_idx, samples_seen,
            metrics={"loss": loss.item()}
        )

        global_step += 1
        samples_seen += batch_size

# Final save
manager.save_checkpoint(
    model, optimizer, scheduler,
    epoch, global_step, batch_idx, samples_seen,
    metrics={"final_loss": loss.item()},
    force=True
)

# Cleanup
manager.cleanup()
"""