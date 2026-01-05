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

        if valid_checkpoints:
            print(f"Found {len(valid_checkpoints)} valid checkpoints")

    def _validate_checkpoint(self, checkpoint_path: Path, metadata: Dict) -> bool:
        """Validate checkpoint integrity using checksum."""
        if 'checksum' not in metadata:
            return True  # No checksum to validate

        try:
            # Compute checksum
            hasher = hashlib.md5()
            with open(checkpoint_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)

            computed_checksum = hasher.hexdigest()
            return computed_checksum == metadata['checksum']
        except Exception as e:
            print(f"Validation error for {checkpoint_path}: {e}")
            return False

    def _compute_checksum(self, checkpoint_path: Path) -> str:
        """Compute MD5 checksum of checkpoint file."""
        hasher = hashlib.md5()
        with open(checkpoint_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def should_save(self) -> bool:
        """Check if enough time has passed for next checkpoint."""
        return time.time() - self.last_save_time >= self.save_interval_seconds

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        tag: Optional[str] = None,
        metadata: Optional[Dict] = None,
        wait: bool = False
    ) -> Path:
        """
        Save checkpoint with atomic write and rotation.

        Args:
            state: State dictionary to save
            tag: Optional tag for checkpoint name
            metadata: Optional metadata to save alongside
            wait: Whether to wait for save to complete (if background saving)

        Returns:
            Path to saved checkpoint
        """
        timestamp = int(time.time())
        tag_str = f"_{tag}" if tag else ""
        checkpoint_name = f"checkpoint_{timestamp}{tag_str}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save function
        def _do_save():
            try:
                # Save to temporary file first (atomic write)
                temp_path = checkpoint_path.with_suffix(".tmp")

                # Use pickle protocol 4 for better performance
                torch.save(state, temp_path, pickle_protocol=4)

                # Compute checksum if validation enabled
                checksum = None
                if self.validate_on_save:
                    checksum = self._compute_checksum(temp_path)

                # Save metadata
                meta_data = metadata or {}
                meta_data.update({
                    'timestamp': timestamp,
                    'datetime': datetime.now().isoformat(),
                    'tag': tag,
                    'checksum': checksum,
                })

                meta_path = checkpoint_path.with_suffix(".json")
                with open(meta_path, 'w') as f:
                    json.dump(meta_data, f, indent=2)

                # Atomic rename
                temp_path.rename(checkpoint_path)

                # Update history and cleanup old checkpoints
                self.checkpoint_history.append(checkpoint_path)
                self._cleanup_old_checkpoints()

                self.last_save_time = time.time()
                print(f"Checkpoint saved: {checkpoint_path}")

            except Exception as e:
                print(f"Error saving checkpoint: {e}")
                raise

        # Execute save
        if self.enable_background_save and not wait:
            # Background save
            if self.save_thread and self.save_thread.is_alive():
                print("Warning: Previous save still in progress, waiting...")
                self.save_thread.join()

            self.save_thread = threading.Thread(target=_do_save)
            self.save_thread.start()
        else:
            # Foreground save
            _do_save()
            if self.save_thread and self.save_thread.is_alive():
                self.save_thread.join()

        return checkpoint_path

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        all_checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime
        )

        # Keep only the most recent checkpoints
        to_remove = all_checkpoints[:-self.max_checkpoints]
        for ckpt_path in to_remove:
            try:
                # Remove checkpoint and metadata
                ckpt_path.unlink()
                meta_path = ckpt_path.with_suffix(".json")
                if meta_path.exists():
                    meta_path.unlink()
                print(f"Removed old checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"Error removing {ckpt_path}: {e}")

    def load_checkpoint(
        self,
        checkpoint_path: Optional[Path] = None,
        map_location: str = 'cpu'
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load checkpoint and metadata.

        Args:
            checkpoint_path: Specific checkpoint to load (or latest if None)
            map_location: Device mapping for torch.load

        Returns:
            Tuple of (state_dict, metadata)
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                raise ValueError("No checkpoint found to load")

        checkpoint_path = Path(checkpoint_path)
        meta_path = checkpoint_path.with_suffix(".json")

        # Load metadata
        metadata = {}
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

        # Validate if requested
        if self.validate_on_load:
            if not self._validate_checkpoint(checkpoint_path, metadata):
                raise ValueError(f"Checkpoint validation failed: {checkpoint_path}")

        # Load state
        state = torch.load(checkpoint_path, map_location=map_location)

        print(f"Loaded checkpoint: {checkpoint_path}")
        if metadata:
            print(f"  Saved at: {metadata.get('datetime', 'unknown')}")
            print(f"  Tag: {metadata.get('tag', 'none')}")

        return state, metadata

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the most recent valid checkpoint."""
        # First check history
        if self.checkpoint_history:
            for ckpt_path in reversed(self.checkpoint_history):
                if ckpt_path.exists():
                    return ckpt_path

        # Fall back to directory scan
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime
        )

        for ckpt_path in reversed(checkpoint_files):
            meta_path = ckpt_path.with_suffix(".json")
            if meta_path.exists():
                if self.validate_on_load:
                    try:
                        with open(meta_path, 'r') as f:
                            metadata = json.load(f)
                        if self._validate_checkpoint(ckpt_path, metadata):
                            return ckpt_path
                    except:
                        continue
                else:
                    return ckpt_path

        return None

    def wait_for_save(self) -> None:
        """Wait for any background save to complete."""
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join()

    def list_checkpoints(self) -> List[Tuple[Path, Dict]]:
        """List all valid checkpoints with metadata."""
        checkpoints = []
        for ckpt_path in sorted(self.checkpoint_dir.glob("checkpoint_*.pt")):
            meta_path = ckpt_path.with_suffix(".json")
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    checkpoints.append((ckpt_path, metadata))
                except:
                    continue
        return checkpoints