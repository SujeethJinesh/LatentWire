"""
Robust checkpoint management system for preemptible jobs - Single Checkpoint Strategy.

Features:
- SINGLE CHECKPOINT STRATEGY: Only maintains 1 checkpoint at a time
- Automatic checkpointing every 5 minutes
- Atomic writes with backup protection to prevent corruption
- Full state preservation (model, optimizer, scheduler, RNG states)
- Fast save/load with pickle protocol 4
- Checkpoint validation and corruption detection
- Automatic backup during saves for rollback protection
- Auto-discovery and restoration of valid checkpoint
- Bulletproof for preemption scenarios

Single Checkpoint Strategy:
- Always saves to 'checkpoint_current.pt'
- Creates temporary backup during save operations
- Atomic rename operations prevent partial writes
- Automatic cleanup of old timestamped checkpoints
- Validation with automatic backup restoration on corruption
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
        max_checkpoints: int = 1,  # Changed default to 1 - keep only current checkpoint
        save_interval_minutes: float = 5.0,
        enable_background_save: bool = False,  # Disable background save for atomicity
        validate_on_save: bool = True,
        validate_on_load: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep (default=1 for single checkpoint)
            save_interval_minutes: Auto-save interval in minutes
            enable_background_save: Whether to save in background thread (default=False for safety)
            validate_on_save: Validate checkpoint after saving
            validate_on_load: Validate checkpoint before loading
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Force max_checkpoints to 1 for single-checkpoint strategy
        self.max_checkpoints = 1
        self.save_interval_seconds = save_interval_minutes * 60
        self.enable_background_save = False  # Always synchronous for safety
        self.validate_on_save = validate_on_save
        self.validate_on_load = validate_on_load

        self.last_save_time = 0
        self.save_thread = None
        self.checkpoint_history = deque(maxlen=1)  # Only track current checkpoint

        # Discover existing checkpoints
        self._discover_checkpoints()

    def _discover_checkpoints(self) -> None:
        """Discover and validate existing checkpoints (single checkpoint strategy)."""
        # Look for the current checkpoint
        checkpoint_path = self.checkpoint_dir / "checkpoint_current.pt"
        meta_path = checkpoint_path.with_suffix(".json")

        self.checkpoint_history.clear()

        if checkpoint_path.exists() and meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    metadata = json.load(f)

                # Validate if requested
                if self.validate_on_load:
                    if self._validate_checkpoint(checkpoint_path, metadata):
                        self.checkpoint_history.append(checkpoint_path)
                        print(f"Found valid checkpoint: {checkpoint_path}")
                    else:
                        print(f"Warning: Current checkpoint corrupted, will check alternatives")
                        self._check_alternative_checkpoints()
                else:
                    self.checkpoint_history.append(checkpoint_path)
                    print(f"Found checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Failed to read checkpoint metadata: {e}")
                self._check_alternative_checkpoints()
        else:
            # No current checkpoint, check alternatives
            self._check_alternative_checkpoints()

        # Clean up any old timestamped checkpoints from previous runs
        self._cleanup_old_checkpoints()

    def _check_alternative_checkpoints(self) -> None:
        """Check for emergency and backup checkpoints."""
        # Priority order: emergency -> backup
        emergency_path = self.checkpoint_dir / "checkpoint_emergency.pt"
        backup_path = self.checkpoint_dir / "checkpoint_backup.pt"

        for alt_path in [emergency_path, backup_path]:
            if alt_path.exists():
                alt_meta = alt_path.with_suffix(".json")
                if alt_meta.exists():
                    try:
                        with open(alt_meta, 'r') as f:
                            alt_metadata = json.load(f)
                        if not self.validate_on_load or self._validate_checkpoint(alt_path, alt_metadata):
                            print(f"Found valid {alt_path.stem}, restoring as current...")
                            # Restore as current
                            current_path = self.checkpoint_dir / "checkpoint_current.pt"
                            alt_path.rename(current_path)
                            alt_meta.rename(current_path.with_suffix(".json"))
                            self.checkpoint_history.append(current_path)
                            return
                    except Exception as e:
                        print(f"Warning: Failed to restore {alt_path}: {e}")

    def _check_backup_checkpoint(self, backup_path: Path) -> None:
        """Check and potentially restore backup checkpoint."""
        backup_meta = backup_path.with_suffix(".json")
        if backup_meta.exists():
            try:
                with open(backup_meta, 'r') as f:
                    backup_metadata = json.load(f)
                if not self.validate_on_load or self._validate_checkpoint(backup_path, backup_metadata):
                    print(f"Found valid backup checkpoint, restoring...")
                    # Restore backup as current
                    current_path = self.checkpoint_dir / "checkpoint_current.pt"
                    backup_path.rename(current_path)
                    backup_meta.rename(current_path.with_suffix(".json"))
                    self.checkpoint_history.append(current_path)
            except Exception as e:
                print(f"Warning: Failed to restore backup: {e}")

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
        wait: bool = True  # Always wait for completion
    ) -> Path:
        """
        Save checkpoint with atomic write, ensuring only 1 checkpoint exists.
        Bulletproof for preemption scenarios.

        Args:
            state: State dictionary to save
            tag: Optional tag for checkpoint name
            metadata: Optional metadata to save alongside
            wait: Whether to wait for save to complete (always True for safety)

        Returns:
            Path to saved checkpoint
        """
        # Use consistent checkpoint name for single-checkpoint strategy
        checkpoint_name = "checkpoint_current.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Backup paths for extra safety
        backup_name = "checkpoint_backup.pt"
        backup_path = self.checkpoint_dir / backup_name

        try:
            # Step 1: If current checkpoint exists, rename it as backup
            if checkpoint_path.exists():
                # First remove any existing backup
                if backup_path.exists():
                    backup_path.unlink()
                    backup_meta = backup_path.with_suffix(".json")
                    if backup_meta.exists():
                        backup_meta.unlink()

                # Rename current to backup
                checkpoint_path.rename(backup_path)
                meta_path = checkpoint_path.with_suffix(".json")
                if meta_path.exists():
                    meta_path.rename(backup_path.with_suffix(".json"))

            # Step 2: Save new checkpoint to temporary file
            temp_path = checkpoint_path.with_suffix(".tmp")

            # Use pickle protocol 4 for better performance
            torch.save(state, temp_path, pickle_protocol=4)

            # Step 3: Compute checksum if validation enabled
            checksum = None
            if self.validate_on_save:
                checksum = self._compute_checksum(temp_path)

            # Step 4: Save metadata to temporary file
            meta_data = metadata or {}
            meta_data.update({
                'timestamp': int(time.time()),
                'datetime': datetime.now().isoformat(),
                'tag': tag,
                'checksum': checksum,
            })

            meta_temp_path = checkpoint_path.with_suffix(".json.tmp")
            with open(meta_temp_path, 'w') as f:
                json.dump(meta_data, f, indent=2)

            # Step 5: Atomic rename of both files
            temp_path.rename(checkpoint_path)
            meta_temp_path.rename(checkpoint_path.with_suffix(".json"))

            # Step 6: Validate the new checkpoint if requested
            if self.validate_on_save:
                if not self._validate_checkpoint(checkpoint_path, meta_data):
                    # Restore backup if validation fails
                    if backup_path.exists():
                        checkpoint_path.unlink()
                        checkpoint_path.with_suffix(".json").unlink()
                        backup_path.rename(checkpoint_path)
                        backup_path.with_suffix(".json").rename(checkpoint_path.with_suffix(".json"))
                    raise ValueError("New checkpoint validation failed")

            # Step 7: Remove backup after successful save
            if backup_path.exists():
                backup_path.unlink()
                backup_meta = backup_path.with_suffix(".json")
                if backup_meta.exists():
                    backup_meta.unlink()

            # Update tracking
            self.checkpoint_history.clear()
            self.checkpoint_history.append(checkpoint_path)
            self.last_save_time = time.time()

            print(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path

        except Exception as e:
            print(f"Error saving checkpoint: {e}")

            # Attempt to restore backup
            if backup_path.exists():
                print("Attempting to restore backup...")
                try:
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                    if checkpoint_path.with_suffix(".json").exists():
                        checkpoint_path.with_suffix(".json").unlink()

                    backup_path.rename(checkpoint_path)
                    backup_meta = backup_path.with_suffix(".json")
                    if backup_meta.exists():
                        backup_meta.rename(checkpoint_path.with_suffix(".json"))
                    print("Backup restored successfully")
                except Exception as restore_error:
                    print(f"Failed to restore backup: {restore_error}")

            raise

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up any old checkpoints (only keep checkpoint_current.pt)."""
        # With single checkpoint strategy, only checkpoint_current.pt should exist
        # Clean up any old timestamped checkpoints from previous runs
        patterns_to_clean = [
            "checkpoint_[0-9]*.pt",  # Old timestamped checkpoints
            "checkpoint_*_*.pt",      # Old tagged checkpoints
        ]

        for pattern in patterns_to_clean:
            for old_ckpt in self.checkpoint_dir.glob(pattern):
                if old_ckpt.name not in ["checkpoint_current.pt", "checkpoint_backup.pt", "checkpoint_emergency.pt"]:
                    try:
                        old_ckpt.unlink()
                        meta_path = old_ckpt.with_suffix(".json")
                        if meta_path.exists():
                            meta_path.unlink()
                        print(f"Cleaned up old checkpoint: {old_ckpt.name}")
                    except Exception as e:
                        print(f"Error removing {old_ckpt}: {e}")

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
        """Get path to the current checkpoint (single checkpoint strategy)."""
        # With single checkpoint strategy, always look for checkpoint_current.pt
        checkpoint_path = self.checkpoint_dir / "checkpoint_current.pt"
        meta_path = checkpoint_path.with_suffix(".json")

        if checkpoint_path.exists() and meta_path.exists():
            if self.validate_on_load:
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    if self._validate_checkpoint(checkpoint_path, metadata):
                        return checkpoint_path
                    else:
                        print(f"Warning: Current checkpoint failed validation")
                        # Try backup
                        backup_path = self.checkpoint_dir / "checkpoint_backup.pt"
                        if backup_path.exists():
                            backup_meta = backup_path.with_suffix(".json")
                            if backup_meta.exists():
                                with open(backup_meta, 'r') as f:
                                    backup_metadata = json.load(f)
                                if self._validate_checkpoint(backup_path, backup_metadata):
                                    print("Using backup checkpoint")
                                    return backup_path
                except Exception as e:
                    print(f"Error reading checkpoint metadata: {e}")
                    return None
            else:
                return checkpoint_path

        # Check for backup if main doesn't exist
        backup_path = self.checkpoint_dir / "checkpoint_backup.pt"
        if backup_path.exists():
            print("Warning: No current checkpoint found, checking backup...")
            backup_meta = backup_path.with_suffix(".json")
            if backup_meta.exists():
                return backup_path

        return None

    def wait_for_save(self) -> None:
        """Wait for any background save to complete."""
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join()

    def emergency_save(self, state: Dict[str, Any]) -> bool:
        """
        Emergency checkpoint save for preemption scenarios.
        Prioritizes speed over validation.

        Args:
            state: State dictionary to save

        Returns:
            True if save succeeded, False otherwise
        """
        try:
            print("EMERGENCY SAVE: Starting rapid checkpoint save...")
            emergency_path = self.checkpoint_dir / "checkpoint_emergency.pt"

            # Save directly without validation for speed
            torch.save(state, emergency_path, pickle_protocol=4)

            # Quick metadata
            meta_data = {
                'timestamp': int(time.time()),
                'datetime': datetime.now().isoformat(),
                'emergency': True,
                'tag': 'preemption'
            }

            meta_path = emergency_path.with_suffix(".json")
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)

            # Try to rename to current (may fail if preemption too immediate)
            try:
                current_path = self.checkpoint_dir / "checkpoint_current.pt"
                if current_path.exists():
                    current_path.unlink()
                    current_path.with_suffix(".json").unlink()
                emergency_path.rename(current_path)
                meta_path.rename(current_path.with_suffix(".json"))
                print("EMERGENCY SAVE: Checkpoint saved as current")
            except:
                print("EMERGENCY SAVE: Saved as emergency checkpoint")

            return True

        except Exception as e:
            print(f"EMERGENCY SAVE FAILED: {e}")
            return False

    def list_checkpoints(self) -> List[Tuple[Path, Dict]]:
        """List valid checkpoints with metadata (single checkpoint strategy)."""
        checkpoints = []

        # Check current checkpoint
        current_path = self.checkpoint_dir / "checkpoint_current.pt"
        if current_path.exists():
            meta_path = current_path.with_suffix(".json")
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    checkpoints.append((current_path, metadata))
                except:
                    pass

        # Check backup checkpoint
        backup_path = self.checkpoint_dir / "checkpoint_backup.pt"
        if backup_path.exists():
            meta_path = backup_path.with_suffix(".json")
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    checkpoints.append((backup_path, metadata))
                except:
                    pass

        return checkpoints