# -*- coding: utf-8 -*-
"""Comprehensive error handling utilities for LatentWire.

This module provides robust error handling, retry mechanisms, and recovery strategies
for common failure modes in distributed training and evaluation.
"""

import os
import gc
import sys
import time
import json
import signal
import logging
import traceback
import functools
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Union, Type
from contextlib import contextmanager
from datetime import datetime, timedelta

import torch
import torch.distributed as dist

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Error tracking
ERROR_LOG_PATH = Path("runs/error_logs")
ERROR_LOG_PATH.mkdir(parents=True, exist_ok=True)


class ErrorTracker:
    """Track and log errors with context for debugging."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.errors: List[Dict[str, Any]] = []
        self.log_file = ERROR_LOG_PATH / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with context."""
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        self.errors.append(error_record)

        # Write to file immediately
        with open(self.log_file, 'w') as f:
            json.dump(self.errors, f, indent=2, default=str)

        logger.error(f"Error logged: {error.__class__.__name__}: {str(error)}")
        if context:
            logger.error(f"Context: {context}")

    def get_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        if not self.errors:
            return {"total_errors": 0}

        error_types = {}
        for error in self.errors:
            error_type = error["type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_errors": len(self.errors),
            "error_types": error_types,
            "first_error": self.errors[0]["timestamp"],
            "last_error": self.errors[-1]["timestamp"],
            "log_file": str(self.log_file)
        }


def retry_on_oom(max_retries: int = 3,
                 reduce_batch_size: bool = True,
                 clear_cache: bool = True,
                 wait_time: float = 5.0):
    """Decorator to retry operations on OOM errors with recovery strategies.

    Args:
        max_retries: Maximum number of retry attempts
        reduce_batch_size: Whether to reduce batch size on retry
        clear_cache: Whether to clear GPU cache before retry
        wait_time: Time to wait between retries (seconds)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            batch_size = kwargs.get('batch_size', None)

            for attempt in range(max_retries + 1):
                try:
                    # Clear cache before retry if requested
                    if attempt > 0 and clear_cache and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(wait_time)

                    # Reduce batch size on retry if requested
                    if attempt > 0 and reduce_batch_size and batch_size:
                        new_batch_size = max(1, batch_size // (2 ** attempt))
                        kwargs['batch_size'] = new_batch_size
                        logger.info(f"Retry {attempt}: Reduced batch size from {batch_size} to {new_batch_size}")

                    return func(*args, **kwargs)

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" in str(e).lower():
                        last_error = e
                        logger.warning(f"OOM error on attempt {attempt + 1}/{max_retries + 1}: {str(e)}")

                        # Log memory stats
                        if torch.cuda.is_available():
                            for i in range(torch.cuda.device_count()):
                                allocated = torch.cuda.memory_allocated(i) / 1024**3
                                reserved = torch.cuda.memory_reserved(i) / 1024**3
                                logger.info(f"GPU {i}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

                        if attempt >= max_retries:
                            logger.error(f"Failed after {max_retries + 1} attempts")
                            raise
                    else:
                        raise

            raise last_error
        return wrapper
    return decorator


def handle_missing_files(default_return=None, create_if_missing: bool = False):
    """Decorator to handle missing file errors gracefully.

    Args:
        default_return: Value to return if file is missing
        create_if_missing: Whether to create parent directories if missing
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Extract file path from args/kwargs
                file_path = None
                for arg in args:
                    if isinstance(arg, (str, Path)):
                        if Path(arg).suffix:  # Likely a file path
                            file_path = Path(arg)
                            break

                for key, value in kwargs.items():
                    if 'path' in key.lower() and isinstance(value, (str, Path)):
                        file_path = Path(value)
                        break

                # Create parent directory if requested
                if file_path and create_if_missing:
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                return func(*args, **kwargs)

            except (FileNotFoundError, OSError) as e:
                logger.warning(f"File operation failed: {str(e)}")
                if default_return is not None:
                    logger.info(f"Returning default value: {default_return}")
                    return default_return
                raise
        return wrapper
    return decorator


@contextmanager
def timeout_handler(seconds: int, error_message: str = "Operation timed out"):
    """Context manager to handle operation timeouts.

    Args:
        seconds: Timeout in seconds
        error_message: Error message to display on timeout
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(error_message)

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class RobustCheckpointer:
    """Robust checkpointing with automatic recovery and validation."""

    def __init__(self, checkpoint_dir: Union[str, Path], max_keep: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep

    @retry_on_oom(max_retries=2)
    def save_checkpoint(self, state_dict: Dict[str, Any], epoch: int,
                       validate: bool = True) -> Path:
        """Save checkpoint with validation and cleanup.

        Args:
            state_dict: Model and optimizer state
            epoch: Current epoch
            validate: Whether to validate saved checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        temp_path = checkpoint_path.with_suffix('.tmp')

        try:
            # Save to temporary file first
            torch.save(state_dict, temp_path)

            # Validate if requested
            if validate:
                test_load = torch.load(temp_path, map_location='cpu')
                assert 'model_state_dict' in test_load or 'model' in test_load
                del test_load
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Move to final location
            temp_path.rename(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load_checkpoint(self, checkpoint_path: Union[str, Path] = None,
                       map_location: str = 'cpu') -> Optional[Dict[str, Any]]:
        """Load checkpoint with automatic recovery.

        Args:
            checkpoint_path: Specific checkpoint to load, or None for latest
            map_location: Device to load checkpoint to

        Returns:
            Loaded checkpoint dict or None if not found
        """
        if checkpoint_path:
            path = Path(checkpoint_path)
        else:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if not checkpoints:
                logger.warning("No checkpoints found")
                return None
            path = checkpoints[-1]

        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return None

        try:
            checkpoint = torch.load(path, map_location=map_location)
            logger.info(f"Loaded checkpoint: {path}")
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint {path}: {str(e)}")

            # Try backup if exists
            backup_path = path.with_suffix('.backup')
            if backup_path.exists():
                logger.info("Attempting to load backup checkpoint")
                try:
                    checkpoint = torch.load(backup_path, map_location=map_location)
                    logger.info("Successfully loaded backup checkpoint")
                    return checkpoint
                except Exception as backup_error:
                    logger.error(f"Backup also failed: {str(backup_error)}")

            return None

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_keep."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))

        if len(checkpoints) > self.max_keep:
            to_remove = checkpoints[:-self.max_keep]
            for checkpoint in to_remove:
                try:
                    checkpoint.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint}")

                    # Also remove backup if exists
                    backup = checkpoint.with_suffix('.backup')
                    if backup.exists():
                        backup.unlink()

                except Exception as e:
                    logger.warning(f"Failed to remove {checkpoint}: {str(e)}")


class DistributedErrorHandler:
    """Handle errors in distributed training environments."""

    @staticmethod
    def sync_error_state(error_occurred: bool) -> bool:
        """Synchronize error state across all processes.

        Args:
            error_occurred: Whether an error occurred in this process

        Returns:
            True if any process had an error
        """
        if not dist.is_initialized():
            return error_occurred

        # Convert to tensor for all_reduce
        error_tensor = torch.tensor([1.0 if error_occurred else 0.0]).cuda()
        dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)

        return error_tensor.item() > 0.5

    @staticmethod
    @contextmanager
    def distributed_catch(rank: int = 0):
        """Context manager for error handling in distributed setting.

        Args:
            rank: Current process rank
        """
        error_occurred = False

        try:
            yield
        except Exception as e:
            error_occurred = True
            if rank == 0:
                logger.error(f"Error in rank {rank}: {str(e)}")
                logger.error(traceback.format_exc())

            # Sync error state
            if dist.is_initialized():
                error_occurred = DistributedErrorHandler.sync_error_state(True)

            raise
        finally:
            # Ensure all processes know if an error occurred
            if dist.is_initialized():
                error_occurred = DistributedErrorHandler.sync_error_state(error_occurred)

                if error_occurred:
                    logger.warning(f"Rank {rank}: Error detected in distributed training")


def safe_json_dump(data: Any, file_path: Union[str, Path],
                   backup: bool = True) -> bool:
    """Safely dump JSON with atomic writes and backup.

    Args:
        data: Data to serialize
        file_path: Output file path
        backup: Whether to create backup of existing file

    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    temp_path = file_path.with_suffix('.tmp')

    try:
        # Backup existing file if requested
        if backup and file_path.exists():
            backup_path = file_path.with_suffix('.backup')
            file_path.rename(backup_path)

        # Write to temporary file
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        # Atomic rename
        temp_path.rename(file_path)
        return True

    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {str(e)}")

        # Restore backup if exists
        if backup:
            backup_path = file_path.with_suffix('.backup')
            if backup_path.exists():
                backup_path.rename(file_path)
                logger.info("Restored from backup")

        return False


class MemoryMonitor:
    """Monitor and manage memory usage during training."""

    def __init__(self, threshold_gb: float = 0.9, cleanup_interval: int = 100):
        """Initialize memory monitor.

        Args:
            threshold_gb: Memory threshold in GB to trigger cleanup
            cleanup_interval: Steps between automatic cleanups
        """
        self.threshold_gb = threshold_gb
        self.cleanup_interval = cleanup_interval
        self.step_count = 0

    def check_memory(self, force_cleanup: bool = False) -> Dict[str, float]:
        """Check memory usage and cleanup if needed.

        Args:
            force_cleanup: Force cleanup regardless of thresholds

        Returns:
            Dictionary with memory statistics
        """
        stats = {}

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3

                stats[f"gpu_{i}_allocated_gb"] = allocated
                stats[f"gpu_{i}_reserved_gb"] = reserved
                stats[f"gpu_{i}_total_gb"] = total
                stats[f"gpu_{i}_free_gb"] = total - allocated

                # Cleanup if threshold exceeded
                if force_cleanup or allocated > self.threshold_gb * total:
                    logger.info(f"GPU {i}: Memory cleanup triggered (allocated={allocated:.2f}GB)")
                    torch.cuda.empty_cache()
                    gc.collect()

        # Periodic cleanup
        self.step_count += 1
        if self.step_count % self.cleanup_interval == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        return stats


def log_system_info():
    """Log system information for debugging."""
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_count"] = torch.cuda.device_count()

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info[f"gpu_{i}_name"] = props.name
            info[f"gpu_{i}_memory_gb"] = props.total_memory / 1024**3

    logger.info("System Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    return info


# Export main utilities
__all__ = [
    'ErrorTracker',
    'retry_on_oom',
    'handle_missing_files',
    'timeout_handler',
    'RobustCheckpointer',
    'DistributedErrorHandler',
    'safe_json_dump',
    'MemoryMonitor',
    'log_system_info',
]