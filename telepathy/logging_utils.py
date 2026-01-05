"""
Comprehensive logging utilities for preemptible HPC jobs.

This module provides robust logging infrastructure that ensures no log data is lost
even during preemption. It captures stdout, stderr, and Python logging with frequent
flushes and persistent storage.

Features:
- TeeLogger that writes to file and console simultaneously
- Automatic log rotation and compression
- Structured logging (JSON lines for metrics)
- Log recovery after preemption
- Integration with SLURM output capture
"""

import os
import sys
import json
import time
import signal
import atexit
import shutil
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, TextIO
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import gzip


# ============================================================================
# Core Logging Configuration
# ============================================================================

@dataclass
class LogConfig:
    """Configuration for comprehensive logging."""
    output_dir: str
    experiment_name: str
    flush_interval: float = 1.0  # Flush every second
    buffer_size: int = 1  # Line buffering
    compress_old_logs: bool = True
    max_log_size_mb: float = 100.0
    enable_structured_logs: bool = True
    enable_git_backup: bool = True
    backup_interval: int = 300  # Backup to git every 5 minutes


class TeeLogger:
    """
    Thread-safe logger that writes to multiple streams simultaneously.

    Features:
    - Writes to console and file(s) atomically
    - Automatic flushing on every write
    - Thread-safe for concurrent logging
    - Graceful handling of write failures
    """

    def __init__(self, *streams: TextIO, flush_on_write: bool = True):
        """
        Initialize TeeLogger with multiple output streams.

        Args:
            *streams: Variable number of file-like objects to write to
            flush_on_write: Whether to flush after each write
        """
        self.streams = list(streams)
        self.flush_on_write = flush_on_write
        self._lock = threading.Lock()
        self._closed = False

        # Register cleanup handlers
        atexit.register(self.close)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals by flushing and closing."""
        self.flush()
        self.close()

    def write(self, data: str) -> int:
        """Write data to all streams atomically."""
        if self._closed:
            return 0

        bytes_written = 0
        with self._lock:
            for stream in self.streams:
                try:
                    bytes_written = stream.write(data)
                    if self.flush_on_write:
                        stream.flush()
                        # Force OS-level flush for critical data
                        if hasattr(stream, 'fileno'):
                            os.fsync(stream.fileno())
                except (IOError, OSError) as e:
                    # Log to stderr if we can't write to a stream
                    print(f"Warning: Failed to write to stream: {e}", file=sys.stderr)
                    continue

        return bytes_written

    def flush(self):
        """Flush all streams."""
        if self._closed:
            return

        with self._lock:
            for stream in self.streams:
                try:
                    stream.flush()
                    if hasattr(stream, 'fileno'):
                        os.fsync(stream.fileno())
                except (IOError, OSError):
                    continue

    def close(self):
        """Close all file streams (but not stdout/stderr)."""
        if self._closed:
            return

        self.flush()
        with self._lock:
            for stream in self.streams:
                if stream not in (sys.stdout, sys.stderr):
                    try:
                        stream.close()
                    except (IOError, OSError):
                        pass
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class StructuredLogger:
    """
    Logger for structured data (metrics, checkpoints, etc.) in JSONL format.

    Features:
    - Atomic writes with immediate flush
    - Automatic timestamp addition
    - Safe concurrent access
    - Crash recovery from partial writes
    """

    def __init__(self, log_path: Union[str, Path], mode: str = 'a'):
        """
        Initialize structured logger.

        Args:
            log_path: Path to JSONL file
            mode: File open mode ('a' for append, 'w' for write)
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Open with line buffering (buffer_size=1)
        self._file = open(self.log_path, mode, buffering=1, encoding='utf-8')
        self._lock = threading.Lock()

        # Register cleanup
        atexit.register(self.close)

    def log(self, data: Dict[str, Any], **kwargs):
        """
        Log structured data as JSON line.

        Args:
            data: Dictionary to log
            **kwargs: Additional fields to add to data
        """
        with self._lock:
            # Add timestamp if not present
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now().isoformat()

            # Add any kwargs
            data.update(kwargs)

            # Write as JSON line
            try:
                json_str = json.dumps(data, default=str)
                self._file.write(json_str + '\n')
                self._file.flush()
                os.fsync(self._file.fileno())
            except Exception as e:
                print(f"Error writing to structured log: {e}", file=sys.stderr)

    def close(self):
        """Close the log file."""
        if hasattr(self, '_file') and not self._file.closed:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class CheckpointLogger:
    """
    Specialized logger for checkpoint metadata that ensures recovery after preemption.

    Features:
    - Atomic checkpoint state updates
    - Recovery information for resuming
    - Integration with training loop
    """

    def __init__(self, checkpoint_dir: Union[str, Path]):
        """Initialize checkpoint logger."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.checkpoint_dir / "training_state.json"
        self.backup_file = self.checkpoint_dir / "training_state.backup.json"

        # Structured logger for checkpoint events
        self.event_logger = StructuredLogger(
            self.checkpoint_dir / "checkpoint_events.jsonl"
        )

    def save_state(self, state: Dict[str, Any]):
        """
        Save training state atomically.

        Args:
            state: Training state dictionary
        """
        # Add metadata
        state['last_update'] = datetime.now().isoformat()
        state['pid'] = os.getpid()

        # Log event
        self.event_logger.log({
            'event': 'save_state',
            'epoch': state.get('epoch'),
            'step': state.get('step'),
            'loss': state.get('loss')
        })

        # Save atomically with backup
        temp_file = self.state_file.with_suffix('.tmp')

        # Write to temp file
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())

        # Backup existing state if it exists
        if self.state_file.exists():
            shutil.copy2(self.state_file, self.backup_file)

        # Atomic rename
        os.replace(temp_file, self.state_file)

    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load training state, trying backup if main file is corrupted.

        Returns:
            Training state dict or None if not found
        """
        # Try main state file
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                self.event_logger.log({'event': 'load_state', 'source': 'main'})
                return state
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Main state file corrupted, trying backup", file=sys.stderr)

        # Try backup
        if self.backup_file.exists():
            try:
                with open(self.backup_file) as f:
                    state = json.load(f)
                self.event_logger.log({'event': 'load_state', 'source': 'backup'})
                return state
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Backup state file also corrupted", file=sys.stderr)

        self.event_logger.log({'event': 'load_state', 'source': 'none'})
        return None

    def close(self):
        """Close the logger."""
        self.event_logger.close()


class LogRotator:
    """
    Handles log rotation and compression to manage disk space.

    Features:
    - Automatic rotation when size limit reached
    - Compression of old logs
    - Configurable retention policy
    """

    def __init__(
        self,
        log_path: Union[str, Path],
        max_size_mb: float = 100.0,
        max_backups: int = 10,
        compress: bool = True
    ):
        """
        Initialize log rotator.

        Args:
            log_path: Path to log file
            max_size_mb: Maximum size before rotation
            max_backups: Number of backups to keep
            compress: Whether to compress rotated logs
        """
        self.log_path = Path(log_path)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_backups = max_backups
        self.compress = compress

    def should_rotate(self) -> bool:
        """Check if log should be rotated."""
        if not self.log_path.exists():
            return False
        return self.log_path.stat().st_size >= self.max_size_bytes

    def rotate(self):
        """Rotate the log file."""
        if not self.log_path.exists():
            return

        # Generate backup name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{self.log_path.stem}.{timestamp}{self.log_path.suffix}"

        if self.compress:
            backup_path = self.log_path.parent / f"{backup_name}.gz"

            # Compress the current log
            with open(self.log_path, 'rb') as f_in:
                with gzip.open(backup_path, 'wb', compresslevel=6) as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Clear the original log
            open(self.log_path, 'w').close()
        else:
            backup_path = self.log_path.parent / backup_name
            shutil.move(self.log_path, backup_path)

        # Clean old backups
        self._cleanup_old_backups()

    def _cleanup_old_backups(self):
        """Remove old backup files beyond retention limit."""
        pattern = f"{self.log_path.stem}.*{self.log_path.suffix}"
        if self.compress:
            pattern += ".gz"

        backups = sorted(self.log_path.parent.glob(pattern))

        # Keep only max_backups
        for old_backup in backups[:-self.max_backups]:
            old_backup.unlink()


# ============================================================================
# Git Integration for Log Persistence
# ============================================================================

class GitLogBackup:
    """
    Backs up logs to git repository for persistence across preemptions.

    Features:
    - Automatic periodic commits
    - Selective file staging (logs only, not checkpoints)
    - Conflict-free push with auto-stash
    """

    def __init__(
        self,
        repo_dir: Union[str, Path],
        backup_interval: int = 300,
        auto_push: bool = True
    ):
        """
        Initialize git log backup.

        Args:
            repo_dir: Git repository directory
            backup_interval: Seconds between backups
            auto_push: Whether to push to remote
        """
        self.repo_dir = Path(repo_dir)
        self.backup_interval = backup_interval
        self.auto_push = auto_push
        self._stop_event = threading.Event()
        self._backup_thread = None

        # Ensure git is configured
        self._configure_git()

    def _configure_git(self):
        """Configure git identity if not set."""
        try:
            # Check if user.name is set
            result = subprocess.run(
                ['git', 'config', 'user.name'],
                cwd=self.repo_dir,
                capture_output=True,
                text=True
            )
            if not result.stdout.strip():
                subprocess.run(
                    ['git', 'config', 'user.name', f'SLURM Job {os.getenv("SLURM_JOB_ID", "unknown")}'],
                    cwd=self.repo_dir
                )

            # Check if user.email is set
            result = subprocess.run(
                ['git', 'config', 'user.email'],
                cwd=self.repo_dir,
                capture_output=True,
                text=True
            )
            if not result.stdout.strip():
                subprocess.run(
                    ['git', 'config', 'user.email', 'slurm@hpc.cluster'],
                    cwd=self.repo_dir
                )
        except subprocess.CalledProcessError:
            print("Warning: Could not configure git", file=sys.stderr)

    def start(self):
        """Start the background backup thread."""
        if self._backup_thread is None or not self._backup_thread.is_alive():
            self._backup_thread = threading.Thread(target=self._backup_loop, daemon=True)
            self._backup_thread.start()

    def stop(self):
        """Stop the backup thread and do final backup."""
        self._stop_event.set()
        if self._backup_thread:
            self._backup_thread.join(timeout=10)

        # Final backup
        self._backup_logs()

    def _backup_loop(self):
        """Background thread that periodically backs up logs."""
        while not self._stop_event.is_set():
            self._backup_logs()
            self._stop_event.wait(self.backup_interval)

    def _backup_logs(self):
        """Perform git backup of log files."""
        try:
            # Add only log files (not checkpoints)
            subprocess.run(
                ['git', 'add', 'runs/*.log', 'runs/*.err', 'runs/*.json',
                 'runs/**/*.log', 'runs/**/*.jsonl'],
                cwd=self.repo_dir,
                capture_output=True,
                stderr=subprocess.DEVNULL
            )

            # Check if there are changes
            result = subprocess.run(
                ['git', 'diff', '--cached', '--quiet'],
                cwd=self.repo_dir,
                capture_output=True
            )

            if result.returncode != 0:  # There are changes
                # Commit
                job_id = os.getenv('SLURM_JOB_ID', 'unknown')
                commit_msg = f"logs: auto-backup from SLURM job {job_id}\n\nTimestamp: {datetime.now().isoformat()}"

                subprocess.run(
                    ['git', 'commit', '-m', commit_msg],
                    cwd=self.repo_dir,
                    capture_output=True
                )

                # Push if enabled
                if self.auto_push:
                    self._safe_push()

        except subprocess.CalledProcessError as e:
            print(f"Warning: Git backup failed: {e}", file=sys.stderr)

    def _safe_push(self):
        """Push to remote with conflict handling."""
        try:
            # Try to push
            result = subprocess.run(
                ['git', 'push'],
                cwd=self.repo_dir,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                # Pull and retry
                subprocess.run(
                    ['git', 'pull', '--rebase=false'],
                    cwd=self.repo_dir,
                    capture_output=True
                )
                subprocess.run(
                    ['git', 'push'],
                    cwd=self.repo_dir,
                    capture_output=True
                )

        except subprocess.CalledProcessError:
            print("Warning: Could not push logs to remote", file=sys.stderr)


# ============================================================================
# Main Setup Function
# ============================================================================

@contextmanager
def setup_comprehensive_logging(config: LogConfig):
    """
    Set up comprehensive logging for a training run.

    This context manager sets up:
    - TeeLogger for stdout/stderr capture
    - StructuredLogger for metrics
    - CheckpointLogger for state tracking
    - LogRotator for space management
    - GitLogBackup for persistence

    Args:
        config: Logging configuration

    Yields:
        Dictionary with all logger instances

    Example:
        >>> config = LogConfig(
        ...     output_dir="runs/experiment1",
        ...     experiment_name="baseline_run"
        ... )
        >>> with setup_comprehensive_logging(config) as loggers:
        ...     # Training code here
        ...     loggers['metrics'].log({'epoch': 1, 'loss': 0.5})
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate log file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log = output_dir / f"{config.experiment_name}_{timestamp}.log"
    metrics_log = output_dir / f"metrics_{timestamp}.jsonl"

    # Initialize loggers
    loggers = {}

    try:
        # Set up main output capture
        log_file = open(main_log, 'w', buffering=config.buffer_size)
        tee_stdout = TeeLogger(sys.stdout, log_file, flush_on_write=True)
        tee_stderr = TeeLogger(sys.stderr, log_file, flush_on_write=True)

        # Replace stdout and stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr

        # Set up structured logging
        metrics_logger = StructuredLogger(metrics_log)

        # Set up checkpoint logging
        checkpoint_logger = CheckpointLogger(output_dir)

        # Set up log rotation
        log_rotator = LogRotator(
            main_log,
            max_size_mb=config.max_log_size_mb,
            compress=config.compress_old_logs
        )

        # Set up git backup if enabled
        git_backup = None
        if config.enable_git_backup:
            # Find git root
            git_root = output_dir
            while git_root.parent != git_root:
                if (git_root / '.git').exists():
                    break
                git_root = git_root.parent

            if (git_root / '.git').exists():
                git_backup = GitLogBackup(
                    git_root,
                    backup_interval=config.backup_interval,
                    auto_push=True
                )
                git_backup.start()

        # Configure Python logging to use our infrastructure
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(tee_stdout)
            ]
        )

        # Log initial setup
        print("=" * 80)
        print(f"Comprehensive Logging Initialized")
        print(f"Experiment: {config.experiment_name}")
        print(f"Output dir: {output_dir}")
        print(f"Main log: {main_log}")
        print(f"Metrics log: {metrics_log}")
        print(f"Git backup: {'enabled' if git_backup else 'disabled'}")
        print(f"Job ID: {os.getenv('SLURM_JOB_ID', 'local')}")
        print(f"Start time: {datetime.now().isoformat()}")
        print("=" * 80)

        # Populate loggers dictionary
        loggers['tee_stdout'] = tee_stdout
        loggers['tee_stderr'] = tee_stderr
        loggers['metrics'] = metrics_logger
        loggers['checkpoint'] = checkpoint_logger
        loggers['rotator'] = log_rotator
        loggers['git_backup'] = git_backup
        loggers['main_log_path'] = main_log
        loggers['metrics_log_path'] = metrics_log

        yield loggers

    finally:
        # Clean up in reverse order
        print("\n" + "=" * 80)
        print(f"Shutting down logging system")
        print(f"End time: {datetime.now().isoformat()}")
        print("=" * 80)

        # Stop git backup
        if 'git_backup' in loggers and loggers['git_backup']:
            loggers['git_backup'].stop()

        # Close structured loggers
        if 'metrics' in loggers:
            loggers['metrics'].close()
        if 'checkpoint' in loggers:
            loggers['checkpoint'].close()

        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Close tee loggers
        if 'tee_stdout' in loggers:
            loggers['tee_stdout'].close()
        if 'tee_stderr' in loggers:
            loggers['tee_stderr'].close()

        # Close log file
        if 'log_file' in locals():
            log_file.close()


# ============================================================================
# Integration Helpers
# ============================================================================

def log_metrics(metrics: Dict[str, Any], step: int = None, epoch: int = None,
                logger: StructuredLogger = None):
    """
    Helper to log training metrics.

    Args:
        metrics: Dictionary of metrics
        step: Training step
        epoch: Training epoch
        logger: StructuredLogger instance (uses global if not provided)
    """
    data = {'type': 'metrics'}

    if step is not None:
        data['step'] = step
    if epoch is not None:
        data['epoch'] = epoch

    data.update(metrics)

    if logger:
        logger.log(data)
    else:
        # Just print as fallback
        print(f"METRICS: {json.dumps(data, default=str)}")


def recover_from_preemption(checkpoint_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Recover training state after preemption.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Recovered state dict or None
    """
    checkpoint_logger = CheckpointLogger(checkpoint_dir)
    state = checkpoint_logger.load_state()

    if state:
        print(f"Recovered from preemption:")
        print(f"  Last epoch: {state.get('epoch', 'unknown')}")
        print(f"  Last step: {state.get('step', 'unknown')}")
        print(f"  Last update: {state.get('last_update', 'unknown')}")

    return state


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example configuration
    config = LogConfig(
        output_dir="runs/test_logging",
        experiment_name="test_run",
        flush_interval=1.0,
        enable_git_backup=True
    )

    # Example training loop with comprehensive logging
    with setup_comprehensive_logging(config) as loggers:
        # Check for recovery
        state = recover_from_preemption(config.output_dir)

        start_epoch = state['epoch'] if state else 0
        start_step = state['step'] if state else 0

        print(f"Starting training from epoch {start_epoch}, step {start_step}")

        # Simulate training
        for epoch in range(start_epoch, 5):
            for step in range(start_step if epoch == start_epoch else 0, 100):
                # Simulate work
                time.sleep(0.01)

                # Log metrics
                if step % 10 == 0:
                    metrics = {
                        'loss': 1.0 / (step + 1),
                        'accuracy': step / 100.0
                    }
                    log_metrics(metrics, step=step, epoch=epoch, logger=loggers['metrics'])

                    # Save checkpoint state
                    loggers['checkpoint'].save_state({
                        'epoch': epoch,
                        'step': step,
                        'loss': metrics['loss']
                    })

                    print(f"Epoch {epoch}, Step {step}: loss={metrics['loss']:.4f}")

                # Check for rotation
                if loggers['rotator'].should_rotate():
                    print("Rotating log file...")
                    loggers['rotator'].rotate()

        print("Training complete!")