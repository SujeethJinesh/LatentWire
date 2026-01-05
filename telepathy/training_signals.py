#!/usr/bin/env python3
"""
Signal handling integration for preemptible training.

This module provides signal handlers that can be integrated into train.py
to support graceful checkpoint saving on preemption signals.

Usage in train.py:
    from telepathy.training_signals import install_signal_handlers, should_save_checkpoint

    # At start of training
    install_signal_handlers()

    # In training loop
    if should_save_checkpoint():
        save_checkpoint(...)
        mark_checkpoint_saved()
"""

import signal
import time
import threading
from typing import Optional, Callable

# Global state for signal handling
_save_requested = False
_save_lock = threading.Lock()
_last_save_time = time.time()
_preemption_received = False
_shutdown_requested = False


def handle_sigterm(signum, frame):
    """Handle SIGTERM (preemption warning)."""
    global _preemption_received, _save_requested
    print("\n🚨 SIGTERM received - preemption detected! Requesting checkpoint save...")
    _preemption_received = True
    _save_requested = True


def handle_sigusr1(signum, frame):
    """Handle SIGUSR1 (manual save request)."""
    global _save_requested
    print("\n📝 SIGUSR1 received - manual checkpoint save requested")
    _save_requested = True


def handle_sigint(signum, frame):
    """Handle SIGINT (Ctrl+C)."""
    global _save_requested, _shutdown_requested
    print("\n⛔ SIGINT received - requesting checkpoint save before exit...")
    _save_requested = True
    _shutdown_requested = True


def install_signal_handlers(
    on_sigterm: Optional[Callable] = None,
    on_sigusr1: Optional[Callable] = None,
    on_sigint: Optional[Callable] = None
):
    """
    Install signal handlers for preemptible training.

    Args:
        on_sigterm: Optional callback for SIGTERM
        on_sigusr1: Optional callback for SIGUSR1
        on_sigint: Optional callback for SIGINT
    """
    # Use custom handlers if provided, otherwise use defaults
    sigterm_handler = on_sigterm or handle_sigterm
    sigusr1_handler = on_sigusr1 or handle_sigusr1
    sigint_handler = on_sigint or handle_sigint

    # Install handlers
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGUSR1, sigusr1_handler)
    signal.signal(signal.SIGINT, sigint_handler)

    print("✅ Signal handlers installed for preemptible training")
    print("   - SIGTERM: Preemption warning (saves checkpoint)")
    print("   - SIGUSR1: Manual checkpoint save")
    print("   - SIGINT: Interrupt (saves before exit)")


def should_save_checkpoint(interval_seconds: int = 0) -> bool:
    """
    Check if a checkpoint should be saved.

    Args:
        interval_seconds: If > 0, also trigger saves at this interval

    Returns:
        True if checkpoint should be saved
    """
    global _save_requested, _last_save_time

    with _save_lock:
        # Check if save was explicitly requested
        if _save_requested:
            return True

        # Check interval-based saving
        if interval_seconds > 0:
            elapsed = time.time() - _last_save_time
            if elapsed >= interval_seconds:
                return True

    return False


def mark_checkpoint_saved():
    """Mark that a checkpoint was just saved."""
    global _save_requested, _last_save_time

    with _save_lock:
        _save_requested = False
        _last_save_time = time.time()


def should_exit_training() -> bool:
    """Check if training should exit due to preemption or interrupt."""
    return _shutdown_requested or _preemption_received


def reset_signal_state():
    """Reset all signal state (useful for testing)."""
    global _save_requested, _preemption_received, _shutdown_requested, _last_save_time

    with _save_lock:
        _save_requested = False
        _preemption_received = False
        _shutdown_requested = False
        _last_save_time = time.time()


# Integration helper for train.py
def integrate_with_training_loop(
    save_function: Callable,
    interval: int = 300,
    verbose: bool = True
) -> Callable:
    """
    Create a wrapper function for integration with training loop.

    Args:
        save_function: Function to call for saving checkpoint
        interval: Checkpoint interval in seconds
        verbose: Print status messages

    Returns:
        Function to call in training loop

    Example:
        # In train.py
        def save_my_checkpoint():
            save_latest_checkpoint(...)

        check_save = integrate_with_training_loop(save_my_checkpoint, interval=300)

        # In training loop
        for batch in dataloader:
            # ... training code ...
            check_save()  # Will save if needed
    """
    def check_and_save():
        if should_save_checkpoint(interval):
            if verbose:
                print(f"\n💾 Saving checkpoint (signal-triggered)...")

            try:
                save_function()
                mark_checkpoint_saved()

                if verbose:
                    print("✅ Checkpoint saved successfully")

                # Exit if preempted
                if should_exit_training():
                    if _preemption_received:
                        print("\n🔄 Exiting for preemption - job will be requeued")
                        exit(99)  # Special exit code for preemption
                    else:
                        print("\n⛔ Exiting due to interrupt")
                        exit(0)

            except Exception as e:
                print(f"❌ Failed to save checkpoint: {e}")
                if should_exit_training():
                    exit(1)

    return check_and_save


# Decorator for making functions preemption-aware
def preemptible(interval: int = 300):
    """
    Decorator to make a training function preemption-aware.

    Args:
        interval: Checkpoint save interval in seconds

    Example:
        @preemptible(interval=300)
        def train_model(model, dataloader):
            for batch in dataloader:
                # Training code
                yield {"step": step, "loss": loss}  # Yield state for checkpointing
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Install signal handlers
            install_signal_handlers()

            # Run the training function
            try:
                generator = func(*args, **kwargs)
                if hasattr(generator, '__next__'):
                    # Function is a generator, iterate through it
                    for state in generator:
                        # Check if we should save
                        if should_save_checkpoint(interval):
                            yield ("save", state)
                            mark_checkpoint_saved()

                            if should_exit_training():
                                break
                        else:
                            yield ("continue", state)
                else:
                    # Regular function, just run it
                    return generator
            except KeyboardInterrupt:
                print("\n⛔ Training interrupted")
                raise

        return wrapper
    return decorator