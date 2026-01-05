#!/usr/bin/env python3
"""
Example training script demonstrating integration with comprehensive logging system.

This shows how to modify existing training scripts to use the robust logging
infrastructure that survives preemption.
"""

import os
import sys
import json
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from telepathy.logging_utils import (
    LogConfig,
    setup_comprehensive_logging,
    log_metrics,
    recover_from_preemption,
    CheckpointLogger
)

# Import actual training components
try:
    import torch
    import torch.nn as nn
    from latentwire.train import (
        prepare_training_data,
        InterlinguaEncoder,
        Adapter,
        LMWrapper
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, running in demo mode")


class PreemptibleTrainer:
    """
    Training wrapper that handles preemption gracefully.
    """

    def __init__(self, args, loggers):
        """
        Initialize trainer with logging infrastructure.

        Args:
            args: Training arguments
            loggers: Dictionary of logger instances from setup_comprehensive_logging
        """
        self.args = args
        self.loggers = loggers
        self.checkpoint_logger = loggers['checkpoint']
        self.metrics_logger = loggers['metrics']

        # State that needs to be saved
        self.current_epoch = 0
        self.current_step = 0
        self.global_step = 0
        self.best_loss = float('inf')

        # Register signal handlers
        signal.signal(signal.SIGUSR1, self._handle_preemption)
        signal.signal(signal.SIGTERM, self._handle_preemption)

    def _handle_preemption(self, signum, frame):
        """Handle preemption signal by saving state."""
        print(f"\n{'='*80}")
        print(f"PREEMPTION SIGNAL {signum} RECEIVED")
        print(f"Saving checkpoint at epoch {self.current_epoch}, step {self.current_step}")
        print(f"{'='*80}")

        # Save comprehensive state
        self._save_checkpoint(is_preemption=True)

        # Log preemption event
        self.metrics_logger.log({
            'event': 'preemption',
            'signal': signum,
            'epoch': self.current_epoch,
            'step': self.current_step,
            'global_step': self.global_step
        })

        # Ensure all logs are flushed
        sys.stdout.flush()
        sys.stderr.flush()

        # Exit gracefully
        sys.exit(0)

    def _save_checkpoint(self, is_preemption: bool = False):
        """
        Save training checkpoint with all necessary state.

        Args:
            is_preemption: Whether this is due to preemption
        """
        checkpoint_data = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'is_preemption': is_preemption,
            'timestamp': datetime.now().isoformat(),
            'job_id': os.getenv('SLURM_JOB_ID', 'local')
        }

        # Add model state if available
        if hasattr(self, 'model') and self.model is not None:
            if TORCH_AVAILABLE:
                checkpoint_data['model_state_dict'] = self.model.state_dict()

        # Add optimizer state if available
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            if TORCH_AVAILABLE:
                checkpoint_data['optimizer_state_dict'] = self.optimizer.state_dict()

        # Save via checkpoint logger
        self.checkpoint_logger.save_state(checkpoint_data)

        # Also save as PyTorch checkpoint if available
        if TORCH_AVAILABLE and hasattr(self, 'model'):
            checkpoint_path = Path(self.args.output_dir) / f"checkpoint_epoch{self.current_epoch}_step{self.current_step}.pt"
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    def recover_from_checkpoint(self) -> bool:
        """
        Attempt to recover from a previous checkpoint.

        Returns:
            True if recovery successful, False otherwise
        """
        state = self.checkpoint_logger.load_state()

        if state is None:
            print("No checkpoint found, starting fresh training")
            return False

        print(f"\n{'='*80}")
        print(f"RECOVERING FROM CHECKPOINT")
        print(f"Last saved: {state.get('timestamp', 'unknown')}")
        print(f"Epoch: {state.get('epoch', 0)}")
        print(f"Step: {state.get('step', 0)}")
        print(f"Global step: {state.get('global_step', 0)}")
        print(f"Was preempted: {state.get('is_preemption', False)}")
        print(f"{'='*80}\n")

        # Restore state
        self.current_epoch = state.get('epoch', 0)
        self.current_step = state.get('step', 0)
        self.global_step = state.get('global_step', 0)
        self.best_loss = state.get('best_loss', float('inf'))

        # Restore model state if available
        if TORCH_AVAILABLE and 'model_state_dict' in state and hasattr(self, 'model'):
            self.model.load_state_dict(state['model_state_dict'])
            print("Restored model state")

        # Restore optimizer state if available
        if TORCH_AVAILABLE and 'optimizer_state_dict' in state and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            print("Restored optimizer state")

        # Log recovery event
        self.metrics_logger.log({
            'event': 'recovery',
            'recovered_epoch': self.current_epoch,
            'recovered_step': self.current_step,
            'recovered_global_step': self.global_step
        })

        return True

    def train(self):
        """
        Main training loop with comprehensive logging.
        """
        print("Starting training with comprehensive logging...")

        # Try to recover from checkpoint
        recovered = self.recover_from_checkpoint()

        # Initialize model and optimizer (if not recovered)
        if TORCH_AVAILABLE and not recovered:
            self._initialize_model()

        # Log training start
        self.metrics_logger.log({
            'event': 'training_start',
            'recovered': recovered,
            'start_epoch': self.current_epoch,
            'start_step': self.current_step,
            'args': vars(self.args)
        })

        # Training loop
        for epoch in range(self.current_epoch, self.args.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            print(f"\n{'='*80}")
            print(f"EPOCH {epoch + 1}/{self.args.epochs}")
            print(f"{'='*80}")

            # Skip completed steps if recovering
            start_step = self.current_step if epoch == self.current_epoch else 0

            for step in range(start_step, self.args.steps_per_epoch):
                self.current_step = step
                self.global_step += 1

                # Simulate training step (replace with actual training)
                step_loss = self._train_step()

                # Log metrics periodically
                if step % self.args.log_interval == 0:
                    metrics = {
                        'loss': step_loss,
                        'learning_rate': self._get_lr(),
                        'epoch': epoch,
                        'step': step,
                        'global_step': self.global_step
                    }

                    # Log to structured logger
                    log_metrics(
                        metrics,
                        step=self.global_step,
                        epoch=epoch,
                        logger=self.metrics_logger
                    )

                    # Print to console (will be captured by TeeLogger)
                    print(f"[Epoch {epoch+1}][Step {step}/{self.args.steps_per_epoch}] "
                          f"Loss: {step_loss:.4f}, LR: {self._get_lr():.6f}")

                # Save checkpoint periodically
                if self.global_step % self.args.checkpoint_interval == 0:
                    print(f"Saving checkpoint at global step {self.global_step}...")
                    self._save_checkpoint()

                # Check if log rotation needed
                if self.loggers['rotator'].should_rotate():
                    print("Rotating log file due to size limit...")
                    self.loggers['rotator'].rotate()

            # End of epoch
            epoch_time = time.time() - epoch_start_time
            self.current_step = 0  # Reset for next epoch

            # Log epoch summary
            epoch_metrics = {
                'event': 'epoch_complete',
                'epoch': epoch,
                'epoch_time': epoch_time,
                'samples_per_second': self.args.steps_per_epoch / epoch_time
            }
            self.metrics_logger.log(epoch_metrics)

            print(f"\nEpoch {epoch+1} complete in {epoch_time:.2f}s")

        # Training complete
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}")

        # Final checkpoint
        self._save_checkpoint()

        # Log completion
        self.metrics_logger.log({
            'event': 'training_complete',
            'total_epochs': self.args.epochs,
            'total_steps': self.global_step,
            'best_loss': self.best_loss
        })

    def _initialize_model(self):
        """Initialize model and optimizer."""
        if not TORCH_AVAILABLE:
            self.model = None
            self.optimizer = None
            return

        print("Initializing model and optimizer...")

        # Example model initialization (replace with actual model)
        self.model = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate
        )

    def _train_step(self) -> float:
        """
        Perform a single training step.

        Returns:
            Loss value for this step
        """
        if TORCH_AVAILABLE and self.model is not None:
            # Actual training step
            self.optimizer.zero_grad()

            # Dummy forward pass (replace with actual training)
            input_data = torch.randn(32, 768)
            if torch.cuda.is_available():
                input_data = input_data.cuda()

            output = self.model(input_data)
            loss = output.mean()  # Dummy loss

            loss.backward()
            self.optimizer.step()

            return loss.item()
        else:
            # Simulation for demo
            import random
            time.sleep(0.01)  # Simulate computation
            return random.random() * 2.0

    def _get_lr(self) -> float:
        """Get current learning rate."""
        if TORCH_AVAILABLE and hasattr(self, 'optimizer'):
            return self.optimizer.param_groups[0]['lr']
        else:
            return self.args.learning_rate


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Example training with comprehensive logging')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                        help='Number of steps per epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log metrics every N steps')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='Save checkpoint every N steps')

    # Logging arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of the experiment')
    parser.add_argument('--enable_git_backup', action='store_true',
                        help='Enable automatic git backup of logs')

    args = parser.parse_args()

    # Set up logging configuration
    config = LogConfig(
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        flush_interval=1.0,
        buffer_size=1,  # Line buffering
        compress_old_logs=True,
        max_log_size_mb=100.0,
        enable_structured_logs=True,
        enable_git_backup=args.enable_git_backup,
        backup_interval=300  # 5 minutes
    )

    # Start comprehensive logging
    with setup_comprehensive_logging(config) as loggers:
        # Create trainer with logging infrastructure
        trainer = PreemptibleTrainer(args, loggers)

        # Start training
        trainer.train()


if __name__ == "__main__":
    main()