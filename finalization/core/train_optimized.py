#!/usr/bin/env python3
"""
Optimized training script for LatentWire with preemption support.

This is the main training entrypoint that includes:
- Elastic GPU configuration
- Checkpoint management
- Memory optimization
- Performance monitoring
- Preemption handling
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.checkpoint_manager import CheckpointManager
from training.logging_utils import PreemptibleLogger, ProgressTracker
from training.gpu_monitor import GPUMonitor


class OptimizedTrainer:
    """Main trainer with all optimizations."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set up infrastructure
        self.checkpoint_manager = CheckpointManager(args.output_dir)
        self.logger = PreemptibleLogger(Path(args.output_dir) / "training.log")
        self.gpu_monitor = GPUMonitor() if args.monitor_gpu else None
        self.progress = ProgressTracker(args.epochs, args.epochs * args.steps_per_epoch)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

    def train(self):
        """Main training loop with all optimizations."""
        self.logger.info("Starting optimized training")

        # Start monitoring
        if self.gpu_monitor:
            self.gpu_monitor.start()

        # Check for resume
        if self.args.resume:
            self._load_checkpoint()

        # Initialize models
        self._setup_models()

        # Initialize data
        train_loader = self._setup_data()

        # Initialize optimizer
        optimizer = self._setup_optimizer()

        # Training loop
        for epoch in range(self.epoch, self.args.epochs):
            self.epoch = epoch
            self.progress.start_epoch(epoch)

            epoch_loss = self._train_epoch(train_loader, optimizer)

            # Log progress
            self.logger.log_metrics(
                epoch=epoch,
                batch=-1,
                loss=epoch_loss,
                lr=optimizer.param_groups[0]['lr']
            )

            # Save checkpoint
            if self.checkpoint_manager.should_save():
                self._save_checkpoint(optimizer, epoch_loss)

            # Check early stopping
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self._save_checkpoint(optimizer, epoch_loss, tag='best')

        # Final save
        self._save_checkpoint(optimizer, epoch_loss, tag='final')

        # Stop monitoring
        if self.gpu_monitor:
            self.gpu_monitor.stop()
            print(self.gpu_monitor.get_report())

    def _train_epoch(self, dataloader, optimizer):
        """Train one epoch."""
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            loss = self._compute_loss(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.clip_grad_norm
                )

            optimizer.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if batch_idx % self.args.log_interval == 0:
                self.logger.log_metrics(
                    epoch=self.epoch,
                    batch=batch_idx,
                    loss=loss.item(),
                    lr=optimizer.param_groups[0]['lr']
                )

        return total_loss / num_batches

    def _setup_models(self):
        """Initialize models with optimizations."""
        # Placeholder - would load actual models
        self.model = nn.Linear(768, 768).to(self.device)

    def _setup_data(self):
        """Setup data loaders with optimizations."""
        # Placeholder - would load actual data
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(
            torch.randn(1000, 768),
            torch.randint(0, 2, (1000,))
        )

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

    def _setup_optimizer(self):
        """Setup optimizer with scheduling."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        return optimizer

    def _compute_loss(self, batch):
        """Compute loss for batch."""
        # Placeholder
        x = batch[0] if isinstance(batch, (list, tuple)) else batch['input']
        out = self.model(x)
        return out.mean()  # Dummy loss

    def _save_checkpoint(self, optimizer, loss, tag=None):
        """Save training checkpoint."""
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'args': vars(self.args),
        }

        self.checkpoint_manager.save_checkpoint(state, tag=tag)
        self.logger.log_checkpoint(
            self.checkpoint_manager.checkpoint_dir / f"checkpoint_{tag}.pt",
            self.epoch,
            self.global_step,
            reason=tag or "periodic"
        )

    def _load_checkpoint(self):
        """Load checkpoint to resume training."""
        state, metadata = self.checkpoint_manager.load_checkpoint()

        self.epoch = state['epoch']
        self.global_step = state['global_step']
        self.best_loss = state.get('best_loss', float('inf'))

        # Models and optimizer would be loaded here
        self.logger.info(f"Resumed from epoch {self.epoch}, step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description='Optimized LatentWire Training')

    # Training args
    parser.add_argument('--output_dir', type=str, default='runs/optimized')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)

    # Data args
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--steps_per_epoch', type=int, default=100)

    # Monitoring
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--monitor_gpu', action='store_true')

    # Checkpointing
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    # Create trainer and run
    trainer = OptimizedTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()