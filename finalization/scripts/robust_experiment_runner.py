#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robust experiment runner with comprehensive error handling.

This script demonstrates best practices for running experiments with:
- Automatic OOM recovery
- Checkpoint management
- Error tracking and reporting
- Memory monitoring
- Distributed training support
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import torch
import torch.distributed as dist

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from latentwire.error_handling import (
    ErrorTracker,
    retry_on_oom,
    handle_missing_files,
    RobustCheckpointer,
    DistributedErrorHandler,
    safe_json_dump,
    MemoryMonitor,
    log_system_info,
)


class RobustExperimentRunner:
    """Run experiments with comprehensive error handling and recovery."""

    def __init__(self, experiment_name: str, output_dir: str = "runs"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.error_tracker = ErrorTracker(experiment_name)
        self.checkpointer = RobustCheckpointer(self.output_dir / "checkpoints")
        self.memory_monitor = MemoryMonitor(threshold_gb=0.85)

        # Log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"experiment_{timestamp}.log"

        # Results storage
        self.results = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "system_info": {},
            "config": {},
            "metrics": {},
            "errors": [],
        }

    @retry_on_oom(max_retries=3, reduce_batch_size=True)
    def train_epoch(self, epoch: int, model, dataloader, optimizer,
                   batch_size: int = None) -> Dict[str, float]:
        """Train one epoch with OOM recovery.

        Args:
            epoch: Current epoch number
            model: Model to train
            dataloader: Training dataloader
            optimizer: Optimizer
            batch_size: Current batch size (for OOM recovery)

        Returns:
            Dictionary of training metrics
        """
        model.train()
        total_loss = 0.0
        num_batches = 0

        try:
            progress = tqdm(dataloader, desc=f"Epoch {epoch}")
            for batch_idx, batch in enumerate(progress):
                # Memory check every 10 batches
                if batch_idx % 10 == 0:
                    mem_stats = self.memory_monitor.check_memory()
                    if batch_idx % 50 == 0:  # Log less frequently
                        print(f"Memory: GPU0={mem_stats.get('gpu_0_allocated_gb', 0):.2f}GB")

                # Forward pass
                optimizer.zero_grad()
                loss = self._compute_loss(model, batch)

                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Update statistics
                total_loss += loss.item()
                num_batches += 1

                progress.set_postfix({"loss": loss.item()})

                # Periodic checkpoint
                if batch_idx > 0 and batch_idx % 500 == 0:
                    self._save_checkpoint(model, optimizer, epoch, batch_idx)

        except Exception as e:
            self.error_tracker.log_error(e, {
                "epoch": epoch,
                "batch_idx": batch_idx if 'batch_idx' in locals() else None,
                "batch_size": batch_size,
            })
            raise

        metrics = {
            "epoch": epoch,
            "avg_loss": total_loss / max(num_batches, 1),
            "num_batches": num_batches,
        }

        return metrics

    @retry_on_oom(max_retries=2)
    def evaluate(self, model, dataloader) -> Dict[str, float]:
        """Evaluate model with OOM recovery.

        Args:
            model: Model to evaluate
            dataloader: Validation dataloader

        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Evaluating"):
                    loss, correct = self._evaluate_batch(model, batch)
                    total_loss += loss
                    total_correct += correct
                    total_samples += len(batch)

                    # Clear cache periodically
                    if total_samples % 1000 == 0:
                        torch.cuda.empty_cache()

        except Exception as e:
            self.error_tracker.log_error(e, {
                "phase": "evaluation",
                "samples_processed": total_samples,
            })
            raise

        metrics = {
            "eval_loss": total_loss / max(len(dataloader), 1),
            "accuracy": total_correct / max(total_samples, 1),
            "total_samples": total_samples,
        }

        return metrics

    def _compute_loss(self, model, batch) -> torch.Tensor:
        """Compute loss for a batch (placeholder - implement actual loss)."""
        # This is a placeholder - implement your actual loss computation
        outputs = model(batch['input'])
        loss = torch.nn.functional.mse_loss(outputs, batch['target'])
        return loss

    def _evaluate_batch(self, model, batch) -> tuple:
        """Evaluate a single batch (placeholder - implement actual evaluation)."""
        # This is a placeholder - implement your actual evaluation
        outputs = model(batch['input'])
        loss = torch.nn.functional.mse_loss(outputs, batch['target'])
        correct = (outputs.argmax(-1) == batch['target'].argmax(-1)).sum().item()
        return loss.item(), correct

    def _save_checkpoint(self, model, optimizer, epoch: int,
                        batch_idx: Optional[int] = None):
        """Save checkpoint with validation."""
        state_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "batch_idx": batch_idx,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            suffix = f"_batch{batch_idx}" if batch_idx else ""
            checkpoint_path = self.checkpointer.save_checkpoint(
                state_dict, epoch, validate=True
            )
            print(f"Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            self.error_tracker.log_error(e, {
                "action": "save_checkpoint",
                "epoch": epoch,
                "batch_idx": batch_idx,
            })
            print(f"Failed to save checkpoint: {str(e)}")

    def resume_from_checkpoint(self, model, optimizer) -> int:
        """Resume training from latest checkpoint.

        Args:
            model: Model to restore
            optimizer: Optimizer to restore

        Returns:
            Starting epoch number
        """
        checkpoint = self.checkpointer.load_checkpoint()

        if checkpoint is None:
            print("No checkpoint found, starting from scratch")
            return 0

        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1

            print(f"Resumed from checkpoint: epoch {start_epoch}")
            return start_epoch

        except Exception as e:
            self.error_tracker.log_error(e, {
                "action": "resume_checkpoint"
            })
            print(f"Failed to resume from checkpoint: {str(e)}")
            return 0

    @handle_missing_files(default_return={}, create_if_missing=True)
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file with error handling.

        Args:
            config_path: Path to configuration JSON file

        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.results["config"] = config
        return config

    def run_experiment(self, config: Dict[str, Any]):
        """Run complete experiment with error handling.

        Args:
            config: Experiment configuration
        """
        print(f"Starting experiment: {self.experiment_name}")
        self.results["system_info"] = log_system_info()

        try:
            # Initialize model, optimizer, dataloaders
            # (These would be actual implementations)
            model = self._create_model(config)
            optimizer = self._create_optimizer(model, config)
            train_loader = self._create_dataloader(config, "train")
            val_loader = self._create_dataloader(config, "val")

            # Resume if checkpoint exists
            start_epoch = self.resume_from_checkpoint(model, optimizer)

            # Training loop
            for epoch in range(start_epoch, config["num_epochs"]):
                print(f"\n{'='*50}")
                print(f"Epoch {epoch}/{config['num_epochs']}")
                print(f"{'='*50}")

                # Train
                train_metrics = self.train_epoch(
                    epoch, model, train_loader, optimizer,
                    batch_size=config.get("batch_size", 32)
                )
                self.results["metrics"][f"epoch_{epoch}_train"] = train_metrics

                # Evaluate
                eval_metrics = self.evaluate(model, val_loader)
                self.results["metrics"][f"epoch_{epoch}_eval"] = eval_metrics

                # Save checkpoint
                self._save_checkpoint(model, optimizer, epoch)

                # Log progress
                print(f"Train Loss: {train_metrics['avg_loss']:.4f}")
                print(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
                print(f"Accuracy: {eval_metrics['accuracy']:.4f}")

                # Save intermediate results
                safe_json_dump(self.results, self.output_dir / "results.json")

        except KeyboardInterrupt:
            print("\nExperiment interrupted by user")
            self.results["status"] = "interrupted"

        except Exception as e:
            print(f"\nExperiment failed with error: {str(e)}")
            self.error_tracker.log_error(e, {"phase": "main_loop"})
            self.results["status"] = "failed"
            traceback.print_exc()

        finally:
            # Final cleanup and save
            self.results["end_time"] = datetime.now().isoformat()
            self.results["errors"] = self.error_tracker.get_summary()

            # Save final results
            safe_json_dump(self.results, self.output_dir / "final_results.json")

            print(f"\nExperiment complete. Results saved to: {self.output_dir}")
            print(f"Total errors: {len(self.error_tracker.errors)}")

    def _create_model(self, config: Dict[str, Any]):
        """Create model (placeholder - implement actual model creation)."""
        # This is a placeholder - implement your actual model creation
        import torch.nn as nn
        model = nn.Linear(config.get("input_dim", 100),
                         config.get("output_dim", 10))
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def _create_optimizer(self, model, config: Dict[str, Any]):
        """Create optimizer (placeholder - implement actual optimizer)."""
        import torch.optim as optim
        return optim.Adam(model.parameters(), lr=config.get("lr", 0.001))

    def _create_dataloader(self, config: Dict[str, Any], split: str):
        """Create dataloader (placeholder - implement actual dataloader)."""
        # This is a placeholder - implement your actual dataloader
        from torch.utils.data import DataLoader, TensorDataset

        # Dummy data
        size = 1000 if split == "train" else 200
        data = TensorDataset(
            torch.randn(size, config.get("input_dim", 100)),
            torch.randn(size, config.get("output_dim", 10))
        )

        return DataLoader(
            data,
            batch_size=config.get("batch_size", 32),
            shuffle=(split == "train")
        )


def main():
    parser = argparse.ArgumentParser(description="Robust experiment runner")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="Name of the experiment")
    parser.add_argument("--config", type=str,
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="runs",
                       help="Output directory")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from latest checkpoint")

    args = parser.parse_args()

    # Create runner
    runner = RobustExperimentRunner(
        args.experiment_name,
        args.output_dir
    )

    # Load config or use defaults
    if args.config:
        config = runner.load_config(args.config)
    else:
        config = {
            "num_epochs": 10,
            "batch_size": 32,
            "lr": 0.001,
            "input_dim": 100,
            "output_dim": 10,
        }

    # Run experiment
    runner.run_experiment(config)


if __name__ == "__main__":
    main()