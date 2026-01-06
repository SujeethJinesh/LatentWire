#!/usr/bin/env python3
"""
Main Orchestrator for Preemptible HPC Experiments
==================================================

This script orchestrates all preemptible training experiments on the HPC cluster,
integrating all components for robust, efficient execution.

Features:
- Single command to run all experiments
- Handles all preemption scenarios (SIGTERM, OOM, crashes)
- Maximizes GPU utilization (1-4 GPUs with elastic scaling)
- Saves checkpoints every 5 minutes
- Resumes from exact stopping point
- Automatic batch size optimization
- Mixed precision training
- Comprehensive logging and monitoring
- Auto-recovery from failures

Usage:
    # Basic usage (runs default experiment suite)
    python telepathy/run_preemptible_experiments.py

    # Run specific experiment
    python telepathy/run_preemptible_experiments.py --experiment sst2

    # Resume from checkpoint
    python telepathy/run_preemptible_experiments.py --resume --checkpoint_dir runs/exp_001

    # Custom configuration
    python telepathy/run_preemptible_experiments.py --config configs/custom.yaml

    # Test mode (small dataset, quick validation)
    python telepathy/run_preemptible_experiments.py --test

SLURM Integration:
    Submit with: sbatch telepathy/submit_preemptible_orchestrator.slurm
"""

import os
import sys
import json
import yaml
import time
import signal
import argparse
import subprocess
import traceback
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import our components
from telepathy.checkpoint_manager import CheckpointManager
from telepathy.preemptible_training import PreemptionHandler
from telepathy.gpu_monitor import GPUMonitor
from telepathy.logging_utils import setup_experiment_logging, log_metrics


class ExperimentPhase(Enum):
    """Experiment phases for systematic execution."""
    WARMUP = "warmup"
    BASELINE = "baseline"
    TELEPATHY = "telepathy"
    ABLATION = "ablation"
    SCALING = "scaling"
    TRANSFER = "transfer"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    phase: ExperimentPhase
    dataset: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    eval_config: Dict[str, Any]
    priority: int = 1  # Lower is higher priority
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        config = asdict(self)
        config['phase'] = self.phase.value
        return config


class PreemptibleOrchestrator:
    """Main orchestrator for preemptible experiments."""

    def __init__(self, args: argparse.Namespace):
        """
        Initialize orchestrator.

        Args:
            args: Command line arguments
        """
        self.args = args
        self.start_time = time.time()

        # Setup directories
        self.setup_directories()

        # Initialize components
        self.preemption_handler = PreemptionHandler(grace_period=120)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            max_checkpoints=3,
            save_interval_minutes=5.0,
            enable_background_save=True
        )

        # GPU monitoring (if not in test mode)
        if not args.test and torch.cuda.is_available():
            self.gpu_monitor = GPUMonitor(
                log_dir=self.log_dir,
                interval=30,
                detailed=True
            )
            self.gpu_monitor.start()
        else:
            self.gpu_monitor = None

        # Setup logging
        self.logger = setup_experiment_logging(
            experiment_name=f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            log_dir=self.log_dir
        )

        # Load or create experiment queue
        self.experiment_queue = self.load_experiment_queue()

        # Track completed experiments
        self.completed_experiments = set()
        self.failed_experiments = {}

        # Distributed training setup
        self.world_size = None
        self.rank = None
        self.setup_distributed()

    def setup_directories(self):
        """Create necessary directories."""
        base_dir = Path(self.args.output_dir)

        self.checkpoint_dir = base_dir / "checkpoints"
        self.log_dir = base_dir / "logs"
        self.results_dir = base_dir / "results"
        self.figures_dir = base_dir / "figures"

        for directory in [self.checkpoint_dir, self.log_dir,
                         self.results_dir, self.figures_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_distributed(self):
        """Setup distributed training if multiple GPUs available."""
        if not torch.cuda.is_available():
            self.logger.info("CUDA not available, using CPU")
            return

        num_gpus = torch.cuda.device_count()
        if num_gpus <= 1:
            self.logger.info(f"Single GPU mode: {num_gpus} GPU(s)")
            return

        # Initialize distributed training
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
        else:
            self.rank = 0
            self.world_size = num_gpus
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
            self.logger.info(f"Distributed training initialized: rank {self.rank}/{self.world_size}")

    def load_experiment_queue(self) -> List[ExperimentConfig]:
        """Load or create experiment queue."""
        queue_file = self.checkpoint_dir / "experiment_queue.json"

        # Try to load existing queue (for resume)
        if queue_file.exists() and self.args.resume:
            try:
                with open(queue_file, 'r') as f:
                    data = json.load(f)

                queue = []
                for exp_dict in data['queue']:
                    exp_dict['phase'] = ExperimentPhase(exp_dict['phase'])
                    queue.append(ExperimentConfig(**exp_dict))

                self.completed_experiments = set(data.get('completed', []))
                self.failed_experiments = data.get('failed', {})

                self.logger.info(f"Loaded {len(queue)} experiments from queue")
                self.logger.info(f"Completed: {len(self.completed_experiments)}")
                self.logger.info(f"Failed: {len(self.failed_experiments)}")

                return queue
            except Exception as e:
                self.logger.warning(f"Failed to load queue: {e}")

        # Create new experiment queue
        return self.create_experiment_queue()

    def create_experiment_queue(self) -> List[ExperimentConfig]:
        """Create experiment queue based on configuration."""
        experiments = []

        if self.args.experiment == "all" or self.args.experiment == "sst2":
            experiments.extend(self.create_sst2_experiments())

        if self.args.experiment == "all" or self.args.experiment == "agnews":
            experiments.extend(self.create_agnews_experiments())

        if self.args.experiment == "all" or self.args.experiment == "trec":
            experiments.extend(self.create_trec_experiments())

        if self.args.experiment == "all" or self.args.experiment == "scaling":
            experiments.extend(self.create_scaling_experiments())

        if self.args.experiment == "all" or self.args.experiment == "ablation":
            experiments.extend(self.create_ablation_experiments())

        # Sort by priority
        experiments.sort(key=lambda x: (x.priority, x.name))

        self.logger.info(f"Created queue with {len(experiments)} experiments")
        return experiments

    def create_sst2_experiments(self) -> List[ExperimentConfig]:
        """Create SST-2 sentiment analysis experiments."""
        experiments = []

        # Base configuration
        base_model_config = {
            "source_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "target_model": "Qwen/Qwen2.5-7B-Instruct",
            "latent_len": 32,
            "d_z": 256,
        }

        base_training_config = {
            "dataset": "sst2",
            "samples": 1000 if self.args.test else 67349,
            "epochs": 2 if self.args.test else 10,
            "batch_size": 32,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-4,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "mixed_precision": "bf16",
        }

        base_eval_config = {
            "eval_samples": 100 if self.args.test else 872,
            "eval_batch_size": 64,
            "metrics": ["accuracy", "f1", "first_token_accuracy"],
        }

        # Baseline experiments
        experiments.append(ExperimentConfig(
            name="sst2_baseline_text",
            phase=ExperimentPhase.BASELINE,
            dataset="sst2",
            model_config=base_model_config,
            training_config={**base_training_config, "baseline": "text"},
            eval_config=base_eval_config,
            priority=1
        ))

        experiments.append(ExperimentConfig(
            name="sst2_baseline_linear",
            phase=ExperimentPhase.BASELINE,
            dataset="sst2",
            model_config=base_model_config,
            training_config={**base_training_config, "baseline": "linear_probe"},
            eval_config=base_eval_config,
            priority=1
        ))

        # Main telepathy experiment
        experiments.append(ExperimentConfig(
            name="sst2_telepathy_main",
            phase=ExperimentPhase.TELEPATHY,
            dataset="sst2",
            model_config=base_model_config,
            training_config=base_training_config,
            eval_config=base_eval_config,
            priority=0,
            dependencies=["sst2_baseline_text"]
        ))

        # Ablation studies
        for latent_len in [16, 32, 64, 128]:
            config = base_model_config.copy()
            config["latent_len"] = latent_len

            experiments.append(ExperimentConfig(
                name=f"sst2_ablation_latent_{latent_len}",
                phase=ExperimentPhase.ABLATION,
                dataset="sst2",
                model_config=config,
                training_config=base_training_config,
                eval_config=base_eval_config,
                priority=2,
                dependencies=["sst2_telepathy_main"]
            ))

        return experiments

    def create_agnews_experiments(self) -> List[ExperimentConfig]:
        """Create AG News experiments."""
        experiments = []

        base_model_config = {
            "source_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "target_model": "Qwen/Qwen2.5-7B-Instruct",
            "latent_len": 48,
            "d_z": 256,
        }

        base_training_config = {
            "dataset": "agnews",
            "samples": 1000 if self.args.test else 120000,
            "epochs": 2 if self.args.test else 5,
            "batch_size": 24,
            "gradient_accumulation_steps": 2,
            "learning_rate": 8e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "mixed_precision": "bf16",
        }

        base_eval_config = {
            "eval_samples": 100 if self.args.test else 7600,
            "eval_batch_size": 48,
            "metrics": ["accuracy", "f1_macro", "first_token_accuracy"],
        }

        # Main experiment
        experiments.append(ExperimentConfig(
            name="agnews_telepathy_main",
            phase=ExperimentPhase.TELEPATHY,
            dataset="agnews",
            model_config=base_model_config,
            training_config=base_training_config,
            eval_config=base_eval_config,
            priority=0
        ))

        return experiments

    def create_trec_experiments(self) -> List[ExperimentConfig]:
        """Create TREC question classification experiments."""
        experiments = []

        base_model_config = {
            "source_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "target_model": "Qwen/Qwen2.5-7B-Instruct",
            "latent_len": 32,
            "d_z": 256,
        }

        base_training_config = {
            "dataset": "trec",
            "samples": 500 if self.args.test else 5452,
            "epochs": 2 if self.args.test else 15,
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.5e-4,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "mixed_precision": "bf16",
        }

        base_eval_config = {
            "eval_samples": 50 if self.args.test else 500,
            "eval_batch_size": 64,
            "metrics": ["accuracy", "f1_macro", "first_token_accuracy"],
        }

        experiments.append(ExperimentConfig(
            name="trec_telepathy_main",
            phase=ExperimentPhase.TELEPATHY,
            dataset="trec",
            model_config=base_model_config,
            training_config=base_training_config,
            eval_config=base_eval_config,
            priority=1
        ))

        return experiments

    def create_scaling_experiments(self) -> List[ExperimentConfig]:
        """Create scaling experiments across model sizes."""
        experiments = []

        model_pairs = [
            ("meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
            ("meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"),
            ("meta-llama/Meta-Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
        ]

        for source, target in model_pairs:
            model_size = source.split("-")[-2].replace("B", "")

            model_config = {
                "source_model": source,
                "target_model": target,
                "latent_len": 32,
                "d_z": 256,
            }

            training_config = {
                "dataset": "sst2",
                "samples": 500 if self.args.test else 10000,
                "epochs": 2 if self.args.test else 5,
                "batch_size": 32,
                "gradient_accumulation_steps": 2,
                "learning_rate": 1e-4,
                "warmup_ratio": 0.1,
                "mixed_precision": "bf16",
            }

            eval_config = {
                "eval_samples": 100 if self.args.test else 872,
                "eval_batch_size": 64,
                "metrics": ["accuracy", "latency_ms", "memory_mb"],
            }

            experiments.append(ExperimentConfig(
                name=f"scaling_{model_size}b",
                phase=ExperimentPhase.SCALING,
                dataset="sst2",
                model_config=model_config,
                training_config=training_config,
                eval_config=eval_config,
                priority=3
            ))

        return experiments

    def create_ablation_experiments(self) -> List[ExperimentConfig]:
        """Create ablation experiments for key components."""
        experiments = []

        base_model_config = {
            "source_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "target_model": "Qwen/Qwen2.5-7B-Instruct",
            "latent_len": 32,
            "d_z": 256,
        }

        base_training_config = {
            "dataset": "sst2",
            "samples": 500 if self.args.test else 10000,
            "epochs": 2 if self.args.test else 5,
            "batch_size": 32,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-4,
            "mixed_precision": "bf16",
        }

        base_eval_config = {
            "eval_samples": 100 if self.args.test else 872,
            "eval_batch_size": 64,
            "metrics": ["accuracy", "f1", "first_token_accuracy"],
        }

        # Ablation: No adapter
        experiments.append(ExperimentConfig(
            name="ablation_no_adapter",
            phase=ExperimentPhase.ABLATION,
            dataset="sst2",
            model_config={**base_model_config, "use_adapter": False},
            training_config=base_training_config,
            eval_config=base_eval_config,
            priority=4
        ))

        # Ablation: No auxiliary losses
        experiments.append(ExperimentConfig(
            name="ablation_no_aux_loss",
            phase=ExperimentPhase.ABLATION,
            dataset="sst2",
            model_config=base_model_config,
            training_config={**base_training_config, "aux_losses": []},
            eval_config=base_eval_config,
            priority=4
        ))

        # Ablation: Different encoder types
        for encoder_type in ["lstm", "transformer", "conv"]:
            experiments.append(ExperimentConfig(
                name=f"ablation_encoder_{encoder_type}",
                phase=ExperimentPhase.ABLATION,
                dataset="sst2",
                model_config={**base_model_config, "encoder_type": encoder_type},
                training_config=base_training_config,
                eval_config=base_eval_config,
                priority=4
            ))

        return experiments

    def save_experiment_queue(self):
        """Save current experiment queue to disk."""
        queue_file = self.checkpoint_dir / "experiment_queue.json"

        data = {
            "queue": [exp.to_dict() for exp in self.experiment_queue],
            "completed": list(self.completed_experiments),
            "failed": self.failed_experiments,
            "timestamp": datetime.now().isoformat(),
        }

        with open(queue_file, 'w') as f:
            json.dump(data, f, indent=2)

    def run_experiment(self, experiment: ExperimentConfig) -> bool:
        """
        Run a single experiment with preemption handling.

        Args:
            experiment: Experiment configuration

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Starting experiment: {experiment.name}")
        exp_start_time = time.time()

        # Create experiment-specific directories
        exp_dir = Path(self.args.output_dir) / experiment.name
        exp_checkpoint_dir = exp_dir / "checkpoints"
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Build command based on experiment type
        if experiment.phase == ExperimentPhase.BASELINE:
            cmd = self.build_baseline_command(experiment, exp_dir)
        else:
            cmd = self.build_telepathy_command(experiment, exp_dir)

        # Add checkpoint management
        cmd.extend([
            "--checkpoint_dir", str(exp_checkpoint_dir),
            "--checkpoint_interval", "300",  # 5 minutes
            "--auto_resume",
        ])

        # Run with monitoring and preemption handling
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

            # Log the command
            self.logger.info(f"Command: {' '.join(cmd)}")

            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env
            )

            # Monitor process
            while True:
                # Check for preemption
                if self.preemption_handler.should_exit():
                    self.logger.warning("Preemption signal received!")
                    process.send_signal(signal.SIGUSR1)  # Request checkpoint save
                    time.sleep(10)  # Give time to save
                    process.terminate()
                    self.save_experiment_queue()
                    sys.exit(0)

                # Check process status
                retcode = process.poll()
                if retcode is not None:
                    if retcode == 0:
                        self.logger.info(f"Experiment {experiment.name} completed successfully")
                        self.completed_experiments.add(experiment.name)

                        # Save results
                        self.save_experiment_results(experiment, exp_dir)
                        return True
                    else:
                        self.logger.error(f"Experiment {experiment.name} failed with code {retcode}")
                        self.failed_experiments[experiment.name] = {
                            "retcode": retcode,
                            "timestamp": datetime.now().isoformat()
                        }
                        return False

                # Read output
                line = process.stdout.readline()
                if line:
                    self.logger.info(line.rstrip())

                time.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error running experiment {experiment.name}: {e}")
            self.logger.error(traceback.format_exc())
            self.failed_experiments[experiment.name] = {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return False

        finally:
            # Ensure process is terminated
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

            # Log experiment duration
            exp_duration = time.time() - exp_start_time
            self.logger.info(f"Experiment {experiment.name} took {exp_duration:.1f}s")

    def build_baseline_command(self, experiment: ExperimentConfig, exp_dir: Path) -> List[str]:
        """Build command for baseline experiment."""
        baseline_type = experiment.training_config.get("baseline", "text")

        if baseline_type == "text":
            script = "telepathy/eval_telepathy_sst2.py"
        elif baseline_type == "linear_probe":
            script = "telepathy/train_linear_probe_baseline.py"
        elif baseline_type == "prompt_tuning":
            script = "telepathy/train_prompt_tuning_baseline.py"
        else:
            script = "telepathy/train_full_finetune_baseline.py"

        cmd = [sys.executable, script]

        # Add model configs
        cmd.extend([
            "--source_model", experiment.model_config["source_model"],
            "--target_model", experiment.model_config["target_model"],
        ])

        # Add training configs
        for key, value in experiment.training_config.items():
            if key != "baseline":
                cmd.extend([f"--{key}", str(value)])

        # Add output directory
        cmd.extend(["--output_dir", str(exp_dir)])

        return cmd

    def build_telepathy_command(self, experiment: ExperimentConfig, exp_dir: Path) -> List[str]:
        """Build command for telepathy experiment."""
        dataset = experiment.dataset
        script = f"telepathy/train_telepathy_{dataset}.py"

        cmd = [sys.executable, script]

        # Add model configs
        for key, value in experiment.model_config.items():
            cmd.extend([f"--{key}", str(value)])

        # Add training configs
        for key, value in experiment.training_config.items():
            cmd.extend([f"--{key}", str(value)])

        # Add output directory
        cmd.extend(["--output_dir", str(exp_dir)])

        return cmd

    def save_experiment_results(self, experiment: ExperimentConfig, exp_dir: Path):
        """Save experiment results to central location."""
        results_file = self.results_dir / f"{experiment.name}_results.json"

        # Collect all result files from experiment directory
        results = {
            "experiment": experiment.to_dict(),
            "completed": datetime.now().isoformat(),
            "files": {}
        }

        # Look for eval results
        for eval_file in exp_dir.glob("**/eval_*.json"):
            relative_path = eval_file.relative_to(exp_dir)
            with open(eval_file, 'r') as f:
                results["files"][str(relative_path)] = json.load(f)

        # Save consolidated results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Saved results to {results_file}")

    def run(self):
        """Main orchestration loop."""
        self.logger.info("=" * 60)
        self.logger.info("PREEMPTIBLE EXPERIMENT ORCHESTRATOR")
        self.logger.info("=" * 60)
        self.logger.info(f"Output directory: {self.args.output_dir}")
        self.logger.info(f"Experiment type: {self.args.experiment}")
        self.logger.info(f"Test mode: {self.args.test}")
        self.logger.info(f"Resume: {self.args.resume}")
        self.logger.info(f"Total experiments: {len(self.experiment_queue)}")
        self.logger.info("=" * 60)

        # Process experiment queue
        while self.experiment_queue:
            # Check for preemption
            if self.preemption_handler.should_exit():
                self.logger.warning("Preemption detected, saving state...")
                self.save_experiment_queue()
                self.cleanup()
                sys.exit(0)

            # Get next experiment
            experiment = self.experiment_queue.pop(0)

            # Skip if already completed
            if experiment.name in self.completed_experiments:
                self.logger.info(f"Skipping completed experiment: {experiment.name}")
                continue

            # Check dependencies
            unsatisfied = [dep for dep in experiment.dependencies
                          if dep not in self.completed_experiments]
            if unsatisfied:
                self.logger.info(f"Deferring {experiment.name}, waiting for: {unsatisfied}")
                self.experiment_queue.append(experiment)
                continue

            # Run experiment
            success = self.run_experiment(experiment)

            # Handle failures
            if not success:
                retry_count = self.failed_experiments[experiment.name].get("retries", 0)
                if retry_count < 3:
                    self.logger.info(f"Retrying {experiment.name} (attempt {retry_count + 2}/3)")
                    self.failed_experiments[experiment.name]["retries"] = retry_count + 1
                    self.experiment_queue.append(experiment)
                else:
                    self.logger.error(f"Experiment {experiment.name} failed after 3 attempts")

            # Save queue after each experiment
            self.save_experiment_queue()

        # Generate final report
        self.generate_report()

        # Cleanup
        self.cleanup()

        self.logger.info("=" * 60)
        self.logger.info("ALL EXPERIMENTS COMPLETED")
        self.logger.info(f"Total time: {(time.time() - self.start_time) / 3600:.1f} hours")
        self.logger.info(f"Completed: {len(self.completed_experiments)}")
        self.logger.info(f"Failed: {len(self.failed_experiments)}")
        self.logger.info("=" * 60)

    def generate_report(self):
        """Generate final experiment report."""
        report_file = self.results_dir / "experiment_report.md"

        with open(report_file, 'w') as f:
            f.write("# Experiment Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"- Total experiments: {len(self.completed_experiments) + len(self.failed_experiments)}\n")
            f.write(f"- Completed: {len(self.completed_experiments)}\n")
            f.write(f"- Failed: {len(self.failed_experiments)}\n")
            f.write(f"- Total time: {(time.time() - self.start_time) / 3600:.1f} hours\n\n")

            # Completed experiments
            f.write("## Completed Experiments\n\n")
            for exp_name in sorted(self.completed_experiments):
                results_file = self.results_dir / f"{exp_name}_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as rf:
                        results = json.load(rf)

                    f.write(f"### {exp_name}\n\n")

                    # Extract key metrics
                    for file_path, data in results.get("files", {}).items():
                        if "metrics" in data:
                            f.write(f"**{file_path}**:\n")
                            for metric, value in data["metrics"].items():
                                f.write(f"- {metric}: {value}\n")
                            f.write("\n")

            # Failed experiments
            if self.failed_experiments:
                f.write("## Failed Experiments\n\n")
                for exp_name, failure_info in self.failed_experiments.items():
                    f.write(f"### {exp_name}\n\n")
                    f.write(f"- Error: {failure_info.get('error', failure_info.get('retcode'))}\n")
                    f.write(f"- Timestamp: {failure_info['timestamp']}\n")
                    f.write(f"- Retries: {failure_info.get('retries', 0)}\n\n")

        self.logger.info(f"Report generated: {report_file}")

    def cleanup(self):
        """Clean up resources."""
        if self.gpu_monitor:
            self.gpu_monitor.stop()

        if self.checkpoint_manager:
            self.checkpoint_manager.cleanup()

        # Clean up distributed training
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Orchestrate preemptible experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Experiment selection
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "sst2", "agnews", "trec", "scaling", "ablation"],
        help="Which experiments to run"
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/orchestrated",
        help="Base output directory"
    )

    # Resume and checkpointing
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Specific checkpoint directory to resume from"
    )

    # Test mode
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with small datasets"
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file"
    )

    args = parser.parse_args()

    # Load config file if provided
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Override args with config
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Run orchestrator
    orchestrator = PreemptibleOrchestrator(args)

    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("\n⛔ Interrupted! Saving state...")
        orchestrator.save_experiment_queue()
        orchestrator.cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        orchestrator.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()