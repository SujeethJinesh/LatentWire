#!/usr/bin/env python3
"""
Unified experiment runner for LatentWire.

This is the main entry point for running all experiments.
It handles:
- Configuration management
- Experiment scheduling
- Resource allocation
- Result aggregation
- Report generation
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str

    # Model config
    llama_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    encoder_type: str = "byte"

    # Training config
    samples: int = 87599
    epochs: int = 24
    batch_size: int = 64
    learning_rate: float = 1e-4

    # Latent config
    latent_len: int = 32
    d_z: int = 256

    # Dataset
    dataset: str = "squad"

    # Evaluation config
    eval_samples: int = 1000
    max_new_tokens: int = 12

    # Hardware config
    gpus: int = 4
    memory_gb: int = 256

    def to_train_args(self) -> List[str]:
        """Convert to training script arguments."""
        args = [
            '--llama_id', self.llama_id,
            '--encoder_type', self.encoder_type,
            '--samples', str(self.samples),
            '--epochs', str(self.epochs),
            '--batch_size', str(self.batch_size),
            '--learning_rate', str(self.learning_rate),
            '--latent_len', str(self.latent_len),
            '--d_z', str(self.d_z),
            '--dataset', self.dataset,
            '--sequential_models',
            '--warm_anchor_text', "Answer: ",
            '--first_token_ce_weight', '0.5',
        ]
        return args

    def to_eval_args(self, checkpoint_path: str) -> List[str]:
        """Convert to evaluation script arguments."""
        args = [
            '--ckpt', checkpoint_path,
            '--samples', str(self.eval_samples),
            '--max_new_tokens', str(self.max_new_tokens),
            '--dataset', self.dataset,
            '--sequential_eval',
            '--fresh_eval',
            '--calibration', 'embed_rms',
            '--latent_anchor_mode', 'text',
            '--latent_anchor_text', "Answer: ",
            '--append_bos_after_prefix', 'yes',
        ]
        return args


class ExperimentRunner:
    """Manages and runs experiments."""

    def __init__(self, base_dir: str = "runs/experiments", dry_run: bool = False):
        """Initialize experiment runner."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run

        # Experiment tracking
        self.experiment_log = self.base_dir / "experiment_log.json"
        self.experiments_run = []

    def get_default_experiments(self) -> List[ExperimentConfig]:
        """Get default set of experiments to run."""
        experiments = [
            # Baseline configuration
            ExperimentConfig(
                name="baseline",
                description="Standard configuration with moderate compression",
                latent_len=32,
                d_z=256
            ),

            # High compression
            ExperimentConfig(
                name="high_compression",
                description="Aggressive compression with smaller latent",
                latent_len=16,
                d_z=128
            ),

            # Large latent
            ExperimentConfig(
                name="large_latent",
                description="Larger latent space for better quality",
                latent_len=64,
                d_z=512
            ),

            # Fast training (for testing)
            ExperimentConfig(
                name="fast_debug",
                description="Quick training for debugging",
                samples=1000,
                epochs=2,
                eval_samples=100
            ),
        ]
        return experiments

    def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment (train + eval)."""
        print(f"\n{'=' * 60}")
        print(f"Running experiment: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'=' * 60}\n")

        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.base_dir / f"{config.name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)

        result = {
            'name': config.name,
            'description': config.description,
            'config': asdict(config),
            'directory': str(exp_dir),
            'start_time': datetime.now().isoformat(),
        }

        try:
            # Phase 1: Training
            print(f"Phase 1: Training", flush=True)
            train_output_dir = exp_dir / "checkpoints"

            if self.dry_run:
                print(f"  [DRY RUN] Would train with config: {config.name}")
                checkpoint_path = train_output_dir / "epoch23"
            else:
                train_result = self._run_training(config, train_output_dir)
                result['train_result'] = train_result
                checkpoint_path = self._find_best_checkpoint(train_output_dir)

            # Phase 2: Evaluation
            print(f"Phase 2: Evaluation")
            eval_output_dir = exp_dir / "evaluation"

            if self.dry_run:
                print(f"  [DRY RUN] Would evaluate checkpoint: {checkpoint_path}", flush=True)
                eval_result = {'status': 'dry_run'}
            else:
                eval_result = self._run_evaluation(config, checkpoint_path, eval_output_dir)
                result['eval_result'] = eval_result

            result['status'] = 'completed'

        except Exception as e:
            print(f"Error in experiment {config.name}: {e}", flush=True)
            result['status'] = 'failed'
            result['error'] = str(e)

        result['end_time'] = datetime.now().isoformat()

        # Save experiment result
        self.experiments_run.append(result)
        self._save_experiment_log()

        return result

    def _run_training(self, config: ExperimentConfig, output_dir: Path) -> Dict:
        """Run training phase."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            'latentwire/train.py',
            '--output_dir', str(output_dir),
        ] + config.to_train_args()

        # Run training
        log_file = output_dir / "train.log"
        print(f"  Running: {' '.join(cmd[:3])}...")
        print(f"  Log: {log_file}")

        with open(log_file, 'w') as f:
            process = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )

        return {
            'return_code': process.returncode,
            'log_file': str(log_file),
        }

    def _run_evaluation(self, config: ExperimentConfig, checkpoint_path: Path, output_dir: Path) -> Dict:
        """Run evaluation phase."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            'latentwire/eval.py',
            '--output_dir', str(output_dir),
        ] + config.to_eval_args(str(checkpoint_path))

        # Run evaluation
        log_file = output_dir / "eval.log"
        print(f"  Running evaluation...")
        print(f"  Log: {log_file}")

        with open(log_file, 'w') as f:
            process = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )

        # Try to load results
        results_file = output_dir / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            results = {}

        return {
            'return_code': process.returncode,
            'log_file': str(log_file),
            'results': results,
        }

    def _find_best_checkpoint(self, checkpoint_dir: Path) -> Path:
        """Find the best checkpoint in directory."""
        # Look for final epoch checkpoint
        checkpoints = sorted(checkpoint_dir.glob("epoch*"))
        if checkpoints:
            return checkpoints[-1]

        # Fall back to any checkpoint
        checkpoints = sorted(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            return checkpoints[-1]

        return checkpoint_dir / "epoch23"  # Default

    def run_all(self, experiments: Optional[List[ExperimentConfig]] = None) -> None:
        """Run all experiments."""
        if experiments is None:
            experiments = self.get_default_experiments()

        print(f"Running {len(experiments)} experiments")

        for i, config in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] {config.name}")
            self.run_experiment(config)

        # Generate final report
        self.generate_report()

    def _save_experiment_log(self) -> None:
        """Save experiment log to disk."""
        with open(self.experiment_log, 'w') as f:
            json.dump(self.experiments_run, f, indent=2)

    def generate_report(self) -> None:
        """Generate final experiment report."""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)

        for exp in self.experiments_run:
            status_emoji = "✅" if exp['status'] == 'completed' else "❌"
            print(f"\n{status_emoji} {exp['name']}")
            print(f"   Status: {exp['status']}")

            if 'eval_result' in exp and 'results' in exp['eval_result']:
                results = exp['eval_result']['results']
                if 'f1_score' in results:
                    print(f"   F1 Score: {results['f1_score']:.3f}")
                if 'compression_ratio' in results:
                    print(f"   Compression: {results['compression_ratio']:.2f}x")

        # Save summary
        summary_path = self.base_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Experiments run: {len(self.experiments_run)}\n")
            completed = sum(1 for e in self.experiments_run if e['status'] == 'completed')
            f.write(f"Completed: {completed}/{len(self.experiments_run)}\n")

        print(f"\nReports saved to {self.base_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run LatentWire Experiments')

    parser.add_argument('--experiments', nargs='+',
                       help='Specific experiments to run (default: all)')
    parser.add_argument('--base_dir', default='runs/experiments',
                       help='Base directory for experiments')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print what would be run without executing')

    # Quick experiment configs
    parser.add_argument('--quick', action='store_true',
                       help='Run quick debug experiments only')
    parser.add_argument('--full', action='store_true',
                       help='Run full experiment suite')

    args = parser.parse_args()

    # Create runner
    runner = ExperimentRunner(base_dir=args.base_dir, dry_run=args.dry_run)

    # Determine which experiments to run
    if args.quick:
        experiments = [
            ExperimentConfig(
                name="quick_test",
                description="Quick test run",
                samples=100,
                epochs=1,
                eval_samples=10
            )
        ]
    elif args.full:
        experiments = runner.get_default_experiments()
    elif args.experiments:
        # Create configs for specified experiments
        experiments = []
        default = runner.get_default_experiments()
        for name in args.experiments:
            matching = [e for e in default if e.name == name]
            if matching:
                experiments.append(matching[0])
            else:
                print(f"Warning: Unknown experiment '{name}'", flush=True)
    else:
        experiments = runner.get_default_experiments()

    # Run experiments
    runner.run_all(experiments)


if __name__ == '__main__':
    main()