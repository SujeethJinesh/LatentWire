#!/usr/bin/env python3
"""
MAIN_EXPERIMENT.py - Main experimental framework for LatentWire/Telepathy research.

This is a self-contained module that provides the core experimental infrastructure
for running compression and interlingua experiments across multiple models.
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
import warnings

# Import checkpoint manager for resume capability
try:
    from checkpoint_manager import CheckpointManager, ExperimentCheckpointer
    HAS_CHECKPOINT_MANAGER = True
except ImportError:
    print("Warning: checkpoint_manager not available. Resume functionality disabled.")
    HAS_CHECKPOINT_MANAGER = False

# Try importing scientific packages with fallbacks
try:
    import numpy as np
except ImportError:
    print("Warning: numpy not available. Some features may be limited.")
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("Warning: PyTorch not available. Neural network features disabled.")
    torch = None
    nn = None
    F = None

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        LlamaForCausalLM,
        Qwen2ForCausalLM,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    print("Warning: transformers not available. Language model features disabled.")
    HAS_TRANSFORMERS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    print("Warning: pandas not available. Data analysis features limited.")
    HAS_PANDAS = False
    pd = None

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    print("Warning: matplotlib not available. Plotting disabled.")
    HAS_MATPLOTLIB = False
    plt = None

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a language model."""
    model_id: str
    model_type: str = "llama"  # llama, qwen, or other
    device: str = "cuda"
    dtype: str = "float16"
    max_length: int = 2048
    use_cache: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    # Model settings
    models: List[ModelConfig] = field(default_factory=list)

    # Compression settings
    latent_dim: int = 256
    latent_len: int = 32
    compression_type: str = "telepathy"  # telepathy, llmlingua, baseline, linear_probe

    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1

    # Evaluation settings
    eval_batch_size: int = 16
    num_eval_samples: int = 1000
    metrics: List[str] = field(default_factory=lambda: ["perplexity", "accuracy", "compression_ratio"])

    # Dataset settings
    dataset_name: str = "squad"
    dataset_split: str = "validation"
    max_samples: Optional[int] = None

    # Output settings
    output_dir: str = "runs/experiment"
    save_checkpoints: bool = True
    log_interval: int = 100

    # Hardware settings
    num_gpus: int = 1
    mixed_precision: bool = True

    def to_dict(self) -> Dict:
        config_dict = asdict(self)
        config_dict['models'] = [m.to_dict() for m in self.models]
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ExperimentConfig':
        if 'models' in config_dict:
            config_dict['models'] = [ModelConfig(**m) for m in config_dict['models']]
        return cls(**config_dict)


# ============================================================================
# Core Experimental Components
# ============================================================================

class ExperimentLogger:
    """Simple logging utility for experiments."""

    def __init__(self, output_dir: str, experiment_name: str = "experiment"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"{experiment_name}_{timestamp}.log"
        self.metrics_file = self.output_dir / f"metrics_{timestamp}.json"

        self.metrics_history = []
        self.start_time = time.time()

    def log(self, message: str, level: str = "INFO"):
        """Log a message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"

        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to file."""
        metrics_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": time.time() - self.start_time,
            "step": step,
            "metrics": metrics
        }

        self.metrics_history.append(metrics_entry)

        # Save incrementally
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        return {
            "total_time": time.time() - self.start_time,
            "num_steps": len(self.metrics_history),
            "log_file": str(self.log_file),
            "metrics_file": str(self.metrics_file),
        }


class DataLoader:
    """Mock data loader for demonstration."""

    def __init__(self, dataset_name: str, split: str, max_samples: Optional[int] = None):
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.samples = []

        # Mock data generation
        self._load_mock_data()

    def _load_mock_data(self):
        """Generate mock data for testing."""
        num_samples = min(self.max_samples or 100, 100)

        for i in range(num_samples):
            self.samples.append({
                "id": f"sample_{i}",
                "input": f"This is input text {i} for the {self.dataset_name} dataset.",
                "target": f"This is target text {i}.",
                "metadata": {"index": i, "dataset": self.dataset_name}
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_batch(self, batch_size: int):
        """Get a batch of samples."""
        for i in range(0, len(self.samples), batch_size):
            yield self.samples[i:i + batch_size]


class CompressionModule:
    """Base class for compression modules."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.initialized = False

    def initialize(self):
        """Initialize the compression module."""
        self.initialized = True

    def compress(self, text: str) -> Dict[str, Any]:
        """Compress input text."""
        if not self.initialized:
            raise RuntimeError("Module not initialized")

        # Mock compression
        compressed_size = len(text) // 4  # Simulate 4x compression
        return {
            "original_size": len(text),
            "compressed_size": compressed_size,
            "compression_ratio": len(text) / compressed_size,
            "method": self.config.compression_type,
        }

    def decompress(self, compressed_data: Dict[str, Any]) -> str:
        """Decompress data back to text."""
        # Mock decompression
        return "Decompressed text placeholder"


class TelepathyCompressor(CompressionModule):
    """Telepathy-specific compression implementation."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.latent_dim = config.latent_dim
        self.latent_len = config.latent_len

    def compress(self, text: str) -> Dict[str, Any]:
        """Compress using Telepathy method."""
        result = super().compress(text)
        result.update({
            "latent_dim": self.latent_dim,
            "latent_len": self.latent_len,
            "latent_bytes": self.latent_dim * self.latent_len * 2,  # fp16
        })
        return result


class LinearProbeBaseline:
    """
    Linear Probe baseline integration for MAIN_EXPERIMENT.py.

    This provides integration with the Linear Probe baseline from telepathy.linear_probe_baseline,
    which uses sklearn's LogisticRegression for a scientifically rigorous baseline comparison.
    This is a CRITICAL baseline demanded by reviewers for comparing against the Bridge method.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.probe = None
        self.initialized = False
        self.LinearProbeClass = None
        self.train_func = None
        self.eval_func = None

        # Try to import the actual LinearProbeBaseline
        try:
            import sys
            # Try to import from local copy first, then fall back to telepathy module
            try:
                from linear_probe_baseline import (
                    LinearProbeBaseline as LPB,
                    train_linear_probe,
                    eval_linear_probe
                )
                print("LinearProbeBaseline imported from local finalization directory")
            except ImportError:
                import os
                # Add parent directory to path to import telepathy module
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)

                from telepathy.linear_probe_baseline import (
                    LinearProbeBaseline as LPB,
                    train_linear_probe,
                    eval_linear_probe
                )
                print("LinearProbeBaseline imported from telepathy module")
            self.LinearProbeClass = LPB
            self.train_func = train_linear_probe
            self.eval_func = eval_linear_probe
            self.has_implementation = True
            print("LinearProbeBaseline successfully imported from telepathy module")
        except ImportError as e:
            print(f"Warning: LinearProbeBaseline not available: {e}")
            self.has_implementation = False

    def initialize(self, hidden_dim: int = 4096, num_classes: int = 2, layer_idx: int = 24):
        """
        Initialize the linear probe baseline.

        Args:
            hidden_dim: Dimension of hidden states (e.g., 4096 for Llama-8B)
            num_classes: Number of output classes for classification
            layer_idx: Which layer to extract from (24 often works well)
        """
        if not self.has_implementation:
            print("Cannot initialize: LinearProbeBaseline implementation not available")
            return False

        self.probe = self.LinearProbeClass(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            layer_idx=layer_idx,
            pooling="mean",  # Mean pooling over sequence
            normalize=True,  # Standardize features
            C=1.0,  # Regularization strength
            max_iter=1000,
            n_jobs=-1,  # Use all CPU cores
            random_state=42,
        )
        self.initialized = True
        print(f"LinearProbe initialized: layer={layer_idx}, dim={hidden_dim}, classes={num_classes}")
        return True

    def train_and_evaluate(self, model, tokenizer, train_data, test_data,
                          dataset_name: str = "test", device: str = "cuda",
                          dataset_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train and evaluate the linear probe baseline.

        Args:
            model: Frozen LLM to extract hidden states from
            tokenizer: Tokenizer for the model
            train_data: Training dataset
            test_data: Test/evaluation dataset
            dataset_name: Name of the dataset for logging
            device: Device to run on
            dataset_config: Configuration for the dataset

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.initialized or not self.has_implementation:
            return {
                "error": "LinearProbeBaseline not initialized or not available",
                "accuracy": 0.0,
                "f1_score": 0.0,
                "status": "failed",
            }

        # Default dataset config if not provided
        if dataset_config is None:
            dataset_config = {
                "text_field": "text",
                "label_field": "label",
                "num_classes": 2,
            }

        try:
            # Train the linear probe with cross-validation
            print(f"\nTraining LinearProbeBaseline on {dataset_name}...")
            train_results = self.train_func(
                probe=self.probe,
                model=model,
                tokenizer=tokenizer,
                train_ds=train_data,
                dataset_name=dataset_name,
                device=device,
                dataset_config=dataset_config,
                batch_size=4,  # Small batch to avoid OOM
                cv_folds=5,  # 5-fold cross-validation
            )

            # Evaluate on test set
            print(f"Evaluating LinearProbeBaseline on {dataset_name}...")
            eval_results = self.eval_func(
                probe=self.probe,
                model=model,
                tokenizer=tokenizer,
                eval_ds=test_data,
                dataset_name=dataset_name,
                device=device,
                dataset_config=dataset_config,
                batch_size=4,
            )

            # Combine results
            results = {
                **eval_results,
                "train_info": train_results,
                "method": "linear_probe_baseline",
                "layer_idx": self.probe.layer_idx,
                "pooling": self.probe.pooling,
                "status": "success",
            }

            print(f"LinearProbe Results: Accuracy={results['accuracy']:.1f}%, F1={results['f1_score']:.1f}%")
            if 'cv_mean' in train_results:
                print(f"  Cross-validation: {train_results['cv_mean']:.1f}% ± {train_results['cv_std']:.1f}%")

            return results

        except Exception as e:
            print(f"Error in LinearProbeBaseline: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "accuracy": 0.0,
                "f1_score": 0.0,
                "status": "failed",
            }

    def save_probe(self, filepath: str):
        """Save the trained probe for reproducibility."""
        if self.initialized and self.probe is not None:
            self.probe.save(filepath)
            print(f"Saved LinearProbe to {filepath}")

    def get_mock_results(self) -> Dict[str, Any]:
        """Return mock results for testing when implementation not available."""
        return {
            "accuracy": np.random.uniform(70, 90) if np else 80,
            "f1_score": np.random.uniform(65, 85) if np else 75,
            "method": "linear_probe_baseline",
            "layer_idx": 24,
            "pooling": "mean",
            "status": "mock",
            "note": "Mock results - actual implementation not available"
        }


class ModelEvaluator:
    """Evaluate model performance."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.metrics = {}

    def evaluate(self, model, data_loader) -> Dict[str, float]:
        """Run evaluation."""
        total_loss = 0
        total_accuracy = 0
        num_samples = 0

        for batch in data_loader.get_batch(self.config.eval_batch_size):
            # Mock evaluation
            batch_size = len(batch)
            total_loss += np.random.random() * batch_size if np else batch_size
            total_accuracy += np.random.random() * batch_size if np else batch_size
            num_samples += batch_size

        return {
            "loss": total_loss / num_samples,
            "accuracy": total_accuracy / num_samples,
            "perplexity": np.exp(total_loss / num_samples) if np else 0,
            "num_samples": num_samples,
        }


# ============================================================================
# Main Experiment Runner
# ============================================================================

class ExperimentRunner:
    """Main experiment orchestrator."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = ExperimentLogger(config.output_dir)
        self.compressor = self._create_compressor()
        self.evaluator = ModelEvaluator(config)
        self.results = {}

        # Initialize checkpoint manager if available
        self.checkpointer = None
        if HAS_CHECKPOINT_MANAGER and config.save_checkpoints:
            self.checkpointer = ExperimentCheckpointer(config.to_dict())
            self.logger.log("Checkpoint manager initialized")

    def _create_compressor(self):
        """Create appropriate compressor or baseline based on config."""
        if self.config.compression_type == "telepathy":
            return TelepathyCompressor(self.config)
        elif self.config.compression_type == "linear_probe":
            return LinearProbeBaseline(self.config)
        else:
            return CompressionModule(self.config)

    def setup(self):
        """Set up experiment environment."""
        self.logger.log("Setting up experiment...")
        self.logger.log(f"Configuration: {json.dumps(self.config.to_dict(), indent=2)}")

        # Initialize components
        self.compressor.initialize()

        # Create output directories
        (Path(self.config.output_dir) / "checkpoints").mkdir(exist_ok=True)
        (Path(self.config.output_dir) / "figures").mkdir(exist_ok=True)

        self.logger.log("Setup complete!")

    def run_compression_experiment(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Run compression experiments."""
        self.logger.log("Running compression experiment...")

        compression_results = []
        for sample in data_loader.samples[:10]:  # Test on subset
            result = self.compressor.compress(sample["input"])
            result["sample_id"] = sample["id"]
            compression_results.append(result)

        # Calculate statistics
        if compression_results:
            avg_ratio = sum(r["compression_ratio"] for r in compression_results) / len(compression_results)
            self.logger.log(f"Average compression ratio: {avg_ratio:.2f}x")

        return {
            "compression_results": compression_results,
            "avg_compression_ratio": avg_ratio if compression_results else 0,
        }

    def run_model_evaluation(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Run model evaluation."""
        self.logger.log("Running model evaluation...")

        # Mock model evaluation
        eval_results = self.evaluator.evaluate(None, data_loader)

        self.logger.log(f"Evaluation results: {eval_results}")
        return eval_results

    def run(self) -> Dict[str, Any]:
        """Run the full experiment."""
        try:
            self.setup()

            # Load data
            self.logger.log("Loading data...")
            train_loader = DataLoader(
                self.config.dataset_name,
                "train",
                self.config.max_samples
            )
            eval_loader = DataLoader(
                self.config.dataset_name,
                self.config.dataset_split,
                self.config.num_eval_samples
            )

            # Run experiments
            self.results["compression"] = self.run_compression_experiment(train_loader)
            self.results["evaluation"] = self.run_model_evaluation(eval_loader)

            # Log final metrics
            self.logger.log_metrics(self.results)

            # Save results
            results_file = Path(self.config.output_dir) / "final_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)

            self.logger.log(f"Results saved to {results_file}")
            self.logger.log("Experiment completed successfully!")

            return self.results

        except Exception as e:
            self.logger.log(f"Experiment failed: {str(e)}", level="ERROR")
            self.logger.log(traceback.format_exc(), level="ERROR")
            raise


# ============================================================================
# Utility Functions
# ============================================================================

def load_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """Load configuration from file or create default."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        return ExperimentConfig.from_dict(config_dict)

    # Default configuration
    return ExperimentConfig(
        models=[
            ModelConfig("meta-llama/Llama-3.2-1B", "llama"),
        ],
        latent_dim=256,
        latent_len=32,
        batch_size=16,
        num_epochs=5,
        output_dir="runs/default_experiment",
    )


def validate_environment() -> Dict[str, bool]:
    """Check available libraries and features."""
    return {
        "numpy": np is not None,
        "torch": torch is not None,
        "transformers": HAS_TRANSFORMERS,
        "pandas": HAS_PANDAS,
        "matplotlib": HAS_MATPLOTLIB,
        "cuda": torch.cuda.is_available() if torch else False,
    }


def run_quick_test():
    """Run a quick test to verify everything works."""
    print("Running quick test...")

    # Check environment
    env_status = validate_environment()
    print("Environment status:")
    for lib, available in env_status.items():
        status = "✓" if available else "✗"
        print(f"  {lib}: {status}")

    # Create minimal config
    config = ExperimentConfig(
        output_dir="runs/test",
        num_epochs=1,
        max_samples=10,
        num_eval_samples=5,
    )

    # Run experiment
    runner = ExperimentRunner(config)
    results = runner.run()

    print("\nTest completed successfully!")
    print(f"Results: {json.dumps(results, indent=2, default=str)}")

    return results


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="LatentWire/Telepathy Main Experiment Runner"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/experiment",
        help="Output directory for results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="squad",
        help="Dataset to use"
    )
    parser.add_argument(
        "--compression-type",
        type=str,
        choices=["telepathy", "llmlingua", "baseline", "linear_probe"],
        default="telepathy",
        help="Compression method to use"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate environment"
    )

    # Add checkpoint/resume arguments
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in output directory"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to specific checkpoint to resume from"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable checkpoint saving"
    )

    args = parser.parse_args()

    if args.validate:
        env_status = validate_environment()
        print("Environment validation:")
        for lib, available in env_status.items():
            status = "Available" if available else "Not available"
            print(f"  {lib}: {status}")
        return 0

    if args.test:
        run_quick_test()
        return 0

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = ExperimentConfig(
            output_dir=args.output_dir,
            dataset_name=args.dataset,
            compression_type=args.compression_type,
        )

    # Run experiment
    runner = ExperimentRunner(config)
    results = runner.run()

    print(f"\nExperiment complete. Results saved to {config.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())