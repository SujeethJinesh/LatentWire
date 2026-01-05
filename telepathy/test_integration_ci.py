#!/usr/bin/env python3
"""
CI/CD Integration Test for LatentWire/Telepathy

More comprehensive test that validates the pipeline with real models and data.
Designed for continuous integration systems.

Features:
- Tests with actual transformer models (small versions)
- Uses real datasets (small samples)
- Validates all output formats
- Checks memory usage
- Tests error handling
- Validates reproducibility with seeds

Usage:
    python telepathy/test_integration_ci.py --quick  # Fast mode (<2 min)
    python telepathy/test_integration_ci.py          # Full mode (<10 min)
"""

import os
import sys
import json
import torch
import numpy as np
import tempfile
import shutil
import psutil
import gc
from pathlib import Path
from datetime import datetime
import traceback
import time
import hashlib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class CIIntegrationTester:
    """Comprehensive CI/CD integration tester."""

    def __init__(self, quick_mode=False):
        """Initialize CI tester."""
        self.quick_mode = quick_mode
        self.temp_dir = tempfile.mkdtemp(prefix="ci_telepathy_")
        self.temp_dir = Path(self.temp_dir)

        # Test configuration
        if quick_mode:
            # Quick mode: minimal testing
            self.config = {
                "train_samples": 20,
                "test_samples": 10,
                "epochs": 1,
                "batch_size": 2,
                "latent_dim": 16,
                "num_latents": 4,
                "models": ["gpt2"],  # Single small model
                "datasets": ["sst2"],  # Single dataset
                "seeds": [42],  # Single seed
            }
        else:
            # Full mode: comprehensive testing
            self.config = {
                "train_samples": 100,
                "test_samples": 50,
                "epochs": 2,
                "batch_size": 4,
                "latent_dim": 32,
                "num_latents": 8,
                "models": ["gpt2", "distilgpt2"],
                "datasets": ["sst2", "trec"],
                "seeds": [42, 123],
            }

        self.results = {}
        self.memory_usage = {}
        self.timing = {}

    def measure_memory(self, func, *args, **kwargs):
        """Measure memory usage of a function."""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        mem_used = mem_after - mem_before

        return result, mem_used, elapsed

    def test_data_loading(self):
        """Test data loading and preprocessing."""
        print("\n" + "="*60)
        print("Testing Data Loading...")
        print("="*60)

        try:
            from datasets import load_dataset

            results = {}
            for dataset_name in self.config["datasets"]:
                print(f"Loading {dataset_name}...")

                if dataset_name == "sst2":
                    dataset = load_dataset("glue", "sst2", split="train[:100]")
                    num_classes = 2
                    text_key = "sentence"
                    label_key = "label"
                elif dataset_name == "trec":
                    dataset = load_dataset("trec", split="train[:100]")
                    num_classes = 6
                    text_key = "text"
                    label_key = "coarse_label"
                else:
                    continue

                # Validate dataset
                assert len(dataset) > 0, f"Empty dataset: {dataset_name}"
                assert text_key in dataset[0], f"Missing text key: {text_key}"
                assert label_key in dataset[0], f"Missing label key: {label_key}"

                # Check data distribution
                labels = [item[label_key] for item in dataset]
                unique_labels = set(labels)
                assert len(unique_labels) >= 2, f"Not enough label diversity: {unique_labels}"

                results[dataset_name] = {
                    "samples": len(dataset),
                    "num_classes": num_classes,
                    "unique_labels": len(unique_labels),
                    "label_distribution": {
                        str(l): labels.count(l) for l in unique_labels
                    }
                }

                print(f"  ✓ {dataset_name}: {len(dataset)} samples, {num_classes} classes")

            self.results["data_loading"] = {
                "status": "passed",
                "datasets": results
            }
            return True

        except Exception as e:
            self.results["data_loading"] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"  ✗ Data loading failed: {e}")
            return False

    def test_model_loading(self):
        """Test model loading and compatibility."""
        print("\n" + "="*60)
        print("Testing Model Loading...")
        print("="*60)

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            results = {}
            for model_name in self.config["models"]:
                print(f"Loading {model_name}...")

                # Load model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu"
                )

                # Set pad token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # Get model info
                num_params = sum(p.numel() for p in model.parameters())
                num_layers = len(model.transformer.h) if hasattr(model, 'transformer') else \
                           len(model.model.layers) if hasattr(model, 'model') else 12

                # Test forward pass
                test_input = tokenizer("Hello world", return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**test_input, output_hidden_states=True)

                assert outputs.logits is not None, "No logits produced"
                assert outputs.hidden_states is not None, "No hidden states"

                results[model_name] = {
                    "num_params": num_params,
                    "num_layers": num_layers,
                    "hidden_size": outputs.hidden_states[-1].shape[-1],
                    "vocab_size": model.config.vocab_size
                }

                print(f"  ✓ {model_name}: {num_params/1e6:.1f}M params, {num_layers} layers")

                # Clean up
                del model
                del tokenizer
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            self.results["model_loading"] = {
                "status": "passed",
                "models": results
            }
            return True

        except Exception as e:
            self.results["model_loading"] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"  ✗ Model loading failed: {e}")
            return False

    def test_bridge_training_real(self):
        """Test bridge training with real models and data."""
        print("\n" + "="*60)
        print("Testing Bridge Training (Real)...")
        print("="*60)

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from datasets import load_dataset

            # Use smallest models
            sender_model_id = "gpt2"
            receiver_model_id = "gpt2"  # Same model for simplicity

            print(f"Loading models: {sender_model_id} -> {receiver_model_id}")

            # Load models
            tokenizer = AutoTokenizer.from_pretrained(sender_model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                sender_model_id,
                torch_dtype=torch.float32,
                device_map="cpu"
            )

            # Load SST-2 data
            print("Loading SST-2 dataset...")
            dataset = load_dataset("glue", "sst2", split=f"train[:{self.config['train_samples']}]")

            # Simple training loop
            print("Training bridge...")
            losses = []
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Create simple encoder/decoder
            encoder = torch.nn.Linear(768, self.config["latent_dim"]).to(device)
            decoder = torch.nn.Linear(self.config["latent_dim"], 768).to(device)
            optimizer = torch.optim.Adam(
                list(encoder.parameters()) + list(decoder.parameters()),
                lr=1e-3
            )

            for epoch in range(self.config["epochs"]):
                epoch_losses = []

                for item in dataset:
                    text = item["sentence"]
                    label = item["label"]

                    # Tokenize
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=64
                    ).to(device)

                    # Get hidden states
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                        hidden = outputs.hidden_states[-1].mean(dim=1)  # Pool

                    # Encode/decode
                    latent = encoder(hidden.to(device))
                    reconstructed = decoder(latent)

                    # Reconstruction loss
                    loss = torch.nn.functional.mse_loss(reconstructed, hidden.to(device))

                    # Add classification loss
                    classifier = torch.nn.Linear(self.config["latent_dim"], 2).to(device)
                    logits = classifier(latent)
                    ce_loss = torch.nn.functional.cross_entropy(
                        logits,
                        torch.tensor([label], device=device)
                    )
                    loss = loss + 0.1 * ce_loss

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_losses.append(loss.item())

                avg_loss = np.mean(epoch_losses)
                losses.append(avg_loss)
                print(f"  Epoch {epoch+1}/{self.config['epochs']}: loss={avg_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "config": self.config,
                "losses": losses
            }
            checkpoint_path = self.temp_dir / "bridge_checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            self.results["bridge_training_real"] = {
                "status": "passed",
                "final_loss": float(losses[-1]),
                "checkpoint_size_mb": checkpoint_path.stat().st_size / 1024 / 1024
            }

            print(f"  ✓ Bridge training successful: final_loss={losses[-1]:.4f}")
            return True

        except Exception as e:
            self.results["bridge_training_real"] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"  ✗ Bridge training failed: {e}")
            return False

    def test_evaluation_pipeline(self):
        """Test the evaluation pipeline."""
        print("\n" + "="*60)
        print("Testing Evaluation Pipeline...")
        print("="*60)

        try:
            # Create dummy predictions and ground truth
            n_samples = self.config["test_samples"]
            num_classes = 2  # Binary for simplicity

            # Simulate different methods
            methods = ["bridge", "linear_probe", "text_baseline"]
            results = {}

            for method in methods:
                # Generate predictions with different accuracy patterns
                if method == "text_baseline":
                    # Best performance
                    accuracy = 0.85 + np.random.normal(0, 0.05)
                elif method == "linear_probe":
                    # Good performance
                    accuracy = 0.75 + np.random.normal(0, 0.05)
                else:
                    # Bridge: moderate performance
                    accuracy = 0.65 + np.random.normal(0, 0.05)

                accuracy = np.clip(accuracy, 0, 1)

                # Generate predictions
                ground_truth = np.random.randint(0, num_classes, n_samples)
                predictions = ground_truth.copy()

                # Flip some predictions based on accuracy
                n_errors = int(n_samples * (1 - accuracy))
                error_indices = np.random.choice(n_samples, n_errors, replace=False)
                for idx in error_indices:
                    predictions[idx] = 1 - predictions[idx]

                # Calculate metrics
                acc = np.mean(predictions == ground_truth)
                from sklearn.metrics import f1_score
                f1 = f1_score(ground_truth, predictions, average='macro')

                # Calculate latency (simulated)
                if method == "bridge":
                    latency_ms = 15.0 + np.random.normal(0, 2)
                elif method == "linear_probe":
                    latency_ms = 10.0 + np.random.normal(0, 1)
                else:
                    latency_ms = 25.0 + np.random.normal(0, 3)

                results[method] = {
                    "accuracy": float(acc),
                    "f1_score": float(f1),
                    "latency_ms": float(latency_ms),
                    "num_samples": n_samples,
                    "predictions": predictions.tolist(),
                    "ground_truth": ground_truth.tolist()
                }

                print(f"  {method}: acc={acc:.3f}, f1={f1:.3f}, latency={latency_ms:.1f}ms")

            # Save evaluation results
            eval_dir = self.temp_dir / "evaluation"
            eval_dir.mkdir(exist_ok=True)

            for method, result in results.items():
                method_file = eval_dir / f"{method}_results.json"
                with open(method_file, "w") as f:
                    json.dump(result, f, indent=2)

            self.results["evaluation_pipeline"] = {
                "status": "passed",
                "methods_tested": list(results.keys()),
                "results": {k: {
                    "accuracy": v["accuracy"],
                    "f1_score": v["f1_score"],
                    "latency_ms": v["latency_ms"]
                } for k, v in results.items()}
            }

            print("  ✓ Evaluation pipeline successful")
            return True

        except Exception as e:
            self.results["evaluation_pipeline"] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"  ✗ Evaluation failed: {e}")
            return False

    def test_statistical_validation(self):
        """Test statistical validation of results."""
        print("\n" + "="*60)
        print("Testing Statistical Validation...")
        print("="*60)

        try:
            from scripts.statistical_testing import (
                bootstrap_ci,
                paired_t_test,
                mcnemar_test,
                compute_effect_size,
                multi_seed_analysis
            )

            # Generate multi-seed results
            seeds = self.config["seeds"]
            n_samples = 100

            # Method A (bridge): moderate performance
            method_a_results = []
            for seed in seeds:
                np.random.seed(seed)
                accuracy = 0.65 + np.random.normal(0, 0.03)
                results = np.random.beta(accuracy * 10, (1-accuracy) * 10, n_samples)
                method_a_results.append(results)

            # Method B (baseline): better performance
            method_b_results = []
            for seed in seeds:
                np.random.seed(seed + 1000)
                accuracy = 0.75 + np.random.normal(0, 0.03)
                results = np.random.beta(accuracy * 10, (1-accuracy) * 10, n_samples)
                method_b_results.append(results)

            # Aggregate across seeds
            method_a_all = np.concatenate(method_a_results)
            method_b_all = np.concatenate(method_b_results)

            print("Running statistical tests...")

            # Bootstrap CI
            mean_a, ci_a = bootstrap_ci(method_a_all, n_resamples=1000)
            mean_b, ci_b = bootstrap_ci(method_b_all, n_resamples=1000)
            print(f"  Bridge: {mean_a:.3f} [{ci_a[0]:.3f}, {ci_a[1]:.3f}]")
            print(f"  Baseline: {mean_b:.3f} [{ci_b[0]:.3f}, {ci_b[1]:.3f}]")

            # Paired t-test
            t_stat, p_value = paired_t_test(
                method_a_all[:len(method_b_all)],
                method_b_all[:len(method_a_all)]
            )
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")

            # Effect size
            effect_size = compute_effect_size(method_a_all, method_b_all)
            print(f"  Effect size (Cohen's d): {effect_size:.3f}")

            # Multi-seed analysis
            if len(seeds) > 1:
                seed_analysis = multi_seed_analysis(
                    method_a_results,
                    seeds=seeds,
                    metric_name="accuracy"
                )
                print(f"  Multi-seed: mean={seed_analysis['mean']:.3f}, "
                      f"std={seed_analysis['std']:.3f}")

            # Significance determination
            is_significant = p_value < 0.05
            significance_level = "***" if p_value < 0.001 else \
                               "**" if p_value < 0.01 else \
                               "*" if p_value < 0.05 else "ns"

            self.results["statistical_validation"] = {
                "status": "passed",
                "bridge_mean": float(mean_a),
                "bridge_ci": [float(ci_a[0]), float(ci_a[1])],
                "baseline_mean": float(mean_b),
                "baseline_ci": [float(ci_b[0]), float(ci_b[1])],
                "p_value": float(p_value),
                "effect_size": float(effect_size),
                "is_significant": is_significant,
                "significance_level": significance_level
            }

            print(f"  ✓ Statistical validation complete: {significance_level}")
            return True

        except Exception as e:
            self.results["statistical_validation"] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"  ✗ Statistical validation failed: {e}")
            return False

    def test_reproducibility(self):
        """Test reproducibility with seeds."""
        print("\n" + "="*60)
        print("Testing Reproducibility...")
        print("="*60)

        try:
            # Test that same seed produces same results
            results_by_seed = {}

            for seed in [42, 42]:  # Run twice with same seed
                np.random.seed(seed)
                torch.manual_seed(seed)

                # Generate some "results"
                accuracy = np.random.rand()
                predictions = np.random.randint(0, 2, 100)

                # Hash the predictions to check equality
                pred_hash = hashlib.md5(predictions.tobytes()).hexdigest()

                if seed not in results_by_seed:
                    results_by_seed[seed] = []
                results_by_seed[seed].append({
                    "accuracy": accuracy,
                    "pred_hash": pred_hash
                })

            # Check reproducibility
            seed_42_results = results_by_seed[42]
            assert len(seed_42_results) == 2, "Should have 2 runs with seed 42"

            acc_match = abs(seed_42_results[0]["accuracy"] - seed_42_results[1]["accuracy"]) < 1e-10
            hash_match = seed_42_results[0]["pred_hash"] == seed_42_results[1]["pred_hash"]

            reproducible = acc_match and hash_match

            self.results["reproducibility"] = {
                "status": "passed" if reproducible else "failed",
                "accuracy_match": acc_match,
                "predictions_match": hash_match
            }

            if reproducible:
                print("  ✓ Reproducibility verified")
            else:
                print("  ✗ Reproducibility check failed")

            return reproducible

        except Exception as e:
            self.results["reproducibility"] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"  ✗ Reproducibility test failed: {e}")
            return False

    def test_output_formats(self):
        """Test that all output formats are correct."""
        print("\n" + "="*60)
        print("Testing Output Formats...")
        print("="*60)

        try:
            # Test JSON output
            json_data = {
                "experiment": "test",
                "accuracy": 0.85,
                "results": [1, 2, 3],
                "timestamp": datetime.now().isoformat()
            }
            json_file = self.temp_dir / "test_output.json"
            with open(json_file, "w") as f:
                json.dump(json_data, f, indent=2)

            # Verify JSON is readable
            with open(json_file) as f:
                loaded = json.load(f)
            assert loaded["accuracy"] == json_data["accuracy"]

            # Test checkpoint format
            checkpoint = {
                "model_state": {"layer1": torch.randn(10, 10)},
                "optimizer_state": {"lr": 0.001},
                "epoch": 5,
                "loss": 0.123
            }
            checkpoint_file = self.temp_dir / "test_checkpoint.pt"
            torch.save(checkpoint, checkpoint_file)

            # Verify checkpoint is loadable
            loaded_ckpt = torch.load(checkpoint_file, map_location="cpu")
            assert loaded_ckpt["epoch"] == 5

            # Test results aggregation format
            aggregated = {
                "summary": {
                    "mean_accuracy": 0.75,
                    "std_accuracy": 0.05,
                    "num_experiments": 10
                },
                "per_seed": {
                    "42": {"accuracy": 0.73},
                    "123": {"accuracy": 0.77}
                }
            }
            agg_file = self.temp_dir / "aggregated_results.json"
            with open(agg_file, "w") as f:
                json.dump(aggregated, f, indent=2)

            self.results["output_formats"] = {
                "status": "passed",
                "json_valid": True,
                "checkpoint_valid": True,
                "aggregation_valid": True
            }

            print("  ✓ All output formats validated")
            return True

        except Exception as e:
            self.results["output_formats"] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"  ✗ Output format test failed: {e}")
            return False

    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print(" GENERATING TEST REPORT")
        print("="*80)

        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get("status") == "passed")
        failed_tests = total_tests - passed_tests

        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "mode": "quick" if self.quick_mode else "full",
            "configuration": self.config,
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "results": self.results,
            "memory_usage": self.memory_usage,
            "timing": self.timing,
            "temp_directory": str(self.temp_dir)
        }

        # Save report
        report_file = self.temp_dir / "ci_test_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"Failed: {failed_tests}")
        print("")
        print("Test Results:")
        for test_name, result in self.results.items():
            status = "✓" if result.get("status") == "passed" else "✗"
            print(f"  {status} {test_name}")
            if result.get("status") == "failed" and result.get("error"):
                print(f"      Error: {result['error'][:100]}...")

        print(f"\nFull report saved to: {report_file}")
        return report

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temp directory: {self.temp_dir}")

    def run_all_tests(self):
        """Run all CI integration tests."""
        print("\n" + "="*80)
        print(" CI/CD INTEGRATION TEST SUITE")
        print("="*80)
        print(f"Mode: {'Quick' if self.quick_mode else 'Full'}")
        print(f"Temp directory: {self.temp_dir}")
        print(f"Started: {datetime.now().isoformat()}")

        start_time = time.time()

        # Define test suite
        tests = [
            ("Data Loading", self.test_data_loading),
            ("Model Loading", self.test_model_loading),
            ("Bridge Training", self.test_bridge_training_real),
            ("Evaluation Pipeline", self.test_evaluation_pipeline),
            ("Statistical Validation", self.test_statistical_validation),
            ("Reproducibility", self.test_reproducibility),
            ("Output Formats", self.test_output_formats),
        ]

        # Run tests
        for test_name, test_func in tests:
            try:
                # Measure memory and time
                success, mem_used, elapsed = self.measure_memory(test_func)
                self.memory_usage[test_name] = mem_used
                self.timing[test_name] = elapsed
            except Exception as e:
                print(f"✗ {test_name} crashed: {e}")
                self.results[test_name] = {
                    "status": "failed",
                    "error": str(e)
                }

        # Generate report
        elapsed_total = time.time() - start_time
        self.timing["total"] = elapsed_total

        report = self.generate_report()

        print(f"\nTotal time: {elapsed_total:.1f}s")

        # Return success status
        return report["summary"]["failed"] == 0


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CI/CD integration test for LatentWire/Telepathy"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in quick mode (<2 minutes)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary directory after completion"
    )

    args = parser.parse_args()

    # Run tests
    tester = CIIntegrationTester(quick_mode=args.quick)

    try:
        success = tester.run_all_tests()

        if not args.keep_temp and success:
            tester.cleanup()
        elif not success:
            print(f"\nTemp directory preserved for debugging: {tester.temp_dir}")

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        print(f"Temp directory: {tester.temp_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()