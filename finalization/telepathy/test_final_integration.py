#!/usr/bin/env python3
"""
Final integration test for telepathy system.

This script validates that all components work together:
1. Linear probe baseline training
2. Bridge training with 2 models
3. ROUGE evaluation on XSUM
4. Statistical testing
5. Result aggregation

Should complete in <5 minutes on 1 GPU.
"""

import sys
import json
import time
import torch
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from telepathy.train_bridge import train_bridge
from telepathy.train_linear_probe import train_linear_probe
from telepathy.eval_rouge import evaluate_rouge
from scripts.statistical_testing import run_statistical_tests


def print_section(title):
    """Print a section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def test_linear_probe(temp_dir):
    """Test linear probe baseline training."""
    print_section("Testing Linear Probe Baseline")

    output_dir = temp_dir / "linear_probe"
    output_dir.mkdir(exist_ok=True)

    config = {
        "model_name": "meta-llama/Llama-3.1-1B",
        "dataset_name": "xsum",
        "num_samples": 10,
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "layer": 8,
        "output_dir": str(output_dir),
        "seed": 42,
        "wandb_project": None
    }

    print(f"Config: {json.dumps(config, indent=2)}")

    # Train linear probe
    results = train_linear_probe(**config)

    assert results is not None, "Linear probe training failed"
    assert "test_accuracy" in results, "Missing test accuracy"
    assert results["test_accuracy"] >= 0.0, "Invalid accuracy"

    print(f"✓ Linear probe test accuracy: {results['test_accuracy']:.4f}")

    # Check that checkpoint was saved
    checkpoint_file = output_dir / "best_model.pt"
    assert checkpoint_file.exists(), "Checkpoint not saved"

    # Check results file
    results_file = output_dir / "results.json"
    assert results_file.exists(), "Results not saved"

    with open(results_file) as f:
        saved_results = json.load(f)
        assert "config" in saved_results, "Config not saved"
        assert "metrics" in saved_results, "Metrics not saved"

    print("✓ Linear probe training passed all checks")
    return results


def test_bridge_training(temp_dir):
    """Test bridge training with 2 models."""
    print_section("Testing Bridge Training")

    output_dir = temp_dir / "bridge"
    output_dir.mkdir(exist_ok=True)

    config = {
        "encoder_model": "meta-llama/Llama-3.1-1B",
        "decoder_models": ["meta-llama/Llama-3.1-1B", "Qwen/Qwen2.5-0.5B"],
        "dataset_name": "xsum",
        "num_samples": 10,
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "latent_dim": 64,
        "num_latents": 8,
        "output_dir": str(output_dir),
        "seed": 42,
        "wandb_project": None,
        "gradient_checkpointing": False,
        "mixed_precision": False
    }

    print(f"Config: {json.dumps(config, indent=2)}")

    # Train bridge
    results = train_bridge(**config)

    assert results is not None, "Bridge training failed"
    assert "final_loss" in results, "Missing final loss"
    assert results["final_loss"] > 0, "Invalid loss"

    print(f"✓ Bridge final loss: {results['final_loss']:.4f}")

    # Check that checkpoint was saved
    checkpoint_file = output_dir / "best_model.pt"
    assert checkpoint_file.exists(), "Checkpoint not saved"

    # Load and verify checkpoint
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    assert "encoder_state_dict" in checkpoint, "Missing encoder state"
    assert "bridge_state_dict" in checkpoint, "Missing bridge state"
    assert "config" in checkpoint, "Missing config in checkpoint"

    # Check results file
    results_file = output_dir / "results.json"
    assert results_file.exists(), "Results not saved"

    with open(results_file) as f:
        saved_results = json.load(f)
        assert "config" in saved_results, "Config not saved"
        assert "metrics" in saved_results, "Metrics not saved"
        assert "training_history" in saved_results["metrics"], "History not saved"

    print("✓ Bridge training passed all checks")
    return results, str(checkpoint_file)


def test_rouge_evaluation(temp_dir, checkpoint_path):
    """Test ROUGE evaluation on XSUM."""
    print_section("Testing ROUGE Evaluation")

    output_dir = temp_dir / "rouge"
    output_dir.mkdir(exist_ok=True)

    config = {
        "checkpoint_path": checkpoint_path,
        "dataset_name": "xsum",
        "num_samples": 5,
        "max_new_tokens": 20,
        "output_dir": str(output_dir),
        "batch_size": 1,
        "seed": 42
    }

    print(f"Config: {json.dumps(config, indent=2)}")

    # Run evaluation
    results = evaluate_rouge(**config)

    assert results is not None, "ROUGE evaluation failed"
    assert "rouge1" in results, "Missing ROUGE-1 scores"
    assert "rouge2" in results, "Missing ROUGE-2 scores"
    assert "rougeL" in results, "Missing ROUGE-L scores"

    print(f"✓ ROUGE scores:")
    print(f"  ROUGE-1: {results['rouge1']:.4f}")
    print(f"  ROUGE-2: {results['rouge2']:.4f}")
    print(f"  ROUGE-L: {results['rougeL']:.4f}")

    # Check outputs file
    outputs_file = output_dir / "outputs.json"
    assert outputs_file.exists(), "Outputs not saved"

    with open(outputs_file) as f:
        outputs = json.load(f)
        assert "config" in outputs, "Config not saved"
        assert "results" in outputs, "Results not saved"
        assert "examples" in outputs, "Examples not saved"
        assert len(outputs["examples"]) > 0, "No examples generated"

    print("✓ ROUGE evaluation passed all checks")
    return results


def test_statistical_testing(temp_dir):
    """Test statistical testing infrastructure."""
    print_section("Testing Statistical Analysis")

    output_dir = temp_dir / "stats"
    output_dir.mkdir(exist_ok=True)

    # Create dummy results for testing
    dummy_results = {
        "baseline": {
            "rouge1": [0.30, 0.32, 0.31, 0.33, 0.29],
            "rouge2": [0.10, 0.12, 0.11, 0.13, 0.09],
            "rougeL": [0.25, 0.27, 0.26, 0.28, 0.24]
        },
        "telepathy": {
            "rouge1": [0.35, 0.37, 0.36, 0.38, 0.34],
            "rouge2": [0.15, 0.17, 0.16, 0.18, 0.14],
            "rougeL": [0.30, 0.32, 0.31, 0.33, 0.29]
        }
    }

    # Save dummy results
    results_file = output_dir / "dummy_results.json"
    with open(results_file, "w") as f:
        json.dump(dummy_results, f, indent=2)

    # Run statistical tests
    stats_results = run_statistical_tests(
        results_file=str(results_file),
        output_dir=str(output_dir)
    )

    assert stats_results is not None, "Statistical testing failed"
    assert "summary" in stats_results, "Missing summary"
    assert "tests" in stats_results, "Missing test results"

    print("✓ Statistical test results:")
    for metric in ["rouge1", "rouge2", "rougeL"]:
        if metric in stats_results["tests"]:
            test = stats_results["tests"][metric]
            print(f"  {metric}: p-value={test.get('p_value', 'N/A'):.4f}, "
                  f"significant={test.get('significant', 'N/A')}")

    # Check output files
    stats_file = output_dir / "statistical_analysis.json"
    assert stats_file.exists(), "Stats file not saved"

    report_file = output_dir / "statistical_report.md"
    assert report_file.exists(), "Report not saved"

    print("✓ Statistical testing passed all checks")
    return stats_results


def test_result_aggregation(temp_dir, all_results):
    """Test result aggregation across all components."""
    print_section("Testing Result Aggregation")

    output_dir = temp_dir / "final"
    output_dir.mkdir(exist_ok=True)

    # Aggregate all results
    aggregated = {
        "timestamp": datetime.now().isoformat(),
        "linear_probe": all_results["linear_probe"],
        "bridge": all_results["bridge"],
        "rouge": all_results["rouge"],
        "statistical": all_results["statistical"],
        "summary": {
            "linear_probe_accuracy": all_results["linear_probe"].get("test_accuracy", 0),
            "bridge_loss": all_results["bridge"].get("final_loss", float("inf")),
            "rouge1": all_results["rouge"].get("rouge1", 0),
            "rouge2": all_results["rouge"].get("rouge2", 0),
            "rougeL": all_results["rouge"].get("rougeL", 0),
            "tests_passed": True
        }
    }

    # Save aggregated results
    results_file = output_dir / "integration_test_results.json"
    with open(results_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print("✓ Results aggregated successfully")
    print(f"\nSummary:")
    print(f"  Linear Probe Accuracy: {aggregated['summary']['linear_probe_accuracy']:.4f}")
    print(f"  Bridge Loss: {aggregated['summary']['bridge_loss']:.4f}")
    print(f"  ROUGE-1: {aggregated['summary']['rouge1']:.4f}")
    print(f"  ROUGE-2: {aggregated['summary']['rouge2']:.4f}")
    print(f"  ROUGE-L: {aggregated['summary']['rougeL']:.4f}")

    return aggregated


def main():
    """Run full integration test."""
    print("\n" + "="*60)
    print(" TELEPATHY INTEGRATION TEST")
    print("="*60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        print(f"\nUsing temp directory: {temp_dir}")

        all_results = {}
        start_time = time.time()

        try:
            # Test 1: Linear Probe Baseline
            linear_results = test_linear_probe(temp_dir)
            all_results["linear_probe"] = linear_results

            # Test 2: Bridge Training
            bridge_results, checkpoint_path = test_bridge_training(temp_dir)
            all_results["bridge"] = bridge_results

            # Test 3: ROUGE Evaluation
            rouge_results = test_rouge_evaluation(temp_dir, checkpoint_path)
            all_results["rouge"] = rouge_results

            # Test 4: Statistical Testing
            stats_results = test_statistical_testing(temp_dir)
            all_results["statistical"] = stats_results

            # Test 5: Result Aggregation
            final_results = test_result_aggregation(temp_dir, all_results)

            # Calculate total time
            total_time = time.time() - start_time

            print("\n" + "="*60)
            print(" ALL TESTS PASSED!")
            print("="*60)
            print(f"Total time: {total_time:.1f} seconds")
            print(f"End time: {datetime.now().isoformat()}")

            # Save final results to project directory
            output_dir = Path(__file__).parent / "integration_test_results"
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_file = output_dir / f"test_run_{timestamp}.json"

            with open(final_file, "w") as f:
                json.dump(final_results, f, indent=2)

            print(f"\nResults saved to: {final_file}")

            return 0

        except Exception as e:
            print("\n" + "="*60)
            print(" TEST FAILED!")
            print("="*60)
            print(f"Error: {e}")

            import traceback
            traceback.print_exc()

            return 1


if __name__ == "__main__":
    sys.exit(main())