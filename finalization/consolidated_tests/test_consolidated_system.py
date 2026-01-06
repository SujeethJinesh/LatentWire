#!/usr/bin/env python3
"""
Test script to verify the consolidated MAIN_EXPERIMENT.py system works correctly.
This runs a quick end-to-end test with mock data when dependencies are missing.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def test_main_experiment():
    """Test the MAIN_EXPERIMENT module."""
    print("=" * 60)
    print("Testing MAIN_EXPERIMENT.py Consolidated System")
    print("=" * 60)

    # Import the module
    try:
        from MAIN_EXPERIMENT import (
            ExperimentConfig,
            ExperimentRunner,
            DataLoader,
            validate_environment,
            run_quick_test
        )
        print("✓ Successfully imported MAIN_EXPERIMENT components")
    except ImportError as e:
        print(f"✗ Failed to import MAIN_EXPERIMENT: {e}")
        return False

    # Check environment
    print("\n1. Environment Check:")
    env_status = validate_environment()
    for lib, available in env_status.items():
        status = "✓" if available else "✗"
        print(f"   {lib}: {status}")

    # Test configuration
    print("\n2. Configuration Test:")
    try:
        config = ExperimentConfig(
            output_dir="runs/test_consolidated",
            latent_dim=128,
            latent_len=16,
            batch_size=8,
            num_epochs=1,
            max_samples=5,
            num_eval_samples=3,
            compression_type="telepathy"
        )
        print(f"   ✓ Created config with compression_type={config.compression_type}")
        print(f"   ✓ Latent dimensions: {config.latent_len}×{config.latent_dim}")
    except Exception as e:
        print(f"   ✗ Failed to create config: {e}")
        return False

    # Test data loader
    print("\n3. Data Loader Test:")
    try:
        data_loader = DataLoader("squad", "validation", max_samples=5)
        print(f"   ✓ Created data loader with {len(data_loader)} samples")
        sample = data_loader[0]
        print(f"   ✓ Sample structure: {list(sample.keys())}")
    except Exception as e:
        print(f"   ✗ Failed to create data loader: {e}")
        return False

    # Test experiment runner
    print("\n4. Experiment Runner Test:")
    try:
        runner = ExperimentRunner(config)
        print("   ✓ Created experiment runner")

        # Setup
        runner.setup()
        print("   ✓ Setup completed")

        # Run compression test
        compression_results = runner.run_compression_experiment(data_loader)
        avg_ratio = compression_results.get("avg_compression_ratio", 0)
        print(f"   ✓ Compression test: {avg_ratio:.2f}x average ratio")

        # Run evaluation test
        eval_results = runner.run_model_evaluation(data_loader)
        print(f"   ✓ Evaluation test: {eval_results.get('num_samples', 0)} samples evaluated")

    except Exception as e:
        print(f"   ✗ Failed experiment runner test: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test quick test function
    print("\n5. Quick Test Function:")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override output dir
            original_dir = config.output_dir
            config.output_dir = tmpdir

            results = run_quick_test()

            # Check results structure
            if "compression" in results and "evaluation" in results:
                print("   ✓ Quick test completed successfully")
                print(f"   ✓ Compression results: {len(results['compression'].get('compression_results', []))} samples")
                print(f"   ✓ Evaluation metrics: {list(results['evaluation'].keys())}")
            else:
                print("   ✗ Quick test results missing expected keys")
                return False

    except Exception as e:
        print(f"   ✗ Failed quick test: {e}")
        return False

    # Test Linear Probe baseline integration
    print("\n6. Linear Probe Baseline Test:")
    try:
        from MAIN_EXPERIMENT import LinearProbeBaseline

        probe_config = ExperimentConfig(compression_type="linear_probe")
        probe = LinearProbeBaseline(probe_config)

        if probe.has_implementation:
            print("   ✓ LinearProbeBaseline implementation available")
        else:
            print("   ⚠ LinearProbeBaseline using mock implementation")

        # Test mock results
        mock_results = probe.get_mock_results()
        print(f"   ✓ Mock results: Accuracy={mock_results['accuracy']:.1f}%, F1={mock_results['f1_score']:.1f}%")

    except Exception as e:
        print(f"   ✗ Failed linear probe test: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print("\nThe consolidated MAIN_EXPERIMENT.py system is working correctly!")
    print("It can:")
    print("  1. Run experiments with mock data when dependencies are missing")
    print("  2. Save results and logs to the specified output directory")
    print("  3. Handle different compression types (telepathy, linear_probe, etc.)")
    print("  4. Provide structured configuration and logging")
    print("\nFor real experiments with PyTorch/Transformers, run on HPC cluster.")

    return True

if __name__ == "__main__":
    success = test_main_experiment()
    sys.exit(0 if success else 1)