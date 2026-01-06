#!/usr/bin/env python
"""
End-to-end test for the LatentWire pipeline.

This script validates the entire training and evaluation pipeline:
1. Train for 1 epoch with 10 samples
2. Save checkpoint
3. Resume from checkpoint
4. Evaluate the model
5. Validate results are reasonable

Usage:
    python test_end_to_end.py

This should complete in ~1-2 minutes on a single GPU.
"""

import os
import sys
import json
import tempfile
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

def run_command(cmd: str, description: str) -> tuple[int, str, str]:
    """Run a shell command and capture output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    start_time = time.time()
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        env={**os.environ, 'PYTHONPATH': '.'}
    )
    elapsed = time.time() - start_time

    print(f"Completed in {elapsed:.2f}s with return code {result.returncode}")

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")

    return result.returncode, result.stdout, result.stderr


def test_training_phase(output_dir: Path) -> bool:
    """Test the training phase."""
    print("\n" + "="*80)
    print("PHASE 1: TRAINING")
    print("="*80)

    # Create minimal training command
    train_cmd = f"""python latentwire/train.py \
        --llama_id meta-llama/Meta-Llama-3.1-8B-Instruct \
        --qwen_id Qwen/Qwen2.5-7B-Instruct \
        --samples 10 \
        --epochs 1 \
        --batch_size 2 \
        --latent_len 8 \
        --d_z 64 \
        --encoder_type byte \
        --dataset squad \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --output_dir {output_dir}/train \
        --save_every_epoch 1 \
        --test_mode"""

    returncode, stdout, stderr = run_command(train_cmd, "Training with minimal samples")

    if returncode != 0:
        print("❌ Training failed!")
        return False

    # Check if checkpoint was created
    checkpoint_path = output_dir / "train" / "epoch0"
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return False

    # Validate checkpoint contents
    expected_files = [
        "encoder.pt",
        "llama_adapter.pt",
        "qwen_adapter.pt",
        "config.json",
        "training_state.json"
    ]

    missing_files = []
    for file in expected_files:
        if not (checkpoint_path / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing checkpoint files: {missing_files}")
        return False

    print("✅ Training phase completed successfully")
    print(f"✅ Checkpoint saved at: {checkpoint_path}")
    return True


def test_resume_training(output_dir: Path) -> bool:
    """Test resuming from checkpoint."""
    print("\n" + "="*80)
    print("PHASE 2: RESUME TRAINING")
    print("="*80)

    checkpoint_path = output_dir / "train" / "epoch0"
    resume_output_dir = output_dir / "resume"

    # Resume training for one more epoch
    resume_cmd = f"""python latentwire/train.py \
        --llama_id meta-llama/Meta-Llama-3.1-8B-Instruct \
        --qwen_id Qwen/Qwen2.5-7B-Instruct \
        --samples 10 \
        --epochs 2 \
        --batch_size 2 \
        --latent_len 8 \
        --d_z 64 \
        --encoder_type byte \
        --dataset squad \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --output_dir {resume_output_dir} \
        --resume_from {checkpoint_path} \
        --save_every_epoch 1 \
        --test_mode"""

    returncode, stdout, stderr = run_command(resume_cmd, "Resuming from checkpoint")

    if returncode != 0:
        print("❌ Resume training failed!")
        return False

    # Check if new checkpoint was created
    new_checkpoint_path = resume_output_dir / "epoch1"
    if not new_checkpoint_path.exists():
        print(f"❌ New checkpoint not found at {new_checkpoint_path}")
        return False

    print("✅ Resume training completed successfully")
    print(f"✅ New checkpoint saved at: {new_checkpoint_path}")
    return True


def test_evaluation(output_dir: Path) -> bool:
    """Test evaluation on the trained model."""
    print("\n" + "="*80)
    print("PHASE 3: EVALUATION")
    print("="*80)

    checkpoint_path = output_dir / "train" / "epoch0"
    eval_output_dir = output_dir / "eval"

    # Run evaluation
    eval_cmd = f"""python latentwire/eval.py \
        --ckpt {checkpoint_path} \
        --samples 5 \
        --max_new_tokens 12 \
        --dataset squad \
        --sequential_eval \
        --fresh_eval \
        --calibration embed_rms \
        --latent_anchor_mode text \
        --latent_anchor_text "Answer: " \
        --append_bos_after_prefix yes \
        --output_dir {eval_output_dir}"""

    returncode, stdout, stderr = run_command(eval_cmd, "Evaluating checkpoint")

    if returncode != 0:
        print("❌ Evaluation failed!")
        return False

    # Check if results were saved
    results_file = eval_output_dir / "eval_results.json"
    if not results_file.exists():
        print(f"❌ Results file not found at {results_file}")
        return False

    # Load and validate results
    with open(results_file, 'r') as f:
        results = json.load(f)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    # Check that we have results for both models
    if "llama" not in results or "qwen" not in results:
        print("❌ Missing model results in output")
        return False

    # Print key metrics
    for model_name in ["llama", "qwen"]:
        model_results = results[model_name]
        print(f"\n{model_name.upper()} Results:")

        for method in ["text", "latent", "token_budget"]:
            if method in model_results:
                metrics = model_results[method]
                print(f"  {method:12s}: EM={metrics.get('exact_match', 0):.3f}, "
                      f"F1={metrics.get('f1', 0):.3f}, "
                      f"NLL={metrics.get('nll_per_token', 'N/A')}")

    # Check compression metrics if available
    if "compression" in results:
        comp = results["compression"]
        print(f"\nCompression: {comp.get('ratio', 'N/A')}x")
        print(f"Text bytes: {comp.get('text_bytes', 'N/A')}")
        print(f"Latent bytes: {comp.get('latent_bytes', 'N/A')}")

    print("\n✅ Evaluation completed successfully")
    return True


def test_diagnostic_tools(output_dir: Path) -> bool:
    """Test diagnostic tools work with checkpoint."""
    print("\n" + "="*80)
    print("PHASE 4: DIAGNOSTIC TOOLS")
    print("="*80)

    checkpoint_path = output_dir / "train" / "epoch0"

    # Test that we can load and inspect the checkpoint
    test_script = f"""
import torch
import json
from pathlib import Path

checkpoint_path = Path("{checkpoint_path}")

# Load config
with open(checkpoint_path / "config.json", 'r') as f:
    config = json.load(f)

print(f"Checkpoint config:")
print(f"  - Latent length: {{config['latent_len']}}")
print(f"  - Latent dimension: {{config['d_z']}}")
print(f"  - Encoder type: {{config['encoder_type']}}")

# Load encoder
encoder_state = torch.load(checkpoint_path / "encoder.pt", map_location='cpu')
print(f"\\nEncoder state dict keys: {{list(encoder_state.keys())[:5]}}...")

# Load adapters
llama_adapter = torch.load(checkpoint_path / "llama_adapter.pt", map_location='cpu')
qwen_adapter = torch.load(checkpoint_path / "qwen_adapter.pt", map_location='cpu')
print(f"Llama adapter keys: {{list(llama_adapter.keys())[:5]}}...")
print(f"Qwen adapter keys: {{list(qwen_adapter.keys())[:5]}}...")

# Load training state
with open(checkpoint_path / "training_state.json", 'r') as f:
    state = json.load(f)
print(f"\\nTraining state:")
print(f"  - Epoch: {{state.get('epoch', 'N/A')}}")
print(f"  - Global step: {{state.get('global_step', 'N/A')}}")
print(f"  - Best F1: {{state.get('best_f1', 'N/A')}}")

print("\\n✅ Checkpoint structure validated")
"""

    # Write and run test script
    test_file = output_dir / "test_load.py"
    with open(test_file, 'w') as f:
        f.write(test_script)

    returncode, stdout, stderr = run_command(
        f"python {test_file}",
        "Testing checkpoint loading"
    )

    if returncode != 0:
        print("❌ Failed to load checkpoint")
        return False

    print(stdout)
    print("✅ Diagnostic tools working correctly")
    return True


def main():
    """Run the complete end-to-end test."""
    print("\n" + "="*80)
    print("LATENTWIRE END-TO-END TEST")
    print("="*80)
    print("\nThis test validates the entire training and evaluation pipeline.")
    print("Expected runtime: ~1-2 minutes on a single GPU\n")

    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        print(f"Using temporary directory: {output_dir}\n")

        # Track test results
        results = {
            "training": False,
            "resume": False,
            "evaluation": False,
            "diagnostics": False
        }

        # Run all test phases
        try:
            # Phase 1: Training
            results["training"] = test_training_phase(output_dir)
            if not results["training"]:
                print("\n⚠️  Skipping remaining tests due to training failure")
            else:
                # Phase 2: Resume training
                results["resume"] = test_resume_training(output_dir)

                # Phase 3: Evaluation
                results["evaluation"] = test_evaluation(output_dir)

                # Phase 4: Diagnostics
                results["diagnostics"] = test_diagnostic_tools(output_dir)

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()

        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        all_passed = True
        for phase, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{phase.capitalize():15s}: {status}")
            all_passed = all_passed and passed

        print("\n" + "="*80)
        if all_passed:
            print("✅ ALL TESTS PASSED - Pipeline is working correctly!")
            return 0
        else:
            print("❌ SOME TESTS FAILED - Please review the output above")
            return 1


if __name__ == "__main__":
    sys.exit(main())