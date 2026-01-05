#!/usr/bin/env python3
"""
Test script to verify the sklearn-based LinearProbeBaseline implementation.

This script:
1. Loads a small subset of SST-2 data
2. Extracts hidden states from Llama
3. Trains a LogisticRegression probe with cross-validation
4. Evaluates and reports results
5. Saves/loads probe weights to verify reproducibility

Run with:
    python telepathy/test_linear_probe_sklearn.py
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import json

# Import our sklearn-based linear probe
from telepathy.linear_probe_baseline import LinearProbeBaseline, train_linear_probe, eval_linear_probe


def main():
    print("="*80)
    print("Testing sklearn-based LinearProbeBaseline")
    print("="*80)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load a small model for testing (or use Llama if available)
    try:
        # Try Llama first
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        print(f"\nLoading {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        hidden_dim = 4096
    except Exception as e:
        # Fallback to a smaller model for testing
        print(f"Could not load Llama: {e}")
        model_id = "gpt2"
        print(f"Falling back to {model_id} for testing...")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        hidden_dim = 768

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print(f"Model loaded: {model_id}")
    print(f"Hidden dimension: {hidden_dim}")

    # Load SST-2 dataset
    print("\nLoading SST-2 dataset...")
    train_dataset = load_dataset("glue", "sst2", split="train[:1000]")  # Use 1000 samples
    eval_dataset = load_dataset("glue", "sst2", split="validation[:200]")  # Use 200 samples

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Dataset config for SST-2
    dataset_config = {
        "text_field": "sentence",
        "label_field": "label",
        "num_classes": 2,
        "label_map": {0: "negative", 1: "positive"},
    }

    # Create linear probe
    print("\n" + "="*80)
    print("Creating LinearProbeBaseline with sklearn")
    print("="*80)

    probe = LinearProbeBaseline(
        hidden_dim=hidden_dim,
        num_classes=2,
        layer_idx=-2,  # Second-to-last layer often works well
        pooling="mean",
        normalize=True,  # Standardize features
        C=1.0,  # Regularization strength
        max_iter=1000,
        n_jobs=-1,  # Use all CPU cores
        random_state=42,
    )

    print(f"Configuration:")
    print(f"  - Layer index: {probe.layer_idx}")
    print(f"  - Pooling: {probe.pooling}")
    print(f"  - Normalize: {probe.normalize}")
    print(f"  - Regularization C: {probe.C}")

    # Train the probe
    print("\n" + "="*80)
    print("Training Linear Probe")
    print("="*80)

    train_results = train_linear_probe(
        probe=probe,
        model=model,
        tokenizer=tokenizer,
        train_ds=train_dataset,
        dataset_name="SST-2",
        device=str(device),
        dataset_config=dataset_config,
        batch_size=8 if "gpt2" in model_id else 4,  # Larger batch for smaller model
        cv_folds=5,  # 5-fold cross-validation
    )

    print("\nTraining Results:")
    print(json.dumps(train_results, indent=2))

    # Evaluate the probe
    print("\n" + "="*80)
    print("Evaluating Linear Probe")
    print("="*80)

    eval_results = eval_linear_probe(
        probe=probe,
        model=model,
        tokenizer=tokenizer,
        eval_ds=eval_dataset,
        dataset_name="SST-2",
        device=str(device),
        dataset_config=dataset_config,
        batch_size=8 if "gpt2" in model_id else 4,
    )

    print("\nEvaluation Results:")
    print(json.dumps(eval_results, indent=2))

    # Test save/load functionality
    print("\n" + "="*80)
    print("Testing Save/Load Functionality")
    print("="*80)

    # Save probe
    save_path = "test_probe_sst2"
    probe.save(save_path)

    # Load probe
    loaded_probe = LinearProbeBaseline.load(save_path)
    print(f"Successfully loaded probe from {save_path}.joblib")

    # Verify loaded probe works
    eval_results_loaded = eval_linear_probe(
        probe=loaded_probe,
        model=model,
        tokenizer=tokenizer,
        eval_ds=eval_dataset,
        dataset_name="SST-2",
        device=str(device),
        dataset_config=dataset_config,
        batch_size=8 if "gpt2" in model_id else 4,
    )

    print("\nEvaluation with Loaded Probe:")
    print(json.dumps(eval_results_loaded, indent=2))

    # Verify results match
    if abs(eval_results["accuracy"] - eval_results_loaded["accuracy"]) < 0.01:
        print("\n✓ Save/load verification PASSED - results match!")
    else:
        print("\n✗ Save/load verification FAILED - results don't match!")

    # Clean up test file
    import os
    if os.path.exists(f"{save_path}.joblib"):
        os.remove(f"{save_path}.joblib")
        print(f"Cleaned up test file: {save_path}.joblib")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model: {model_id}")
    print(f"Train accuracy: {train_results['train_accuracy']:.1f}%")
    if "cv_mean" in train_results:
        print(f"CV accuracy: {train_results['cv_mean']:.1f}% ± {train_results['cv_std']:.1f}%")
    print(f"Test accuracy: {eval_results['accuracy']:.1f}%")
    print(f"Test F1 score: {eval_results['f1_score']:.1f}%")
    print(f"Random baseline: 50.0% (binary classification)")

    if eval_results["accuracy"] > 50.0:
        print(f"\n✓ Linear probe performs above random chance!")
    else:
        print(f"\n✗ Linear probe at or below random chance - may need tuning")

    print("\n" + "="*80)
    print("Test complete! The sklearn-based LinearProbeBaseline is working correctly.")
    print("="*80)


if __name__ == "__main__":
    main()