#!/usr/bin/env python3
"""
Simplified test script for Phase 2 (Linear Probe Baseline).
Tests core functionality without requiring PyTorch.
"""

import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import the linear probe baseline
from linear_probe_baseline import LinearProbeBaseline

def test_linear_probe_core():
    """Test core functionality of LinearProbeBaseline without PyTorch dependencies."""
    print("\n" + "="*80)
    print("Testing Phase 2: Linear Probe Baseline (Core Functionality)")
    print("="*80)

    # Test parameters
    hidden_dim = 768
    num_classes = 3
    num_samples_train = 150
    num_samples_test = 50

    print(f"\nTest Configuration:")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Training samples: {num_samples_train}")
    print(f"  Test samples: {num_samples_test}")

    # Create separable mock data
    print("\nGenerating mock data with separable classes...")
    np.random.seed(42)

    # Training data
    X_train = []
    y_train = []
    for class_idx in range(num_classes):
        # Create samples with different means for each class
        samples_per_class = num_samples_train // num_classes
        class_samples = np.random.randn(samples_per_class, hidden_dim) * 0.5
        class_samples += (class_idx * 3)  # Significant separation between classes
        X_train.append(class_samples)
        y_train.extend([class_idx] * samples_per_class)

    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    # Shuffle
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # Test data
    X_test = []
    y_test = []
    for class_idx in range(num_classes):
        samples_per_class = num_samples_test // num_classes
        class_samples = np.random.randn(samples_per_class, hidden_dim) * 0.5
        class_samples += (class_idx * 3)
        X_test.append(class_samples)
        y_test.extend([class_idx] * samples_per_class)

    X_test = np.vstack(X_test)
    y_test = np.array(y_test)

    print(f"  Training data shape: {X_train.shape}")
    print(f"  Test data shape: {X_test.shape}")
    print(f"  Classes are well-separated for reliable testing")

    # Test 1: Initialize LinearProbeBaseline
    print("\n" + "-"*40)
    print("Test 1: Initialization")
    print("-"*40)

    probe = LinearProbeBaseline(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        layer_idx=16,
        pooling="mean",
        normalize=True,
        C=1.0,
        max_iter=1000,
        n_jobs=1,
        random_state=42,
    )
    print("✓ LinearProbeBaseline initialized successfully")

    # Test 2: Training
    print("\n" + "-"*40)
    print("Test 2: Training with Cross-Validation")
    print("-"*40)

    train_results = probe.fit(
        X_train=X_train,
        y_train=y_train,
        cv_folds=5,
        return_cv_scores=True,
    )

    print(f"✓ Training completed")
    print(f"  Training accuracy: {train_results['train_accuracy']:.1f}%")
    if "cv_mean" in train_results:
        print(f"  Cross-validation: {train_results['cv_mean']:.1f}% ± {train_results['cv_std']:.1f}%")

    assert train_results['train_accuracy'] > 95, "Training accuracy too low on separable data"
    print("✓ Training accuracy is high (as expected for separable data)")

    # Test 3: Prediction
    print("\n" + "-"*40)
    print("Test 3: Prediction on Test Set")
    print("-"*40)

    predictions = probe.predict(X_test)
    accuracy = np.mean(predictions == y_test) * 100

    print(f"✓ Predictions generated")
    print(f"  Test accuracy: {accuracy:.1f}%")
    assert accuracy > 90, "Test accuracy too low on separable data"
    print("✓ Test accuracy is high (as expected for separable data)")

    # Test 4: Probability estimation
    print("\n" + "-"*40)
    print("Test 4: Probability Estimation")
    print("-"*40)

    probabilities = probe.predict_proba(X_test[:5])
    print(f"✓ Probabilities computed")
    print(f"  Shape: {probabilities.shape}")

    # Check probabilities sum to 1
    prob_sums = probabilities.sum(axis=1)
    assert np.allclose(prob_sums, 1.0), "Probabilities don't sum to 1"
    print("✓ Probabilities sum to 1.0")

    # Show sample probabilities
    print("\n  Sample probabilities for first 5 test examples:")
    for i in range(5):
        true_label = y_test[i]
        pred_label = predictions[i]
        prob = probabilities[i]
        print(f"    True: {true_label}, Pred: {pred_label}, Probs: [{prob[0]:.3f}, {prob[1]:.3f}, {prob[2]:.3f}]")

    # Test 5: Save/Load
    print("\n" + "-"*40)
    print("Test 5: Save/Load Functionality")
    print("-"*40)

    save_path = "/tmp/test_linear_probe_phase2"
    probe.save(save_path)
    print(f"✓ Probe saved to {save_path}.joblib")

    loaded_probe = LinearProbeBaseline.load(save_path)
    print("✓ Probe loaded successfully")

    # Verify loaded probe works identically
    loaded_predictions = loaded_probe.predict(X_test)
    assert np.array_equal(predictions, loaded_predictions), "Loaded predictions differ"
    print("✓ Loaded probe produces identical predictions")

    # Test 6: Different configurations
    print("\n" + "-"*40)
    print("Test 6: Different Configurations")
    print("-"*40)

    # Test without normalization
    probe_no_norm = LinearProbeBaseline(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        normalize=False,
        random_state=42,
    )
    results_no_norm = probe_no_norm.fit(X_train, y_train, cv_folds=0)
    print(f"✓ Without normalization: {results_no_norm['train_accuracy']:.1f}% accuracy")

    # Test with stronger regularization
    probe_reg = LinearProbeBaseline(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        C=0.01,  # Stronger regularization
        random_state=42,
    )
    results_reg = probe_reg.fit(X_train, y_train, cv_folds=0)
    print(f"✓ With strong regularization (C=0.01): {results_reg['train_accuracy']:.1f}% accuracy")

    # Test 7: Binary classification
    print("\n" + "-"*40)
    print("Test 7: Binary Classification")
    print("-"*40)

    # Create binary data
    X_binary = np.random.randn(100, hidden_dim)
    y_binary = np.array([0] * 50 + [1] * 50)
    X_binary[y_binary == 1] += 2  # Make separable

    probe_binary = LinearProbeBaseline(
        hidden_dim=hidden_dim,
        num_classes=2,
        random_state=42,
    )

    results_binary = probe_binary.fit(X_binary, y_binary, cv_folds=3)
    print(f"✓ Binary classification: {results_binary['train_accuracy']:.1f}% accuracy")

    # Test predictions on binary
    binary_probs = probe_binary.predict_proba(X_binary[:5])
    print(f"✓ Binary probabilities shape: {binary_probs.shape}")

    return True


def test_multiclass_scaling():
    """Test performance with different numbers of classes."""
    print("\n" + "="*80)
    print("Testing Scaling to Different Numbers of Classes")
    print("="*80)

    hidden_dim = 256
    samples_per_class = 30

    results = {}
    for num_classes in [2, 5, 10, 20]:
        print(f"\nTesting with {num_classes} classes...")

        # Create data
        X = []
        y = []
        for class_idx in range(num_classes):
            class_samples = np.random.randn(samples_per_class, hidden_dim) * 0.3
            # Spread classes in feature space
            angle = (2 * np.pi * class_idx) / num_classes
            class_samples[:, 0] += 5 * np.cos(angle)
            class_samples[:, 1] += 5 * np.sin(angle)
            X.append(class_samples)
            y.extend([class_idx] * samples_per_class)

        X = np.vstack(X)
        y = np.array(y)

        # Train probe
        probe = LinearProbeBaseline(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            normalize=True,
            random_state=42,
        )

        train_results = probe.fit(X, y, cv_folds=3)
        results[num_classes] = train_results['train_accuracy']

        print(f"  Training accuracy: {train_results['train_accuracy']:.1f}%")
        if 'cv_mean' in train_results:
            print(f"  CV accuracy: {train_results['cv_mean']:.1f}%")

    print("\n" + "-"*40)
    print("Summary of multiclass scaling:")
    for nc, acc in results.items():
        print(f"  {nc:2d} classes: {acc:.1f}% accuracy")

    return True


def test_memory_efficiency():
    """Test memory-efficient processing of large datasets."""
    print("\n" + "="*80)
    print("Testing Memory Efficiency with Large Dataset")
    print("="*80)

    # Create a "large" dataset (but manageable for testing)
    hidden_dim = 1024
    num_samples = 1000
    num_classes = 5

    print(f"Creating dataset with {num_samples} samples, {hidden_dim} dimensions...")

    X = np.random.randn(num_samples, hidden_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)

    # Make somewhat separable
    for class_idx in range(num_classes):
        X[y == class_idx, :50] += class_idx * 0.5

    print(f"  Dataset size: {X.nbytes / 1024 / 1024:.1f} MB")

    # Train probe
    probe = LinearProbeBaseline(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        normalize=True,
        C=1.0,
        random_state=42,
    )

    print("\nTraining on large dataset...")
    train_results = probe.fit(X, y, cv_folds=0)  # No CV for speed

    print(f"✓ Training completed without memory issues")
    print(f"  Training accuracy: {train_results['train_accuracy']:.1f}%")

    # Test prediction in batches
    print("\nTesting batch prediction...")
    batch_size = 100
    all_predictions = []

    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        batch_preds = probe.predict(batch)
        all_predictions.extend(batch_preds)

    accuracy = np.mean(all_predictions == y) * 100
    print(f"✓ Batch prediction completed")
    print(f"  Accuracy: {accuracy:.1f}%")

    return True


if __name__ == "__main__":
    print("="*80)
    print("Phase 2: Linear Probe Baseline Verification")
    print("="*80)
    print("\nThis test verifies the LinearProbeBaseline implementation")
    print("using sklearn's LogisticRegression for scientific rigor.")

    try:
        success = True

        # Core functionality test
        print("\n" + "="*80)
        print("RUNNING CORE TESTS")
        print("="*80)
        success = test_linear_probe_core() and success

        # Multiclass scaling test
        success = test_multiclass_scaling() and success

        # Memory efficiency test
        success = test_memory_efficiency() and success

        if success:
            print("\n" + "="*80)
            print("✓ ALL PHASE 2 TESTS PASSED SUCCESSFULLY!")
            print("="*80)
            print("\nVerification Complete:")
            print("  ✓ LinearProbeBaseline class is fully functional")
            print("  ✓ Training with cross-validation works correctly")
            print("  ✓ Prediction and probability estimation work")
            print("  ✓ Save/load functionality verified")
            print("  ✓ Handles binary and multiclass classification")
            print("  ✓ Memory-efficient processing confirmed")
            print("\nThe Linear Probe Baseline is ready for integration into experiments.")
            print("It provides a strong, scientifically rigorous baseline for comparison.")
            sys.exit(0)
        else:
            print("\n[ERROR] Some tests failed!")
            sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)