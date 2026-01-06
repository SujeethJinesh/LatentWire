#!/usr/bin/env python3
"""
Test script to verify Phase 2 (Linear Probe Baseline) functionality.
Tests the LinearProbeBaseline class with mock data to ensure it works correctly.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import the linear probe baseline
from linear_probe_baseline import LinearProbeBaseline

def test_linear_probe_basic():
    """Test basic functionality of LinearProbeBaseline with mock data."""
    print("\n" + "="*80)
    print("Testing Phase 2: Linear Probe Baseline")
    print("="*80)

    # Test parameters
    hidden_dim = 768  # Smaller dimension for testing
    num_classes = 3
    num_samples_train = 100
    num_samples_test = 30

    print(f"\nTest Configuration:")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Training samples: {num_samples_train}")
    print(f"  Test samples: {num_samples_test}")

    # Create mock data
    print("\nGenerating mock data...")
    np.random.seed(42)

    # Create separable mock features (to ensure the probe can learn)
    X_train = []
    y_train = []
    for class_idx in range(num_classes):
        # Create samples for this class with different means
        class_samples = np.random.randn(num_samples_train // num_classes, hidden_dim)
        class_samples += (class_idx * 2)  # Shift mean for each class
        X_train.append(class_samples)
        y_train.extend([class_idx] * (num_samples_train // num_classes))

    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    # Shuffle training data
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # Create test data similarly
    X_test = []
    y_test = []
    for class_idx in range(num_classes):
        class_samples = np.random.randn(num_samples_test // num_classes, hidden_dim)
        class_samples += (class_idx * 2)
        X_test.append(class_samples)
        y_test.extend([class_idx] * (num_samples_test // num_classes))

    X_test = np.vstack(X_test)
    y_test = np.array(y_test)

    print(f"  Training data shape: {X_train.shape}")
    print(f"  Test data shape: {X_test.shape}")

    # Initialize LinearProbeBaseline
    print("\nInitializing LinearProbeBaseline...")
    probe = LinearProbeBaseline(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        layer_idx=16,  # Arbitrary layer for testing
        pooling="mean",
        normalize=True,
        C=1.0,
        max_iter=1000,
        n_jobs=1,  # Use single thread for testing
        random_state=42,
    )
    print("  Probe initialized successfully!")

    # Test training
    print("\nTraining linear probe...")
    train_results = probe.fit(
        X_train=X_train,
        y_train=y_train,
        cv_folds=3,  # Use 3-fold CV for speed
        return_cv_scores=True,
    )

    print(f"\nTraining Results:")
    print(f"  Training accuracy: {train_results['train_accuracy']:.1f}%")
    if "cv_mean" in train_results:
        print(f"  Cross-validation: {train_results['cv_mean']:.1f}% ± {train_results['cv_std']:.1f}%")
        print(f"  CV scores: {train_results['cv_scores']}")

    # Test prediction
    print("\nTesting prediction...")
    predictions = probe.predict(X_test)
    accuracy = np.mean(predictions == y_test) * 100
    print(f"  Test accuracy: {accuracy:.1f}%")

    # Test probability prediction
    print("\nTesting probability prediction...")
    probabilities = probe.predict_proba(X_test)
    print(f"  Probability shape: {probabilities.shape}")
    print(f"  Probability sum check (should be ~1.0): {probabilities[0].sum():.3f}")

    # Test save/load functionality
    print("\nTesting save/load functionality...")
    save_path = "/tmp/test_linear_probe"
    probe.save(save_path)
    print(f"  Saved to {save_path}.joblib")

    # Load and verify
    loaded_probe = LinearProbeBaseline.load(save_path)
    print("  Loaded successfully!")

    # Verify loaded probe works
    loaded_predictions = loaded_probe.predict(X_test)
    assert np.array_equal(predictions, loaded_predictions), "Loaded predictions don't match!"
    print("  Loaded probe produces identical predictions!")

    # Test feature extraction mock (without real model)
    print("\nTesting feature extraction interface...")

    class MockModel(nn.Module):
        """Mock model that returns random hidden states."""
        def __init__(self, hidden_dim):
            super().__init__()
            self.hidden_dim = hidden_dim

        def forward(self, input_ids, attention_mask, output_hidden_states=True, return_dict=True):
            batch_size, seq_len = input_ids.shape
            # Create mock hidden states for each layer (25 layers like Llama)
            hidden_states = []
            for i in range(25):
                hidden = torch.randn(batch_size, seq_len, self.hidden_dim)
                hidden_states.append(hidden)

            class MockOutput:
                pass

            output = MockOutput()
            output.hidden_states = hidden_states
            return output

    class MockTokenizer:
        """Mock tokenizer that returns random tokens."""
        def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=512):
            batch_size = len(texts)
            seq_len = 10  # Mock sequence length

            return {
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
            }

    # Test extraction with mock model
    mock_model = MockModel(hidden_dim)
    mock_tokenizer = MockTokenizer()

    texts = ["Test text 1", "Test text 2", "Test text 3"]
    features = probe.extract_hidden_states_batch(
        model=mock_model,
        tokenizer=mock_tokenizer,
        texts=texts,
        batch_size=2,
        device="cpu",
        show_progress=False,
    )

    print(f"  Extracted features shape: {features.shape}")
    assert features.shape == (3, hidden_dim), f"Expected shape (3, {hidden_dim}), got {features.shape}"
    print("  Feature extraction works correctly!")

    print("\n" + "="*80)
    print("Phase 2 (Linear Probe Baseline) Test: PASSED")
    print("="*80)
    print("\nAll tests completed successfully!")
    print("The LinearProbeBaseline class is working correctly with:")
    print("  - sklearn LogisticRegression training")
    print("  - Cross-validation support")
    print("  - Prediction and probability estimation")
    print("  - Save/load functionality")
    print("  - Feature extraction interface")

    return True


def test_different_pooling_methods():
    """Test different pooling methods."""
    print("\n" + "="*80)
    print("Testing Different Pooling Methods")
    print("="*80)

    hidden_dim = 256
    num_classes = 2

    # Create simple linearly separable data
    X = np.random.randn(50, hidden_dim)
    y = np.array([0] * 25 + [1] * 25)
    X[y == 1] += 3  # Make classes separable

    pooling_methods = ["mean", "first_token", "last_token"]

    for pooling in pooling_methods:
        print(f"\nTesting pooling method: {pooling}")

        probe = LinearProbeBaseline(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            pooling=pooling,
            normalize=True,
            random_state=42,
        )

        results = probe.fit(X, y, cv_folds=0, return_cv_scores=False)
        print(f"  Training accuracy: {results['train_accuracy']:.1f}%")

        # Should achieve high accuracy on linearly separable data
        assert results['train_accuracy'] > 90, f"Low accuracy for {pooling} pooling"

    print("\nAll pooling methods work correctly!")
    return True


def test_multiclass_classification():
    """Test multiclass classification with different numbers of classes."""
    print("\n" + "="*80)
    print("Testing Multiclass Classification")
    print("="*80)

    hidden_dim = 512

    for num_classes in [2, 5, 10]:
        print(f"\nTesting with {num_classes} classes")

        # Create mock data
        X = np.random.randn(num_classes * 20, hidden_dim)
        y = np.repeat(np.arange(num_classes), 20)

        # Make classes separable
        for i in range(num_classes):
            X[y == i] += i * 2

        probe = LinearProbeBaseline(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            normalize=True,
            random_state=42,
        )

        results = probe.fit(X, y, cv_folds=0)
        print(f"  Training accuracy: {results['train_accuracy']:.1f}%")

        # Test predictions
        predictions = probe.predict(X[:10])
        assert len(predictions) == 10, "Wrong number of predictions"

        # Test probabilities
        probs = probe.predict_proba(X[:10])
        assert probs.shape == (10, num_classes), f"Wrong probability shape"
        assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities don't sum to 1"

        print(f"  Predictions and probabilities work correctly!")

    print("\nMulticlass classification works for all tested class counts!")
    return True


if __name__ == "__main__":
    print("="*80)
    print("Linear Probe Baseline (Phase 2) Verification")
    print("="*80)

    try:
        # Run all tests
        success = True

        # Basic functionality test
        success = test_linear_probe_basic() and success

        # Pooling methods test
        success = test_different_pooling_methods() and success

        # Multiclass test
        success = test_multiclass_classification() and success

        if success:
            print("\n" + "="*80)
            print("ALL PHASE 2 TESTS PASSED SUCCESSFULLY!")
            print("="*80)
            print("\nThe LinearProbeBaseline is fully functional and ready for use.")
            print("It can be integrated into experiments as a strong baseline.")
            sys.exit(0)
        else:
            print("\nSome tests failed!")
            sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)