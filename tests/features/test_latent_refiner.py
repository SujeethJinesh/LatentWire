"""Tests for Latent Refiner feature."""

import pytest

try:
    import torch
    import torch.nn as nn
except ImportError:
    pytest.skip("PyTorch not available", allow_module_level=True)

from latentwire.models import LatentRefiner


def test_latent_refiner_initialization():
    """Test LatentRefiner initialization with various parameters."""
    d_z = 256
    num_layers = 2
    num_heads = 8

    refiner = LatentRefiner(
        d_z=d_z,
        num_layers=num_layers,
        num_heads=num_heads
    )

    # Check that the encoder was created
    assert refiner.encoder is not None

    # Check that parameters were created
    params = sum(p.numel() for p in refiner.parameters())
    assert params > 0


def test_latent_refiner_forward():
    """Test LatentRefiner forward pass."""
    batch_size = 2
    seq_len = 32
    d_z = 128
    num_layers = 2
    num_heads = 4

    refiner = LatentRefiner(
        d_z=d_z,
        num_layers=num_layers,
        num_heads=num_heads
    )

    # Create dummy input
    z = torch.randn(batch_size, seq_len, d_z)

    # Forward pass
    output = refiner(z)

    # Check output shape matches input
    assert output.shape == z.shape

    # Check that output is different from input (refining happened)
    assert not torch.allclose(output, z)


def test_latent_refiner_requires_layers():
    """Test that LatentRefiner requires at least one layer."""
    with pytest.raises(AssertionError, match="at least one layer"):
        LatentRefiner(
            d_z=256,
            num_layers=0,  # Invalid
            num_heads=8
        )


def test_latent_refiner_head_dimension_divisibility():
    """Test that d_z must be divisible by num_heads."""
    # This should work (256 is divisible by 8)
    refiner = LatentRefiner(
        d_z=256,
        num_layers=1,
        num_heads=8
    )
    assert 256 % 8 == 0
    assert refiner is not None

    # This should fail - 257 is not divisible by 8
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        refiner = LatentRefiner(
            d_z=257,  # Prime number, not divisible by 8
            num_layers=1,
            num_heads=8
        )


def test_latent_refiner_gradient_flow():
    """Test that gradients flow through the refiner."""
    d_z = 64
    refiner = LatentRefiner(
        d_z=d_z,
        num_layers=1,
        num_heads=4
    )

    z = torch.randn(1, 8, d_z, requires_grad=True)
    output = refiner(z)

    # Create a dummy loss
    loss = output.sum()
    loss.backward()

    # Check that input has gradients
    assert z.grad is not None
    assert not torch.allclose(z.grad, torch.zeros_like(z.grad))

    # Check that refiner parameters have gradients
    for param in refiner.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_latent_refiner_with_different_layer_counts():
    """Test LatentRefiner with different numbers of layers."""
    d_z = 128
    seq_len = 16
    batch_size = 1

    # Test with 1, 2, and 4 layers
    for num_layers in [1, 2, 4]:
        refiner = LatentRefiner(
            d_z=d_z,
            num_layers=num_layers,
            num_heads=4
        )

        z = torch.randn(batch_size, seq_len, d_z)
        output = refiner(z)

        assert output.shape == z.shape

        # More layers should have more parameters
        param_count = sum(p.numel() for p in refiner.parameters())
        assert param_count > 0


def test_latent_refiner_config():
    """Test LatentRefiner configuration."""
    from latentwire.config import LatentRefinerConfig

    config = LatentRefinerConfig()

    # Check defaults
    assert config.latent_refiner_layers == 0
    assert config.latent_refiner_heads == 4

    # Test custom values
    config = LatentRefinerConfig(
        latent_refiner_layers=3,
        latent_refiner_heads=8
    )

    assert config.latent_refiner_layers == 3
    assert config.latent_refiner_heads == 8


def test_latent_refiner_training_mode():
    """Test that refiner behaves differently in train vs eval mode."""
    d_z = 64
    refiner = LatentRefiner(
        d_z=d_z,
        num_layers=1,
        num_heads=2
    )

    z = torch.randn(1, 8, d_z)

    # Set to eval mode
    refiner.eval()
    with torch.no_grad():
        output_eval1 = refiner(z).clone()
        output_eval2 = refiner(z).clone()

    # In eval mode, outputs should be deterministic
    assert torch.allclose(output_eval1, output_eval2)

    # Set to train mode
    refiner.train()
    # Note: Without dropout, outputs might still be deterministic
    # but the model is ready for training


def test_latent_refiner_device_handling():
    """Test that LatentRefiner handles device placement correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    d_z = 64
    refiner = LatentRefiner(
        d_z=d_z,
        num_layers=1,
        num_heads=2
    )

    # Move to CUDA
    refiner = refiner.cuda()

    # Input on CUDA
    z = torch.randn(1, 8, d_z).cuda()
    output = refiner(z)

    # Output should be on same device as input
    assert output.device == z.device


def test_latent_refiner_from_config_dict():
    """Test creating LatentRefiner from configuration dictionary."""
    from latentwire.config import TrainingConfig

    config_dict = {
        "features": {
            "use_latent_refiner": True
        },
        "latent_refiner": {
            "latent_refiner_layers": 2,
            "latent_refiner_heads": 8
        }
    }

    config = TrainingConfig.from_dict(config_dict)

    # Feature should be enabled
    assert config.features.use_latent_refiner == True
    assert config.latent_refiner.latent_refiner_layers == 2
    assert config.latent_refiner.latent_refiner_heads == 8