"""Tests for Gist reconstruction head feature."""

import pytest

try:
    import torch
    import torch.nn as nn
except ImportError:
    pytest.skip("PyTorch not available", allow_module_level=True)

from latentwire.models import GistReconstructionHead


def test_gist_reconstruction_head_forward():
    """Test GistReconstructionHead forward pass."""
    batch_size = 2
    seq_len = 10
    d_latent = 256
    d_model = 512
    target_len = 48
    hidden = 512
    num_layers = 2

    head = GistReconstructionHead(
        d_latent=d_latent,
        d_model=d_model,
        target_len=target_len,
        hidden=hidden,
        num_layers=num_layers,
        dropout=0.1
    )

    # Create dummy input (latent)
    z = torch.randn(batch_size, seq_len, d_latent)

    # Forward pass
    output = head(z)

    # Check output shape
    assert output.shape == (batch_size, target_len, d_model)


def test_gist_reconstruction_head_layers():
    """Test GistReconstructionHead with different layer counts."""
    d_latent = 128
    d_model = 256
    target_len = 32

    # Test with 1 layer
    head1 = GistReconstructionHead(
        d_latent=d_latent,
        d_model=d_model,
        target_len=target_len,
        hidden=256,
        num_layers=1,
        dropout=0.0
    )

    # Test with 3 layers
    head3 = GistReconstructionHead(
        d_latent=d_latent,
        d_model=d_model,
        target_len=target_len,
        hidden=256,
        num_layers=3,
        dropout=0.0
    )

    # Check that more layers means more parameters
    params1 = sum(p.numel() for p in head1.parameters())
    params3 = sum(p.numel() for p in head3.parameters())
    assert params3 > params1


def test_gist_reconstruction_head_dropout():
    """Test that dropout produces stochastic outputs in training mode and deterministic outputs in eval mode."""
    d_latent = 64
    d_model = 128
    target_len = 16

    head = GistReconstructionHead(
        d_latent=d_latent,
        d_model=d_model,
        target_len=target_len,
        hidden=128,
        num_layers=2,
        dropout=0.5  # High dropout for testing
    )

    x = torch.randn(1, 8, d_latent)

    head.train()
    torch.manual_seed(0)
    out_train_1 = head(x)
    torch.manual_seed(1)
    out_train_2 = head(x)
    assert not torch.allclose(out_train_1, out_train_2), "Dropout should introduce stochasticity in training mode"

    head.eval()
    torch.manual_seed(0)
    out_eval_1 = head(x)
    torch.manual_seed(1)
    out_eval_2 = head(x)
    assert torch.allclose(out_eval_1, out_eval_2), "Eval mode should be deterministic"


def test_gist_reconstruction_head_gradient_flow():
    """Test that gradients flow through the head."""
    d_latent = 64
    d_model = 128
    target_len = 16

    head = GistReconstructionHead(
        d_latent=d_latent,
        d_model=d_model,
        target_len=target_len,
        hidden=128,
        num_layers=1,
        dropout=0.0
    )

    x = torch.randn(1, 8, d_latent, requires_grad=True)
    output = head(x)

    # Create a dummy loss
    loss = output.sum()
    loss.backward()

    # Check that input has gradients
    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


def test_gist_config_integration():
    """Test Gist configuration integration."""
    from latentwire.config import GistConfig

    config = GistConfig()

    # Check defaults
    assert config.gist_target_len == 48
    assert config.gist_hidden == 512
    assert config.gist_layers == 2
    assert config.gist_dropout == 0.1
    assert config.gist_weight == 0.0
    assert config.gist_mask_prob == 0.15

    # Test custom values
    config = GistConfig(
        gist_target_len=64,
        gist_hidden=1024,
        gist_layers=3,
        gist_dropout=0.2,
        gist_weight=1.0,
        gist_mask_prob=0.2
    )

    assert config.gist_target_len == 64
    assert config.gist_hidden == 1024
    assert config.gist_layers == 3
    assert config.gist_dropout == 0.2
    assert config.gist_weight == 1.0
    assert config.gist_mask_prob == 0.2


def test_gist_dropout_masks_attention():
    """Ensure the internal dropout zeros out some attention weights when training."""
    head = GistReconstructionHead(
        d_latent=32,
        d_model=64,
        target_len=8,
        hidden=64,
        num_layers=1,
        dropout=0.5,
    )

    assert head.dropout is not None
    head.train()

    attn = torch.ones(4, 8, 10)
    torch.manual_seed(0)
    dropped = head.dropout(attn)

    # Dropout should zero some entries and rescale others
    zero_count = (dropped == 0).sum().item()
    assert zero_count > 0, "Expected dropout to zero attention weights"
    assert torch.unique(dropped).numel() > 1, "Dropout should rescale surviving weights"


def test_gist_loss_computation():
    """Test gist reconstruction loss computation."""
    batch_size = 2
    target_len = 16
    d_model = 64

    # Create dummy predictions and targets
    pred = torch.randn(batch_size, target_len, d_model)
    target = torch.randn(batch_size, target_len, d_model)
    mask = torch.ones(batch_size, target_len, dtype=torch.bool)
    mask[0, -4:] = False  # Mask out last 4 positions of first sample

    # Compute MSE loss manually
    mse = ((pred - target) ** 2).mean(dim=-1)  # [B, T]
    mse = mse * mask  # Apply mask
    loss = mse.sum() / mask.sum()

    # This should match the gist loss computation
    assert loss.item() > 0
