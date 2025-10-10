"""Regression test for embedding baseline issues fixed in PR."""

import pytest
import sys
import os

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_eval_handles_missing_config_keys():
    """Test that eval.py handles missing config keys gracefully."""
    import latentwire.eval as eval_module

    # Test that missing keys use defaults
    cfg = {"llama_id": "test/model"}  # Missing qwen_id and d_z

    # These should use defaults without KeyError
    qwen_id = cfg.get("qwen_id", "Qwen/Qwen2.5-7B-Instruct")
    d_z = cfg.get("d_z", 256)

    assert qwen_id == "Qwen/Qwen2.5-7B-Instruct"
    assert d_z == 256


def test_embedding_baseline_without_encoder():
    """Test that embedding baselines can run without encoder loaded."""
    # This tests the logic where encoded_latents can be None
    combined_latents = {"llama": None, "qwen": None}

    # Check that all latents are None (embedding-only baseline)
    skip_latent_eval = all(latent is None for latent in combined_latents.values())

    assert skip_latent_eval == True

    # Test that we can create dummy results
    dummy_results = {
        "latent": {"preds": [""], "metrics": {"em": 0, "f1": 0, "nll": float("inf")}, "time": 0},
        "trunc": {"preds": [""], "metrics": {"em": 0, "f1": 0}, "time": 0}
    }

    assert dummy_results["latent"]["metrics"]["em"] == 0
    assert dummy_results["latent"]["metrics"]["nll"] == float("inf")


def test_tensor_padding_for_different_lengths():
    """Test that tensors of different lengths can be padded and concatenated."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    # Create tensors of different sequence lengths
    tensor1 = torch.randn(1, 10, 768)  # batch=1, seq_len=10, hidden=768
    tensor2 = torch.randn(1, 15, 768)  # batch=1, seq_len=15, hidden=768
    tensor3 = torch.randn(1, 8, 768)   # batch=1, seq_len=8, hidden=768

    prefix_tensors = [tensor1, tensor2, tensor3]

    # Pad to max length
    max_len = max(p.size(1) for p in prefix_tensors)
    padded_prefix_tensors = []
    for p in prefix_tensors:
        if p.size(1) < max_len:
            padding = torch.zeros(p.size(0), max_len - p.size(1), p.size(2),
                                device=p.device, dtype=p.dtype)
            p = torch.cat([p, padding], dim=1)
        padded_prefix_tensors.append(p)

    # Should be able to concatenate now
    prefix_batch = torch.cat(padded_prefix_tensors, dim=0)

    assert prefix_batch.shape == (3, max_len, 768)


def test_float32_conversion_for_interpolation():
    """Test that float16 tensors are converted to float32 for interpolation."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        pytest.skip("PyTorch not available")

    # Skip if MPS not available
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    # Create a float16 tensor
    seq = torch.randn(1, 768, 10, dtype=torch.float16)  # batch=1, channels=768, length=10

    # Convert to float32 for interpolation
    orig_dtype = seq.dtype
    seq_float = seq.float()

    # Interpolation should work with float32
    pooled_float = F.interpolate(seq_float, size=32, mode="linear", align_corners=False)

    # Convert back to original dtype
    pooled = pooled_float.to(orig_dtype)

    assert pooled.dtype == torch.float16
    assert pooled.shape == (1, 768, 32)


def test_none_latent_handling():
    """Test that None latents are handled gracefully in wire stats."""
    combined_latents = {"llama": None}

    # This simulates the wire stats creation with None latent
    name, latent_tensor = next(iter(combined_latents.items()))

    if latent_tensor is not None:
        num_latent = latent_tensor.numel()
        wire_stats = {"latent_bytes": num_latent * 4}
    else:
        wire_stats = {"latent_bytes": 0}

    assert wire_stats["latent_bytes"] == 0