#!/usr/bin/env python3
"""
Comprehensive unit tests for Stage 1 Phase 1 training.
All tests run on CPU (MacBook compatible).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_adapter_only_phase1 import (
    EmbeddingCompressor,
    compute_reconstruction_metrics,
    log_diagnostics
)
from latentwire.models import Adapter
from latentwire.data import load_squad_subset


class TestEmbeddingCompressor:
    """Test PCA-based embedding compressor"""

    def test_init(self):
        """Test compressor initialization"""
        compressor = EmbeddingCompressor(input_dim=4096, output_dim=512, method="pca")
        assert compressor.input_dim == 4096
        assert compressor.output_dim == 512
        assert compressor.method == "pca"
        assert compressor.projection is None

    def test_pca_fit_shape(self):
        """Test that PCA fitting produces correct shapes"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")

        # Create sample embeddings [num_vectors, dim]
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        assert compressor.projection is not None
        assert compressor.projection.shape == (128, 32)
        assert compressor.mean.shape == (128,)
        assert compressor.explained_variance_ratio is not None
        assert 0 < compressor.explained_variance_ratio <= 1.0

    def test_pca_fit_explained_variance(self):
        """Test that PCA captures meaningful variance"""
        compressor = EmbeddingCompressor(input_dim=64, output_dim=32, method="pca")

        # Create correlated data (should have high explained variance)
        base = torch.randn(500, 32)
        embeddings = torch.cat([base, base + torch.randn(500, 32) * 0.1], dim=1)

        compressor.fit(embeddings)

        # Should explain >90% of variance for this correlated data
        assert compressor.explained_variance_ratio > 0.90

    def test_compress_shape_preservation(self):
        """Test that compression produces correct output shapes"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")

        # Fit on sample data
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        # Test compression with batched sequences
        batch_embeds = torch.randn(16, 50, 128)  # [batch, seq_len, dim]
        compressed = compressor.compress(batch_embeds)

        assert compressed.shape == (16, 50, 32)

    def test_compress_dtype_preservation(self):
        """Test that compression preserves input dtype"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")

        # Fit on float32
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        # Compress bfloat16 input
        batch_embeds = torch.randn(4, 10, 128).to(torch.bfloat16)
        compressed = compressor.compress(batch_embeds)

        assert compressed.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_compress_device_handling(self):
        """Test that compression works on GPU"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")

        # Fit on CPU
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        # Compress on GPU
        batch_embeds = torch.randn(4, 10, 128).cuda()
        compressed = compressor.compress(batch_embeds)

        assert compressed.is_cuda
        assert compressed.shape == (4, 10, 32)

    def test_compress_reconstruction_quality(self):
        """Test that compression preserves meaningful information"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=64, method="pca")

        # Create structured data
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        # Compress and check information is preserved
        test_embeds = torch.randn(10, 20, 128)
        compressed = compressor.compress(test_embeds)

        # Compressed should have different scale but preserve relative structure
        assert compressed.std() > 0  # Not degenerate
        assert not torch.isnan(compressed).any()
        assert not torch.isinf(compressed).any()


class TestReconstructionMetrics:
    """Test reconstruction quality metrics"""

    def test_perfect_reconstruction(self):
        """Test metrics for perfect reconstruction"""
        original = torch.randn(8, 20, 128)
        reconstructed = original.clone()
        mask = torch.ones(8, 20)

        metrics = compute_reconstruction_metrics(reconstructed, original, mask)

        assert metrics["recon_mse"] < 1e-6  # Nearly zero
        assert metrics["recon_cosine_sim"] > 0.999  # Nearly 1
        assert metrics["recon_rel_error"] < 1e-5  # Nearly zero

    def test_poor_reconstruction(self):
        """Test metrics for poor reconstruction"""
        original = torch.randn(8, 20, 128)
        reconstructed = torch.randn(8, 20, 128)  # Random, unrelated
        mask = torch.ones(8, 20)

        metrics = compute_reconstruction_metrics(reconstructed, original, mask)

        assert metrics["recon_mse"] > 0.5  # High error
        assert -0.5 < metrics["recon_cosine_sim"] < 0.5  # Random alignment
        assert metrics["recon_rel_error"] > 0.5  # High relative error

    def test_mask_handling(self):
        """Test that metrics respect attention mask"""
        original = torch.randn(4, 10, 128)
        reconstructed = torch.randn(4, 10, 128)

        # Mask out last 5 positions
        mask = torch.ones(4, 10)
        mask[:, 5:] = 0

        metrics = compute_reconstruction_metrics(reconstructed, original, mask)

        # Should still compute without errors
        assert isinstance(metrics["recon_mse"], float)
        assert isinstance(metrics["recon_cosine_sim"], float)
        assert isinstance(metrics["recon_rel_error"], float)

    def test_bfloat16_compatibility(self):
        """Test metrics work with bfloat16 tensors"""
        original = torch.randn(4, 10, 128).to(torch.bfloat16)
        reconstructed = torch.randn(4, 10, 128).to(torch.bfloat16)
        mask = torch.ones(4, 10)

        metrics = compute_reconstruction_metrics(reconstructed, original, mask)

        # Should handle dtype conversion internally
        assert not np.isnan(metrics["recon_mse"])
        assert not np.isnan(metrics["recon_cosine_sim"])
        assert not np.isnan(metrics["recon_rel_error"])


class TestAdapterIntegration:
    """Test adapter with compressor integration"""

    def test_adapter_forward_pass(self):
        """Test adapter produces correct output shape"""
        adapter = Adapter(
            d_z=1024,
            d_model=4096,
            latent_length=32,
            hidden_mult=4,
            dropout=0.0,
            enable_metadata=False,
            colorize=False
        )

        compressed = torch.randn(8, 20, 1024)
        output = adapter(compressed)

        assert output.shape == (8, 20, 4096)

    def test_adapter_dtype_preservation(self):
        """Test adapter handles different dtypes"""
        adapter = Adapter(
            d_z=1024,
            d_model=4096,
            latent_length=32,
            hidden_mult=4,
            dropout=0.0,
            enable_metadata=False,
            colorize=False
        ).to(torch.bfloat16)

        compressed = torch.randn(4, 10, 1024).to(torch.bfloat16)
        output = adapter(compressed)

        assert output.dtype == torch.bfloat16
        assert output.shape == (4, 10, 4096)

    def test_full_pipeline(self):
        """Test complete compress -> adapt pipeline"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")
        adapter = Adapter(
            d_z=32,
            d_model=128,
            latent_length=16,
            hidden_mult=4,
            dropout=0.0,
            enable_metadata=False,
            colorize=False
        )

        # Fit compressor
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        # Test pipeline
        orig_embeds = torch.randn(4, 10, 128)
        compressed = compressor.compress(orig_embeds)
        reconstructed = adapter(compressed)

        assert reconstructed.shape == orig_embeds.shape

        # Check reconstruction improves with gradient updates
        optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)

        initial_loss = F.mse_loss(reconstructed, orig_embeds)

        # Train for a few steps
        for _ in range(10):
            compressed = compressor.compress(orig_embeds)
            reconstructed = adapter(compressed)
            loss = F.mse_loss(reconstructed, orig_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = F.mse_loss(reconstructed, orig_embeds)

        # Loss should decrease
        assert final_loss < initial_loss


class TestDataLoading:
    """Test data loading and formatting"""

    def test_squad_loader(self):
        """Test SQuAD data loads correctly"""
        dataset = load_squad_subset("train", samples=10, seed=42)

        assert len(dataset) == 10
        assert all("source" in item for item in dataset)
        assert all("answer" in item for item in dataset)

        # Check format
        sample = dataset[0]
        assert "Context:" in sample["source"]
        assert "Question:" in sample["source"]
        assert isinstance(sample["answer"], str)
        assert len(sample["answer"]) > 0

    def test_squad_reproducibility(self):
        """Test that same seed gives same data"""
        dataset1 = load_squad_subset("train", samples=5, seed=42)
        dataset2 = load_squad_subset("train", samples=5, seed=42)

        assert dataset1[0]["answer"] == dataset2[0]["answer"]
        assert dataset1[0]["source"] == dataset2[0]["source"]

    def test_squad_different_seeds(self):
        """Test that different seeds give different data"""
        dataset1 = load_squad_subset("train", samples=5, seed=42)
        dataset2 = load_squad_subset("train", samples=5, seed=123)

        # At least one should be different
        different = any(
            d1["answer"] != d2["answer"]
            for d1, d2 in zip(dataset1, dataset2)
        )
        assert different


class TestDiagnosticLogging:
    """Test diagnostic logging functionality"""

    def test_log_diagnostics(self, tmp_path):
        """Test that diagnostics log correctly"""
        import json

        log_file = tmp_path / "test_diagnostics.jsonl"

        # Log some metrics
        log_diagnostics(str(log_file), step=1, epoch=0, metrics={"loss": 0.5, "f1": 0.8})
        log_diagnostics(str(log_file), step=2, epoch=0, metrics={"loss": 0.4, "f1": 0.82})

        # Read back
        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Parse first entry
        entry = json.loads(lines[0])
        assert entry["step"] == 1
        assert entry["epoch"] == 0
        assert entry["loss"] == 0.5
        assert entry["f1"] == 0.8
        assert "timestamp" in entry

    def test_log_diagnostics_none(self):
        """Test that None log_file doesn't crash"""
        # Should not raise
        log_diagnostics(None, step=1, epoch=0, metrics={"loss": 0.5})


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_batch(self):
        """Test handling of empty batches"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        # Empty batch
        empty_batch = torch.randn(0, 10, 128)
        compressed = compressor.compress(empty_batch)
        assert compressed.shape == (0, 10, 32)

    def test_single_sample(self):
        """Test handling of single sample"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        single_sample = torch.randn(1, 10, 128)
        compressed = compressor.compress(single_sample)
        assert compressed.shape == (1, 10, 32)

    def test_long_sequence(self):
        """Test handling of very long sequences"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        long_seq = torch.randn(2, 512, 128)
        compressed = compressor.compress(long_seq)
        assert compressed.shape == (2, 512, 32)

    def test_compression_without_fit(self):
        """Test that compression without fitting uses fallback"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")

        # Compress without fitting
        batch = torch.randn(4, 10, 128)
        compressed = compressor.compress(batch)

        # Should use truncation fallback
        assert compressed.shape == (4, 10, 32)
        # Verify it's actually truncating
        assert torch.allclose(compressed, batch[..., :32])


def test_import_dependencies():
    """Test that all required dependencies can be imported"""
    try:
        import torch
        import transformers
        import sklearn
        import numpy as np
        from latentwire.data import load_squad_subset
        from latentwire.models import Adapter
        from latentwire.core_utils import batch_metrics
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import dependency: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
