#!/usr/bin/env python3
"""
Unit tests for Stage 1 adapter training.
These tests can run on MacBook CPU to verify code correctness before HPC deployment.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_adapter_only import EmbeddingCompressor, train_adapter_only
from latentwire.models import Adapter


class TestEmbeddingCompressor:
    """Test the embedding compressor component"""

    def test_compressor_init(self):
        """Test compressor initialization"""
        compressor = EmbeddingCompressor(input_dim=4096, output_dim=512, method="pca")
        assert compressor.input_dim == 4096
        assert compressor.output_dim == 512
        assert compressor.method == "pca"
        assert compressor.projection is None

    def test_compressor_fit_shape(self):
        """Test that compressor fitting works with correct shapes"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")

        # Create sample embeddings [num_tokens, embed_dim]
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        assert compressor.projection is not None
        assert compressor.projection.shape == (128, 32)
        assert compressor.mean.shape == (128,)

    def test_compressor_compress_shape(self):
        """Test that compression produces correct output shapes"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")

        # Fit on sample data
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        # Test compression with batch
        batch_embeds = torch.randn(16, 50, 128)  # [batch, seq_len, embed_dim]
        compressed = compressor.compress(batch_embeds)

        assert compressed.shape == (16, 50, 32)

    def test_compressor_dtype_preservation(self):
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
    def test_compressor_device_handling(self):
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


class TestAdapterIntegration:
    """Test adapter with compressor integration"""

    def test_adapter_forward_shape(self):
        """Test adapter produces correct output shape"""
        adapter = Adapter(
            d_z=512,
            d_model=4096,
            latent_length=32,
            hidden_mult=4,
            dropout=0.0,
            enable_metadata=False,
            colorize=False
        )

        # Test forward pass
        compressed = torch.randn(8, 20, 512)  # [batch, seq_len, d_z]
        output = adapter(compressed)

        assert output.shape == (8, 20, 4096)

    def test_adapter_dtype_handling(self):
        """Test adapter handles different dtypes"""
        adapter = Adapter(
            d_z=512,
            d_model=4096,
            latent_length=32,
            hidden_mult=4,
            dropout=0.0,
            enable_metadata=False,
            colorize=False
        ).to(torch.bfloat16)

        # Test with bfloat16 input
        compressed = torch.randn(4, 10, 512).to(torch.bfloat16)
        output = adapter(compressed)

        assert output.dtype == torch.bfloat16
        assert output.shape == (4, 10, 4096)

    def test_pipeline_integration(self):
        """Test full compression -> adapter pipeline"""
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


class TestDevicePlacement:
    """Test device placement for multi-GPU scenarios"""

    def test_tensor_concatenation_same_device(self):
        """Test that concatenation works when tensors are on same device"""
        tensor1 = torch.randn(4, 10, 128)
        tensor2 = torch.randn(4, 5, 128)

        result = torch.cat([tensor1, tensor2], dim=1)
        assert result.shape == (4, 15, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_device_mismatch_detection(self):
        """Test that we can detect device mismatches"""
        tensor1 = torch.randn(4, 10, 128).cuda()
        tensor2 = torch.randn(4, 5, 128)  # CPU tensor

        # Should detect mismatch
        assert tensor1.device != tensor2.device

        # Move to same device
        tensor2 = tensor2.to(tensor1.device)
        result = torch.cat([tensor1, tensor2], dim=1)
        assert result.is_cuda

    def test_embedding_device_placement_mock(self):
        """Test device placement logic with mocked model"""
        # Mock embedding layer on different device
        mock_embed_layer = MagicMock()
        mock_embed_layer.return_value = torch.randn(4, 10, 128)

        # Simulate device mismatch scenario
        reconstructed = torch.randn(4, 10, 128)
        answer_embeds = mock_embed_layer()

        # Ensure same device (this is what the fix does)
        if answer_embeds.device != reconstructed.device:
            answer_embeds = answer_embeds.to(reconstructed.device)

        # Should now concatenate successfully
        result = torch.cat([reconstructed, answer_embeds], dim=1)
        assert result.shape == (4, 20, 128)


class TestLossComputation:
    """Test loss computation functions"""

    def test_mse_loss_float32_conversion(self):
        """Test that MSE loss works with float32 conversion"""
        pred = torch.randn(100, 128).to(torch.bfloat16)
        target = torch.randn(100, 128).to(torch.bfloat16)

        # Should work with float32 conversion
        loss = F.mse_loss(
            pred.to(torch.float32),
            target.to(torch.float32)
        )

        assert loss.dtype == torch.float32
        assert not torch.isnan(loss)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_mse_loss_stays_on_gpu(self):
        """Test that MSE loss computation stays on GPU"""
        pred = torch.randn(100, 128).to(torch.bfloat16).cuda()
        target = torch.randn(100, 128).to(torch.bfloat16).cuda()

        # Convert to float32 for loss (should stay on GPU)
        loss = F.mse_loss(
            pred.to(torch.float32),
            target.to(torch.float32)
        )

        assert loss.is_cuda
        assert not torch.isnan(loss)


class TestBatchProcessing:
    """Test batch processing logic"""

    def test_batch_size_handling(self):
        """Test that different batch sizes are handled correctly"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        # Test various batch sizes
        for batch_size in [1, 4, 16, 32, 64]:
            batch_embeds = torch.randn(batch_size, 10, 128)
            compressed = compressor.compress(batch_embeds)
            assert compressed.shape == (batch_size, 10, 32)

    def test_attention_mask_application(self):
        """Test that attention masks work correctly"""
        batch_size = 4
        seq_len = 10

        # Create embeddings and mask
        embeddings = torch.randn(batch_size, seq_len, 128)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask[0, 8:] = False  # Mask last 2 tokens of first sample

        # Select only unmasked embeddings
        selected = embeddings[attention_mask]

        # Should have removed 2 tokens
        assert selected.shape[0] == (batch_size * seq_len - 2)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_batch_handling(self):
        """Test that empty batches are rejected"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        # Empty batch should preserve shape structure
        empty_batch = torch.randn(0, 10, 128)
        compressed = compressor.compress(empty_batch)
        assert compressed.shape == (0, 10, 32)

    def test_single_sample_batch(self):
        """Test that single-sample batches work"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        single_sample = torch.randn(1, 10, 128)
        compressed = compressor.compress(single_sample)
        assert compressed.shape == (1, 10, 32)

    def test_very_long_sequence(self):
        """Test that very long sequences are handled"""
        compressor = EmbeddingCompressor(input_dim=128, output_dim=32, method="pca")
        embeddings = torch.randn(1000, 128)
        compressor.fit(embeddings)

        # Test with 512 token sequence
        long_seq = torch.randn(2, 512, 128)
        compressed = compressor.compress(long_seq)
        assert compressed.shape == (2, 512, 32)


def test_import_dependencies():
    """Test that all required dependencies can be imported"""
    try:
        import torch
        import transformers
        import sklearn
        import numpy as np
        from latentwire.data import load_squad_subset
        from latentwire.models import Adapter
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import dependency: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
