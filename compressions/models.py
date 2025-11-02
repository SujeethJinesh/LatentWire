"""Compression model architectures.

Includes:
- CrossAttentionCompressor: Learnable query-based compression
- ConvolutionalCompressor: Strided conv-based compression
- WeightedPoolingCompressor: Window-based weighted pooling
- GistCompressor: Gist tokens for prompt compression (Mu et al., 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionCompressor(nn.Module):
    """
    Compress via learned cross-attention queries.
    Most flexible - can select information from anywhere.

    Reference: "Perceiver: General Perception with Iterative Attention" (Jaegle et al., 2021)
    """
    def __init__(self, target_length=64, hidden_dim=4096):
        super().__init__()
        self.target_length = target_length
        self.hidden_dim = hidden_dim

        # Learnable queries that will extract information
        self.queries = nn.Parameter(torch.randn(target_length, hidden_dim) * 0.02)

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=32,  # Llama-3.1 8B has 32 heads
            batch_first=True,
            dropout=0.1
        )

        # Layer norm and residual refinement
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Small MLP for refinement
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Initialize MLP near identity
        with torch.no_grad():
            self.mlp[-1].weight.data *= 0.01
            self.mlp[-1].bias.data.zero_()

    def forward(self, full_embeds):
        """
        Args:
            full_embeds: [batch, seq_len, 4096] - full sequence embeddings
        Returns:
            compressed: [batch, target_length, 4096]
            attn_weights: Attention weights from cross-attention
        """
        batch_size = full_embeds.size(0)

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: queries attend to full sequence
        compressed, attn_weights = self.cross_attn(
            query=queries,
            key=full_embeds,
            value=full_embeds
        )

        # Residual + norm
        compressed = self.norm1(compressed + queries)

        # MLP refinement
        refined = self.mlp(compressed)
        compressed = self.norm2(compressed + refined)

        return compressed, attn_weights


class ConvolutionalCompressor(nn.Module):
    """
    Compress via strided 1D convolution.
    Better for local patterns and maintaining order.

    Reference: "WaveNet: A Generative Model for Raw Audio" (van den Oord et al., 2016)
    """
    def __init__(self, target_length=64, hidden_dim=4096):
        super().__init__()
        self.target_length = target_length
        self.hidden_dim = hidden_dim

        # Calculate stride needed for compression
        # Assuming input ~512 tokens, we need stride ~512/target_length
        self.stride = max(1, 512 // target_length)
        self.kernel_size = self.stride * 2

        # Depthwise convolution (each channel processed separately)
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=hidden_dim  # Depthwise
        )

        # Pointwise convolution to mix information
        self.pointwise = nn.Linear(hidden_dim, hidden_dim)

        # Adaptive pooling to exact target length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_length)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, full_embeds):
        """
        Args:
            full_embeds: [batch, seq_len, 4096]
        Returns:
            compressed: [batch, target_length, 4096]
            None: No attention weights for this architecture
        """
        # Transpose for Conv1d: [batch, seq, hidden] -> [batch, hidden, seq]
        x = full_embeds.transpose(1, 2)

        # Convolutional compression
        compressed = self.conv(x)

        # Ensure exact target length
        compressed = self.adaptive_pool(compressed)

        # Transpose back: [batch, hidden, target_len] -> [batch, target_len, hidden]
        compressed = compressed.transpose(1, 2)

        # Mix information across dimensions
        compressed = self.pointwise(compressed)
        compressed = self.norm(compressed)

        return compressed, None  # No attention weights


class WeightedPoolingCompressor(nn.Module):
    """
    Compress via learned weighted pooling over windows.
    Simplest and most interpretable.

    Reference: "Attention is All You Need" (Vaswani et al., 2017) - adapted pooling
    """
    def __init__(self, target_length=64, hidden_dim=4096, max_seq_len=512):
        super().__init__()
        self.target_length = target_length
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Learn importance weights for each position in each window
        window_size = max_seq_len // target_length + 1
        self.importance_weights = nn.Parameter(
            torch.ones(target_length, window_size) / window_size
        )

        # Optional: learn different projections for each compressed position
        self.position_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(target_length)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, full_embeds):
        """
        Args:
            full_embeds: [batch, seq_len, 4096]
        Returns:
            compressed: [batch, target_length, 4096]
            importance_weights: Learned pooling weights
        """
        batch_size, seq_len, hidden_dim = full_embeds.shape
        window_size = seq_len // self.target_length + 1

        compressed = []

        for i in range(self.target_length):
            # Define window boundaries
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, seq_len)
            window_len = end_idx - start_idx

            if window_len > 0:
                # Extract window
                window = full_embeds[:, start_idx:end_idx, :]

                # Apply importance weights
                weights = F.softmax(
                    self.importance_weights[i, :window_len], dim=0
                ).unsqueeze(0).unsqueeze(-1)

                # Weighted average
                pooled = (window * weights).sum(dim=1)

                # Position-specific projection
                pooled = self.position_projections[i](pooled)
                compressed.append(pooled)

        compressed = torch.stack(compressed, dim=1)
        compressed = self.norm(compressed)

        return compressed, self.importance_weights


class GistCompressor(nn.Module):
    """
    Compress via learnable "gist" tokens inserted into the sequence.

    Reference: "Learning to Compress Prompts with Gist Tokens" (Mu et al., NeurIPS 2023)
    Paper: https://arxiv.org/abs/2304.08467

    Key idea:
    - Insert learnable gist tokens at the beginning of the sequence
    - Gist tokens attend to full sequence and compress information
    - Rest of model only needs to attend to gist tokens
    - Trained via masked instruction finetuning

    Implementation:
    - Learnable gist token embeddings
    - Bidirectional attention (gist tokens can see everything)
    - Position-aware to maintain sequence order
    """
    def __init__(self, target_length=64, hidden_dim=4096):
        super().__init__()
        self.target_length = target_length  # Number of gist tokens
        self.hidden_dim = hidden_dim

        # Learnable gist token embeddings
        # Initialize similar to word embeddings
        self.gist_embeddings = nn.Parameter(
            torch.randn(target_length, hidden_dim) * 0.02
        )

        # Transformer layer for gist token refinement
        # Gist tokens attend to full sequence to compress information
        self.gist_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=32,  # Match Llama architecture
            batch_first=True,
            dropout=0.1
        )

        # Self-attention among gist tokens for refinement
        self.gist_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=32,
            batch_first=True,
            dropout=0.1
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Feed-forward network for gist token refinement
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )

        # Positional encoding for gist tokens
        self.gist_position_embed = nn.Parameter(
            torch.randn(target_length, hidden_dim) * 0.02
        )

    def forward(self, full_embeds):
        """
        Args:
            full_embeds: [batch, seq_len, hidden_dim] - full sequence embeddings
        Returns:
            compressed: [batch, target_length, hidden_dim] - gist tokens
            attn_weights: Attention weights showing what gist tokens attend to
        """
        batch_size = full_embeds.size(0)

        # Initialize gist tokens with learned embeddings + positional encoding
        gist_tokens = self.gist_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        gist_tokens = gist_tokens + self.gist_position_embed.unsqueeze(0)

        # Step 1: Gist tokens attend to full sequence (compression step)
        # Query: gist tokens, Key/Value: full sequence
        compressed, attn_weights = self.gist_attention(
            query=gist_tokens,
            key=full_embeds,
            value=full_embeds
        )
        compressed = self.norm1(compressed + gist_tokens)

        # Step 2: Self-attention among gist tokens (refinement step)
        # This allows gist tokens to exchange information
        refined, _ = self.gist_self_attention(
            query=compressed,
            key=compressed,
            value=compressed
        )
        compressed = self.norm2(compressed + refined)

        # Step 3: Feed-forward network for final refinement
        output = self.ffn(compressed)
        compressed = self.norm3(compressed + output)

        return compressed, attn_weights


def create_compressor(architecture: str, target_length: int = 64, hidden_dim: int = 4096, **kwargs):
    """
    Factory function to create a compressor based on architecture name.

    Args:
        architecture: One of 'cross_attention', 'conv', 'pooling', 'gist'
        target_length: Number of compressed tokens
        hidden_dim: Hidden dimension (4096 for Llama-3.1-8B)
        **kwargs: Additional architecture-specific arguments

    Returns:
        Compressor module
    """
    if architecture == "cross_attention":
        return CrossAttentionCompressor(target_length, hidden_dim)
    elif architecture == "conv":
        return ConvolutionalCompressor(target_length, hidden_dim)
    elif architecture == "pooling":
        return WeightedPoolingCompressor(target_length, hidden_dim, **kwargs)
    elif architecture == "gist":
        return GistCompressor(target_length, hidden_dim)
    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: cross_attention, conv, pooling, gist"
        )
