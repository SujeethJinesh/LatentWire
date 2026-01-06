# telepathy/latent_bridge.py
"""
Latent Bridge: Neural adapter for cross-model latent communication.
Enables Llama 3.1 8B to inject hidden states directly into Mistral 0.3 7B.

Solves four physical incompatibilities:
- Magnitude Shock: StatisticalNormalizer (affine transformation)
- Vocab Density: PerceiverResampler (cross-attention compression)
- RoPE Geometry: Implicit de-rotation via learned queries
- KV Cache Gap: Handled by prefix priming in training loop
"""
import torch
import torch.nn as nn


class StatisticalNormalizer(nn.Module):
    """
    Solves Challenge A: Magnitude Shock.
    Applies affine transformation to match Source distribution to Target distribution.
    Parameters are frozen buffers (calculated in Phase 1 calibration).

    Llama values are ~±20; Mistral embeddings are ~±100.
    Direct injection causes gradient explosion without this normalization.
    """
    def __init__(self, stats_path: str):
        super().__init__()
        try:
            stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            self.register_buffer("l_mean", stats["l_mean"].float())
            self.register_buffer("l_std", stats["l_std"].float())
            self.register_buffer("m_mean", stats["m_mean"].float())
            self.register_buffer("m_std", stats["m_std"].float())
            print(f"[StatisticalNormalizer] Loaded stats from {stats_path}")
            print(f"  Source mean norm: {self.l_mean.norm().item():.4f}")
            print(f"  Target mean norm: {self.m_mean.norm().item():.4f}")
        except Exception as e:
            print(f"WARNING: Could not load stats ({e}). Using Identity transform.")
            self.register_buffer("l_mean", torch.tensor(0.0))
            self.register_buffer("l_std", torch.tensor(1.0))
            self.register_buffer("m_mean", torch.tensor(0.0))
            self.register_buffer("m_std", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] - Source hidden states
        Returns:
            [B, T, D] - Normalized hidden states matching target distribution
        """
        # 1. Whiten (Remove Source stats)
        x = (x.float() - self.l_mean) / (self.l_std + 1e-8)
        # 2. Color (Apply Target stats)
        x = (x * self.m_std) + self.m_mean
        return x.to(torch.bfloat16)


class PerceiverResampler(nn.Module):
    """
    Solves Challenge B (Vocab Density) & C (RoPE Mismatch).

    Compresses variable-length source sequence (T tokens) into fixed-length
    latent sequence (K soft tokens) via cross-attention.

    - Llama (128k vocab) may encode "bioluminescence" as 1 vector
    - Mistral (32k vocab) expects 3 vectors ("bio", "lum", "inescence")
    - The resampler handles this vocab density mismatch

    RoPE de-rotation is implicit: learned queries attend to source keys/values
    without enforcing source positional encoding on output.
    """
    def __init__(
        self,
        src_dim: int,
        tgt_dim: int,
        num_latents: int = 64,
        heads: int = 8,
        depth: int = 4
    ):
        super().__init__()
        self.num_latents = num_latents
        self.tgt_dim = tgt_dim

        # Learned latent queries (the "soft tokens")
        self.latents = nn.Parameter(torch.randn(num_latents, tgt_dim) * 0.02)

        # Project source dim to target dim if needed
        self.input_proj = (
            nn.Linear(src_dim, tgt_dim)
            if src_dim != tgt_dim
            else nn.Identity()
        )

        # Perceiver layers: cross-attn -> self-attn -> FFN
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(tgt_dim, heads, batch_first=True),
                "ln1": nn.LayerNorm(tgt_dim),
                "self_attn": nn.MultiheadAttention(tgt_dim, heads, batch_first=True),
                "ln2": nn.LayerNorm(tgt_dim),
                "ffn": nn.Sequential(
                    nn.Linear(tgt_dim, 4 * tgt_dim),
                    nn.GELU(),
                    nn.Linear(4 * tgt_dim, tgt_dim)
                ),
                "ln3": nn.LayerNorm(tgt_dim)
            }) for _ in range(depth)
        ])

    def forward(
        self,
        src_hidden: torch.Tensor,
        src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            src_hidden: [B, T, D_src] - Source model hidden states
            src_mask: [B, T] - Attention mask (1 = attend, 0 = ignore)
        Returns:
            [B, K, D_tgt] - Compressed soft tokens
        """
        B = src_hidden.shape[0]

        # Project source to target dimension
        keys = self.input_proj(src_hidden)

        # Expand latent queries for batch
        x = self.latents.unsqueeze(0).expand(B, -1, -1)

        # Invert mask for PyTorch MHA (True = Ignore)
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None

        for layer in self.layers:
            # Cross Attention: Read from source hidden states
            attn_out, _ = layer["cross_attn"](
                query=layer["ln1"](x),
                key=keys,
                value=keys,
                key_padding_mask=key_padding_mask
            )
            x = x + attn_out

            # Self Attention: Mix information across latent tokens
            attn_out, _ = layer["self_attn"](
                query=layer["ln2"](x),
                key=layer["ln2"](x),
                value=layer["ln2"](x)
            )
            x = x + attn_out

            # FFN: Nonlinear transformation
            x = x + layer["ffn"](layer["ln3"](x))

        return x


class LatentBridge(nn.Module):
    """
    Complete Latent Bridge module.

    Combines StatisticalNormalizer and PerceiverResampler to transform
    source model hidden states into soft tokens consumable by target model.
    """
    def __init__(
        self,
        args,
        src_dim: int,
        tgt_dim: int
    ):
        super().__init__()
        self.normalizer = StatisticalNormalizer(args.stats_path)
        self.resampler = PerceiverResampler(
            src_dim,
            tgt_dim,
            args.soft_tokens,
            args.heads,
            args.depth
        )

    def forward(
        self,
        src_hidden: torch.Tensor,
        src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            src_hidden: [B, T, D_src] - Source model hidden states
            src_mask: [B, T] - Attention mask
        Returns:
            [B, K, D_tgt] - Soft tokens for target model
        """
        normed = self.normalizer(src_hidden)
        compressed = self.resampler(normed, src_mask)
        return compressed
