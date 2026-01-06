# telepathy/latent_bridge_v3.py
"""
Latent Bridge V3: Manifold Anchoring

Phase 2 failed because contrastive learning pushed vectors into "dead zones" -
mathematically unique but semantically meaningless regions of embedding space.

V3 Fixes:
1. LearnableNormalizer: Unfrozen affine parameters for fine-tuning
2. Output Clamping: Prevents value explosion (10^100 garbage)
3. Architecture supports Batch Anchor Loss (implemented in training)
"""
import torch
import torch.nn as nn


class LearnableNormalizer(nn.Module):
    """
    V3: Learnable Statistical Normalizer.

    Unlike V1/V2 which used frozen buffers, this version uses nn.Parameter
    so the model can fine-tune the scale/shift during training.

    The initial values come from calibration, but gradients flow through.
    """
    def __init__(self, stats_path: str, hidden_dim: int = 4096):
        super().__init__()

        try:
            stats = torch.load(stats_path, map_location="cpu", weights_only=True)

            # Initialize from calibration stats but make learnable
            l_mean = stats["l_mean"].float()
            l_std = stats["l_std"].float()
            m_mean = stats["m_mean"].float()
            m_std = stats["m_std"].float()

            # Learnable parameters (initialized from calibration)
            self.l_mean = nn.Parameter(l_mean)
            self.l_std = nn.Parameter(l_std)
            self.m_mean = nn.Parameter(m_mean)
            self.m_std = nn.Parameter(m_std)

            # Learnable scale factor for fine-tuning amplitude
            self.scale = nn.Parameter(torch.ones(1))

            print(f"[LearnableNormalizer] Loaded stats from {stats_path}")
            print(f"  Source mean norm: {l_mean.norm().item():.4f}")
            print(f"  Target mean norm: {m_mean.norm().item():.4f}")
            print(f"  Parameters are LEARNABLE (gradients enabled)")

        except Exception as e:
            print(f"WARNING: Could not load stats ({e}). Using identity init.")
            self.l_mean = nn.Parameter(torch.zeros(hidden_dim))
            self.l_std = nn.Parameter(torch.ones(hidden_dim))
            self.m_mean = nn.Parameter(torch.zeros(hidden_dim))
            self.m_std = nn.Parameter(torch.ones(hidden_dim))
            self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] - Source hidden states
        Returns:
            [B, T, D] - Normalized hidden states
        """
        # 1. Whiten (Remove Source stats)
        x = (x.float() - self.l_mean) / (self.l_std.abs() + 1e-8)

        # 2. Color (Apply Target stats)
        x = (x * self.m_std.abs()) + self.m_mean

        # 3. Learnable scale adjustment
        x = x * self.scale

        return x.to(torch.bfloat16)


class ClampedPerceiverResampler(nn.Module):
    """
    V3: Perceiver Resampler with Output Clamping.

    Prevents value explosion by applying tanh + learnable scale on output.
    This keeps soft tokens in a "safe" numerical range.
    """
    def __init__(
        self,
        src_dim: int,
        tgt_dim: int,
        num_latents: int = 128,
        heads: int = 8,
        depth: int = 4,
        output_scale: float = 1.0
    ):
        super().__init__()
        self.num_latents = num_latents
        self.tgt_dim = tgt_dim

        # Learned latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, tgt_dim) * 0.02)

        # Project source dim to target dim
        self.input_proj = (
            nn.Linear(src_dim, tgt_dim)
            if src_dim != tgt_dim
            else nn.Identity()
        )

        # Perceiver layers
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

        # Output normalization and clamping
        self.output_ln = nn.LayerNorm(tgt_dim)

        # Learnable output scale (initialized to match target embedding RMS)
        self.output_scale = nn.Parameter(torch.tensor(output_scale))

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
            [B, K, D_tgt] - Clamped soft tokens
        """
        B = src_hidden.shape[0]

        # Project source to target dimension
        keys = self.input_proj(src_hidden)

        # Expand latent queries for batch
        x = self.latents.unsqueeze(0).expand(B, -1, -1)

        # Invert mask for PyTorch MHA
        key_padding_mask = ~src_mask.bool() if src_mask is not None else None

        for layer in self.layers:
            # Cross Attention
            attn_out, _ = layer["cross_attn"](
                query=layer["ln1"](x),
                key=keys,
                value=keys,
                key_padding_mask=key_padding_mask
            )
            x = x + attn_out

            # Self Attention
            attn_out, _ = layer["self_attn"](
                query=layer["ln2"](x),
                key=layer["ln2"](x),
                value=layer["ln2"](x)
            )
            x = x + attn_out

            # FFN
            x = x + layer["ffn"](layer["ln3"](x))

        # OUTPUT CLAMPING: Prevent value explosion
        # 1. Layer norm to stabilize
        x = self.output_ln(x)

        # 2. Tanh to bound values to [-1, 1]
        x = torch.tanh(x)

        # 3. Scale to target embedding magnitude
        x = x * self.output_scale.abs()

        return x


class LatentBridgeV3(nn.Module):
    """
    V3 Latent Bridge with Manifold Anchoring support.

    Changes from V2:
    - LearnableNormalizer (unfrozen parameters)
    - ClampedPerceiverResampler (prevents explosion)
    - get_target_rms() helper for anchor loss
    """
    def __init__(
        self,
        args,
        src_dim: int,
        tgt_dim: int,
        target_rms: float = 0.1
    ):
        super().__init__()

        self.normalizer = LearnableNormalizer(args.stats_path, src_dim)
        self.resampler = ClampedPerceiverResampler(
            src_dim,
            tgt_dim,
            args.soft_tokens,
            args.heads,
            args.depth,
            output_scale=target_rms  # Initialize to target embedding RMS
        )

        # Store target RMS for reference
        self.register_buffer("target_rms", torch.tensor(target_rms))

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
            [B, K, D_tgt] - Clamped soft tokens
        """
        normed = self.normalizer(src_hidden)
        clamped = self.resampler(normed, src_mask)
        return clamped

    def get_output_stats(self, soft_tokens: torch.Tensor) -> dict:
        """Debug helper: Get statistics of soft token outputs."""
        return {
            "mean": soft_tokens.mean().item(),
            "std": soft_tokens.std().item(),
            "min": soft_tokens.min().item(),
            "max": soft_tokens.max().item(),
            "rms": soft_tokens.pow(2).mean().sqrt().item(),
        }
