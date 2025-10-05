
# sample_code/multi_depth_adapters.py
"""
Multi-depth latent adapters (IAA-style).
"""
import torch
import torch.nn as nn

class LatentAdapterBlock(nn.Module):
    def __init__(self, d_model: int, d_z: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_z, d_model)
        self.v_proj = nn.Linear(d_z, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _split_heads(self, x, B):
        D = x.shape[-1]; H = self.n_heads; d_h = D // H
        return x.view(B, -1, H, d_h).transpose(1,2)  # [B,H,T,d_h]

    def forward(self, h: torch.Tensor, z: torch.Tensor):
        """
        h: [B,T,D] hidden; z: [B,M,d_z] latent
        """
        B,T,D = h.shape; H = self.n_heads
        h_ = self.ln(h)
        Q = self._split_heads(self.q_proj(h_), B)      # [B,H,T,d_h]
        K = self._split_heads(self.k_proj(z), B)       # [B,H,M,d_h]
        V = self._split_heads(self.v_proj(z), B)       # [B,H,M,d_h]
        attn = torch.matmul(Q, K.transpose(-2,-1)) / (Q.shape[-1] ** 0.5)
        w = torch.softmax(attn, dim=-1)
        w = self.dropout(w)
        ctx = torch.matmul(w, V)                       # [B,H,T,d_h]
        ctx = ctx.transpose(1,2).contiguous().view(B,T,D)
        out = self.out(ctx)
        return h + torch.tanh(self.alpha) * out
