
# sample_code/latent_coprocessor.py
"""
Latent Coprocessor: writes small {K,V} deltas into the LLM cache per layer.

Integration points:
- Call coprocessor.augment_past_kv(...) before each forward pass (train & eval).
- Disable adapters for teacher KD to get clean text distributions.
"""
from typing import List, Tuple
import torch
import torch.nn as nn

KV = Tuple[torch.Tensor, torch.Tensor]  # (K, V)

class LatentCoprocessor(nn.Module):
    def __init__(self, d_z: int, d_model: int, n_layers: int, heads_per_layer: List[int],
                 width: int = 256, z_dropout: float = 0.1, kv_scale: float = 1.0):
        super().__init__()
        self.n_layers = n_layers
        self.heads_per_layer = heads_per_layer
        self.kv_scale = kv_scale
        self.z_proj = nn.Sequential(
            nn.LayerNorm(d_z),
            nn.Linear(d_z, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.to_k = nn.ModuleList()
        self.to_v = nn.ModuleList()
        for h in heads_per_layer:
            d_head = d_model // h
            self.to_k.append(nn.Linear(width, h * d_head))
            self.to_v.append(nn.Linear(width, h * d_head))
        self.dropout = nn.Dropout(z_dropout)

    def forward(self, z: torch.Tensor):
        """
        z: [B, M, d_z] latent tokens
        returns per-layer K,V blocks shaped [B, 1, H*D_head]
        """
        B, M, Dz = z.shape
        z_pool = z.mean(dim=1)  # [B, d_z]
        h = self.dropout(self.z_proj(z_pool))  # [B, width]
        K_layers, V_layers = [], []
        for to_k, to_v in zip(self.to_k, self.to_v):
            K_layers.append(self.kv_scale * to_k(h))  # [B, H*D_head]
            V_layers.append(self.kv_scale * to_v(h))
        return K_layers, V_layers

    @torch.no_grad()
    def augment_past_kv(self, past_key_values, K_layers, V_layers, heads_per_layer: List[int], seq_dim: int = 2):
        """
        Append one 'virtual' position to each layer's KV cache.
        past_key_values: list[(k,v)] with shapes k,v=[B,H,S,Dh]
        """
        new_pkv = []
        for (k, v), Kflat, Vflat, H in zip(past_key_values, K_layers, V_layers, heads_per_layer):
            B, Hc, S, Dh = k.shape
            assert Hc == H, f"heads mismatch: cache={Hc} vs cfg={H}"
            Kpos = Kflat.view(B, H, Dh)[:, :, None, :]  # [B,H,1,Dh]
            Vpos = Vflat.view(B, H, Dh)[:, :, None, :]
            k_aug = torch.cat([k, Kpos], dim=seq_dim)   # [B,H,S+1,Dh]
            v_aug = torch.cat([v, Vpos], dim=seq_dim)
            new_pkv.append((k_aug, v_aug))
        return tuple(new_pkv)
