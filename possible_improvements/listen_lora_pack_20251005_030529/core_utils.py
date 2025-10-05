
# core_utils.py (drop-in)
# Shared utilities for PEFT-LoRA, multi-depth adapters, coprocessor, and anchor helpers.
import torch
import torch.nn as nn

# ---------- PEFT (LoRA) helpers ----------
try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    def get_peft_model(model, cfg): return model

def apply_listen_lora(model, target_modules, r=8, alpha=16, dropout=0.05):
    if LoraConfig is None:
        return model
    peft_cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, task_type="CAUSAL_LM"
    )
    return get_peft_model(model, peft_cfg)

# ---------- Anchor utilities ----------
def build_anchor_mask(attn_mask: torch.Tensor, is_anchor: torch.Tensor, window: int = 64) -> torch.Tensor:
    B, T = attn_mask.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=attn_mask.device))
    out = causal[None, :, :].repeat(B, 1, 1)  # [B,T,T]
    for b in range(B):
        for t in range(T):
            if not is_anchor[b, t]:
                left = max(0, t - window + 1)
                out[b, t, :left] = False
            else:
                prev_anchors = (is_anchor[b, :t] == 1)
                out[b, t, :t] = prev_anchors
    out &= attn_mask[:, None, :].bool()
    out &= attn_mask[:, :, None].bool()
    return out

def select_anchor_kv(past_key_values, is_anchor: torch.Tensor):
    new_pkv = []
    for k, v in past_key_values:
        B,H,T,D = k.shape
        mask = is_anchor[:, None, :, None].expand(B,H,T,D)
        k_new = k[mask].view(B,H,-1,D)
        v_new = v[mask].view(B,H,-1,D)
        new_pkv.append((k_new, v_new))
    return tuple(new_pkv)

# ---------- Multi-depth adapter block ----------
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
        B,T,D = h.shape
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

# ---------- Latent coprocessor (KV augmentation) ----------
class LatentCoprocessor(nn.Module):
    def __init__(self, d_z: int, d_model: int, n_layers: int, heads_per_layer, width=256, z_dropout=0.1, kv_scale=0.8):
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
        B,M,Dz = z.shape
        z_pool = z.mean(dim=1)
        h = self.dropout(self.z_proj(z_pool))
        K_layers, V_layers = [], []
        for to_k, to_v in zip(self.to_k, self.to_v):
            K_layers.append(self.kv_scale * to_k(h))  # [B,H*Dh]
            V_layers.append(self.kv_scale * to_v(h))
        return K_layers, V_layers

    @torch.no_grad()
    def augment_past_kv(self, past_key_values, K_layers, V_layers, heads_per_layer, seq_dim=2):
        new_pkv = []
        for (k, v), Kflat, Vflat, H in zip(past_key_values, K_layers, V_layers, heads_per_layer):
            B,Hc,S,Dh = k.shape
            assert Hc == H, f"heads mismatch: cache={Hc} vs cfg={H}"
            Kpos = Kflat.view(B, H, Dh)[:, :, None, :]
            Vpos = Vflat.view(B, H, Dh)[:, :, None, :]
            k_aug = torch.cat([k, Kpos], dim=seq_dim)
            v_aug = torch.cat([v, Vpos], dim=seq_dim)
            new_pkv.append((k_aug, v_aug))
        return tuple(new_pkv)
