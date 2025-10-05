
# sample_code/anchor_tokens.py
"""
Anchor masking utilities:
- build_anchor_mask: modifies attention so anchors summarize segments.
- select_anchor_kv: drops non-anchor kv from cache.
"""
import torch

def build_anchor_mask(attn_mask: torch.Tensor, is_anchor: torch.Tensor) -> torch.Tensor:
    """
    attn_mask: [B, T] (bool or 0/1), is_anchor: [B, T] bool
    Returns anchor-aware causal mask [B, T, T].
    """
    B, T = attn_mask.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=attn_mask.device))
    out = causal[None, :, :].repeat(B, 1, 1)  # [B,T,T]
    W = 64  # local segment window
    for b in range(B):
        for t in range(T):
            if not is_anchor[b, t]:
                left = max(0, t - W + 1)
                out[b, t, :left] = False
            else:
                prev_anchors = (is_anchor[b, :t] == 1)
                out[b, t, :t] = prev_anchors
    out &= attn_mask[:, None, :].bool()
    out &= attn_mask[:, :, None].bool()
    return out

def select_anchor_kv(past_key_values, is_anchor: torch.Tensor):
    """
    Keep only kv entries corresponding to anchors (during inference).
    """
    new_pkv = []
    for k, v in past_key_values:
        B,H,T,D = k.shape
        mask = is_anchor[:, None, :, None].expand(B,H,T,D)
        k_new = k[mask].view(B,H,-1,D)
        v_new = v[mask].view(B,H,-1,D)
        new_pkv.append((k_new, v_new))
    return tuple(new_pkv)
