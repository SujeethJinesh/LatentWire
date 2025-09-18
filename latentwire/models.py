from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class STQueryEncoder(nn.Module):
    """
    Sentence-Transformer based encoder (MiniLM backbone) with learned cross-attention
    queries to produce M slots; projection to d_z.
    By default, backbone is frozen; we train the query bank and projection.
    """
    def __init__(self, hf_id: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_enc_tokens: int = 512, m_slots: int = 32, d_z: int = 256,
                 freeze_backbone: bool = True, slot_sinusoid: bool = True):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.backbone = AutoModel.from_pretrained(hf_id)
        self.hidden = self.backbone.config.hidden_size
        self.max_len = max_enc_tokens
        self.m = m_slots
        self.d_z = d_z

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Learned queries
        self.query = nn.Parameter(torch.randn(self.m, self.hidden) * 0.02)

        # Cross-attention pooler (single-head for efficiency)
        self.q_proj = nn.Linear(self.hidden, self.hidden, bias=False)
        self.k_proj = nn.Linear(self.hidden, self.hidden, bias=False)
        self.v_proj = nn.Linear(self.hidden, self.hidden, bias=False)
        self.o_proj = nn.Linear(self.hidden, self.hidden, bias=False)

        # Projection to latent space
        self.to_latent = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, d_z),
        )

        self.slot_sinusoid = slot_sinusoid
        if slot_sinusoid:
            self.register_buffer("pos", self._build_sinusoid(self.m, self.hidden), persistent=False)

    def _build_sinusoid(self, L: int, D: int) -> torch.Tensor:
        pe = torch.zeros(L, D)
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def encode_text(self, text: str, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns token_embeddings [1, T, H] and attention mask [1, T]
        """
        toks = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_len)
        if device is not None:
            toks = {k: v.to(device) for k, v in toks.items()}
        out = self.backbone(**toks, output_hidden_states=False)
        return out.last_hidden_state, toks["attention_mask"]

    def forward(self, text: str) -> torch.Tensor:
        device = next(self.parameters()).device
        x, attn_mask = self.encode_text(text, device=device)   # [1, T, H], [1, T]
        Q = self.query
        if self.slot_sinusoid:
            Q = Q + self.pos.to(Q.device)
        Q = self.q_proj(Q).unsqueeze(0)                        # [1, M, H]
        K = self.k_proj(x)                                     # [1, T, H]
        V = self.v_proj(x)                                     # [1, T, H]

        attn = torch.softmax((Q @ K.transpose(-1, -2)) / math.sqrt(K.size(-1)), dim=-1)  # [1, M, T]
        attn = attn * attn_mask.unsqueeze(1) + 1e-6
        attn = attn / attn.sum(dim=-1, keepdim=True)

        pooled = attn @ V                                      # [1, M, H]
        pooled = self.o_proj(pooled)                           # [1, M, H]
        z = self.to_latent(pooled)                             # [1, M, d_z]
        return z.squeeze(0)                                    # [M, d_z]

class Adapter(nn.Module):
    """LayerNorm -> Linear -> tanh clip to map from d_z -> d_model"""
    def __init__(self, d_z: int, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_z)
        self.proj = nn.Linear(d_z, d_model)
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm(z))
        return self.tanh(x)

@torch.no_grad()
def embedding_rms(embedding: torch.nn.Embedding) -> float:
    w = embedding.weight
    return float(torch.sqrt(torch.mean(w.float() ** 2)).item())
