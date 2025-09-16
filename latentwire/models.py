# latentwire/models.py
import math
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Sequence

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from latentwire.common import clean_pred  # global cleaner used across the project

# ---------------------------
# Small helpers
# ---------------------------

STOP_STRINGS = [
    "<|eot_id|>", "<|im_end|>", "</s>",  # common chat EOS-ish markers
    "<|system|>", "<|user|>", "<|assistant|>",  # guardrails: if the model starts a chat block, cut it
    "\n\n\n", "\n\nAssistant:", "\nAssistant:"
]

def _local_clean_pred(s: str) -> str:
    """
    Defensive cleaner for short generations; uses STOP_STRINGS as hard stops and
    trims role echoes. We still rely on latentwire.common.clean_pred in eval.py,
    but keep a local version for LMWrapper utilities.
    """
    if not s:
        return s
    for ss in STOP_STRINGS:
        idx = s.find(ss)
        if idx >= 0:
            s = s[:idx]
    s = re.sub(r"^\s*(assistant|assistant:|Assistant:)\s*", "", s)
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    s = lines[0].strip(" \t\r\n.:;,'\"-–—")
    return s


# ---------------------------
# Tokenizers & Encoders
# ---------------------------

class ByteTokenizer:
    def __init__(self, max_bytes: int = 512):
        self.max_bytes = max_bytes
    def encode(self, text: str) -> torch.Tensor:
        b = text.encode("utf-8")[: self.max_bytes]
        return torch.tensor(list(b), dtype=torch.long)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)

class ByteEncoder(nn.Module):
    def __init__(self, d_z: int = 256, n_layers: int = 6, n_heads: int = 8, ff_mult: int = 4, max_len: int = 2048):
        super().__init__()
        assert d_z % n_heads == 0, "d_z must be divisible by n_heads"
        self.byte_emb = nn.Embedding(256, d_z)
        self.pos = PositionalEncoding(d_z, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_z, nhead=n_heads, dim_feedforward=ff_mult*d_z, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_z)
    def forward(self, byte_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if byte_ids.numel() == 0:
            return torch.zeros((0, 0, self.ln.normalized_shape[0]), device=byte_ids.device, dtype=torch.float32)
        x = self.byte_emb(byte_ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=None)
        x = self.ln(x)
        return x

class LatentPooler(nn.Module):
    def __init__(self, d_z: int = 256, latent_len: int = 8, n_heads: int = 8):
        super().__init__()
        assert d_z % n_heads == 0
        self.latent = nn.Parameter(torch.randn(latent_len, d_z) / math.sqrt(d_z))
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_z, num_heads=n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_z),
            nn.Linear(d_z, 2*d_z),
            nn.GELU(),
            nn.Linear(2*d_z, d_z),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(d_z),
            nn.Linear(d_z, d_z),
            nn.Sigmoid(),
        )
    def forward(self, byte_feats: torch.Tensor) -> torch.Tensor:
        B = byte_feats.size(0)
        q = self.latent.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.cross_attn(q, byte_feats, byte_feats, need_weights=False)
        out = out + q
        gate = self.gate(out)
        out = out * gate + q * (1 - gate)
        out = out + self.ff(out)
        return out

class InterlinguaEncoder(nn.Module):
    def __init__(
        self,
        d_z: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_mult: int = 4,
        latent_len: int = 8,
        latent_shared_len: Optional[int] = None,
        latent_private_len: int = 0,
        model_keys: Sequence[str] = ("llama", "qwen"),
    ):
        super().__init__()
        self.model_keys = list(model_keys)
        self.backbone = ByteEncoder(d_z=d_z, n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult)

        if latent_shared_len is None:
            if latent_private_len > 0:
                latent_shared_len = max(latent_len - latent_private_len * len(self.model_keys), 0)
            else:
                latent_shared_len = latent_len
        self.latent_shared_len = int(latent_shared_len)
        self.latent_private_len = int(latent_private_len)
        self.total_latent_len = int(self.latent_shared_len + self.latent_private_len * len(self.model_keys))

        if self.latent_shared_len > 0:
            self.shared_pooler = LatentPooler(d_z=d_z, latent_len=self.latent_shared_len, n_heads=n_heads)
        else:
            self.shared_pooler = None

        if self.latent_private_len > 0:
            self.private_poolers = nn.ModuleDict(
                {
                    key: LatentPooler(d_z=d_z, latent_len=self.latent_private_len, n_heads=n_heads)
                    for key in self.model_keys
                }
            )
        else:
            self.private_poolers = nn.ModuleDict()

    def forward(self, byte_ids: torch.Tensor, return_components: bool = False) -> torch.Tensor:
        feats = self.backbone(byte_ids)

        if self.latent_shared_len > 0:
            shared = self.shared_pooler(feats)
        else:
            shared = feats.new_zeros(feats.size(0), 0, feats.size(-1))

        private: Dict[str, torch.Tensor] = {}
        for key in self.model_keys:
            if self.latent_private_len > 0:
                private[key] = self.private_poolers[key](feats)
            else:
                private[key] = feats.new_zeros(feats.size(0), 0, feats.size(-1))

        if return_components:
            return {"shared": shared, "private": private}

        parts = [shared] + [private[key] for key in self.model_keys]
        return torch.cat(parts, dim=1)

class SimpleEncoder(nn.Module):
    """
    Frozen SentenceTransformer -> linear projection -> learned queries.
    """
    def __init__(self, d_z: int = 256, latent_len: int = 8, backbone: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError("sentence-transformers is required for SimpleEncoder. Install with: pip install sentence-transformers") from e
        self.st = SentenceTransformer(backbone)
        try:
            for p in self.st._first_module().parameters():
                p.requires_grad_(False)
        except Exception:
            for p in self.st.parameters():
                p.requires_grad_(False)
        try:
            self.st_dim = self.st.get_sentence_embedding_dimension()
        except Exception:
            self.st_dim = 384
        self.proj = nn.Linear(self.st_dim, d_z)
        self.queries = nn.Parameter(torch.randn(latent_len, d_z) / math.sqrt(d_z))
        self.ln = nn.LayerNorm(d_z)

    def forward(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            emb = self.st.encode(texts, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=False)
        emb = emb.detach().clone()
        z0 = self.proj(emb)         # [B, d_z]
        B = z0.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, M, d_z]
        z = self.ln(q + z0.unsqueeze(1))
        return z  # [B, M, d_z]

class Adapter(nn.Module):
    """
    Maps Z (d_z) -> model embedding space (d_model).
    Includes a learnable scalar 'scale' that we regularize to ~1.0 in train.py.
    """
    def __init__(self, d_z: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_z), nn.Linear(d_z, d_model))
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z) * self.scale
        # Keep a soft bound to avoid blowing up logits; calibration will adjust RMS
        return torch.tanh(x / 3.0) * 3.0


# ---------------------------
# LM Wrapper
# ---------------------------

@dataclass
class LMConfig:
    model_id: str
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    load_4bit: bool = False

class LMWrapper(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg

        compute_dtype = cfg.dtype
        load_kwargs = dict(torch_dtype=compute_dtype)

        if cfg.load_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                load_kwargs["quantization_config"] = bnb_config
                load_kwargs["device_map"] = "auto"
            except Exception as e:
                print("bitsandbytes not available or failed; falling back to full precision:", e)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True, trust_remote_code=True)
        try:
            self.tokenizer.padding_side = "left"
        except Exception:
            pass

        # Ensure pad/eos exist
        if self.tokenizer.pad_token_id is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        if self.tokenizer.eos_token_id is None:
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            self.tokenizer.eos_token = self.tokenizer.pad_token

        if cfg.device == "cuda" and "device_map" not in load_kwargs:
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, trust_remote_code=True, **load_kwargs)
        if cfg.device in ("mps", "cpu"):
            try:
                self.model.to(cfg.device)
            except Exception:
                pass
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.input_embed = self.model.get_input_embeddings()

        d_model = getattr(self.model.config, "hidden_size", None)
        if d_model is None:
            d_model = getattr(self.model.config, "n_embd", None)
        self.d_model = int(d_model)

        self._stop_token_ids = self._collect_eos_token_ids()

    # ---- utility ----

    def _collect_eos_token_ids(self) -> List[int]:
        ids = set()
        try:
            if self.tokenizer.eos_token_id is not None:
                ids.add(int(self.tokenizer.eos_token_id))
        except Exception:
            pass
        for tok in ["<|eot_id|>", "<|im_end|>", "</s>"]:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    ids.add(int(tid))
            except Exception:
                pass
        return list(sorted(ids))

    def _encode_anchor_text(self, text: Optional[str]) -> List[int]:
        if not text:
            return []
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            ids = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False).get("input_ids", [])
            if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
                ids = ids[0]
            return ids or []

    def enable_gradient_checkpointing(self):
        try:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                if hasattr(self.model.config, "use_cache"):
                    self.model.config.use_cache = False
                self.model.gradient_checkpointing_enable()
                return True
        except Exception:
            pass
        return False

    @torch.no_grad()
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.input_embed(input_ids)

    @torch.no_grad()
    def input_embedding_rms(self, sample_rows: int = 65536) -> float:
        W = self.input_embed.weight.detach()
        if W.dim() != 2:
            W = W.view(W.size(0), -1)
        if W.size(0) > sample_rows:
            idx = torch.randint(0, W.size(0), (sample_rows,), device=W.device)
            sample = W[idx]
        else:
            sample = W
        return float(sample.float().pow(2).mean().sqrt().item())

    # ---- decoding helpers ----

    @torch.no_grad()
    def decode_batch_then_clean(self, batches: List[List[int]]) -> List[str]:
        outs: List[str] = []
        for ids in batches:
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            txt = self.tokenizer.decode(ids, skip_special_tokens=True)
            outs.append(_local_clean_pred(txt))
        return outs

    @staticmethod
    def _nucleus_sample(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            dist = torch.distributions.Categorical(probs)
            return dist.sample()
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= top_p
        keep[..., 0] = True
        filtered = sorted_probs * keep
        filtered_sum = filtered.sum(dim=-1, keepdim=True)
        filtered = torch.where(filtered_sum > 0, filtered / filtered_sum, sorted_probs)
        dist = torch.distributions.Categorical(filtered)
        idx_in_sorted = dist.sample()
        next_ids = sorted_idx.gather(-1, idx_in_sorted.unsqueeze(-1)).squeeze(-1)
        return next_ids

    @staticmethod
    def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature is None or temperature <= 0.0:
            return logits
        return logits / max(1e-6, float(temperature))

    @torch.no_grad()
    def _sample_top_p(self, logits: torch.Tensor, top_p: float = 1.0, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1)
        logits = self._apply_temperature(logits, temperature)
        probs = torch.softmax(logits, dim=-1)
        return self._nucleus_sample(probs, top_p=top_p)

    # ---- losses / scoring ----

    def forward_with_prefix_loss(
        self,
        prefix_embeds: torch.Tensor,
        target_ids: torch.Tensor,
        anchor_token_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Teacher-forced CE over the gold answer conditioned on a latent prefix,
        optionally warm-started by one or more anchor tokens.

        Now PAD-aware: labels that are PAD/EOS (when PAD==EOS) are ignored,
        and TF attention masks zero-out padded inputs.
        """
        B, M, D = prefix_embeds.shape
        model_device = next(self.model.parameters()).device
        emb_dtype = self.input_embed.weight.dtype if hasattr(self.input_embed, 'weight') else None
        if emb_dtype is not None:
            prefix_embeds = prefix_embeds.to(model_device, dtype=emb_dtype)
        else:
            prefix_embeds = prefix_embeds.to(model_device)

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        # Optional anchor
        A = 0
        anchor_embeds = None
        if anchor_token_ids:
            anchor_ids = torch.tensor(anchor_token_ids, dtype=torch.long, device=model_device).unsqueeze(0).expand(B, -1)
            anchor_embeds = self.input_embed(anchor_ids)
            A = anchor_embeds.size(1)

        # Teacher-forcing inputs (shift)
        #   We still feed TF inputs (may include PADs), but we will (1) mask labels at PADs
        #   and (2) zero the attention_mask over padded TF positions so they don't serve as context.
        tf_inputs = target_ids[:, :-1].to(model_device)
        tf_embeds = self.input_embed(tf_inputs)

        # Compose input embeddings
        if anchor_embeds is not None:
            inputs_embeds = torch.cat([prefix_embeds, anchor_embeds, tf_embeds], dim=1)
        else:
            inputs_embeds = torch.cat([prefix_embeds, tf_embeds], dim=1)

        # Build labels (shifted by one) and mask PAD/EOS (when PAD==EOS) to -100
        labels = target_ids[:, 1:].to(model_device).clone()
        if pad_id is not None:
            ignore = labels.eq(int(pad_id))
            # If PAD and EOS are distinct, you may choose to keep EOS as a supervised symbol.
            # If they are equal (common in chat LMs when pad->eos), EOS gets ignored too.
            if (eos_id is not None) and (int(eos_id) != int(pad_id)):
                # Optional: uncomment to also ignore EOS
                # ignore = ignore | labels.eq(int(eos_id))
                pass
            labels = torch.where(ignore, torch.full_like(labels, -100), labels)

        # Prepend ignore masks for prefix/anchor
        ignore_prefix = torch.full((B, M), -100, dtype=torch.long, device=model_device)
        ignore_anchor = torch.full((B, A), -100, dtype=torch.long, device=model_device) if A > 0 else None
        labels_full = torch.cat([ignore_prefix, ignore_anchor, labels] if A > 0 else [ignore_prefix, labels], dim=1)

        # Attention mask: start as ones; zero-out TF padded positions if any
        attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=model_device)
        if pad_id is not None:
            tf_pad = tf_inputs.eq(int(pad_id))
            if tf_pad.any():
                # slice covering the TF region inside the composed sequence
                start = M + (A if A > 0 else 0)
                stop  = start + tf_inputs.size(1)
                attn_mask[:, start:stop] = (~tf_pad).long()

        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels_full)
        return out.loss


    def loss_with_text_prompt(self, prompt_ids: torch.Tensor, target_ids: torch.Tensor):
        """
        PAD-aware text loss for diagnostics. We ignore PADs in the target labels
        and zero the attention over padded TF inputs.
        """
        device = next(self.model.parameters()).device
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        B = prompt_ids.size(0)
        tf_inputs_tail = target_ids[:, :-1]
        tf_inputs = torch.cat([prompt_ids, tf_inputs_tail], dim=1)

        # Attention mask: ones, but zero-out padded TF positions if any
        attn_mask = torch.ones_like(tf_inputs, dtype=torch.long, device=device)
        if pad_id is not None:
            tail_pad = tf_inputs_tail.eq(int(pad_id))
            if tail_pad.any():
                S = prompt_ids.size(1)
                attn_mask[:, S:S + tf_inputs_tail.size(1)] = (~tail_pad).long()

        # Labels: ignore the prompt part entirely; mask PADs in the target part
        labels_tail = target_ids[:, 1:].clone().to(device)
        if pad_id is not None:
            ignore = labels_tail.eq(int(pad_id))
            if (eos_id is not None) and (int(eos_id) != int(pad_id)):
                # Optional: also ignore EOS
                # ignore = ignore | labels_tail.eq(int(eos_id))
                pass
            labels_tail = torch.where(ignore, torch.full_like(labels_tail, -100), labels_tail)

        labels = torch.cat([
            torch.full((B, prompt_ids.size(1)), -100, dtype=torch.long, device=device),
            labels_tail
        ], dim=1)

        out = self.model(input_ids=tf_inputs.to(device), attention_mask=attn_mask, labels=labels)
        n_tokens = (labels != -100).sum()
        return out.loss, int(n_tokens.item())

    def score_prefix_logprob(self, prefix_embeds: torch.Tensor, target_ids: torch.Tensor, anchor_token_ids: Optional[List[int]] = None) -> float:
        loss = self.forward_with_prefix_loss(prefix_embeds, target_ids, anchor_token_ids=anchor_token_ids)
        n_tokens = target_ids.size(1) - 1
        total_nll = loss * n_tokens
        return -float(total_nll.item())

    # ---- diagnostics: first-step distribution ----

    def first_token_logits_from_prefix(
        self,
        prefix_embeds: torch.Tensor,
        anchor_token_text: Optional[str] = None,
        append_bos_after_prefix: Optional[bool] = None,  # None => auto: only if NO anchor
    ) -> torch.Tensor:
        """
        Returns logits for the very first token conditioned ONLY on:
           [latent prefix] + optional [anchor tokens] + optional [BOS]
        Used in training (first-token CE) and eval diagnostics.
        """
        model_device = next(self.model.parameters()).device
        emb_dtype = self.input_embed.weight.dtype if hasattr(self.input_embed, 'weight') else None
        if emb_dtype is not None:
            prefix_embeds = prefix_embeds.to(model_device, dtype=emb_dtype)
        else:
            prefix_embeds = prefix_embeds.to(model_device)

        B = prefix_embeds.size(0)
        parts = [prefix_embeds]

        # Anchor
        anchor_ids = self._encode_anchor_text(anchor_token_text) if anchor_token_text else []
        if anchor_ids:
            anchor_ids_tensor = torch.tensor(anchor_ids, dtype=torch.long, device=model_device).unsqueeze(0).expand(B, -1)
            parts.append(self.input_embed(anchor_ids_tensor))

        # BOS policy
        if append_bos_after_prefix is None:
            append_bos_after_prefix = (len(anchor_ids) == 0)
        if append_bos_after_prefix:
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_id is not None:
                bos = torch.full((B, 1), int(bos_id), dtype=torch.long, device=model_device)
                parts.append(self.input_embed(bos))

        inputs_embeds = torch.cat(parts, dim=1)
        attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=model_device)
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=False, return_dict=True)
        logits = out.logits[:, -1, :]
        return logits

    @torch.no_grad()
    def peek_first_step_from_prefix(
        self,
        prefix_embeds: torch.Tensor,
        anchor_token_text: Optional[str] = None,
        append_bos_after_prefix: Optional[bool] = None,
        topk: int = 10
    ) -> List[List[Tuple[int, float, str]]]:
        model_device = next(self.model.parameters()).device
        emb_dtype = self.input_embed.weight.dtype if hasattr(self.input_embed, 'weight') else None
        if emb_dtype is not None:
            prefix_embeds = prefix_embeds.to(model_device, dtype=emb_dtype)
        else:
            prefix_embeds = prefix_embeds.to(model_device)

        B = prefix_embeds.size(0)

        # Anchor (optional)
        anchor_ids = self._encode_anchor_text(anchor_token_text) if anchor_token_text else []
        if anchor_ids:
            anchor_ids_tensor = torch.tensor(anchor_ids, dtype=torch.long, device=model_device).unsqueeze(0).expand(B, -1)
            anchor_embeds = self.input_embed(anchor_ids_tensor)
            inputs_embeds = torch.cat([prefix_embeds, anchor_embeds], dim=1)
        else:
            inputs_embeds = prefix_embeds

        # BOS auto-policy
        if append_bos_after_prefix is None:
            append_bos_after_prefix = (len(anchor_ids) == 0)
        if append_bos_after_prefix:
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_id is not None:
                bos = torch.full((B, 1), int(bos_id), dtype=torch.long, device=model_device)
                bos_embeds = self.input_embed(bos)
                inputs_embeds = torch.cat([inputs_embeds, bos_embeds], dim=1)

        attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=model_device)
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=True, return_dict=True)
        logits = out.logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, k=min(topk, probs.size(-1)), dim=-1)
        results: List[List[Tuple[int, float, str]]] = []
        for b in range(B):
            ids_b = topk_ids[b].tolist()
            probs_b = topk_probs[b].tolist()
            toks = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in ids_b]
            results.append([(ids_b[i], float(probs_b[i]), toks[i]) for i in range(len(ids_b))])
        return results

    # ---- generation: latent prefix ----

    @torch.no_grad()
    def generate_from_prefix(
        self,
        prefix_embeds: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        anchor_token_text: Optional[str] = None,
        min_new_tokens: int = 0,
        eos_ban_steps: int = 0,
        first_token_top_p: float = 1.0,
        first_token_temperature: float = 0.0,
        append_bos_after_prefix: Optional[bool] = None,  # None => auto: only if NO anchor
    ) -> List[List[int]]:
        model_device = next(self.model.parameters()).device
        emb_dtype = self.input_embed.weight.dtype if hasattr(self.input_embed, 'weight') else None
        if emb_dtype is not None:
            prefix_embeds = prefix_embeds.to(model_device, dtype=emb_dtype)
        else:
            prefix_embeds = prefix_embeds.to(model_device)

        B = prefix_embeds.size(0)

        # Optional anchor tokens
        anchor_ids = self._encode_anchor_text(anchor_token_text) if anchor_token_text else []
        if anchor_ids:
            anchor_ids_tensor = torch.tensor(anchor_ids, dtype=torch.long, device=model_device).unsqueeze(0).expand(B, -1)
            anchor_embeds = self.input_embed(anchor_ids_tensor)
            inputs_embeds = torch.cat([prefix_embeds, anchor_embeds], dim=1)
        else:
            inputs_embeds = prefix_embeds

        # BOS priming (auto policy: only if no anchor provided)
        if append_bos_after_prefix is None:
            append_bos_after_prefix = (len(anchor_ids) == 0)
        if append_bos_after_prefix:
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_id is not None:
                bos = torch.full((B, 1), int(bos_id), dtype=torch.long, device=model_device)
                bos_embeds = self.input_embed(bos)
                inputs_embeds = torch.cat([inputs_embeds, bos_embeds], dim=1)

        # Initial forward
        attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=model_device)
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_token_logits = out.logits[:, -1, :]

        pad_id = self.tokenizer.pad_token_id or 0
        stop_ids = set(self._stop_token_ids)

        # Build extended early stop list for aggressive banning
        early_stop_ids = set(stop_ids)
        for tok in ["<|eot_id|>", "<|eom_id|>", "<|endoftext|>", "<|end_of_text|>", "<|im_end|>", "</s>"]:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    early_stop_ids.add(int(tid))
            except Exception:
                pass

        generated = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=model_device)

        for t in range(max_new_tokens):
            step_logits = next_token_logits.clone()
            step_logits[finished] = -1e9

            # Early EOS ban / min tokens - use extended set
            if t < max(min_new_tokens, eos_ban_steps):
                for sid in early_stop_ids:
                    step_logits[:, sid] = -1e9

            # First-token exploration if requested
            if t == 0 and (first_token_temperature > 0.0 or first_token_top_p < 1.0):
                next_tokens = self._sample_top_p(step_logits, top_p=first_token_top_p, temperature=first_token_temperature)
            else:
                if temperature <= 0.0:
                    next_tokens = torch.argmax(step_logits, dim=-1)
                else:
                    next_tokens = self._sample_top_p(step_logits, top_p=top_p, temperature=temperature)

            for b in range(B):
                if finished[b]:
                    continue
                nid = int(next_tokens[b].item())
                if (t >= min_new_tokens) and (nid in stop_ids):  # Use regular stop_ids for actual stopping
                    finished[b] = True
                else:
                    generated[b].append(nid)

            if bool(torch.all(finished).item()):
                break

            feed_tokens = next_tokens.clone()
            feed_tokens[finished] = pad_id
            attn_mask_step = torch.ones((B, 1), dtype=torch.long, device=model_device)
            out = self.model(
                input_ids=feed_tokens.unsqueeze(-1), attention_mask=attn_mask_step,
                use_cache=True, past_key_values=past, return_dict=True
            )
            past = out.past_key_values
            next_token_logits = out.logits[:, -1, :]

        return generated

    # ---- generation: text prompt ----

    @torch.no_grad()
    def generate_from_text(
        self,
        prompt_texts: List[str],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0
    ) -> List[List[int]]:
        enc = self.tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)
        input_ids = enc["input_ids"].to(next(self.model.parameters()).device)
        attn_mask = enc["attention_mask"].to(input_ids.device)

        B = input_ids.size(0)
        out = self.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_token_logits = out.logits[:, -1, :]

        pad_id = self.tokenizer.pad_token_id or 0
        stop_ids = set(self._stop_token_ids)

        generated = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            step_logits = next_token_logits.clone()
            step_logits[finished] = -1e9

            if temperature <= 0.0:
                next_tokens = torch.argmax(step_logits, dim=-1)
            else:
                next_tokens = self._sample_top_p(step_logits, top_p=top_p, temperature=temperature)

            for b in range(B):
                if finished[b]:
                    continue
                nid = int(next_tokens[b].item())
                if nid in stop_ids:
                    finished[b] = True
                else:
                    generated[b].append(nid)

            if bool(torch.all(finished).item()):
                break

            feed_tokens = next_tokens.clone()
            feed_tokens[finished] = pad_id

            attn_mask_step = torch.ones((B, 1), dtype=torch.long, device=input_ids.device)
            out = self.model(input_ids=feed_tokens.unsqueeze(-1), attention_mask=attn_mask_step,
                             use_cache=True, past_key_values=past, return_dict=True)
            next_token_logits = out.logits[:, -1, :]
            past = out.past_key_values
        return generated
