# latentwire/models.py
import math
import re
from dataclasses import dataclass
from typing import Optional, List, Iterable

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# Small helpers
# ---------------------------

STOP_STRINGS = [
    "<|eot_id|>", "<|im_end|>", "</s>",  # common chat EOS-ish markers
    "<|system|>", "<|user|>", "<|assistant|>",  # guardrails: if the model starts a chat block, cut it
    "\n\n\n", "\n\nAssistant:", "\nAssistant:"
]

def clean_pred(s: str) -> str:
    """
    Normalize short-span generations to a clean answer phrase.
    - Hard-stop at known markers
    - Remove role echoes like 'assistant:' at the front
    - Keep the first non-empty line
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
    s = lines[0]
    s = s.strip(" \t\r\n.:;,'\"-–—")
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
    def __init__(self, d_z: int = 256, n_layers: int = 2, n_heads: int = 8, ff_mult: int = 4, max_len: int = 1024):
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
    def forward(self, byte_feats: torch.Tensor) -> torch.Tensor:
        B = byte_feats.size(0)
        q = self.latent.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.cross_attn(q, byte_feats, byte_feats, need_weights=False)
        out = out + q
        out = out + self.ff(out)
        return out

class InterlinguaEncoder(nn.Module):
    def __init__(self, d_z: int = 256, n_layers: int = 2, n_heads: int = 8, ff_mult: int = 4, latent_len: int = 8):
        super().__init__()
        self.backbone = ByteEncoder(d_z=d_z, n_layers=n_layers, n_heads=n_heads, ff_mult=ff_mult)
        self.pooler = LatentPooler(d_z=d_z, latent_len=latent_len, n_heads=n_heads)
    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(byte_ids)
        latents = self.pooler(feats)
        return latents

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
                p.requires_grad_((False))
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
        """
        B, M, D = prefix_embeds.shape
        model_device = next(self.model.parameters()).device
        emb_dtype = self.input_embed.weight.dtype if hasattr(self.input_embed, 'weight') else None
        if emb_dtype is not None:
            prefix_embeds = prefix_embeds.to(model_device, dtype=emb_dtype)
        else:
            prefix_embeds = prefix_embeds.to(model_device)

        # Optional anchor
        A = 0
        if anchor_token_ids:
            anchor_ids = torch.tensor(anchor_token_ids, dtype=torch.long, device=model_device).unsqueeze(0).expand(B, -1)
            anchor_embeds = self.input_embed(anchor_ids)
            A = anchor_embeds.size(1)
        else:
            anchor_embeds = None

        # Teacher-forcing inputs (shift)
        tf_inputs = target_ids[:, :-1].to(model_device)
        tf_embeds = self.input_embed(tf_inputs)

        # Compose
        if anchor_embeds is not None:
            inputs_embeds = torch.cat([prefix_embeds, anchor_embeds, tf_embeds], dim=1)
        else:
            inputs_embeds = torch.cat([prefix_embeds, tf_embeds], dim=1)

        # Labels (ignore prefix + optional anchor)
        labels = target_ids[:, 1:].to(model_device)
        ignore_prefix = torch.full((B, M), -100, dtype=torch.long, device=model_device)
        ignore_anchor = torch.full((B, A), -100, dtype=torch.long, device=model_device) if A > 0 else None
        labels_full = torch.cat([ignore_prefix, ignore_anchor, labels] if A > 0 else [ignore_prefix, labels], dim=1)

        attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=model_device)
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels_full)
        return out.loss

    def loss_with_text_prompt(self, prompt_ids: torch.Tensor, target_ids: torch.Tensor):
        device = next(self.model.parameters()).device
        B = prompt_ids.size(0)
        tf_inputs = torch.cat([prompt_ids, target_ids[:, :-1]], dim=1)
        attn_mask = torch.ones_like(tf_inputs, dtype=torch.long, device=device)
        labels = torch.cat([
            torch.full((B, prompt_ids.size(1)), -100, dtype=torch.long, device=device),
            target_ids[:, 1:]
        ], dim=1)
        out = self.model(input_ids=tf_inputs.to(device), attention_mask=attn_mask, labels=labels)
        n_tokens = (labels != -100).sum()
        return out.loss, int(n_tokens.item())

    def score_prefix_logprob(self, prefix_embeds: torch.Tensor, target_ids: torch.Tensor, anchor_token_ids: Optional[List[int]] = None) -> float:
        """Return total log-prob (negative NLL) of target_ids under the latent prefix (with optional anchor)."""
        loss = self.forward_with_prefix_loss(prefix_embeds, target_ids, anchor_token_ids=anchor_token_ids)
        n_tokens = target_ids.size(1) - 1
        total_nll = loss * n_tokens
        return -float(total_nll.item())

    # ---- decoding helpers ----

    def _decode_one(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def decode_then_clean(self, ids: Iterable[int]) -> str:
        return clean_pred(self._decode_one(ids))

    def decode_batch_then_clean(self, batch_ids: List[List[int]]) -> List[str]:
        return [self.decode_then_clean(x) for x in batch_ids]

    # ---- nucleus sampling helper ----
    @staticmethod
    def _sample_top_p(logits: torch.Tensor, top_p: float, temperature: float) -> torch.Tensor:
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits / max(1e-5, temperature), dim=-1)
        if top_p >= 1.0:
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        mask = cdf <= top_p
        mask[..., 0] = True
        masked_probs = sorted_probs * mask
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        sampled_sorted = torch.multinomial(masked_probs, num_samples=1).squeeze(-1)
        return sorted_idx.gather(-1, sampled_sorted.unsqueeze(-1)).squeeze(-1)

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
        append_bos_after_prefix: Optional[bool] = None,  # NEW: control BOS insertion
    ) -> List[List[int]]:
        """
        Generation from latent prefix with optional anchor text.
        If append_bos_after_prefix is None (auto), we append BOS *only if* there is no anchor text.
        """
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

        generated = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=model_device)

        for t in range(max_new_tokens):
            step_logits = next_token_logits.clone()
            step_logits[finished] = -1e9

            # Early EOS ban / min tokens
            if t < max(min_new_tokens, eos_ban_steps):
                for sid in stop_ids:
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
                if (t >= min_new_tokens) and (nid in stop_ids):
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
