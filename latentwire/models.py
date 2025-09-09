import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    Fast-path encoder using a frozen SentenceTransformer embedding plus a
    lightweight projection and learned latent queries. This preserves the
    interlingua concept while avoiding long byte sequences (useful on MPS).
    """
    def __init__(self, d_z: int = 256, latent_len: int = 8, backbone: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError("sentence-transformers is required for SimpleEncoder. Install with: pip install sentence-transformers") from e
        self.st = SentenceTransformer(backbone)
        # Keep ST frozen
        try:
            for p in self.st._first_module().parameters():
                p.requires_grad_(False)
        except Exception:
            pass
        try:
            self.st_dim = self.st.get_sentence_embedding_dimension()
        except Exception:
            self.st_dim = 384
        self.proj = nn.Linear(self.st_dim, d_z)
        self.queries = nn.Parameter(torch.randn(latent_len, d_z) / math.sqrt(d_z))
        self.ln = nn.LayerNorm(d_z)

    def forward(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            emb = self.st.encode(texts, convert_to_tensor=True)
        # Convert inference-mode tensor to a regular tensor for autograd downstream
        emb = emb.detach().clone()
        z0 = self.proj(emb)
        B = z0.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        z = self.ln(q + z0.unsqueeze(1))
        return z


class Adapter(nn.Module):
    def __init__(self, d_z: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_z), nn.Linear(d_z, d_model))
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z) * self.scale
        # tame extremes: helps with stability on non-text inputs
        return torch.tanh(x / 3.0) * 3.0


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
        load_kwargs = dict(torch_dtype=compute_dtype, device_map="auto")

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
            except Exception as e:
                print("bitsandbytes not available or failed; falling back to full precision:", e)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True, trust_remote_code=True)
        # Ensure pad/eos exist for generation
        if self.tokenizer.pad_token_id is None:
            # Prefer eos as pad if available
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # As a last resort, set a dummy pad token
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        if self.tokenizer.eos_token_id is None:
            # Fall back to pad if eos is missing
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            self.tokenizer.eos_token = self.tokenizer.pad_token

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, trust_remote_code=True, **load_kwargs)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.input_embed = self.model.get_input_embeddings()

        d_model = getattr(self.model.config, "hidden_size", None)
        if d_model is None:
            d_model = getattr(self.model.config, "n_embd", None)
        self.d_model = int(d_model)

    def enable_gradient_checkpointing(self):
        try:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                # Disable cache as recommended by HF when using checkpointing
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

    def forward_with_prefix_loss(self, prefix_embeds: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        B, M, D = prefix_embeds.shape
        model_device = next(self.model.parameters()).device
        prefix_embeds = prefix_embeds.to(model_device)
        tf_inputs = target_ids[:, :-1]
        tf_embeds = self.input_embed(tf_inputs.to(model_device))
        inputs_embeds = torch.cat([prefix_embeds, tf_embeds], dim=1)
        labels = target_ids[:, 1:]
        ignore = torch.full((B, M), -100, dtype=torch.long, device=model_device)
        labels_full = torch.cat([ignore, labels], dim=1)
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

    def score_prefix_logprob(self, prefix_embeds: torch.Tensor, target_ids: torch.Tensor) -> float:
        loss = self.forward_with_prefix_loss(prefix_embeds, target_ids)
        n_tokens = target_ids.size(1) - 1
        total_nll = loss * n_tokens
        return -float(total_nll.item())

    @torch.no_grad()
    def generate_from_prefix(self, prefix_embeds: torch.Tensor, max_new_tokens: int = 64, temperature: float = 0.0, top_p: float = 1.0) -> List[List[int]]:
        model_device = next(self.model.parameters()).device
        prefix_embeds = prefix_embeds.to(model_device)
        B = prefix_embeds.size(0)
        attn_mask = torch.ones(prefix_embeds.size()[:-1], dtype=torch.long, device=model_device)
        out = self.model(inputs_embeds=prefix_embeds, attention_mask=attn_mask, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_token_logits = out.logits[:, -1, :]
        generated = [[] for _ in range(B)]
        for _ in range(max_new_tokens):
            if temperature == 0.0:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            else:
                probs = torch.softmax(next_token_logits / max(1e-5, temperature), dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum > top_p
                    mask[..., 0] = False
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-8)
                    idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
                    next_tokens = sorted_idx.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
                else:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            for b in range(B):
                generated[b].append(int(next_tokens[b].item()))

            attn_mask_step = torch.ones((B, 1), dtype=torch.long, device=model_device)
            out = self.model(input_ids=next_tokens.unsqueeze(-1), attention_mask=attn_mask_step, use_cache=True, past_key_values=past, return_dict=True)
            past = out.past_key_values
            next_token_logits = out.logits[:, -1, :]
        return generated

    @torch.no_grad()
    def generate_from_text(self, prompt_texts: List[str], max_new_tokens: int = 64, temperature: float = 0.0, top_p: float = 1.0) -> List[List[int]]:
        enc = self.tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(next(self.model.parameters()).device)
        attn_mask = enc["attention_mask"].to(input_ids.device)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else pad_id
        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature>0),
            temperature=max(1e-5, temperature),
            top_p=top_p,
            pad_token_id=[pad_id],
            eos_token_id=[eos_id],
        )
        outs = []
        for i in range(gen.size(0)):
            new_tokens = gen[i, input_ids.size(1):].tolist()
            outs.append(new_tokens)
        return outs

    @torch.no_grad()
    def generate_from_text_manual(self, prompt_texts: List[str], max_new_tokens: int = 64, temperature: float = 0.0, top_p: float = 1.0) -> List[List[int]]:
        enc = self.tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(next(self.model.parameters()).device)
        attn_mask = enc["attention_mask"].to(input_ids.device)
        out = self.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_token_logits = out.logits[:, -1, :]
        B = input_ids.size(0)
        generated = [[] for _ in range(B)]
        for _ in range(max_new_tokens):
            if temperature == 0.0:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            else:
                probs = torch.softmax(next_token_logits / max(1e-5, temperature), dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum > top_p
                    mask[..., 0] = False
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-8)
                    idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
                    next_tokens = sorted_idx.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
                else:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            for b in range(B):
                generated[b].append(int(next_tokens[b].item()))

            attn_mask_step = torch.ones((B, 1), dtype=torch.long, device=input_ids.device)
            out = self.model(input_ids=next_tokens.unsqueeze(-1), attention_mask=attn_mask_step, use_cache=True, past_key_values=past, return_dict=True)
            past = out.past_key_values
            next_token_logits = out.logits[:, -1, :]
        return generated
