# latentwire/models.py
import math
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Sequence, Union, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from latentwire.core_utils import (
    clean_pred,
    apply_lora,
    apply_prefix_tuning,
    maybe_merge_lora,
)


def default_lora_targets_for(model_name_or_path: str) -> List[str]:
    """Return canonical attention + MLP module names for Llama/Qwen stacks."""
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def resolve_lora_targets(arg: str, model_name_or_path: str) -> Tuple[List[str], Optional[int]]:
    """Parse '--lora_target_modules' style specifiers into module names and optional layer limits."""
    spec = (arg or "auto").strip()
    if spec == "auto":
        return default_lora_targets_for(model_name_or_path), None
    if spec.startswith("attn_mlp_firstN:"):
        try:
            first_n = int(spec.split(":", 1)[1])
        except ValueError:
            first_n = None
        return default_lora_targets_for(model_name_or_path), first_n
    modules = [tok.strip() for tok in spec.split(",") if tok.strip()]
    return modules or default_lora_targets_for(model_name_or_path), None


def apply_lora_if_requested(model, lora_args: Dict[str, Any], model_name_or_path: str):
    """Attach LoRA adapters to *model* using the supplied hyper-parameters."""
    target_modules, first_n = resolve_lora_targets(lora_args.get("target_modules", "auto"), model_name_or_path)
    try:
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("peft is required for --use_lora runs") from exc

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        try:
            model = prepare_model_for_kbit_training(model)
        except Exception:
            pass

    cfg = LoraConfig(
        r=int(lora_args.get("r", 8)),
        lora_alpha=int(lora_args.get("alpha", 16)),
        lora_dropout=float(lora_args.get("dropout", 0.05)),
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    lora_model = get_peft_model(model, cfg)

    if first_n is not None:
        try:
            base_layers = getattr(lora_model.base_model, "model", None)
            base_layers = getattr(base_layers, "layers", None)
            if base_layers is not None:
                for idx, layer in enumerate(base_layers):
                    allow = idx < first_n
                    for name, module in layer.named_modules():
                        if any(tok in name for tok in target_modules):
                            for param in module.parameters():
                                param.requires_grad = allow
        except Exception:
            pass

    try:
        lora_model.print_trainable_parameters()
    except Exception:
        pass
    return lora_model


def apply_prefix_if_requested(model, prefix_args: Dict[str, Any], tokenizer=None):
    """Attach Prefix-Tuning adapters across transformer layers when requested."""
    try:
        from peft import PrefixTuningConfig, TaskType, get_peft_model
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("peft is required for --use_prefix runs") from exc

    hidden = getattr(model.config, "hidden_size", getattr(model.config, "n_embd", None))
    num_layers = getattr(model.config, "num_hidden_layers", None)
    num_heads = getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", None))
    if hidden is None or num_layers is None or num_heads is None:
        raise ValueError("Cannot infer model dimensions for Prefix-Tuning; please provide a compatible model config.")
    depth = prefix_args.get("depth", None)
    depth_int = num_layers
    if depth is not None:
        try:
            depth_int = max(1, min(int(depth), num_layers))
        except Exception:
            depth_int = num_layers
    all_layers_flag = bool(prefix_args.get("all_layers", True))
    target_layers = depth_int if all_layers_flag else min(depth_int, 1)

    cfg = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=int(prefix_args.get("tokens", 16)),
        prefix_projection=bool(prefix_args.get("projection", True)),
        encoder_hidden_size=hidden,
        token_dim=hidden,
        num_layers=target_layers,
        num_attention_heads=num_heads,
        num_transformer_submodules=1,
    )
    prefix_model = get_peft_model(model, cfg)
    try:
        prefix_model.print_trainable_parameters()
    except Exception:
        pass
    return prefix_model

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
    trims role echoes. We still rely on latentwire.core_utils.clean_pred in eval.py,
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
# Model loading helpers (PEFT-aware)
# ---------------------------

def load_llm_pair(
    llama_id: str,
    qwen_id: str,
    load_4bit: bool = False,
    device_map: Union[str, Dict[str, Union[str, int]]] = "auto",
    **kw,
):
    """Load a Llama/Qwen pair along with their tokenizers.

    The helper supports optional PEFT hooks (LoRA / Prefix tuning) when
    latentwire.core_utils module is available and the caller sets the
    "use_lora" / "use_prefix" flags on an argparse-style namespace passed via
    ``args`` or ``cfg`` in ``kw``.
    """

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs: Dict[str, Any] = dict(device_map=device_map)

    if load_4bit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("bitsandbytes is required for 4-bit loading") from exc
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quant_cfg
    else:
        model_kwargs["torch_dtype"] = dtype

    # Allow callers to thread additional kwargs (e.g., max_memory)
    extra_model_kw = kw.pop("model_kwargs", {})
    if extra_model_kw:
        model_kwargs.update(extra_model_kw)

    llama = AutoModelForCausalLM.from_pretrained(llama_id, **model_kwargs)
    qwen = AutoModelForCausalLM.from_pretrained(qwen_id, **model_kwargs)

    tok_llama = AutoTokenizer.from_pretrained(llama_id, use_fast=True)
    tok_qwen = AutoTokenizer.from_pretrained(qwen_id, use_fast=True)

    for tok in (tok_llama, tok_qwen):
        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({"pad_token": "<|pad|>"})

    args = kw.get("args") or kw.get("cfg") or None

    def maybe(name: str, default=None):
        return getattr(args, name, default) if args is not None else default

    if maybe("use_lora", False):
        target = maybe("lora_target_modules", "attn_mlp_firstN:12")
        r = maybe("lora_r", 8)
        alpha = maybe("lora_alpha", 16)
        dropout = maybe("lora_dropout", 0.05)
        llama = apply_lora(llama, r=r, alpha=alpha, dropout=dropout, target_modules=target)
        qwen = apply_lora(qwen, r=r, alpha=alpha, dropout=dropout, target_modules=target)
    if maybe("use_prefix", False):
        tokens = maybe("prefix_tokens", 16)
        projection = maybe("prefix_projection", True)
        llama = apply_prefix_tuning(llama, num_virtual_tokens=tokens, projection=projection)
        qwen = apply_prefix_tuning(qwen, num_virtual_tokens=tokens, projection=projection)

    return llama, tok_llama, qwen, tok_qwen


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

        ids = byte_ids.clamp_min(0).clamp_max(255).long()
        x = self.byte_emb(ids)
        x = self.pos(x)

        key_padding = None
        if attn_mask is not None:
            key_padding = attn_mask.eq(0)

        x = self.encoder(x, src_key_padding_mask=key_padding)
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

class _EmbedColor(nn.Module):
    """Per-dimension affine transform that matches target μ/σ statistics."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (x - mu) / sd
        return x * self.gamma + self.beta


class Adapter(nn.Module):
    """
    Maps Z (d_z) -> model embedding space (d_model).
    Uses a small MLP with LayerNorm, optional metadata hints, and an optional
    "colorizer" that aligns per-dimension statistics with the LM's embedding table.
    """

    def __init__(
        self,
        d_z: int,
        d_model: int,
        latent_length: int,
        enable_metadata: bool = False,
        length_norm: float = 32.0,
        hidden_mult: int = 2,
        colorize: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.enable_metadata = enable_metadata
        self.length_norm = max(length_norm, 1.0)
        self.hidden_mult = max(int(hidden_mult), 1)
        self.dropout = float(max(dropout, 0.0))

        if self.enable_metadata:
            self.position_emb = nn.Parameter(torch.randn(latent_length, d_z) * 0.02)
            self.length_proj = nn.Sequential(nn.Linear(1, d_z), nn.Tanh())
        else:
            self.register_parameter("position_emb", None)
            self.length_proj = None

        hidden_dim = d_z * self.hidden_mult
        if self.hidden_mult <= 1:
            self.input_norm = nn.LayerNorm(d_z)
            self.proj_out = nn.Linear(d_z, d_model)
            self.skip = None
            self.mid = None
            self.dropout_layer = None
        else:
            self.input_norm = nn.LayerNorm(d_z)
            self.proj_in = nn.Linear(d_z, hidden_dim)
            self.mid = nn.Linear(hidden_dim, hidden_dim)
            self.proj_out = nn.Linear(hidden_dim, d_model)
            self.skip = nn.Linear(d_z, d_model)
            self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0.0 else None
        self.out_norm = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(torch.ones(1))
        self.color = _EmbedColor(d_model) if colorize else None

        # FiLM modulation heads: per-slot scale/shift conditioned on latent features.
        self.film_scale = nn.Linear(d_z, d_model)
        self.film_shift = nn.Linear(d_z, d_model)

        def _init_linear(module: Optional[nn.Linear]) -> None:
            if module is None:
                return
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        _init_linear(getattr(self, "proj_in", None))
        _init_linear(getattr(self, "mid", None))
        _init_linear(self.proj_out)
        _init_linear(getattr(self, "skip", None))
        _init_linear(self.film_scale)
        _init_linear(self.film_shift)

    def install_color_from_wrapper(self, wrapper: "LMWrapper") -> None:
        """Initialize the colorizer parameters from LM embedding statistics."""
        if self.color is None:
            return
        with torch.no_grad():
            mu, sd = wrapper.embedding_stats()
            dev = next(self.parameters()).device
            self.color.beta.data = mu.to(dev)
            self.color.gamma.data = sd.to(dev).clamp_min(1e-6)

    def forward(self, z: torch.Tensor, answer_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_metadata and self.position_emb is not None:
            z = z + self.position_emb.unsqueeze(0).to(z.dtype)
            if answer_lengths is not None:
                lengths = answer_lengths.float().unsqueeze(-1) / self.length_norm
                length_emb = self.length_proj(lengths).unsqueeze(1)
                z = z + length_emb

        if self.hidden_mult <= 1:
            x_norm = self.input_norm(z)
            x = self.proj_out(x_norm)
        else:
            x_norm = self.input_norm(z)
            hidden = self.proj_in(x_norm)
            hidden = F.gelu(hidden)
            hidden = self.mid(hidden)
            hidden = F.gelu(hidden)
            if self.dropout_layer is not None:
                hidden = self.dropout_layer(hidden)
            out_main = self.proj_out(hidden)
            skip = self.skip(x_norm)
            x = out_main + skip
        # FiLM modulation: scale and shift each slot using the original latent features.
        film_scale = torch.sigmoid(self.film_scale(z))
        film_shift = self.film_shift(z)
        x = x * film_scale + film_shift
        x = self.out_norm(x)
        x = x * self.scale
        if self.color is not None:
            x = self.color(x)
        return x


# ---------------------------
# LM Wrapper
# ---------------------------

@dataclass
class LMConfig:
    model_id: str
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    load_4bit: bool = False
    device_map: Optional[Union[str, int, Dict[str, int]]] = None
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None

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
                if cfg.device_map is not None:
                    load_kwargs["device_map"] = cfg.device_map
                elif cfg.device == "cuda":
                    load_kwargs.setdefault("device_map", "auto")
                if cfg.max_memory is not None:
                    load_kwargs["max_memory"] = cfg.max_memory
            except Exception as e:
                print("bitsandbytes not available or failed; falling back to full precision:", e)

        if cfg.device == "cuda":
            if cfg.device_map is not None:
                load_kwargs["device_map"] = cfg.device_map
            else:
                load_kwargs.setdefault("device_map", "auto")
            if cfg.max_memory is not None and "max_memory" not in load_kwargs:
                load_kwargs["max_memory"] = cfg.max_memory
        else:
            if cfg.device_map is not None:
                load_kwargs["device_map"] = cfg.device_map
            if cfg.max_memory is not None and "max_memory" not in load_kwargs:
                load_kwargs["max_memory"] = cfg.max_memory

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

        try:
            dmap = getattr(self.model, "hf_device_map", None)
            print(f"[{cfg.model_id}] hf_device_map: {dmap}")
        except Exception:
            pass

        d_model = getattr(self.model.config, "hidden_size", None)
        if d_model is None:
            d_model = getattr(self.model.config, "n_embd", None)
        self.d_model = int(d_model)

        self._stop_token_ids = self._collect_eos_token_ids()

    # ---- utility ----

    def _collect_eos_token_ids(self) -> List[int]:
        ids: Set[int] = set()

        tid = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(tid, int) and tid >= 0:
            ids.add(int(tid))

        for tok in ("<|eot_id|>", "<|im_end|>", "</s>", "<|eom_id|>", "<|endoftext|>"):
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                continue
            if isinstance(token_id, int) and token_id >= 0:
                ids.add(int(token_id))

        return sorted(ids)

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
            try:
                # Flash/SDPA kernels are incompatible with non-reentrant checkpointing; disable globally.
                import torch.backends.cuda
                if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                    torch.backends.cuda.enable_flash_sdp(False)
                if hasattr(torch.backends.cuda, "enable_math_sdp"):
                    torch.backends.cuda.enable_math_sdp(True)
                if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
            except Exception:
                pass
            if hasattr(self.model, "gradient_checkpointing_enable"):
                if hasattr(self.model.config, "use_cache"):
                    self.model.config.use_cache = False
                try:
                    self.model.gradient_checkpointing_enable(use_reentrant=False)
                except TypeError:
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

    @torch.no_grad()
    def embedding_stats(self, sample_rows: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-dimension mean and std of the token embedding matrix."""
        W = self.input_embed.weight.detach().float()
        if sample_rows and sample_rows > 0 and W.size(0) > sample_rows:
            idx = torch.randint(0, W.size(0), (sample_rows,), device=W.device)
            W = W.index_select(0, idx)
        mu = W.mean(dim=0).cpu()
        sd = W.std(dim=0).clamp_min(1e-6).cpu()
        return mu, sd

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

    def _compose_inputs_from_prefix(
        self,
        prefix_embeds: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor],
        *,
        anchor_ids: Optional[Union[Sequence[int], torch.Tensor]] = None,
        append_bos_after_prefix: Optional[bool] = None,
    ) -> torch.Tensor:
        """Compose latent prefix + optional anchor/BOS + previous token ids into embeddings."""
        model_device = next(self.model.parameters()).device
        emb_dtype = self.input_embed.weight.dtype if hasattr(self.input_embed, "weight") else None
        if emb_dtype is not None:
            prefix = prefix_embeds.to(model_device, dtype=emb_dtype)
        else:
            prefix = prefix_embeds.to(model_device)

        parts = [prefix]
        B = prefix.size(0)

        anchor_tensor: Optional[torch.Tensor] = None
        if anchor_ids is not None:
            if isinstance(anchor_ids, torch.Tensor):
                anchor_tensor = anchor_ids.to(model_device, dtype=torch.long)
            else:
                ids_list = list(anchor_ids)
                if ids_list:
                    anchor_tensor = torch.tensor(ids_list, dtype=torch.long, device=model_device)
        if anchor_tensor is not None and anchor_tensor.numel() > 0:
            anchor_tensor = anchor_tensor.view(1, -1).expand(B, -1)
            parts.append(self.input_embed(anchor_tensor))

        if append_bos_after_prefix is None:
            append_bos_after_prefix = not (anchor_tensor is not None and anchor_tensor.numel() > 0)
        if append_bos_after_prefix:
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_id is not None:
                bos = torch.full((B, 1), int(bos_id), dtype=torch.long, device=model_device)
                parts.append(self.input_embed(bos))

        if prev_token_ids is not None and prev_token_ids.numel() > 0:
            prev = prev_token_ids.to(model_device)
            parts.append(self.input_embed(prev))

        full = torch.cat(parts, dim=1)
        if hasattr(self.input_embed, "weight"):
            full = full.to(dtype=self.input_embed.weight.dtype)
        return full

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
        enc = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
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


class STQueryEncoder(nn.Module):
    """Sentence-Transformer encoder with learned query pooling to produce latent slots."""
    def __init__(self,
                 d_z: int = 256,
                 latent_len: int = 32,
                 hf_encoder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_tokens: int = 1024,
                 freeze_backbone: bool = True,
                 slot_sinusoid: bool = True,
                 attn_heads: int = 8,
                 gate: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_encoder_id, use_fast=True)
        self.encoder = AutoModel.from_pretrained(hf_encoder_id)
        self.hidden = int(self.encoder.config.hidden_size)
        self.latent_len = int(latent_len)
        self.d_z = int(d_z)
        self.max_tokens = int(max_tokens)
        self.use_gate = bool(gate)

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        self.query = nn.Parameter(torch.randn(self.latent_len, self.hidden) * 0.02)
        self.q_proj = nn.Linear(self.hidden, self.hidden, bias=False)
        self.k_proj = nn.Linear(self.hidden, self.hidden, bias=False)
        self.v_proj = nn.Linear(self.hidden, self.hidden, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim=self.hidden, num_heads=attn_heads, batch_first=True)
        self.o_proj = nn.Linear(self.hidden, self.hidden, bias=False)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.d_z),
        )

        if self.use_gate:
            self.slot_gate = nn.Sequential(
                nn.LayerNorm(self.hidden),
                nn.Linear(self.hidden, self.hidden),
                nn.Sigmoid(),
            )
        else:
            self.slot_gate = None

        self.slot_sinusoid = bool(slot_sinusoid)
        if self.slot_sinusoid:
            slot_pe = torch.zeros(self.latent_len, self.hidden)
            position = torch.arange(0, self.latent_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.hidden, 2, dtype=torch.float32) * (-math.log(10000.0) / self.hidden))
            slot_pe[:, 0::2] = torch.sin(position * div_term)
            slot_pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("slot_pe", slot_pe, persistent=False)
        else:
            self.register_buffer("slot_pe", torch.zeros(self.latent_len, self.hidden), persistent=False)

    def _encode_tokens(self, texts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        if "input_ids" in toks and toks["input_ids"].size(-1) > self.max_tokens:
            toks["input_ids"] = toks["input_ids"][:, : self.max_tokens]
        if "attention_mask" in toks and toks["attention_mask"].size(-1) > self.max_tokens:
            toks["attention_mask"] = toks["attention_mask"][:, : self.max_tokens]
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            encoder_out = self.encoder(
                input_ids=toks.get("input_ids"),
                attention_mask=toks.get("attention_mask"),
                output_hidden_states=False,
                return_dict=True,
            )
        hidden = encoder_out.last_hidden_state  # [B, T, hidden]
        if "attention_mask" in toks:
            attn_mask = toks["attention_mask"].bool()
        else:
            attn_mask = torch.ones(hidden.size()[:2], dtype=torch.bool, device=device)
        return hidden, attn_mask

    def forward(self, texts: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        feats, amask = self._encode_tokens(texts, device)
        B, T, H = feats.shape
        q = self.q_proj(self.query + self.slot_pe.to(device)).unsqueeze(0).expand(B, -1, -1)
        k = self.k_proj(feats)
        v = self.v_proj(feats)
        key_padding_mask = ~amask
        pooled, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        pooled = pooled + q
        if self.slot_gate is not None:
            gate = self.slot_gate(pooled)
            pooled = pooled * gate + q * (1.0 - gate)
        pooled = self.o_proj(pooled)
        z = self.to_latent(pooled)
        return z
