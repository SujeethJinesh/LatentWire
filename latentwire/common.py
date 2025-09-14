# latentwire/common.py
from typing import List, Optional
import re
import torch

# ---------- Cleaning ----------

STOP_STRINGS = [
    "<|eot_id|>", "<|im_end|>", "</s>",              # common chat EOS-ish markers
    "<|system|>", "<|user|>", "<|assistant|>",       # guardrails: if the model starts a chat block, cut it
    "\n\n\n", "\n\nAssistant:", "\nAssistant:",
]

def clean_pred(s: str) -> str:
    """Normalize short-span generations to a clean answer phrase."""
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

# ---------- Prompt builders ----------

SYSTEM_PROMPT = (
    "You are a concise QA assistant. Use the context to answer with a short phrase only. "
    "Answer in English. Respond with the answer phrase only."
)
NEUTRAL_SYSTEM_PROMPT = "You are a concise QA assistant. Use the context to answer with a short phrase only."

def build_chat_prompts(tokenizer, raw_sources: List[str]) -> List[str]:
    outs = []
    for s in raw_sources:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": s},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outs.append(text)
    return outs

def build_neutral_encoder_texts(raw_sources: List[str]) -> List[str]:
    return [f"System: {NEUTRAL_SYSTEM_PROMPT}\nUser: {s}\nAssistant:" for s in raw_sources]

def truncate_chat_to_k_tokens(tokenizer, chat_prompts: List[str], k: int) -> List[str]:
    outs = []
    for cp in chat_prompts:
        enc = tokenizer(cp, add_special_tokens=False, return_attention_mask=False)
        ids_k = enc["input_ids"][:k]
        outs.append(tokenizer.decode(ids_k, skip_special_tokens=True))
    return outs

def content_only_m_token_chat_prompt(tokenizer, raw_source: str, k: int) -> str:
    enc = tokenizer(raw_source, add_special_tokens=True, return_attention_mask=False)
    ids_k = enc["input_ids"][:k]
    truncated_content = tokenizer.decode(ids_k, skip_special_tokens=True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": truncated_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def build_token_budget_prompts(tokenizer, raw_sources: List[str], chat_prompts: List[str], k: int, mode: str) -> List[str]:
    if mode == "chat_full":
        return truncate_chat_to_k_tokens(tokenizer, chat_prompts, k)
    return [content_only_m_token_chat_prompt(tokenizer, s, k) for s in raw_sources]

# ---------- Byte collation ----------

class _ByteTokenizerShim:
    """Type shim so the signature matches the ByteTokenizer in models."""
    def __init__(self, max_bytes: int = 512):
        self.max_bytes = max_bytes
    def encode(self, text: str) -> torch.Tensor:
        b = text.encode("utf-8")[: self.max_bytes]
        return torch.tensor(list(b), dtype=torch.long)

def collate_bytes(texts: List[str], byte_tok, device: str) -> torch.Tensor:
    ids = [byte_tok.encode(t) for t in texts]
    maxT = max([x.size(0) for x in ids]) if ids else 0
    if maxT == 0:
        # keep batch dimension
        return torch.zeros((len(texts), 1), dtype=torch.long, device=device)
    batch = torch.stack([torch.cat([x, torch.zeros(maxT - x.size(0), dtype=torch.long)], dim=0) for x in ids], dim=0)
    return batch.to(device)

# ---------- Anchors ----------

def assistant_header_anchor(tokenizer) -> str:
    """Extract the assistant header string for this model family, if available."""
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": ""}],
            tokenize=False, add_generation_prompt=False
        )
        return text or ""
    except Exception:
        return ""

def make_anchor_text(mode: str, wrapper, explicit_text: str) -> str:
    if mode == "none":
        return ""
    if mode == "text":
        return explicit_text or ""
    if mode == "chat":
        return assistant_header_anchor(wrapper.tokenizer)
    raise ValueError(f"Unknown latent_anchor_mode: {mode}")

def infer_anchor_mode_and_text(wrapper, cfg: dict, cli_mode: str, cli_text: str):
    """
    If cli_mode == 'auto':
      - If training used warm_anchor_text (non-empty) -> ('text', that_string)
      - Else -> ('chat', assistant_header)
    Else: (cli_mode, cli_text)
    """
    if cli_mode != "auto":
        return cli_mode, cli_text
    train_anchor = (cfg.get("warm_anchor_text") or "").strip()
    if train_anchor:
        return "text", train_anchor
    return "chat", ""
