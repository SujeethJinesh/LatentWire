"""Prompt builders for long-context experiments."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import random


def load_longctx_corpus(path: str) -> List[str]:
    chunks: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    text = obj.get("text") if isinstance(obj, dict) else None
                except Exception:
                    text = None
            else:
                text = line
            if text:
                chunks.append(text)
    return chunks


def extend_prompt_to_tokens(prompt: str, tokenizer: Any, target_tokens: int,
                            corpus_chunks: List[str], seed_key: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    if target_tokens <= 0 or not corpus_chunks:
        return prompt, {"actual_tokens": len(tokenizer.encode(prompt, add_special_tokens=False))}
    seed_source = str(seed_key) if seed_key is not None else "0"
    seed = int(hashlib.md5(seed_source.encode("utf-8")).hexdigest(), 16) % (2**31)
    rng = random.Random(seed)
    extra_parts: List[str] = []
    base_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    current_tokens = base_tokens
    max_iters = max(1, len(corpus_chunks)) * 3
    iters = 0
    while current_tokens < target_tokens and iters < max_iters:
        iters += 1
        extra_parts.append(corpus_chunks[rng.randrange(len(corpus_chunks))])
        candidate = prompt + "\n\n" + "\n".join(extra_parts)
        current_tokens = len(tokenizer.encode(candidate, add_special_tokens=False))
    extended = prompt if not extra_parts else candidate
    meta = {
        "target_tokens": target_tokens,
        "actual_tokens": current_tokens,
        "base_tokens": base_tokens,
        "longctx_seed": seed,
    }
    return extended, meta
