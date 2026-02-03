"""Baseline helpers for byte-capped communication."""
from __future__ import annotations

from typing import Dict, Any


def apply_payload_to_receiver(prompt: str, payload: Dict[str, Any], mode: str = "text") -> str:
    """Standardized receiver wrapper for baseline payloads."""
    if mode == "text":
        return f"Teacher message:\n{payload.get('message', '')}\n\n{prompt}"
    return prompt
