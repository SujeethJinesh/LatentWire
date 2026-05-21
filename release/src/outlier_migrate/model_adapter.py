"""Model adapter interfaces and dry-run implementation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from outlier_migrate.data import PromptRecord


@dataclass(frozen=True)
class GeneratedTrace:
    """A generated token trace for one prompt."""

    prompt_index: int
    token_ids: tuple[int, ...]


class ModelAdapter(Protocol):
    """Minimal generation interface used by release runners."""

    def generate_trace(self, prompt: PromptRecord, *, max_new_tokens: int) -> GeneratedTrace:
        """Generate a deterministic trace for a prompt."""


@dataclass(frozen=True)
class DryRunModelAdapter:
    """Deterministic placeholder adapter for CPU-only dry runs."""

    model_id: str
    seed: int

    def generate_trace(self, prompt: PromptRecord, *, max_new_tokens: int) -> GeneratedTrace:
        """Generate stable pseudo-token IDs without loading model weights."""

        seed_payload = f"{self.model_id}:{self.seed}:{prompt.index}:{prompt.text}".encode("utf-8")
        digest = hashlib.sha256(seed_payload).digest()
        tokens = tuple(int(digest[index % len(digest)]) for index in range(max_new_tokens))
        return GeneratedTrace(prompt_index=prompt.index, token_ids=tokens)
