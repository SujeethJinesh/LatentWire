"""Protection method registry for release scripts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

MaskBuilder = Callable[[np.ndarray, int], np.ndarray]


METHODS_REGISTRY: dict[str, MaskBuilder] = {}


def register_method(name: str, builder: MaskBuilder) -> None:
    """Register a protection-mask builder."""

    if name in METHODS_REGISTRY:
        raise ValueError(f"method already registered: {name}")
    METHODS_REGISTRY[name] = builder


def build_mask(name: str, scores: np.ndarray, budget: int) -> np.ndarray:
    """Build a mask from a registered method name."""

    try:
        return METHODS_REGISTRY[name](scores, budget)
    except KeyError as exc:
        raise KeyError(f"unknown method: {name}") from exc
