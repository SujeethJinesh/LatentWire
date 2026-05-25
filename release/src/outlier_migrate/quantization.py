"""Minimal W4A16-style quantization helpers."""

from __future__ import annotations

import numpy as np


def symmetric_int4_quantize(weights: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Quantize weights to signed int4 values with per-axis scales."""

    arr = np.asarray(weights, dtype=np.float32)
    max_abs = np.max(np.abs(arr), axis=axis, keepdims=True)
    scale = np.where(max_abs == 0.0, 1.0, max_abs / 7.0)
    quantized = np.clip(np.rint(arr / scale), -8, 7).astype(np.int8)
    return quantized, scale.astype(np.float32)


def dequantize_int4(qweights: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize signed int4 values represented in int8 storage."""

    return np.asarray(qweights, dtype=np.float32) * np.asarray(scale, dtype=np.float32)


def protected_mask(channel_count: int, protected: list[int]) -> np.ndarray:
    """Build a boolean protected-channel mask."""

    if channel_count <= 0:
        raise ValueError("channel_count must be positive")
    mask = np.zeros(channel_count, dtype=bool)
    for index in protected:
        if index < 0 or index >= channel_count:
            raise ValueError(f"protected channel {index} out of range")
        mask[index] = True
    return mask
