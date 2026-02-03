"""INT4 packing utilities for KVWire."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _to_int8_array(values: Iterable) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int8)
    return arr


def pack_int4(values: Iterable) -> bytes:
    """Pack int4 values (range [-8, 7]) into bytes."""
    arr = _to_int8_array(values).reshape(-1)
    if arr.size == 0:
        return b""
    if arr.min() < -8 or arr.max() > 7:
        raise ValueError("int4 values must be in [-8, 7]")
    # Convert to unsigned nibble representation.
    unsigned = (arr.astype(np.int16) & 0xF).astype(np.uint8)
    if unsigned.size % 2 == 1:
        unsigned = np.concatenate([unsigned, np.zeros(1, dtype=np.uint8)])
    low = unsigned[0::2]
    high = unsigned[1::2] << 4
    packed = (low | high).astype(np.uint8)
    return packed.tobytes()


def unpack_int4(blob: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    """Unpack int4-encoded bytes into int8 ndarray with the requested shape."""
    if not blob:
        return np.zeros(shape, dtype=np.int8)
    packed = np.frombuffer(blob, dtype=np.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    values = np.empty(packed.size * 2, dtype=np.int8)
    values[0::2] = low.astype(np.int8)
    values[1::2] = high.astype(np.int8)
    # Convert unsigned nibble to signed int4.
    values = values.astype(np.int16)
    values[values >= 8] -= 16
    values = values.astype(np.int8)
    total = int(np.prod(shape)) if shape else 0
    if total == 0:
        return np.zeros(shape, dtype=np.int8)
    values = values[:total]
    return values.reshape(shape)


def int4_byte_length(num_values: int) -> int:
    """Return number of bytes needed to store num_values int4 values."""
    if num_values <= 0:
        return 0
    return int((num_values + 1) // 2)
