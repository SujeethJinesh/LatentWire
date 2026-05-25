"""Tests for W4A16 helpers."""

import numpy as np

from outlier_migrate.quantization import dequantize_int4, protected_mask, symmetric_int4_quantize


def test_symmetric_int4_quantize_bounds_and_error() -> None:
    weights = np.array([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]], dtype=np.float32)
    qweights, scale = symmetric_int4_quantize(weights, axis=1)
    restored = dequantize_int4(qweights, scale)
    assert qweights.min() >= -8
    assert qweights.max() <= 7
    assert np.mean(np.abs(restored - weights)) < 0.1


def test_protected_mask_shape_and_values() -> None:
    mask = protected_mask(5, [0, 3])
    assert mask.shape == (5,)
    assert mask.tolist() == [True, False, False, True, False]
