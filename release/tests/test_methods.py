"""Tests for method mask invariants."""

import numpy as np

import outlier_migrate.methods.decdec
import outlier_migrate.methods.m11b
import outlier_migrate.methods.m26
import outlier_migrate.methods.static
from outlier_migrate.methods import build_mask, register_method


def test_registered_methods_enforce_budget() -> None:
    scores = np.array([0.1, 0.9, 0.3, 0.8])
    for name in ["static", "m11b", "m26", "decdec"]:
        mask = build_mask(name, scores, 2)
        assert mask.dtype == bool
        assert mask.shape == scores.shape
        assert int(mask.sum()) == 2


def test_extension_registration_pattern() -> None:
    def channel_zero(scores: np.ndarray, budget: int) -> np.ndarray:
        mask = np.zeros(scores.size, dtype=bool)
        mask[0] = True
        return mask

    register_method("always_channel_zero_test", channel_zero)
    mask = build_mask("always_channel_zero_test", np.ones(4), 1)
    assert mask.tolist() == [True, False, False, False]
