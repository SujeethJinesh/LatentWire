"""Tests for metric definitions."""

import math

from outlier_migrate.metrics import kl_divergence, recovery_fraction, set_leaving_rate


def test_recovery_fraction_hand_computed() -> None:
    assert recovery_fraction(bf16=1.0, static=3.0, candidate=2.0) == 0.5


def test_set_leaving_rate_hand_computed() -> None:
    assert set_leaving_rate({1, 2, 3, 4}, {3, 4, 5, 6}) == 0.5


def test_kl_divergence_zero_for_equal_distributions() -> None:
    assert math.isclose(kl_divergence([0.25, 0.75], [0.25, 0.75]), 0.0, abs_tol=1e-12)
