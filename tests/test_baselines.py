from __future__ import annotations

import pytest

from latent_bridge.baselines import (
    BaselineContext,
    C2CAdapter,
    KVCommAdapter,
    available_baselines,
    get_baseline,
)


def test_baseline_registry_exposes_expected_placeholders() -> None:
    assert available_baselines() == ["c2c", "kvcomm"]
    assert get_baseline("c2c") is C2CAdapter
    assert get_baseline("kvcomm") is KVCommAdapter


def test_unknown_baseline_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_baseline("missing")


def test_placeholder_adapters_fail_explicitly() -> None:
    context = BaselineContext(
        source_model="src",
        target_model="tgt",
        device="cpu",
        dtype="float32",
    )

    c2c = C2CAdapter()
    kvcomm = KVCommAdapter()
    c2c.fit(context)
    kvcomm.fit(context)

    with pytest.raises(NotImplementedError):
        c2c.evaluate_generation()
    with pytest.raises(NotImplementedError):
        kvcomm.evaluate_mcq()
