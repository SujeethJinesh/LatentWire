import pytest

from outlier_migrate.registry import MethodRegistry, MethodSpec, default_registry


def test_default_registry_contains_release_methods() -> None:
    names = default_registry().names()
    assert {"static_1pct", "static_2pct", "m11b", "m26", "decdec", "paroquant"}.issubset(names)


def test_registry_extension() -> None:
    registry = MethodRegistry()
    registry.register(MethodSpec("custom", "custom selector", lambda _scores, _budget: {42}))
    assert registry.get("custom").select({0: [1.0]}, 1) == {42}


def test_method_spec_declares_positions_and_validates_budget() -> None:
    method = default_registry().get("m11b")
    assert method.required_positions == (100, 1000, 5000, 10000)
    with pytest.raises(ValueError, match="budget"):
        method.select({100: [1.0], 1000: [1.0], 5000: [1.0], 10000: [1.0]}, -1)
