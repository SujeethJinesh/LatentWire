from outlier_migrate.registry import MethodRegistry, MethodSpec, default_registry


def test_default_registry_contains_release_methods() -> None:
    names = default_registry().names()
    assert {"static_1pct", "static_2pct", "m11b", "m26", "decdec", "paroquant"}.issubset(names)


def test_registry_extension() -> None:
    registry = MethodRegistry()
    registry.register(MethodSpec("custom", "custom selector", lambda _scores, _budget: {42}))
    assert registry.get("custom").select({}, 1) == {42}
