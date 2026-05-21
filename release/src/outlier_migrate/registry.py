"""Registry for release method definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


ScoreTable = dict[int, list[float]]


@dataclass(frozen=True)
class MethodSpec:
    """A deterministic protected-channel selection method."""

    name: str
    description: str
    selector: Callable[[ScoreTable, int], set[int]]

    def select(self, scores_by_position: ScoreTable, budget: int) -> set[int]:
        """Select protected channel indices."""

        return self.selector(scores_by_position, budget)


class MethodRegistry:
    """Mutable registry for protected-channel methods."""

    def __init__(self) -> None:
        self._methods: dict[str, MethodSpec] = {}

    def register(self, spec: MethodSpec) -> None:
        """Register a method, replacing an existing method of the same name."""

        self._methods[spec.name] = spec

    def get(self, name: str) -> MethodSpec:
        """Return a registered method by name."""

        try:
            return self._methods[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._methods))
            raise KeyError(f"unknown method {name!r}; available: {available}") from exc

    def names(self) -> list[str]:
        """List registered method names."""

        return sorted(self._methods)


def default_registry() -> MethodRegistry:
    """Build the default release method registry."""

    from outlier_migrate.methods.decdec import make_decdec
    from outlier_migrate.methods.m11b import make_m11b
    from outlier_migrate.methods.m26 import make_m26
    from outlier_migrate.methods.paroquant import make_paroquant
    from outlier_migrate.methods.static import make_static_topk

    registry = MethodRegistry()
    registry.register(make_static_topk("static_1pct", position=100))
    registry.register(make_static_topk("static_2pct", position=100))
    registry.register(make_m11b())
    registry.register(make_m26())
    registry.register(make_decdec())
    registry.register(make_paroquant())
    return registry
