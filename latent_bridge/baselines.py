"""Lightweight adapter scaffolding for external communication baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaselineContext:
    source_model: str
    target_model: str
    device: str
    dtype: str
    calibration_file: str | None = None


class BaselineAdapter(ABC):
    """Narrow interface for non-RotAlign baseline integrations."""

    name: str

    def fit(self, context: BaselineContext) -> None:
        """Optional calibration or checkpoint-setup step."""

    @abstractmethod
    def evaluate_mcq(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_generation(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        raise NotImplementedError


_BASELINE_REGISTRY: dict[str, type[BaselineAdapter]] = {}


def register_baseline(adapter_cls: type[BaselineAdapter]) -> type[BaselineAdapter]:
    name = getattr(adapter_cls, "name", None)
    if not name:
        raise ValueError("Baseline adapters must define a non-empty `name`.")
    _BASELINE_REGISTRY[name] = adapter_cls
    return adapter_cls


def available_baselines() -> list[str]:
    return sorted(_BASELINE_REGISTRY)


def get_baseline(name: str) -> type[BaselineAdapter]:
    if name not in _BASELINE_REGISTRY:
        raise KeyError(f"Unknown baseline adapter: {name}")
    return _BASELINE_REGISTRY[name]


@register_baseline
class C2CAdapter(BaselineAdapter):
    name = "c2c"

    def evaluate_mcq(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        raise NotImplementedError("C2C adapter is not wired yet.")

    def evaluate_generation(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        raise NotImplementedError("C2C adapter is not wired yet.")


@register_baseline
class KVCommAdapter(BaselineAdapter):
    name = "kvcomm"

    def evaluate_mcq(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        raise NotImplementedError("KVComm adapter is not wired yet.")

    def evaluate_generation(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        raise NotImplementedError("KVComm adapter is not wired yet.")


@register_baseline
class LatentMASAdapter(BaselineAdapter):
    name = "latentmas"

    def evaluate_mcq(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        raise NotImplementedError("LatentMAS adapter is not wired yet.")

    def evaluate_generation(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        raise NotImplementedError("LatentMAS adapter is not wired yet.")
