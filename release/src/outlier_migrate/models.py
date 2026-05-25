"""Model metadata and dry-run validation helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Minimal model metadata needed by reproduction configs."""

    name: str
    architecture: str
    hidden_size: int
    layers: int


def model_from_config(config: dict[str, object]) -> ModelSpec:
    """Build a model spec from a YAML config mapping."""

    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("config must contain a model mapping")
    return ModelSpec(
        name=str(model["name"]),
        architecture=str(model["architecture"]),
        hidden_size=int(model["hidden_size"]),
        layers=int(model["layers"]),
    )
