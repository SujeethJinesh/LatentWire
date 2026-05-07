"""Helpers for identifying attention/SSM boundaries from module names."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LayerKind(str, Enum):
    ATTENTION = "attention"
    SSM = "ssm"
    MLP = "mlp"
    OTHER = "other"


@dataclass(frozen=True)
class Boundary:
    index: int
    left: LayerKind
    right: LayerKind
    left_name: str
    right_name: str

    @property
    def direction(self) -> str:
        return f"{self.left.value}->{self.right.value}"


def classify_layer_name(name: str) -> LayerKind:
    lowered = name.lower()
    if any(token in lowered for token in ["attn", "attention", "self_attn"]):
        return LayerKind.ATTENTION
    if any(token in lowered for token in ["mamba", "ssm", "state_space", "gated_delta", "gateddeltanet"]):
        return LayerKind.SSM
    if any(token in lowered for token in ["mlp", "ffn", "feed_forward"]):
        return LayerKind.MLP
    return LayerKind.OTHER


def boundaries_from_named_layers(named_layers: list[tuple[str, LayerKind | str]]) -> list[Boundary]:
    """Return adjacent attention/SSM boundaries from a layer sequence."""

    normalized: list[tuple[str, LayerKind]] = []
    for name, kind in named_layers:
        normalized.append((name, kind if isinstance(kind, LayerKind) else LayerKind(str(kind))))

    boundaries: list[Boundary] = []
    for idx, ((left_name, left), (right_name, right)) in enumerate(zip(normalized, normalized[1:])):
        if {left, right} == {LayerKind.ATTENTION, LayerKind.SSM}:
            boundaries.append(
                Boundary(
                    index=idx,
                    left=left,
                    right=right,
                    left_name=left_name,
                    right_name=right_name,
                )
            )
    return boundaries


def boundaries_from_module_names(module_names: list[str]) -> list[Boundary]:
    return boundaries_from_named_layers([(name, classify_layer_name(name)) for name in module_names])
