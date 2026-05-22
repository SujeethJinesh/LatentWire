"""Configuration loading for release reproduction commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Model identity and deterministic execution defaults."""

    name: str
    model_id: str
    seed: int
    revision: str | None = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Release experiment settings shared by dry-run commands."""

    model: ModelConfig
    scoring_position: int = 10000
    scoring_window_tokens: int = 512
    prompt_count: int = 24
    output_dir: Path = Path("results")

    def validate(self) -> None:
        """Raise ``ValueError`` if required release settings are invalid."""

        if not self.model.name:
            raise ValueError("model.name must not be empty")
        if not self.model.model_id:
            raise ValueError("model.model_id must not be empty")
        if self.prompt_count <= 0:
            raise ValueError("prompt_count must be positive")
        if self.scoring_position <= 0:
            raise ValueError("scoring_position must be positive")
        if self.scoring_window_tokens <= 0:
            raise ValueError("scoring_window_tokens must be positive")


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load an experiment config from YAML."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    model_payload = payload.get("model")
    if not isinstance(model_payload, dict):
        raise ValueError("config missing model mapping")
    model = ModelConfig(
        name=str(model_payload["name"]),
        model_id=str(model_payload["model_id"]),
        seed=int(model_payload["seed"]),
        revision=_optional_str(model_payload.get("revision")),
        dtype=str(model_payload.get("dtype", "bfloat16")),
        trust_remote_code=bool(model_payload.get("trust_remote_code", True)),
    )
    config = ExperimentConfig(
        model=model,
        scoring_position=int(payload.get("scoring_position", 10000)),
        scoring_window_tokens=int(payload.get("scoring_window_tokens", 512)),
        prompt_count=int(payload.get("prompt_count", 24)),
        output_dir=Path(str(payload.get("output_dir", "results"))),
    )
    config.validate()
    return config


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
