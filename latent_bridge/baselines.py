"""Lightweight adapter scaffolding for external communication baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PublishedBaselineArtifact:
    baseline: str
    repo_id: str
    source_model: str
    target_model: str
    subdir: str
    config_path: str
    checkpoint_dir: str
    local_root: str | None = None


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
    repo_id = "nics-efc/C2C_Fuser"
    published_pair_subdirs = {
        ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B"): "qwen3_0.6b+qwen2.5_0.5b_Fuser",
        ("meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen3-0.6B"): "qwen3_0.6b+llam3.2_1b_Fuser",
        ("Qwen/Qwen2.5-Math-1.5B", "Qwen/Qwen3-0.6B"): "qwen3_0.6b+qwen2.5_1.5b_math_Fuser",
        ("Qwen/Qwen3-4B", "Qwen/Qwen3-0.6B"): "qwen3_0.6b+qwen3_4b_Fuser",
        ("Qwen/Qwen3-4B-Base", "Qwen/Qwen3-0.6B"): "qwen3_0.6b+qwen3_4b_base_Fuser",
        ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-1.7B"): "qwen3_1.7b+qwen2.5_1.5b_Fuser",
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B"): "qwen3_8b+qwen2.5_7b_Fuser",
    }

    @classmethod
    def published_subdir(cls, source_model: str, target_model: str) -> str:
        key = (source_model, target_model)
        if key not in cls.published_pair_subdirs:
            raise KeyError(f"No published C2C artifact registered for pair: {source_model} -> {target_model}")
        return cls.published_pair_subdirs[key]

    @classmethod
    def published_artifact(
        cls,
        source_model: str,
        target_model: str,
        *,
        local_root: str | None = None,
    ) -> PublishedBaselineArtifact:
        subdir = cls.published_subdir(source_model, target_model)
        config_path = f"{subdir}/config.json"
        checkpoint_dir = f"{subdir}/final"
        return PublishedBaselineArtifact(
            baseline=cls.name,
            repo_id=cls.repo_id,
            source_model=source_model,
            target_model=target_model,
            subdir=subdir,
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            local_root=local_root,
        )

    @classmethod
    def prepare_published_artifact(
        cls,
        source_model: str,
        target_model: str,
        *,
        download: bool = False,
    ) -> PublishedBaselineArtifact:
        artifact = cls.published_artifact(source_model, target_model)
        if not download:
            return artifact

        from huggingface_hub import snapshot_download

        local_root = snapshot_download(
            repo_id=artifact.repo_id,
            allow_patterns=[f"{artifact.subdir}/*"],
        )
        return PublishedBaselineArtifact(
            baseline=artifact.baseline,
            repo_id=artifact.repo_id,
            source_model=artifact.source_model,
            target_model=artifact.target_model,
            subdir=artifact.subdir,
            config_path=artifact.config_path,
            checkpoint_dir=artifact.checkpoint_dir,
            local_root=local_root,
        )

    @staticmethod
    def local_config_path(artifact: PublishedBaselineArtifact) -> str | None:
        if artifact.local_root is None:
            return None
        return str(Path(artifact.local_root) / artifact.config_path)

    @staticmethod
    def local_checkpoint_dir(artifact: PublishedBaselineArtifact) -> str | None:
        if artifact.local_root is None:
            return None
        return str(Path(artifact.local_root) / artifact.checkpoint_dir)

    def fit(self, context: BaselineContext) -> None:
        try:
            self.artifact = self.published_artifact(context.source_model, context.target_model)
        except KeyError:
            self.artifact = None

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
