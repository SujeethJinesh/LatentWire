"""Environment verification helpers for the release package."""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

from outlier_migrate.config import load_experiment_config
from outlier_migrate.registry import default_registry


@dataclass(frozen=True)
class ConfigCheck:
    """Validation result for one release YAML config."""

    path: str
    model_id: str
    ok: bool


@dataclass(frozen=True)
class EnvironmentReport:
    """Serializable environment report for release dry-runs."""

    python_version: str
    platform: str
    package_version: str | None
    pyyaml_version: str | None
    registered_methods: tuple[str, ...]
    config_checks: tuple[ConfigCheck, ...]

    def to_dict(self) -> dict[str, object]:
        """Convert the report to a JSON-serializable mapping."""

        return asdict(self)


def collect_environment(config_paths: tuple[Path, ...] = ()) -> EnvironmentReport:
    """Collect package and config state without requiring model weights."""

    checks: list[ConfigCheck] = []
    for config_path in config_paths:
        config = load_experiment_config(config_path)
        checks.append(
            ConfigCheck(
                path=str(config_path),
                model_id=config.model.model_id,
                ok=True,
            )
        )
    return EnvironmentReport(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        package_version=_package_version("outlier-migrate-release"),
        pyyaml_version=_package_version("PyYAML"),
        registered_methods=tuple(default_registry().names()),
        config_checks=tuple(checks),
    )


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
