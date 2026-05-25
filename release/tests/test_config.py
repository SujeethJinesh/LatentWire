from pathlib import Path

import pytest

from outlier_migrate.config import ExperimentConfig, ModelConfig, load_experiment_config


def test_placeholder_configs_load() -> None:
    for path in sorted(Path("configs").glob("*.yaml")):
        config = load_experiment_config(path)
        assert config.model.model_id
        assert config.prompt_count == 24
        assert config.scoring_window_tokens == 512


def test_config_validation_rejects_empty_model_id() -> None:
    config = ExperimentConfig(model=ModelConfig(name="bad", model_id="", seed=1))
    with pytest.raises(ValueError, match="model.model_id"):
        config.validate()
