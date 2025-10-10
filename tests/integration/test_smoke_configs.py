"""Integration tests for smoke test configurations."""

import json
from pathlib import Path

import pytest

from latentwire.config import TrainingConfig
from latentwire.cli.utils import apply_overrides


# List all smoke configurations
SMOKE_CONFIGS = [
    "base.json",
    "coprocessor.json",
    "deep_prefix.json",
    "gist_head.json",
    "latent_adapters.json",
    "lora.json",
    "prefix.json",
    "refiner.json"
]


@pytest.mark.parametrize("config_name", SMOKE_CONFIGS)
def test_smoke_config_loads(config_name):
    """Test that each smoke config can be loaded successfully."""
    config_path = Path("configs/smoke") / config_name

    if not config_path.exists():
        pytest.skip(f"Config {config_path} not found")

    with open(config_path) as f:
        config_dict = json.load(f)

    # Should be able to create a TrainingConfig from it
    config = TrainingConfig.from_dict(config_dict)
    assert config is not None


@pytest.mark.parametrize("config_name", SMOKE_CONFIGS)
def test_smoke_config_features(config_name):
    """Test that each smoke config has the expected features enabled."""
    config_path = Path("configs/smoke") / config_name

    if not config_path.exists():
        pytest.skip(f"Config {config_path} not found")

    with open(config_path) as f:
        config_dict = json.load(f)

    config = TrainingConfig.from_dict(config_dict)

    # Map config names to expected features
    expected_features = {
        "base.json": {},
        "coprocessor.json": {"use_coprocessor": True},
        "deep_prefix.json": {"use_deep_prefix": True},
        "gist_head.json": {"use_gist_head": True},
        "latent_adapters.json": {"use_latent_adapters": True},
        "lora.json": {"use_lora": True},
        "prefix.json": {"use_prefix": True},
        "refiner.json": {"use_latent_refiner": True}
    }

    expected = expected_features.get(config_name, {})

    for feature, value in expected.items():
        assert getattr(config.features, feature) == value, \
            f"Feature {feature} should be {value} in {config_name}"


def test_smoke_configs_mutual_exclusion():
    """Test that mutually exclusive features are not enabled together."""
    for config_name in SMOKE_CONFIGS:
        config_path = Path("configs/smoke") / config_name

        if not config_path.exists():
            continue

        with open(config_path) as f:
            config_dict = json.load(f)

        config = TrainingConfig.from_dict(config_dict)

        # Check mutual exclusions
        if config.features.use_deep_prefix:
            assert not config.features.use_coprocessor, \
                f"{config_name}: deep_prefix and coprocessor are mutually exclusive"
            assert not config.features.use_prefix, \
                f"{config_name}: deep_prefix and prefix are mutually exclusive"

        if config.features.use_coprocessor:
            assert not config.features.use_deep_prefix, \
                f"{config_name}: coprocessor and deep_prefix are mutually exclusive"


def test_smoke_configs_have_required_fields():
    """Test that all smoke configs have required fields."""
    # These are nested fields in the config structure
    required_nested = [
        ("model", "models"),
        ("data", "dataset"),
        ("data", "samples"),
        ("data", "epochs"),
        ("data", "batch_size")
    ]

    for config_name in SMOKE_CONFIGS:
        config_path = Path("configs/smoke") / config_name

        if not config_path.exists():
            continue

        with open(config_path) as f:
            config_dict = json.load(f)

        for section, field in required_nested:
            assert section in config_dict, \
                f"{config_name} missing section: {section}"
            assert field in config_dict[section], \
                f"{config_name} missing field: {section}.{field}"


def test_smoke_configs_cli_conversion():
    """Test that smoke configs can be converted to CLI arguments."""
    from latentwire.cli.utils import config_to_argv

    for config_name in SMOKE_CONFIGS:
        config_path = Path("configs/smoke") / config_name

        if not config_path.exists():
            continue

        with open(config_path) as f:
            config_dict = json.load(f)

        # Should be able to convert to argv
        argv = config_to_argv(config_dict)
        assert isinstance(argv, list)
        assert len(argv) > 0


def test_base_config_is_minimal():
    """Test that base config has minimal features enabled."""
    config_path = Path("configs/smoke/base.json")

    if not config_path.exists():
        pytest.skip("Base config not found")

    with open(config_path) as f:
        config_dict = json.load(f)

    config = TrainingConfig.from_dict(config_dict)

    # Base should have no special features enabled
    assert not config.features.use_lora
    assert not config.features.use_prefix
    assert not config.features.use_deep_prefix
    assert not config.features.use_latent_adapters
    assert not config.features.use_coprocessor
    assert not config.features.use_gist_head
    assert not config.features.use_latent_refiner


def test_smoke_configs_validation():
    """Test that all smoke configs pass validation."""
    for config_name in SMOKE_CONFIGS:
        config_path = Path("configs/smoke") / config_name

        if not config_path.exists():
            continue

        with open(config_path) as f:
            config_dict = json.load(f)

        config = TrainingConfig.from_dict(config_dict)

        # Validation should not raise errors
        warnings = config.validate()

        # Check for critical warnings
        for warning in warnings:
            # These are acceptable warnings
            if "mutually exclusive" in warning.lower():
                continue
            if "experimental" in warning.lower():
                continue
            # Unexpected warning
            print(f"Warning in {config_name}: {warning}")


def test_smoke_config_overrides():
    """Test that smoke configs can have overrides applied."""
    config_path = Path("configs/smoke/base.json")

    if not config_path.exists():
        pytest.skip("Base config not found")

    with open(config_path) as f:
        config_dict = json.load(f)

    # Apply some overrides - must be strings in key=value format
    overrides = [
        "data.samples=256",
        "data.batch_size=16",
        "optimizer.lr=0.001"
    ]

    config_dict = apply_overrides(config_dict, overrides)

    assert config_dict["data"]["samples"] == 256
    assert config_dict["data"]["batch_size"] == 16
    assert config_dict["optimizer"]["lr"] == 0.001


def test_smoke_configs_consistent_structure():
    """Test that all smoke configs have consistent structure."""
    first_config = None
    first_name = None

    for config_name in SMOKE_CONFIGS:
        config_path = Path("configs/smoke") / config_name

        if not config_path.exists():
            continue

        with open(config_path) as f:
            config_dict = json.load(f)

        if first_config is None:
            first_config = set(config_dict.keys())
            first_name = config_name
        else:
            current_keys = set(config_dict.keys())

            # Check for major structural differences
            # (small differences are OK for feature-specific configs)
            major_keys = {"models", "dataset", "samples", "epochs", "batch_size"}

            first_major = first_config.intersection(major_keys)
            current_major = current_keys.intersection(major_keys)

            assert first_major == current_major, \
                f"Structural difference between {first_name} and {config_name}"