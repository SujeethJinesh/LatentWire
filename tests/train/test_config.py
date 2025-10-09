import argparse

from latentwire.config import TrainingConfig


def test_training_config_defaults_valid():
    cfg = TrainingConfig()
    warnings = cfg.validate()
    assert warnings == []


def test_training_config_mutual_exclusion():
    cfg = TrainingConfig()
    cfg.features.use_deep_prefix = True
    cfg.features.use_coprocessor = True
    warnings = cfg.validate()
    assert any("deep_prefix" in msg for msg in warnings)


def test_baseline_verification_resets_features():
    cfg = TrainingConfig()
    cfg.features.use_deep_prefix = True
    cfg.features.use_latent_adapters = True
    cfg.features.use_coprocessor = True
    cfg.baseline_verification = True
    cfg.apply_baseline_verification()
    assert not cfg.features.use_deep_prefix
    assert not cfg.features.use_latent_adapters
    assert not cfg.features.use_coprocessor


def test_from_args_handles_coprocessor():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_coprocessor", action="store_true")
    parser.add_argument("--coprocessor_len", type=int, default=1)
    parser.add_argument("--coprocessor_pool", type=str, default="mean")
    args = parser.parse_args(["--use_coprocessor", "--coprocessor_len", "2", "--coprocessor_pool", "first"])
    cfg = TrainingConfig.from_args(args)
    assert cfg.features.use_coprocessor is True
    assert cfg.coprocessor.coprocessor_len == 2
    assert cfg.coprocessor.coprocessor_pool == "first"
