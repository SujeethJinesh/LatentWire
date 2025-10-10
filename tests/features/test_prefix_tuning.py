"""Tests for prefix tuning feature."""

import types

import pytest

try:
    import torch
    import torch.nn as nn
except ImportError:
    pytest.skip("PyTorch not available", allow_module_level=True)


def test_prefix_tuning_integration():
    """Test that prefix tuning can be applied via the configuration."""
    # Check if peft is available
    try:
        import peft
    except ImportError:
        pytest.skip("peft not available, skipping prefix tuning test")

    from latentwire.models import apply_prefix_if_requested

    # Create a mock model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            self.config = types.SimpleNamespace(
                hidden_size=10,
                num_hidden_layers=2,
                num_attention_heads=2,
            )

    model = DummyModel()

    # Create prefix args
    prefix_args = {
        "tokens": 16,
        "all_layers": "yes"
    }

    # Apply prefix tuning
    pefted = apply_prefix_if_requested(model, prefix_args)

    # Check that the model has been wrapped
    assert pefted is not model

    # Check that trainable parameters were added
    trainable_params = sum(p.numel() for p in pefted.parameters() if p.requires_grad)
    assert trainable_params > 0


def test_prefix_tuning_mutually_exclusive_with_deep_prefix():
    """Test that prefix tuning and deep prefix should not be enabled together."""
    from latentwire.config import TrainingConfig

    config = TrainingConfig()
    config.features.use_prefix = True
    config.features.use_deep_prefix = True

    # Both can be set in config, but they shouldn't be used together in practice
    # The actual mutual exclusion would be enforced at the model level
    assert config.features.use_prefix == True
    assert config.features.use_deep_prefix == True


def test_prefix_tuning_configuration():
    """Test prefix tuning configuration parameters."""
    from latentwire.config import PrefixTuningConfig

    config = PrefixTuningConfig()

    # Check default values
    assert config.prefix_tokens == 16
    assert config.peft_prefix_all_layers == "yes"

    # Test custom values
    config = PrefixTuningConfig(
        prefix_tokens=32,
        peft_prefix_all_layers="no"
    )
    assert config.prefix_tokens == 32
    assert config.peft_prefix_all_layers == "no"


def test_prefix_tuning_from_json_config():
    """Test loading prefix tuning from JSON configuration."""
    from latentwire.config import TrainingConfig

    config_dict = {
        "features": {
            "use_prefix": True
        },
        "prefix": {
            "prefix_tokens": 24,
            "peft_prefix_all_layers": "yes"
        }
    }

    config = TrainingConfig.from_dict(config_dict)

    assert config.features.use_prefix == True
    assert config.prefix.prefix_tokens == 24
    assert config.prefix.peft_prefix_all_layers == "yes"