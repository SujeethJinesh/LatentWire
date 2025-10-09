import types

import pytest

try:
    import torch
    import torch.nn as nn
except OSError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"PyTorch unavailable: {exc}", allow_module_level=True)

from latentwire.features.deep_prefix import DeepPrefixFeature


class DummyWrapper:
    def __init__(self):
        self.model = nn.Linear(4, 4)
        self.model.config = types.SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        self.d_model = 4


def _context(args, extras=None):
    extras = extras or {}
    return types.SimpleNamespace(args=args, extras=extras, get=extras.get)


def test_deep_prefix_generator_shapes():
    args = types.SimpleNamespace(
        latent_len=4,
        deep_prefix_len=2,
        deep_prefix_dropout=0.0,
        lr=1e-3,
    )
    feature = DeepPrefixFeature()
    context = _context(args, {"latent_shared_len": 2, "latent_private_len": 0})
    wrappers = {"llama": DummyWrapper()}
    extra = {"llama": []}
    feature.apply_post_model_build(context, wrappers, extra)

    generator = feature.generators["llama"]
    latents = torch.randn(3, 2, 4)
    cache = generator(latents)
    assert len(cache) == wrappers["llama"].model.config.num_hidden_layers
    key, value = cache[0]
    assert key.shape == (3, 2, 2, 2)
    assert value.shape == (3, 2, 2, 2)


def test_deep_prefix_state_restore():
    args = types.SimpleNamespace(
        latent_len=4,
        deep_prefix_len=2,
        deep_prefix_dropout=0.0,
        lr=1e-3,
    )
    feature = DeepPrefixFeature()
    dummy_state = {}
    context = _context(
        args,
        {
            "latent_shared_len": 2,
            "latent_private_len": 0,
            "deep_prefix_state": {"llama": dummy_state},
        },
    )
    wrappers = {"llama": DummyWrapper()}
    extra = {"llama": []}
    feature.apply_post_model_build(context, wrappers, extra)
    # After apply_post_model_build the state dict should be loadable (even if empty)
    generator = feature.generators["llama"]
    state = generator.state_dict()
    generator.load_state_dict(state)
