import types

import pytest

try:
    import torch
    import torch.nn as nn
except OSError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"PyTorch unavailable: {exc}", allow_module_level=True)

from latentwire.features.coproc import CoprocessorFeature
from latentwire.train import _merge_kv_caches


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


def test_coprocessor_feature_builds_generators():
    args = types.SimpleNamespace(
        use_coprocessor=True,
        use_deep_prefix=False,
        coprocessor_len=1,
        coprocessor_width=8,
        coprocessor_dropout=0.0,
        coprocessor_kv_scale=0.5,
        coprocessor_pool="mean",
        coprocessor_heads="3,2",
        lr=1e-3,
    )
    feature = CoprocessorFeature()
    context = _context(
        args,
        {
            "latent_shared_len": 2,
            "latent_private_len": 0,
            "coprocessor_state": {},
        },
    )
    wrappers = {"llama": DummyWrapper()}
    extra = {"llama": []}
    feature.apply_post_model_build(context, wrappers, extra)
    module = feature.coprocessors["llama"]
    assert module.heads_per_layer == [3, 2]


def test_coprocessor_feature_mutual_exclusion():
    args = types.SimpleNamespace(
        use_coprocessor=True,
        use_deep_prefix=True,
        coprocessor_len=1,
        coprocessor_width=8,
        coprocessor_dropout=0.0,
        coprocessor_kv_scale=0.5,
        coprocessor_pool="mean",
        coprocessor_heads="",
        lr=1e-3,
    )
    feature = CoprocessorFeature()
    context = _context(args)
    with pytest.raises(ValueError):
        feature.apply_post_model_build(context, {"llama": DummyWrapper()}, {"llama": []})


def test_merge_kv_caches():
    k1 = torch.randn(1, 2, 1, 4)
    v1 = torch.randn(1, 2, 1, 4)
    k2 = torch.randn(1, 2, 1, 4)
    v2 = torch.randn(1, 2, 1, 4)
    merged = _merge_kv_caches([(k1, v1)], [(k2, v2)])
    assert merged[0][0].shape == (1, 2, 2, 4)
    assert merged[0][1].shape == (1, 2, 2, 4)
