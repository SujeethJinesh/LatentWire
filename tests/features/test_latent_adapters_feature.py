import types

import pytest

try:
    import torch.nn as nn
except OSError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"PyTorch unavailable: {exc}", allow_module_level=True)

from latentwire.features.latent_adapters import LatentAdaptersFeature


class WrapperWithAdapters:
    def __init__(self, trainable: bool = True):
        block = nn.Linear(4, 4)
        if not trainable:
            for p in block.parameters():
                p.requires_grad_(False)
        self.latent_adapters = nn.ModuleDict({"0": nn.Sequential(block)})
        self.use_latent_adapters = True
        self.cfg = types.SimpleNamespace(
            latent_adapter_layers=(0,),
            latent_adapter_heads=2,
            latent_adapter_dropout=0.0,
            latent_d_z=4,
        )
        self.latent_adapter_layers = (0,)


class WrapperWithoutAdapters:
    def __init__(self):
        self.latent_adapters = nn.ModuleDict()
        self.use_latent_adapters = True
        self.cfg = types.SimpleNamespace(
            latent_adapter_layers=(),
            latent_adapter_heads=2,
            latent_adapter_dropout=0.0,
            latent_d_z=4,
        )
        self.latent_adapter_layers = ()


def _context(args, extras=None):
    extras = extras or {}
    return types.SimpleNamespace(args=args, extras=extras, get=extras.get)


def test_latent_adapters_feature_collects_params():
    args = types.SimpleNamespace(use_latent_adapters=True, lr=1e-3)
    feature = LatentAdaptersFeature()
    wrappers = {"llama": WrapperWithAdapters()}
    context = _context(args)
    extra_params = {"llama": []}
    feature.apply_post_model_build(context, wrappers, extra_params)
    assert extra_params["llama"]
    groups = feature.optimizer_param_groups()
    assert len(groups) == 1


def test_latent_adapters_feature_raises_on_empty():
    args = types.SimpleNamespace(use_latent_adapters=True, lr=1e-3)
    feature = LatentAdaptersFeature()
    wrappers = {"llama": WrapperWithoutAdapters()}
    context = _context(args)
    extra_params = {"llama": []}
    with pytest.raises(RuntimeError):
        feature.apply_post_model_build(context, wrappers, extra_params)


def test_latent_adapters_feature_raises_on_no_trainable():
    args = types.SimpleNamespace(use_latent_adapters=True, lr=1e-3)
    feature = LatentAdaptersFeature()
    wrappers = {"llama": WrapperWithAdapters(trainable=False)}
    context = _context(args)
    extra_params = {"llama": []}
    with pytest.raises(RuntimeError):
        feature.apply_post_model_build(context, wrappers, extra_params)
