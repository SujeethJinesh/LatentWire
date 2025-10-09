import types

import pytest

try:
    import torch
    import torch.nn as nn
except OSError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"PyTorch unavailable: {exc}", allow_module_level=True)

from latentwire.feature_registry import FeatureRegistry
from latentwire.config import TrainingConfig


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.config = types.SimpleNamespace(
            hidden_size=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )


class DummyWrapper:
    def __init__(self, use_latent_adapters: bool = False):
        self.model = DummyModel()
        self.tokenizer = types.SimpleNamespace(pad_token_id=0)
        self.d_model = 4
        self.use_latent_adapters = use_latent_adapters
        if use_latent_adapters:
            block = nn.Linear(4, 4)
            self.latent_adapters = nn.ModuleDict({"0": nn.Sequential(block)})
            self.latent_adapter_layers = (0,)
            self.cfg = types.SimpleNamespace(
                latent_adapter_layers=(0,),
                latent_adapter_heads=2,
                latent_adapter_dropout=0.0,
                latent_d_z=4,
            )
        else:
            self.latent_adapters = nn.ModuleDict()
            self.latent_adapter_layers = ()
            self.cfg = types.SimpleNamespace(
                latent_adapter_layers=(),
                latent_adapter_heads=2,
                latent_adapter_dropout=0.0,
                latent_d_z=4,
            )


def _base_args(**overrides):
    defaults = dict(
        use_lora=False,
        use_deep_prefix=False,
        use_latent_adapters=False,
        use_coprocessor=False,
        llama_id="llama",
        qwen_id="qwen",
        d_z=4,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_firstN=None,
        lora_target_modules="auto",
        latent_len=4,
        deep_prefix_len=2,
        deep_prefix_dropout=0.0,
        latent_adapter_layers="1",
        latent_adapter_heads=2,
        latent_adapter_dropout=0.0,
        lr=1e-3,
        coprocessor_len=1,
        coprocessor_width=8,
        coprocessor_dropout=0.0,
        coprocessor_kv_scale=0.5,
        coprocessor_pool="mean",
        coprocessor_heads="",
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


@pytest.fixture
def dummy_wrappers():
    return {"llama": DummyWrapper(), "qwen": DummyWrapper()}


def test_mutual_exclusion_validation():
    cfg = TrainingConfig()
    cfg.features.use_deep_prefix = True
    cfg.features.use_coprocessor = True
    warnings = cfg.validate()
    assert any("deep_prefix and coprocessor" in msg for msg in warnings)


def test_registry_with_lora(monkeypatch, dummy_wrappers):
    args = _base_args(use_lora=True)
    registry = FeatureRegistry(args)
    assert registry.names() == ["lora"]

    def fake_apply(model, lora_args, model_name):
        # add a trainable parameter so optimizer groups are non-empty
        class DummyWithEmbed(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.embed = nn.Linear(4, 4)

            def forward(self, *args, **kwargs):  # pragma: no cover - unused
                return self.base(*args, **kwargs)

            def get_input_embeddings(self):
                return self.embed

            def train(self, mode=True):
                self.base.train(mode)
                super().train(mode)
                return self

        return DummyWithEmbed(model)

    monkeypatch.setattr("latentwire.feature_registry.apply_lora_if_requested", fake_apply)

    extra = registry.apply_post_model_build(dummy_wrappers)
    assert extra["llama"] and extra["qwen"]
    groups = registry.optimizer_param_groups()
    assert groups == []


def test_registry_with_deep_prefix(dummy_wrappers):
    args = _base_args(use_deep_prefix=True, latent_len=4, deep_prefix_len=2)
    registry = FeatureRegistry(args)
    registry.set_extra("latent_shared_len", 2)
    registry.set_extra("latent_private_len", 0)
    generators_state = {}
    registry.set_extra("deep_prefix_state", generators_state)

    extra = registry.apply_post_model_build(dummy_wrappers)
    state = registry.state.get("deep_prefix_generators", {})
    assert set(state.keys()) == {"llama", "qwen"}


def test_registry_with_latent_adapters():
    args = _base_args(use_latent_adapters=True)
    registry = FeatureRegistry(args)
    wrappers = {"llama": DummyWrapper(use_latent_adapters=True)}
    registry.set_extra("latent_shared_len", 2)
    registry.set_extra("latent_private_len", 2)
    registry.apply_post_model_build(wrappers)
    groups = registry.optimizer_param_groups()
    assert len(groups) == 1


def test_registry_with_coprocessor(dummy_wrappers):
    args = _base_args(use_coprocessor=True, coprocessor_len=1, coprocessor_width=8)
    registry = FeatureRegistry(args)
    registry.set_extra("latent_shared_len", 2)
    registry.set_extra("latent_private_len", 0)
    registry.set_extra("coprocessor_state", {})
    extra = registry.apply_post_model_build(dummy_wrappers)
    assert extra["llama"] and extra["qwen"]
    groups = registry.optimizer_param_groups()
    assert len(groups) == 1
