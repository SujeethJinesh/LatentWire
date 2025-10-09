import pytest

try:
    import torch
    import torch.nn as nn
except OSError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"PyTorch unavailable: {exc}", allow_module_level=True)

from latentwire.loss_bundles import (
    loss_with_text_prompt_chunked,
    alignment_mse,
    manifold_stat_loss,
    scale_penalty,
    rms_raw_penalty,
)


class StubWrapper:
    def __init__(self):
        self.calls = 0

    def loss_with_text_prompt(self, scaffold_ids, target_ids):
        self.calls += 1
        loss = torch.tensor(0.5, dtype=torch.float32)
        return loss, scaffold_ids.size(0), None


class StubAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))


class StubWrapperRMS:
    def input_embedding_rms(self, sample_rows: int = 0):
        return 0.75


def test_loss_with_text_prompt_chunked():
    wrapper = StubWrapper()
    scaffold = torch.randint(0, 5, (3, 4))
    targets = torch.randint(0, 5, (3, 4))
    loss, _, _ = loss_with_text_prompt_chunked(wrapper, scaffold, targets)
    assert torch.isclose(loss, torch.tensor(0.5))
    assert wrapper.calls == 1


def test_alignment_mse():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    mask = torch.tensor([1, 0])
    loss = alignment_mse(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_manifold_stat_loss():
    prefix = torch.randn(2, 3, 4)
    ref_mu = torch.zeros(4)
    ref_sd = torch.ones(4)
    loss = manifold_stat_loss(prefix, (ref_mu, ref_sd), weight=0.5)
    assert loss >= 0.0


def test_scale_penalty():
    adapter = StubAdapter()
    penalty = scale_penalty(adapter, weight=1.0, device=torch.device("cpu"))
    assert penalty == 0.0
    adapter.scale.data.fill_(2.0)
    penalty = scale_penalty(adapter, weight=1.0, device=torch.device("cpu"))
    assert penalty > 0.0


def test_rms_raw_penalty():
    prefix = torch.full((2, 3, 4), 0.5)
    wrapper = StubWrapperRMS()
    penalty = rms_raw_penalty(prefix, wrapper, weight=1.0)
    assert penalty >= 0.0
