from __future__ import annotations

import torch

import latent_bridge.translator as translator_mod
from latent_bridge.translator import RotAlignKVTranslator, TranslatorConfig


class _TinyQuantizer:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def fit(self, x: torch.Tensor) -> None:
        self.shape = x.shape

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        return codes

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_grouped_adaptive_canonical_transport_records_soft_plan(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _TinyQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    tr = RotAlignKVTranslator(
        TranslatorConfig(
            src_head_dim=2,
            src_num_heads=2,
            num_src_layers=1,
            tgt_head_dim=2,
            tgt_num_heads=2,
            num_tgt_layers=1,
            alignment_method="grouped_adaptive_canonical_transport",
            canonical_subspace_rank=2,
            transport_signature_weight=0.25,
            transport_temperature=0.1,
            transport_sinkhorn_iters=16,
        )
    )

    torch.manual_seed(0)
    src = torch.randn(6, 2, 3, 2)
    tgt = src.flip(1).contiguous()

    tr.fit_from_pairs([(src, src + 0.1)], [(tgt, tgt + 0.1)])
    plan = tr.transport_plan_K[0].detach()

    assert plan.shape == (2, 2)
    assert torch.allclose(plan.sum(dim=1), torch.ones(2), atol=1e-5)
    assert plan[0, 1] > plan[0, 0]
    assert plan[1, 0] > plan[1, 1]
