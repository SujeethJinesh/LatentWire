from __future__ import annotations

from types import SimpleNamespace

import torch

from latent_bridge import (
    RotAlignKVTranslator,
    TranslatorConfig,
    alignment_quality,
    apply_rotation,
    fit_alignment,
    hadamard_matrix,
    dct_matrix,
    make_rotation,
    random_orthogonal,
)
import latent_bridge.translator as translator_mod


class _TinyQuantizer:
    def __init__(self, bits: int = 4) -> None:
        self.bits = bits

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def to(self, *args, **kwargs):
        return self


class _OffsetQuantizer(_TinyQuantizer):
    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.25


def _make_identity_translator(
    monkeypatch,
    *,
    src_layers: int = 1,
    tgt_layers: int = 1,
    **config_overrides,
) -> RotAlignKVTranslator:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _TinyQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    cfg = TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=src_layers,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=tgt_layers,
        quant_bits=2,
        rotation_kind="orthogonal",
        layer_pairing="interp",
        **config_overrides,
    )
    tr = RotAlignKVTranslator(cfg)
    with torch.no_grad():
        tr.R_s.copy_(torch.eye(2))
        tr.R_t.copy_(torch.eye(2))
        for w in list(tr.W_K) + list(tr.W_V):
            w.copy_(torch.eye(4))
        tr._fitted = True
    return tr


def test_rotation_helpers_are_orthogonal() -> None:
    R = random_orthogonal(4, seed=123)
    H = hadamard_matrix(4, seed=123)
    D = dct_matrix(4, seed=123)

    assert R.shape == (4, 4)
    assert H.shape == (4, 4)
    assert D.shape == (4, 4)
    assert torch.allclose(R.T @ R, torch.eye(4), atol=1e-5, rtol=1e-5)
    assert torch.allclose(H.T @ H, torch.eye(4), atol=1e-5, rtol=1e-5)
    assert torch.allclose(D.T @ D, torch.eye(4), atol=1e-5, rtol=1e-5)

    x = torch.randn(2, 3, 4)
    y = apply_rotation(x, R)
    assert y.shape == x.shape

    # `make_rotation` should dispatch to the same shapes for the supported kinds.
    assert make_rotation(4, kind="orthogonal", seed=0).shape == (4, 4)
    assert make_rotation(4, kind="hadamard", seed=0).shape == (4, 4)
    assert make_rotation(4, kind="dct", seed=0).shape == (4, 4)


def test_fit_alignment_and_quality_match_a_known_linear_map() -> None:
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    W_true = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    Y = X @ W_true

    W_hat = fit_alignment(X, Y, method="auto")
    q = alignment_quality(X, Y, W_hat)

    assert torch.allclose(W_hat, W_true, atol=1e-6, rtol=1e-6)
    assert q["relative_frobenius_error"] < 1e-6
    assert q["mean_cosine_similarity"] > 0.999


def test_identity_rotation_and_alignment_ablation_helpers() -> None:
    x = torch.randn(5, 4)
    R = make_rotation(4, kind="identity")
    assert torch.equal(R, torch.eye(4))
    y = apply_rotation(x, R)
    assert torch.equal(y, x)

    W = fit_alignment(torch.randn(3, 4), torch.randn(3, 6), method="identity")
    assert W.shape == (4, 6)
    assert torch.allclose(W[:4, :4], torch.eye(4))


def test_translate_fuse_and_roundtrip_checkpointing(monkeypatch, tmp_path) -> None:
    tr = _make_identity_translator(monkeypatch)

    K = torch.arange(12, dtype=torch.float32).view(1, 2, 3, 2)
    V = K + 100.0
    K_hat, V_hat = tr.translate_layer(K, V, tgt_layer_idx=0, quantize=False)

    assert torch.equal(K_hat, K)
    assert torch.equal(V_hat, V)

    tr.gate_K[0].data.fill_(0.0)
    tr.gate_V[0].data.fill_(torch.logit(torch.tensor(0.8)))
    K_out, V_out = tr.fuse_layer(torch.zeros_like(K), torch.zeros_like(V), K_hat, V_hat, 0)

    assert torch.allclose(K_out, 0.5 * K_hat)
    assert torch.allclose(V_out, 0.8 * V_hat)

    path = tmp_path / "translator.pt"
    tr.save(str(path))
    loaded = RotAlignKVTranslator.load(str(path))

    assert loaded.config == tr.config
    assert loaded._fitted is True
    assert torch.equal(loaded.R_s, tr.R_s)
    assert torch.equal(loaded.W_K[0], tr.W_K[0])


def test_cosine_fusion_rule_suppresses_opposing_translated_kv(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch)
    tr.set_fixed_gates(0.8)

    K_target = torch.ones(1, 2, 2, 2)
    V_target = 2.0 * torch.ones(1, 2, 2, 2)
    K_opposed = -K_target
    V_opposed = -V_target

    K_static, V_static = tr.fuse_layer(
        K_target,
        V_target,
        K_opposed,
        V_opposed,
        0,
        fusion_rule="static",
    )
    K_cos, V_cos = tr.fuse_layer(
        K_target,
        V_target,
        K_opposed,
        V_opposed,
        0,
        fusion_rule="cosine",
    )

    assert torch.allclose(K_static, (1.0 - 2.0 * 0.8) * K_target)
    assert torch.allclose(V_static, (1.0 - 2.0 * 0.8) * V_target)
    assert torch.allclose(K_cos, K_target)
    assert torch.allclose(V_cos, V_target)


def test_js_and_kalman_fusion_rules_downweight_noisy_translation(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch)
    tr.set_fixed_gates(0.8)

    K_target = torch.ones(1, 2, 2, 2)
    V_target = 2.0 * torch.ones(1, 2, 2, 2)
    K_noisy = 5.0 * torch.ones_like(K_target)
    V_noisy = 5.0 * torch.ones_like(V_target)

    K_js, V_js = tr.fuse_layer(K_target, V_target, K_noisy, V_noisy, 0, fusion_rule="js_shrinkage")
    K_kal, V_kal = tr.fuse_layer(K_target, V_target, K_noisy, V_noisy, 0, fusion_rule="kalman")

    assert torch.all(K_js <= K_noisy)
    assert torch.all(K_js >= K_target)
    assert torch.all(V_js <= V_noisy)
    assert torch.all(V_js >= V_target)
    assert torch.all(K_kal <= K_noisy)
    assert torch.all(K_kal >= K_target)
    assert torch.all(V_kal <= V_noisy)
    assert torch.all(V_kal >= V_target)


def test_tokenwise_kalman_fusion_downweights_only_noisy_positions(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch)
    tr.set_fixed_gates(0.8)

    K_target = torch.ones(1, 2, 2, 2)
    V_target = 2.0 * torch.ones(1, 2, 2, 2)
    K_translated = K_target.clone()
    V_translated = V_target.clone()
    K_translated[:, :, 1] = 5.0
    V_translated[:, :, 1] = 6.0

    K_out, V_out = tr.fuse_layer(
        K_target,
        V_target,
        K_translated,
        V_translated,
        0,
        fusion_rule="kalman_tokenwise",
    )

    assert torch.allclose(K_out[:, :, 0], K_target[:, :, 0])
    assert torch.allclose(V_out[:, :, 0], V_target[:, :, 0])
    assert torch.all(K_out[:, :, 1] < 2.0)
    assert torch.all(K_out[:, :, 1] > K_target[:, :, 1])
    assert torch.all(V_out[:, :, 1] < 3.0)
    assert torch.all(V_out[:, :, 1] > V_target[:, :, 1])


def test_matched_noise_quantization_control_keeps_shape(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch)
    tr.quantizer = _OffsetQuantizer(bits=2)
    K = torch.zeros(1, 2, 3, 2)
    V = torch.ones(1, 2, 3, 2)

    K_real, V_real = tr.translate_layer(K, V, tgt_layer_idx=0, quantize=True, quantization_control="real")
    K_noise, V_noise = tr.translate_layer(
        K,
        V,
        tgt_layer_idx=0,
        quantize=True,
        quantization_control="matched_noise",
    )

    assert K_real.shape == K_noise.shape == K.shape
    assert V_real.shape == V_noise.shape == V.shape
    assert torch.allclose(K_real, torch.full_like(K, 0.25))
    assert torch.allclose(V_real, torch.full_like(V, 1.25))


def test_layer_pairing_interpolation_and_explicit_lists(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch, src_layers=3, tgt_layers=5)
    assert tr.layer_map == [0, 1, 1, 2, 2]

    cfg = TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=3,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=3,
        layer_pairing=[2, 0, 1],
    )
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _TinyQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))
    tr2 = RotAlignKVTranslator(cfg)
    assert tr2.layer_map == [2, 0, 1]

    shifted_cfg = TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=3,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=5,
        layer_pairing="shifted",
    )
    shifted = RotAlignKVTranslator(shifted_cfg)
    assert shifted.layer_map == [1, 2, 2, 0, 0]

    reverse_cfg = TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=3,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=5,
        layer_pairing="reverse",
    )
    reverse = RotAlignKVTranslator(reverse_cfg)
    assert reverse.layer_map == [2, 1, 1, 0, 0]


def test_layer_selection_and_fixed_gate_helpers(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch, src_layers=2, tgt_layers=2)
    diagnostics = {
        0: {
            "K": {"mean_cosine_similarity": 0.9, "relative_frobenius_error": 0.1},
            "V": {"mean_cosine_similarity": 0.8, "relative_frobenius_error": 0.2},
        },
        1: {
            "K": {"mean_cosine_similarity": 0.3, "relative_frobenius_error": 0.7},
            "V": {"mean_cosine_similarity": 0.2, "relative_frobenius_error": 0.8},
        },
    }
    tr.config.layer_selection_topk = 1
    tr._apply_layer_selection(diagnostics)
    assert tr.selected_layer_indices() == [0]
    assert tr.is_layer_selected(0) is True
    assert tr.is_layer_selected(1) is False

    tr.set_fixed_gates(0.25, 0.75)
    gk, gv = tr.gate_value(0)
    assert abs(gk - 0.25) < 1e-5
    assert abs(gv - 0.75) < 1e-5


def test_grouped_head_alignment_and_whitening_stay_block_diagonal(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _TinyQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    cfg = TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=1,
        tgt_head_dim=2,
        tgt_num_heads=4,
        num_tgt_layers=1,
        alignment_method="grouped_identity",
        use_whitening=True,
    )
    tr = RotAlignKVTranslator(cfg)

    src_kvs = [(torch.randn(2, 2, 3, 2), torch.randn(2, 2, 3, 2))]
    tgt_kvs = [(torch.randn(2, 4, 3, 2), torch.randn(2, 4, 3, 2))]
    tr.fit_from_pairs(src_kvs, tgt_kvs)

    W = tr.W_K[0].detach()
    expected_block = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=W.dtype,
    )
    assert torch.allclose(W[:2, :4], expected_block)
    assert torch.allclose(W[2:, 4:], expected_block)
    assert torch.allclose(W[:2, 4:], torch.zeros(2, 4), atol=1e-6)
    assert torch.allclose(W[2:, :4], torch.zeros(2, 4), atol=1e-6)

    W_zca = tr.whiten_K_src[0].detach()
    assert torch.allclose(W_zca[:2, 2:], torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(W_zca[2:, :2], torch.zeros(2, 2), atol=1e-6)


def test_grouped_transport_can_swap_head_groups(monkeypatch) -> None:
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
                alignment_method="grouped_transport",
                transport_temperature=0.1,
                transport_sinkhorn_iters=16,
            )
        )
    baseline = RotAlignKVTranslator(
        TranslatorConfig(
            src_head_dim=2,
            src_num_heads=2,
            num_src_layers=1,
            tgt_head_dim=2,
            tgt_num_heads=2,
            num_tgt_layers=1,
            alignment_method="grouped_identity",
        )
    )

    torch.manual_seed(0)
    src = torch.randn(6, 2, 3, 2)
    tgt = src.flip(1).contiguous()

    tr.fit_from_pairs([(src, src + 0.1)], [(tgt, tgt + 0.1)])
    baseline.fit_from_pairs([(src, src + 0.1)], [(tgt, tgt + 0.1)])
    pred, _ = tr.translate_layer(src, src + 0.1, tgt_layer_idx=0, quantize=False)
    pred_base, _ = baseline.translate_layer(src, src + 0.1, tgt_layer_idx=0, quantize=False)

    err = (pred - tgt).pow(2).mean()
    err_base = (pred_base - tgt).pow(2).mean()
    plan = tr.transport_plan_K[0].detach()

    assert err < err_base
    assert plan[0, 1] > plan[0, 0]
    assert plan[1, 0] > plan[1, 1]


def test_grouped_permutation_can_swap_head_groups(monkeypatch) -> None:
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
            alignment_method="grouped_permutation",
        )
    )

    torch.manual_seed(0)
    src = torch.randn(6, 2, 3, 2)
    tgt = src.flip(1).contiguous()

    tr.fit_from_pairs([(src, src + 0.1)], [(tgt, tgt + 0.1)])
    pred, _ = tr.translate_layer(src, src + 0.1, tgt_layer_idx=0, quantize=False)

    err = (pred - tgt).pow(2).mean()
    plan = tr.transport_plan_K[0].detach()

    assert err < 1e-4
    assert torch.equal(plan, torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=plan.dtype))


def test_grouped_signature_transport_records_soft_plan(monkeypatch) -> None:
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
            alignment_method="grouped_signature_transport",
            transport_signature_weight=0.25,
            transport_signature_rank=2,
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


def test_grouped_subspace_transport_records_soft_plan(monkeypatch) -> None:
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
            alignment_method="grouped_subspace_transport",
            transport_signature_weight=0.25,
            transport_signature_rank=2,
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


def test_grouped_canonical_transport_records_soft_plan(monkeypatch) -> None:
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
            alignment_method="grouped_canonical_transport",
            canonical_subspace_rank=2,
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


def test_grouped_covariance_transport_records_soft_plan(monkeypatch) -> None:
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
            alignment_method="grouped_covariance_transport",
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


def test_grouped_template_transport_records_soft_plan(monkeypatch) -> None:
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
            alignment_method="grouped_template_transport",
            transport_signature_weight=0.25,
            transport_temperature=0.1,
            transport_sinkhorn_iters=16,
            transport_template_bins=4,
        )
    )
    tr._transport_src_group_templates = [torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])]
    tr._transport_tgt_group_templates = [torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])]

    torch.manual_seed(0)
    src = torch.randn(6, 2, 3, 2)
    tgt = src.flip(1).contiguous()

    tr.fit_from_pairs([(src, src + 0.1)], [(tgt, tgt + 0.1)])
    plan = tr.transport_plan_K[0].detach()

    assert plan.shape == (2, 2)
    assert torch.allclose(plan.sum(dim=1), torch.ones(2), atol=1e-5)
    assert plan[0, 1] > plan[0, 0]
    assert plan[1, 0] > plan[1, 1]


def test_grouped_template_subspace_transport_records_soft_plan(monkeypatch) -> None:
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
            alignment_method="grouped_template_subspace_transport",
            transport_signature_weight=0.25,
            transport_temperature=0.1,
            transport_sinkhorn_iters=16,
            transport_template_bins=4,
        )
    )
    tr._transport_src_group_templates = [torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])]
    tr._transport_tgt_group_templates = [torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])]

    torch.manual_seed(0)
    src = torch.randn(6, 2, 3, 2)
    tgt = src.flip(1).contiguous()

    tr.fit_from_pairs([(src, src + 0.1)], [(tgt, tgt + 0.1)])
    plan = tr.transport_plan_K[0].detach()

    assert plan.shape == (2, 2)
    assert torch.allclose(plan.sum(dim=1), torch.ones(2), atol=1e-5)
    assert plan[0, 1] > plan[0, 0]
    assert plan[1, 0] > plan[1, 1]


def test_broadcast_template_transport_records_rectangular_plan(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _TinyQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    tr = RotAlignKVTranslator(
        TranslatorConfig(
            src_head_dim=2,
            src_num_heads=2,
            num_src_layers=1,
            tgt_head_dim=2,
            tgt_num_heads=4,
            num_tgt_layers=1,
            alignment_method="broadcast_template_transport",
            transport_signature_weight=0.25,
            transport_temperature=0.1,
            transport_residual_rank=None,
            transport_template_bins=4,
        )
    )
    tr._transport_src_group_templates = [torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])]
    tr._transport_tgt_group_templates = [torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])]

    torch.manual_seed(0)
    src = torch.randn(6, 2, 3, 2)
    tgt = src[:, [1, 1, 0, 0]].contiguous()

    tr.fit_from_pairs([(src, src + 0.1)], [(tgt, tgt + 0.1)])
    assert tr._broadcast_transport_plan_K is not None
    plan = tr._broadcast_transport_plan_K[0].detach()

    assert plan.shape == (2, 4)
    assert torch.allclose(plan.sum(dim=1), torch.ones(2), atol=1e-5)
    assert plan[0, 2] > plan[0, 0]
    assert plan[0, 3] > plan[0, 1]
    assert plan[1, 0] > plan[1, 2]
    assert plan[1, 1] > plan[1, 3]


def test_broadcast_template_ot_transport_balances_target_columns(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _TinyQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    tr = RotAlignKVTranslator(
        TranslatorConfig(
            src_head_dim=2,
            src_num_heads=2,
            num_src_layers=1,
            tgt_head_dim=2,
            tgt_num_heads=4,
            num_tgt_layers=1,
            alignment_method="broadcast_template_ot_transport",
            transport_signature_weight=0.25,
            transport_temperature=0.1,
            transport_sinkhorn_iters=16,
            transport_residual_rank=None,
            transport_template_bins=4,
        )
    )
    tr._transport_src_group_templates = [torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])]
    tr._transport_tgt_group_templates = [torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])]

    torch.manual_seed(0)
    src = torch.randn(6, 2, 3, 2)
    tgt = src[:, [1, 1, 0, 0]].contiguous()

    tr.fit_from_pairs([(src, src + 0.1)], [(tgt, tgt + 0.1)])
    assert tr._broadcast_transport_plan_K is not None
    plan = tr._broadcast_transport_plan_K[0].detach()

    assert plan.shape == (2, 4)
    assert torch.allclose(plan.sum(dim=0), torch.ones(4), atol=1e-4)
    assert torch.allclose(plan.sum(dim=1), torch.full((2,), 2.0), atol=1e-3)
    assert plan[0, 2] > plan[0, 0]
    assert plan[0, 3] > plan[0, 1]
    assert plan[1, 0] > plan[1, 2]
    assert plan[1, 1] > plan[1, 3]


def test_broadcast_retrieval_spectrum_ot_transport_balances_target_columns(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _TinyQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    tr = RotAlignKVTranslator(
        TranslatorConfig(
            src_head_dim=2,
            src_num_heads=2,
            num_src_layers=1,
            tgt_head_dim=2,
            tgt_num_heads=4,
            num_tgt_layers=1,
            alignment_method="broadcast_retrieval_spectrum_ot_transport",
            transport_signature_weight=0.25,
            transport_temperature=0.1,
            transport_sinkhorn_iters=16,
            transport_residual_rank=None,
            transport_signature_rank=4,
        )
    )
    tr._transport_src_group_templates = [torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.3, 0.5]])]
    tr._transport_tgt_group_templates = [torch.tensor([
        [0.0, 0.2, 0.3, 0.5],
        [0.0, 0.2, 0.3, 0.5],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])]

    torch.manual_seed(0)
    src = torch.randn(6, 2, 3, 2)
    tgt = src[:, [1, 1, 0, 0]].contiguous()

    tr.fit_from_pairs([(src, src + 0.1)], [(tgt, tgt + 0.1)])
    assert tr._broadcast_transport_plan_K is not None
    plan = tr._broadcast_transport_plan_K[0].detach()

    assert plan.shape == (2, 4)
    assert torch.allclose(plan.sum(dim=0), torch.ones(4), atol=1e-4)
    assert torch.allclose(plan.sum(dim=1), torch.full((2,), 2.0), atol=1e-3)
    assert plan[0, 2] > plan[0, 0]
    assert plan[0, 3] > plan[0, 1]
    assert plan[1, 0] > plan[1, 2]
    assert plan[1, 1] > plan[1, 3]


def test_broadcast_qk_template_ot_transport_balances_target_columns(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _TinyQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    tr = RotAlignKVTranslator(
        TranslatorConfig(
            src_head_dim=2,
            src_num_heads=2,
            num_src_layers=1,
            tgt_head_dim=2,
            tgt_num_heads=4,
            num_tgt_layers=1,
            alignment_method="broadcast_qk_template_ot_transport",
            transport_signature_weight=0.25,
            transport_temperature=0.1,
            transport_sinkhorn_iters=16,
            transport_residual_rank=None,
            transport_template_bins=4,
        )
    )
    tr._transport_src_group_templates = [torch.tensor([[1.0, -1.0, 0.0, 0.0], [0.0, 0.2, -0.2, 1.0]])]
    tr._transport_tgt_group_templates = [torch.tensor([
        [0.0, 0.2, -0.2, 1.0],
        [0.0, 0.2, -0.2, 1.0],
        [1.0, -1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
    ])]

    torch.manual_seed(0)
    src = torch.randn(6, 2, 3, 2)
    tgt = src[:, [1, 1, 0, 0]].contiguous()

    tr.fit_from_pairs([(src, src + 0.1)], [(tgt, tgt + 0.1)])
    assert tr._broadcast_transport_plan_K is not None
    plan = tr._broadcast_transport_plan_K[0].detach()

    assert plan.shape == (2, 4)
    assert torch.allclose(plan.sum(dim=0), torch.ones(4), atol=1e-4)
    assert torch.allclose(plan.sum(dim=1), torch.full((2,), 2.0), atol=1e-3)
    assert plan[0, 2] > plan[0, 0]
    assert plan[0, 3] > plan[0, 1]
    assert plan[1, 0] > plan[1, 2]
    assert plan[1, 1] > plan[1, 3]


def test_target_whitening_recovers_anisotropic_target_under_procrustes(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _TinyQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    torch.manual_seed(0)
    cfg_base = dict(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=1,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=1,
        alignment_method="procrustes",
    )
    tr_plain = RotAlignKVTranslator(TranslatorConfig(**cfg_base))
    tr_whiten = RotAlignKVTranslator(TranslatorConfig(**cfg_base, use_target_whitening=True))

    src = torch.randn(8, 2, 4, 2, dtype=torch.float32)
    flat = src.permute(0, 2, 1, 3).reshape(-1, 4)
    scale = torch.diag(torch.tensor([3.0, 0.5, 2.0, 0.25], dtype=torch.float32))
    bias = torch.tensor([[0.3, -0.2, 0.15, 0.4]], dtype=torch.float32)
    tgt_flat = flat @ scale + bias
    tgt = tgt_flat.reshape(8, 4, 2, 2).permute(0, 2, 1, 3).contiguous()

    src_kvs = [(src, src + 0.1)]
    tgt_kvs = [(tgt, tgt + 0.1)]

    tr_plain.fit_from_pairs(src_kvs, tgt_kvs)
    diagnostics = tr_whiten.fit_from_pairs(src_kvs, tgt_kvs)

    pred_plain, _ = tr_plain.translate_layer(src, src + 0.1, tgt_layer_idx=0, quantize=False)
    pred_whiten, _ = tr_whiten.translate_layer(src, src + 0.1, tgt_layer_idx=0, quantize=False)

    err_plain = (pred_plain - tgt).pow(2).mean()
    err_whiten = (pred_whiten - tgt).pow(2).mean()

    assert err_whiten < err_plain * 0.25
    assert tr_whiten.whiten_K_tgt is not None
    assert tr_whiten.whiten_K_tgt_inv is not None
    assert "original_space_relative_frobenius_error" in diagnostics[0]["K"]


def test_target_whitening_save_load_preserves_translation(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _TinyQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    torch.manual_seed(1)
    tr = RotAlignKVTranslator(
        TranslatorConfig(
            src_head_dim=2,
            src_num_heads=2,
            num_src_layers=1,
            tgt_head_dim=2,
            tgt_num_heads=2,
            num_tgt_layers=1,
            alignment_method="procrustes",
            use_target_whitening=True,
        )
    )
    src = torch.randn(4, 2, 3, 2, dtype=torch.float32)
    flat = src.permute(0, 2, 1, 3).reshape(-1, 4)
    scale = torch.diag(torch.tensor([2.0, 0.5, 1.5, 0.75], dtype=torch.float32))
    bias = torch.tensor([[0.1, -0.3, 0.2, 0.05]], dtype=torch.float32)
    tgt = (flat @ scale + bias).reshape(4, 3, 2, 2).permute(0, 2, 1, 3).contiguous()
    tr.fit_from_pairs([(src, src + 0.2)], [(tgt, tgt + 0.2)])

    before, _ = tr.translate_layer(src, src + 0.2, tgt_layer_idx=0, quantize=False)

    path = tmp_path / "target_whitening.pt"
    tr.save(str(path))
    loaded = RotAlignKVTranslator.load(str(path))

    after, _ = loaded.translate_layer(src, src + 0.2, tgt_layer_idx=0, quantize=False)

    assert loaded.config.use_target_whitening is True
    assert torch.allclose(after, before)
    assert torch.allclose(loaded.whiten_K_tgt[0], tr.whiten_K_tgt[0])
    assert torch.allclose(loaded.whiten_K_tgt_inv[0], tr.whiten_K_tgt_inv[0])


def test_learned_affine_fusion_uses_stored_coordinatewise_scales(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch)
    with torch.no_grad():
        tr.fusion_src_scale_K[0].fill_(2.0)
        tr.fusion_tgt_scale_K[0].fill_(0.5)
        tr.fusion_bias_K[0].fill_(1.0)
        tr.fusion_src_scale_V[0].fill_(1.5)
        tr.fusion_tgt_scale_V[0].fill_(0.25)
        tr.fusion_bias_V[0].fill_(0.5)

    K_target = torch.ones(1, 2, 2, 2)
    V_target = 2.0 * torch.ones(1, 2, 2, 2)
    K_translated = 3.0 * torch.ones_like(K_target)
    V_translated = 4.0 * torch.ones_like(V_target)

    K_out, V_out = tr.fuse_layer(
        K_target,
        V_target,
        K_translated,
        V_translated,
        0,
        fusion_rule="learned_affine",
    )

    assert torch.allclose(K_out, 2.0 * K_translated + 0.5 * K_target + 1.0)
    assert torch.allclose(V_out, 1.5 * V_translated + 0.25 * V_target + 0.5)


def test_learned_head_ridge_fusion_uses_stored_headwise_projection(monkeypatch) -> None:
    cfg = TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=1,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=1,
        learned_fusion_dropout=0.5,
    )
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))
    tr = RotAlignKVTranslator(cfg)
    with torch.no_grad():
        for head_idx in range(2):
            weight = torch.zeros(4, 2)
            weight[:2] = 2.0 * torch.eye(2)
            weight[2:] = 0.5 * torch.eye(2)
            tr.fusion_head_proj_K[0][head_idx].copy_(weight)
            tr.fusion_head_proj_V[0][head_idx].copy_(weight)
            tr.fusion_head_bias_K[0][head_idx].fill_(1.0)
            tr.fusion_head_bias_V[0][head_idx].fill_(0.5)

    K_target = torch.ones(1, 2, 2, 2)
    V_target = 2.0 * torch.ones(1, 2, 2, 2)
    K_translated = 3.0 * torch.ones_like(K_target)
    V_translated = 4.0 * torch.ones_like(V_target)

    K_out, V_out = tr.fuse_layer(
        K_target,
        V_target,
        K_translated,
        V_translated,
        0,
        fusion_rule="learned_head_ridge",
    )

    assert torch.allclose(K_out, 2.0 * K_translated + 0.5 * K_target + 1.0)
    assert torch.allclose(V_out, 2.0 * V_translated + 0.5 * V_target + 0.5)


def test_head_selection_masks_unselected_heads(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch)
    tr.head_selected_mask[0].copy_(torch.tensor([True, False]))

    kv = torch.arange(8, dtype=torch.float32).view(1, 2, 2, 2)
    masked = tr.apply_head_selection(kv, 0)

    assert torch.equal(masked[:, 0], kv[:, 0])
    assert torch.equal(masked[:, 1], torch.zeros_like(kv[:, 1]))


def test_cosine_shifted_uses_only_selected_heads_for_adaptive_gate(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch)
    tr.set_fixed_gates(0.8)
    tr.head_selected_mask[0].copy_(torch.tensor([True, False]))

    K_target = torch.ones(1, 2, 2, 2)
    V_target = 2.0 * torch.ones(1, 2, 2, 2)
    K_translated = K_target.clone()
    V_translated = V_target.clone()
    K_translated[:, 0] = -K_target[:, 0]
    V_translated[:, 0] = -V_target[:, 0]

    K_out, V_out = tr.fuse_layer(
        K_target,
        V_target,
        K_translated,
        V_translated,
        0,
        fusion_rule="cosine_shifted",
    )

    assert torch.allclose(K_out, K_target)
    assert torch.allclose(V_out, V_target)


def test_cosine_shifted_tokenwise_preserves_agreeing_positions(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch)
    tr.set_fixed_gates(0.8)

    K_target = torch.ones(1, 2, 2, 2)
    V_target = 2.0 * torch.ones(1, 2, 2, 2)
    K_translated = K_target.clone()
    V_translated = V_target.clone()
    K_translated[:, :, 1] = -K_target[:, :, 1]
    V_translated[:, :, 1] = -V_target[:, :, 1]

    K_out, V_out = tr.fuse_layer(
        K_target,
        V_target,
        K_translated,
        V_translated,
        0,
        fusion_rule="cosine_shifted_tokenwise",
    )

    assert torch.allclose(K_out[:, :, 0], K_target[:, :, 0])
    assert torch.allclose(V_out[:, :, 0], V_target[:, :, 0])
    assert torch.allclose(K_out[:, :, 1], K_target[:, :, 1])
    assert torch.allclose(V_out[:, :, 1], V_target[:, :, 1])


def test_pre_quant_rank_zero_disables_filter_even_with_shrinkage(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        pre_quant_rank=0,
        pre_quant_shrinkage=0.25,
    )
    Y = torch.randn(8, tr.d_t)

    filt = tr._fit_pre_quant_filter(Y)

    assert torch.allclose(filt, torch.eye(tr.d_t))


def test_fit_from_pairs_populates_prequant_filter_and_affine_correction(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _OffsetQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    tr = RotAlignKVTranslator(
        TranslatorConfig(
            src_head_dim=2,
            src_num_heads=2,
            num_src_layers=1,
            tgt_head_dim=2,
            tgt_num_heads=2,
            num_tgt_layers=1,
            pre_quant_rank=1,
            pre_quant_shrinkage=0.25,
            quantization_correction="affine",
        )
    )
    base = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.5, 0.0], [0.0, 0.5]],
            ],
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
        ],
        dtype=torch.float32,
    )
    src_kvs = [(base, base + 0.5)]
    tgt_kvs = [(base * 1.5, base * 2.0)]

    tr.fit_from_pairs(src_kvs, tgt_kvs)

    assert not torch.allclose(tr.pre_quant_filter_K[0], torch.eye(tr.d_t))
    assert (
        not torch.allclose(tr.quant_scale_K[0], torch.ones_like(tr.quant_scale_K[0]))
        or not torch.allclose(tr.quant_bias_K[0], torch.zeros_like(tr.quant_bias_K[0]))
    )


def test_fit_from_pairs_ridge_correction_reduces_quantized_error(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _OffsetQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    cfg_none = TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=1,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=1,
        quantization_correction="none",
        ridge_lambda=1e-4,
    )
    cfg_ridge = TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=1,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=1,
        quantization_correction="ridge",
        ridge_lambda=1e-4,
    )
    tr_none = RotAlignKVTranslator(cfg_none)
    tr_ridge = RotAlignKVTranslator(cfg_ridge)

    base = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.5, 0.0], [0.0, 0.5]],
            ],
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
        ],
        dtype=torch.float32,
    )
    src_kvs = [(base, base + 0.5)]
    tgt_kvs = [(base * 1.5, base * 2.0)]

    tr_none.fit_from_pairs(src_kvs, tgt_kvs)
    tr_ridge.fit_from_pairs(src_kvs, tgt_kvs)

    K_none, _ = tr_none.translate_layer(base, base + 0.5, tgt_layer_idx=0, quantize=True)
    K_ridge, _ = tr_ridge.translate_layer(base, base + 0.5, tgt_layer_idx=0, quantize=True)
    target = base * 1.5

    err_none = (K_none - target).pow(2).mean()
    err_ridge = (K_ridge - target).pow(2).mean()

    assert err_ridge < err_none
    assert not torch.allclose(tr_ridge.quant_proj_K[0], torch.eye(tr_ridge.d_t))


def test_fit_from_pairs_with_learned_affine_and_no_quant_correction_does_not_crash(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    cfg = TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=1,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=1,
        quantization_correction="none",
        learned_fusion_dropout=0.5,
        ridge_lambda=1e-4,
    )
    tr = RotAlignKVTranslator(cfg)

    base = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.5, 0.0], [0.0, 0.5]],
            ],
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
        ],
        dtype=torch.float32,
    )
    src_kvs = [(base, base + 0.5)]
    tgt_kvs = [(base * 1.5, base * 2.0)]

    tr.fit_from_pairs(src_kvs, tgt_kvs)

    assert not torch.allclose(tr.fusion_src_scale_K[0], torch.zeros_like(tr.fusion_src_scale_K[0]))


def test_fit_from_pairs_populates_learned_head_ridge_projection(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    cfg = TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=1,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=1,
        quantization_correction="none",
        learned_fusion_dropout=0.5,
        ridge_lambda=1e-4,
    )
    tr = RotAlignKVTranslator(cfg)

    base = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.5, 0.0], [0.0, 0.5]],
            ],
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
        ],
        dtype=torch.float32,
    )
    src_kvs = [(base, base + 0.5)]
    tgt_kvs = [(base * 1.5, base * 2.0)]

    tr.fit_from_pairs(src_kvs, tgt_kvs)

    assert not torch.allclose(tr.fusion_head_proj_K[0], torch.zeros_like(tr.fusion_head_proj_K[0]))
