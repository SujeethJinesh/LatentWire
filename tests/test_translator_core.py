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


def test_fuse_layer_supports_per_head_gate_override(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch)
    tr.set_fixed_gates(0.5)

    K_target = torch.zeros(1, 2, 2, 2)
    V_target = torch.zeros(1, 2, 2, 2)
    K_translated = torch.ones(1, 2, 2, 2)
    V_translated = 2.0 * torch.ones(1, 2, 2, 2)

    K_out, V_out = tr.fuse_layer(
        K_target,
        V_target,
        K_translated,
        V_translated,
        0,
        head_gate_override_K=torch.tensor([1.0, 0.0]),
        head_gate_override_V=torch.tensor([1.0, 0.0]),
    )

    assert torch.allclose(K_out[:, 0], K_translated[:, 0])
    assert torch.allclose(K_out[:, 1], K_target[:, 1])
    assert torch.allclose(V_out[:, 0], V_translated[:, 0])
    assert torch.allclose(V_out[:, 1], V_target[:, 1])


def test_fuse_layer_supports_tokenwise_gate_override(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch)
    tr.set_fixed_gates(0.5)

    K_target = torch.zeros(1, 2, 2, 2)
    V_target = torch.zeros(1, 2, 2, 2)
    K_translated = torch.ones(1, 2, 2, 2)
    V_translated = 2.0 * torch.ones(1, 2, 2, 2)

    token_gate = torch.tensor(
        [[[[1.0], [0.0]], [[0.5], [0.5]]]],
        dtype=torch.float32,
    )
    K_out, V_out = tr.fuse_layer(
        K_target,
        V_target,
        K_translated,
        V_translated,
        0,
        head_gate_override_K=token_gate,
        head_gate_override_V=token_gate,
    )

    assert torch.allclose(K_out[:, 0, 0], K_translated[:, 0, 0])
    assert torch.allclose(K_out[:, 0, 1], K_target[:, 0, 1])
    assert torch.allclose(V_out[:, 0, 0], V_translated[:, 0, 0])
    assert torch.allclose(V_out[:, 0, 1], V_target[:, 0, 1])
    assert torch.allclose(K_out[:, 1], 0.5 * K_translated[:, 1])


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


def test_grouped_rotational_transport_records_soft_plan(monkeypatch) -> None:
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
            alignment_method="grouped_rotational_transport",
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


def test_grouped_fitted_rotation_transport_records_soft_plan(monkeypatch) -> None:
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
            alignment_method="grouped_fitted_rotation_transport",
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


def test_grouped_shared_basis_transport_records_soft_plan(monkeypatch) -> None:
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
            alignment_method="grouped_shared_basis_transport",
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


def test_grouped_qk_retrieval_transport_records_soft_plan(monkeypatch) -> None:
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
            alignment_method="grouped_qk_retrieval_transport",
            transport_signature_weight=0.25,
            transport_temperature=0.1,
            transport_sinkhorn_iters=16,
            transport_template_bins=4,
        )
    )
    tr._transport_src_group_templates = [torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.2, 0.7]])]
    tr._transport_tgt_group_templates = [torch.tensor([[0.0, 0.1, 0.2, 0.7], [1.0, 0.0, 0.0, 0.0]])]

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


def test_grouped_contrastive_template_transport_prefers_contrastive_pairing(monkeypatch) -> None:
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
            alignment_method="grouped_contrastive_template_transport",
            transport_signature_weight=1.0,
            transport_temperature=0.1,
            transport_sinkhorn_iters=16,
            transport_template_bins=4,
        )
    )
    tr._transport_src_group_template_banks = [torch.tensor([
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    ])]
    tr._transport_tgt_group_template_banks = [torch.tensor([
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    ])]

    monkeypatch.setattr(
        translator_mod,
        "fit_alignment",
        lambda X, Y, **kwargs: torch.eye(X.shape[1], Y.shape[1], dtype=X.dtype, device=X.device),
    )
    monkeypatch.setattr(
        translator_mod,
        "alignment_quality",
        lambda X, Y, W: {"mean_cosine_similarity": 0.0, "relative_frobenius_error": 1.0},
    )

    X = torch.randn(8, 4)
    Y = torch.randn(8, 4)
    _, plan = tr._fit_group_transport_alignment(
        X,
        Y,
        lam=1e-3,
        residual_rank=None,
        src_layer_idx=0,
        tgt_layer_idx=0,
    )

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


def test_selective_source_whitening_applies_only_to_v_stream(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        use_whitening=True,
        whitening_streams="v",
    )
    with torch.no_grad():
        tr.whiten_K_src[0].zero_()
        tr.whiten_K_mean[0].zero_()
        tr.whiten_V_src[0].copy_(2.0 * torch.eye(4))
        tr.whiten_V_mean[0].zero_()

    K = torch.arange(12, dtype=torch.float32).view(1, 2, 3, 2)
    V = K + 1.0

    K_hat, V_hat = tr.translate_layer(K, V, tgt_layer_idx=0, quantize=False)

    assert torch.allclose(K_hat, K)
    assert torch.allclose(V_hat, 2.0 * V)


def test_selective_target_whitening_applies_only_to_v_stream(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        use_target_whitening=True,
        target_whitening_streams="v",
    )
    with torch.no_grad():
        tr.whiten_K_tgt_inv[0].zero_()
        tr.whiten_K_tgt_mean[0].zero_()
        tr.whiten_V_tgt_inv[0].copy_(4.0 * torch.eye(4))
        tr.whiten_V_tgt_mean[0].zero_()

    K = torch.arange(12, dtype=torch.float32).view(1, 2, 3, 2)
    V = K + 1.0

    K_hat, V_hat = tr.translate_layer(K, V, tgt_layer_idx=0, quantize=False)

    assert torch.allclose(K_hat, K)
    assert torch.allclose(V_hat, 4.0 * V)


def test_fit_ridge_override_targets_only_requested_layer_and_stream(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        src_layers=1,
        tgt_layers=2,
        alignment_method="ridge",
        ridge_lambda=1e-3,
        fit_ridge_override_lambda=1e-2,
        fit_ridge_override_streams="v",
        fit_ridge_override_layers=(1,),
    )
    calls: list[float] = []

    def _fake_fit_alignment(
        X: torch.Tensor,
        Y: torch.Tensor,
        method: str = "auto",
        lam: float = 1e-3,
        rank: int | None = None,
    ) -> torch.Tensor:
        calls.append(float(lam))
        return torch.full(
            (X.shape[1], Y.shape[1]),
            float(lam),
            dtype=X.dtype,
            device=X.device,
        )

    monkeypatch.setattr(translator_mod, "fit_alignment", _fake_fit_alignment)

    src = torch.arange(24, dtype=torch.float32).view(2, 2, 3, 2)
    tr.fit_from_pairs(
        [(src, src + 1.0)],
        [
            (src + 2.0, src + 3.0),
            (src + 4.0, src + 5.0),
        ],
    )

    assert any(abs(call - 1e-2) < 1e-12 for call in calls)
    assert any(abs(call - 1e-3) < 1e-12 for call in calls)
    assert torch.allclose(tr.W_K[1], torch.full_like(tr.W_K[1], 1e-3))
    assert torch.allclose(tr.W_V[1], torch.full_like(tr.W_V[1], 1e-2))
    assert tr.config.fit_ridge_override_layers == (1,)


def test_fit_alignment_with_protected_outputs_splits_tail_lambda(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch)
    calls: list[tuple[float, int]] = []

    def _fake_fit_alignment(
        X: torch.Tensor,
        Y: torch.Tensor,
        method: str = "auto",
        lam: float = 1e-3,
        rank: int | None = None,
    ) -> torch.Tensor:
        del method, rank
        calls.append((float(lam), int(Y.shape[1])))
        return torch.full(
            (X.shape[1], Y.shape[1]),
            float(lam),
            dtype=X.dtype,
            device=X.device,
        )

    monkeypatch.setattr(translator_mod, "fit_alignment", _fake_fit_alignment)

    X = torch.randn(5, tr.d_s)
    Y = torch.randn(5, tr.d_t)
    mask = torch.tensor([True, False, True, False])

    W = tr._fit_alignment_with_protected_outputs(
        X,
        Y,
        method="ridge",
        lam=1e-2,
        protected_lam=1e-3,
        protected_output_mask=mask,
    )

    assert calls == [(1e-2, 4), (1e-3, 2)]
    assert torch.allclose(W[:, mask], torch.full_like(W[:, mask], 1e-3))
    assert torch.allclose(W[:, ~mask], torch.full_like(W[:, ~mask], 1e-2))


def test_fit_ridge_top_output_mask_selects_innovation_outliers(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch)
    residual = torch.tensor(
        [
            [0.0, 1.0, 10.0, 1.0],
            [0.0, 1.0, -8.0, 1.0],
            [5.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    mask = tr._fit_ridge_top_output_mask(residual, rank=2)

    assert mask.tolist() == [True, False, True, False]


def test_selective_conditioning_targets_only_requested_layers(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _TinyQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    tr = RotAlignKVTranslator(
        TranslatorConfig(
            src_head_dim=2,
            src_num_heads=2,
            num_src_layers=2,
            tgt_head_dim=2,
            tgt_num_heads=2,
            num_tgt_layers=2,
            alignment_method="procrustes",
            use_whitening=True,
            use_target_whitening=True,
            whitening_streams="v",
            target_whitening_streams="v",
            conditioning_target_layers=(1,),
        )
    )

    src0 = torch.randn(4, 2, 3, 2, dtype=torch.float32)
    src1 = torch.randn(4, 2, 3, 2, dtype=torch.float32)
    scale = torch.diag(torch.tensor([2.0, 0.5, 1.5, 0.75], dtype=torch.float32))
    bias = torch.tensor([[0.2, -0.3, 0.1, 0.05]], dtype=torch.float32)
    tgt1_flat = src1.permute(0, 2, 1, 3).reshape(-1, 4) @ scale + bias
    tgt1 = tgt1_flat.reshape(4, 3, 2, 2).permute(0, 2, 1, 3).contiguous()

    src_kvs = [
        (src0, src0 + 0.1),
        (src1, src1 + 0.2),
    ]
    tgt_kvs = [
        (src0, src0 + 0.1),
        (src1, tgt1),
    ]

    tr.fit_from_pairs(src_kvs, tgt_kvs)

    eye_src = torch.eye(4)
    eye_tgt = torch.eye(4)
    zero_mean = torch.zeros(1, 4)

    assert torch.allclose(tr.whiten_K_src[0], eye_src)
    assert torch.allclose(tr.whiten_K_src[1], eye_src)
    assert torch.allclose(tr.whiten_K_tgt[0], eye_tgt)
    assert torch.allclose(tr.whiten_K_tgt[1], eye_tgt)
    assert torch.allclose(tr.whiten_V_src[0], eye_src)
    assert torch.allclose(tr.whiten_V_tgt[0], eye_tgt)
    assert torch.allclose(tr.whiten_V_mean[0], zero_mean)
    assert torch.allclose(tr.whiten_V_tgt_mean[0], zero_mean)
    assert not torch.allclose(tr.whiten_V_src[1], eye_src)
    assert not torch.allclose(tr.whiten_V_tgt[1], eye_tgt)


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


def test_fit_from_pairs_bridge_affine_correction_reduces_quantized_error(monkeypatch) -> None:
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
    cfg_bridge = TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=1,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=1,
        quantization_correction="bridge_affine",
        ridge_lambda=1e-4,
    )
    tr_none = RotAlignKVTranslator(cfg_none)
    tr_bridge = RotAlignKVTranslator(cfg_bridge)

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
    tr_bridge.fit_from_pairs(src_kvs, tgt_kvs)

    K_none, _ = tr_none.translate_layer(base, base + 0.5, tgt_layer_idx=0, quantize=True)
    K_bridge, _ = tr_bridge.translate_layer(base, base + 0.5, tgt_layer_idx=0, quantize=True)
    target = base * 1.5

    err_none = (K_none - target).pow(2).mean()
    err_bridge = (K_bridge - target).pow(2).mean()

    assert err_bridge < err_none
    assert not torch.allclose(tr_bridge.quant_aux_scale_K[0], torch.zeros_like(tr_bridge.quant_aux_scale_K[0]))


def test_fit_from_pairs_bridge_ridge_correction_reduces_quantized_error(monkeypatch) -> None:
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
    cfg_bridge = TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=1,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=1,
        quantization_correction="bridge_ridge",
        ridge_lambda=1e-4,
    )
    tr_none = RotAlignKVTranslator(cfg_none)
    tr_bridge = RotAlignKVTranslator(cfg_bridge)

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
    tr_bridge.fit_from_pairs(src_kvs, tgt_kvs)

    K_none, _ = tr_none.translate_layer(base, base + 0.5, tgt_layer_idx=0, quantize=True)
    K_bridge, _ = tr_bridge.translate_layer(base, base + 0.5, tgt_layer_idx=0, quantize=True)
    target = base * 1.5

    err_none = (K_none - target).pow(2).mean()
    err_bridge = (K_bridge - target).pow(2).mean()

    assert err_bridge < err_none
    assert not torch.allclose(tr_bridge.quant_aux_proj_K[0], torch.zeros_like(tr_bridge.quant_aux_proj_K[0]))


def test_bridge_ridge_query_scales_bridge_by_runtime_template_agreement(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _OffsetQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    cfg_query = TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=1,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=1,
        quantization_correction="bridge_ridge_query",
        ridge_lambda=1e-4,
        transport_template_bins=4,
    )
    tr_query = RotAlignKVTranslator(cfg_query)

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

    tr_query.fit_from_pairs(src_kvs, tgt_kvs)
    tr_query.set_bridge_runtime_templates([torch.tensor([1.0, 0.0, 0.0, 0.0])])

    K_match, _ = tr_query.translate_layer(
        base,
        base + 0.5,
        tgt_layer_idx=0,
        quantize=True,
        runtime_attention_profile=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    K_mismatch, _ = tr_query.translate_layer(
        base,
        base + 0.5,
        tgt_layer_idx=0,
        quantize=True,
        runtime_attention_profile=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )

    target = base * 1.5
    err_match = (K_match - target).pow(2).mean()
    err_mismatch = (K_mismatch - target).pow(2).mean()

    assert err_match <= err_mismatch + 1e-6
    assert not torch.allclose(K_match, K_mismatch)


def test_fit_bridge_ridge_correction_accepts_sample_weights(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))
    tr = RotAlignKVTranslator(
        TranslatorConfig(
            src_head_dim=2,
            src_num_heads=2,
            num_src_layers=1,
            tgt_head_dim=2,
            tgt_num_heads=2,
            num_tgt_layers=1,
            quantization_correction="bridge_ridge_qk_weighted",
            ridge_lambda=1e-4,
        )
    )

    quantized = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    predicted = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 3.0],
        ],
        dtype=torch.float32,
    )

    proj_u, aux_u, bias_u = tr._fit_bridge_ridge_correction(
        quantized,
        predicted,
        target,
        lam=1e-4,
    )
    proj_w, aux_w, bias_w = tr._fit_bridge_ridge_correction(
        quantized,
        predicted,
        target,
        lam=1e-4,
        sample_weights=torch.tensor([1.0, 16.0]),
    )

    pred_u = quantized @ proj_u + predicted @ aux_u + bias_u
    pred_w = quantized @ proj_w + predicted @ aux_w + bias_w

    err_u = (pred_u[1] - target[1]).pow(2).mean()
    err_w = (pred_w[1] - target[1]).pow(2).mean()

    assert err_w <= err_u + 1e-6


def test_fit_bridge_ridge_query_projector_correction_uses_query_features(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))
    tr = RotAlignKVTranslator(
        TranslatorConfig(
            src_head_dim=2,
            src_num_heads=2,
            num_src_layers=1,
            tgt_head_dim=2,
            tgt_num_heads=2,
            num_tgt_layers=1,
            quantization_correction="bridge_ridge_qk_projector",
            ridge_lambda=1e-4,
        )
    )

    quantized = torch.tensor(
        [
            [1.0, 0.0, 0.5, 0.0],
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )
    predicted = quantized.clone()
    query_features = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    target = quantized + quantized * query_features

    proj_b, aux_b, bias_b = tr._fit_bridge_ridge_correction(
        quantized,
        predicted,
        target,
        lam=1e-4,
    )
    proj_q, aux_q, q_proj, q_aux_proj, bias_q = tr._fit_bridge_ridge_query_projector_correction(
        quantized,
        predicted,
        query_features,
        target,
        lam=1e-4,
    )

    pred_b = quantized @ proj_b + predicted @ aux_b + bias_b
    pred_q = (
        quantized @ proj_q
        + predicted @ aux_q
        + (quantized * query_features) @ q_proj
        + (predicted * query_features) @ q_aux_proj
        + bias_q
    )

    err_b = (pred_b - target).pow(2).mean()
    err_q = (pred_q - target).pow(2).mean()

    assert err_q <= err_b + 1e-6


def test_bridge_ridge_qk_adapter_adds_query_conditioned_residual(monkeypatch) -> None:
    for mode in (
        "bridge_ridge_qk_adapter",
        "bridge_ridge_qk_affinity_adapter",
        "bridge_ridge_qk_attnkl_adapter",
        "bridge_ridge_qk_cab_adapter",
        "bridge_ridge_qk_emkd_adapter",
        "bridge_ridge_qk_readout_adapter",
        "bridge_ridge_qk_predkl_adapter",
        "bridge_ridge_qk_asym_projector",
        "bridge_ridge_qk_asym_predkl_adapter",
        "bridge_ridge_qk_asym_dynmap_adapter",
        "bridge_ridge_qk_xattn_adapter",
        "bridge_ridge_qk_xattn_dynmap_adapter",
    ):
        tr = _make_identity_translator(
            monkeypatch,
            quantization_correction=mode,
            quantization_correction_rank=1,
        )
        with torch.no_grad():
            tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
            tr.quant_aux_proj_K[0].zero_()
            tr.quant_bias_K[0].zero_()
            tr.quant_query_resid_K_left[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
            tr.quant_query_resid_K_right[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
            tr.quant_query_aux_resid_K_left[0].zero_()
            tr.quant_query_aux_resid_K_right[0].zero_()
            tr.quant_proj_V[0].zero_()
            tr.quant_query_resid_V_left[0].copy_(torch.tensor([[0.5], [0.0], [0.5], [0.0]], dtype=torch.float32))
            tr.quant_query_resid_V_right[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
            tr.quant_query_aux_resid_V_left[0].zero_()
            tr.quant_query_aux_resid_V_right[0].zero_()

        base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
        out_k, out_v = tr.translate_layer(
            base,
            base,
            tgt_layer_idx=0,
            quantize=True,
            runtime_query_features=torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32),
        )

        assert out_k.shape == base.shape
        assert out_v.shape == base.shape
        assert not torch.allclose(out_k, base)
        assert not torch.allclose(out_v, base)


def test_fit_from_pairs_bridge_ridge_qk_readout_adapter_populates_k_and_v_residuals(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_readout_adapter",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_sample_prompt_ids(torch.tensor([0, 0, 1, 1], dtype=torch.long))

    calls: list[str] = []

    def fake_fit(
        self,
        quantized,
        predicted,
        query_features,
        base_prediction,
        residual_target,
        *,
        rank,
        readout_partner_kind=None,
        **kwargs,
    ):
        calls.append(str(readout_partner_kind))
        fill = float(len(calls))
        left = torch.full((self.d_t, rank), fill, dtype=residual_target.dtype)
        right = torch.zeros(rank, self.d_t, dtype=residual_target.dtype)
        right[0, 0] = fill
        aux_left = torch.zeros_like(left)
        aux_right = torch.zeros_like(right)
        return left, right, aux_left, aux_right

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_residual_adapter", fake_fit)

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

    assert calls == ["V", "K"]
    assert torch.allclose(tr.quant_query_resid_K_left[0], torch.ones_like(tr.quant_query_resid_K_left[0]))
    assert torch.allclose(tr.quant_query_resid_V_left[0], torch.full_like(tr.quant_query_resid_V_left[0], 2.0))


def test_fit_from_pairs_bridge_ridge_qk_predkl_adapter_passes_teacher_and_populates_k_and_v_residuals(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_predkl_adapter",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    calls: list[tuple[float, bool, bool]] = []

    def fake_fit(
        self,
        quantized,
        predicted,
        query_features,
        base_prediction,
        residual_target,
        *,
        rank,
        prediction_distill_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                teacher_topk_log_probs is not None,
                teacher_topk_output_rows is not None,
            )
        )
        fill = float(len(calls))
        left = torch.full((self.d_t, rank), fill, dtype=residual_target.dtype)
        right = torch.zeros(rank, self.d_t, dtype=residual_target.dtype)
        right[0, 0] = fill
        aux_left = torch.zeros_like(left)
        aux_right = torch.zeros_like(right)
        return left, right, aux_left, aux_right

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_residual_adapter", fake_fit)

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

    assert calls == [(0.25, True, True), (0.25, True, True)]
    assert torch.allclose(tr.quant_query_resid_K_left[0], torch.ones_like(tr.quant_query_resid_K_left[0]))
    assert torch.allclose(tr.quant_query_resid_V_left[0], torch.full_like(tr.quant_query_resid_V_left[0], 2.0))


def test_bridge_ridge_qk_asym_adapter_uses_shared_query_residual(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_asym_adapter",
        quantization_correction_rank=1,
    )
    with torch.no_grad():
        tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
        tr.quant_proj_V[0].copy_(torch.eye(tr.d_t))
        tr.quant_aux_proj_K[0].zero_()
        tr.quant_aux_proj_V[0].zero_()
        tr.quant_bias_K[0].zero_()
        tr.quant_bias_V[0].zero_()
        tr.quant_query_resid_K_left[0].zero_()
        tr.quant_query_resid_K_right[0].zero_()
        tr.quant_query_aux_resid_K_left[0].zero_()
        tr.quant_query_aux_resid_K_right[0].zero_()
        tr.quant_query_resid_V_left[0].zero_()
        tr.quant_query_resid_V_right[0].zero_()
        tr.quant_query_aux_resid_V_left[0].zero_()
        tr.quant_query_aux_resid_V_right[0].zero_()
        tr.quant_query_shared_left[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_shared_aux_left[0].zero_()
        tr.quant_query_shared_K_right[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_shared_V_right[0].copy_(torch.tensor([[0.5, 0.0, 0.0, 0.0]], dtype=torch.float32))

    K_base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
    V_base = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]], dtype=torch.float32)
    out_k, out_v = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_query_features=torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32),
    )

    assert out_k.shape == K_base.shape
    assert out_v.shape == V_base.shape
    assert not torch.allclose(out_k, K_base)
    assert not torch.allclose(out_v, V_base)


def test_fit_from_pairs_bridge_ridge_qk_asym_adapter_populates_shared_and_private_residuals(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_asym_adapter",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])

    calls: list[int] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        base_prediction_k,
        base_prediction_v,
        residual_target_k,
        residual_target_v,
        *,
        rank,
        **kwargs,
    ):
        calls.append(rank)
        return (
            torch.full((self.d_t, rank), 3.0, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 5.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=residual_target_v.dtype),
            torch.full((self.d_t, rank), 1.0, dtype=residual_target_k.dtype),
            torch.ones(rank, self.d_t, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 1.5, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 1.5, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=residual_target_v.dtype),
            torch.full((rank, self.d_t), 2.0, dtype=residual_target_v.dtype),
            torch.full((self.d_t, rank), 2.5, dtype=residual_target_v.dtype),
            torch.full((rank, self.d_t), 2.5, dtype=residual_target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_shared_residual_adapter", fake_fit)

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

    assert calls == [1]
    assert torch.allclose(tr.quant_query_shared_left[0], torch.full_like(tr.quant_query_shared_left[0], 3.0))


def test_fit_from_pairs_bridge_ridge_qk_asym_projector_populates_projector_and_shared_residuals(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_asym_projector",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])

    projector_calls: list[str] = []
    shared_calls: list[int] = []

    def fake_projector(self, quantized, predicted, query_features, target, *, lam):
        projector_calls.append(str(query_features.shape))
        fill = float(len(projector_calls))
        return (
            torch.full((self.d_t, self.d_t), fill, dtype=target.dtype),
            torch.full((self.d_t, self.d_t), fill + 1.0, dtype=target.dtype),
            torch.full((self.d_t, self.d_t), fill + 2.0, dtype=target.dtype),
            torch.full((self.d_t, self.d_t), fill + 3.0, dtype=target.dtype),
            torch.full((1, self.d_t), fill + 4.0, dtype=target.dtype),
        )

    def fake_shared(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        base_prediction_k,
        base_prediction_v,
        residual_target_k,
        residual_target_v,
        *,
        rank,
        **kwargs,
    ):
        shared_calls.append(rank)
        return (
            torch.full((self.d_t, rank), 3.0, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 5.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=residual_target_v.dtype),
            torch.full((self.d_t, rank), 1.0, dtype=residual_target_k.dtype),
            torch.ones(rank, self.d_t, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 1.5, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 1.5, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=residual_target_v.dtype),
            torch.full((rank, self.d_t), 2.0, dtype=residual_target_v.dtype),
            torch.full((self.d_t, rank), 2.5, dtype=residual_target_v.dtype),
            torch.full((rank, self.d_t), 2.5, dtype=residual_target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_ridge_query_projector_correction", fake_projector)
    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_shared_residual_adapter", fake_shared)

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

    assert len(projector_calls) == 2
    assert shared_calls == [1]
    assert torch.allclose(tr.quant_query_proj_K[0], torch.full_like(tr.quant_query_proj_K[0], 3.0))
    assert torch.allclose(tr.quant_query_aux_proj_K[0], torch.full_like(tr.quant_query_aux_proj_K[0], 4.0))
    assert torch.allclose(tr.quant_query_proj_V[0], torch.full_like(tr.quant_query_proj_V[0], 4.0))
    assert torch.allclose(tr.quant_query_aux_proj_V[0], torch.full_like(tr.quant_query_aux_proj_V[0], 5.0))
    assert torch.allclose(tr.quant_query_shared_left[0], torch.full_like(tr.quant_query_shared_left[0], 3.0))
    assert torch.allclose(tr.quant_query_shared_K_right[0], torch.full_like(tr.quant_query_shared_K_right[0], 5.0))
    assert torch.allclose(tr.quant_query_shared_V_right[0], torch.full_like(tr.quant_query_shared_V_right[0], 6.0))


def test_bridge_ridge_qk_xattn_adapter_uses_cross_attention_module(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_xattn_adapter",
        quantization_correction_rank=1,
    )
    with torch.no_grad():
        tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
        tr.quant_proj_V[0].copy_(torch.eye(tr.d_t))
        tr.quant_aux_proj_K[0].zero_()
        tr.quant_aux_proj_V[0].zero_()
        tr.quant_bias_K[0].zero_()
        tr.quant_bias_V[0].zero_()
        tr.quant_query_resid_K_left[0].zero_()
        tr.quant_query_resid_K_right[0].zero_()
        tr.quant_query_aux_resid_K_left[0].zero_()
        tr.quant_query_aux_resid_K_right[0].zero_()
        tr.quant_query_resid_V_left[0].zero_()
        tr.quant_query_resid_V_right[0].zero_()
        tr.quant_query_aux_resid_V_left[0].zero_()
        tr.quant_query_aux_resid_V_right[0].zero_()
        tr.quant_query_xattn_q[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_xattn_k[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_xattn_v[0].copy_(torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32))
        tr.quant_query_xattn_K_out[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_xattn_V_out[0].copy_(torch.tensor([[0.5, 0.0, 0.0, 0.0]], dtype=torch.float32))

    K_base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
    V_base = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]], dtype=torch.float32)
    out_k, out_v = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_query_features=torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32),
    )

    assert out_k.shape == K_base.shape
    assert out_v.shape == V_base.shape
    assert not torch.allclose(out_k, K_base)
    assert not torch.allclose(out_v, V_base)


def test_fit_from_pairs_bridge_ridge_qk_xattn_adapter_populates_cross_attention_module(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_xattn_adapter",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])

    calls: list[int] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        base_prediction_k,
        base_prediction_v,
        residual_target_k,
        residual_target_v,
        *,
        rank,
        **kwargs,
    ):
        calls.append(rank)
        return (
            torch.full((self.d_t, rank), 1.0, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 4.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 5.0, dtype=residual_target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_xattn_adapter", fake_fit)

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

    assert calls == [1]
    assert torch.allclose(tr.quant_query_xattn_q[0], torch.ones_like(tr.quant_query_xattn_q[0]))
    assert torch.allclose(tr.quant_query_xattn_k[0], torch.full_like(tr.quant_query_xattn_k[0], 2.0))
    assert torch.allclose(tr.quant_query_xattn_v[0], torch.full_like(tr.quant_query_xattn_v[0], 3.0))
    assert torch.allclose(tr.quant_query_xattn_K_out[0], torch.full_like(tr.quant_query_xattn_K_out[0], 4.0))
    assert torch.allclose(tr.quant_query_xattn_V_out[0], torch.full_like(tr.quant_query_xattn_V_out[0], 5.0))


def test_fit_from_pairs_bridge_ridge_qk_xattn_dynmap_adapter_passes_dynamic_teacher(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_xattn_dynmap_adapter",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    calls: list[tuple[float, bool, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        base_prediction_k,
        base_prediction_v,
        residual_target_k,
        residual_target_v,
        *,
        rank,
        dynamic_prediction_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        calls.append(
            (
                float(dynamic_prediction_weight),
                teacher_topk_log_probs is not None,
                teacher_topk_output_rows is not None,
            )
        )
        return (
            torch.full((self.d_t, rank), 1.0, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 4.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 5.0, dtype=residual_target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_xattn_adapter", fake_fit)

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

    assert calls == [(0.25, True, True)]


def test_bridge_ridge_qk_module_adapter_uses_attention_module(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_module_adapter",
        quantization_correction_rank=1,
    )
    with torch.no_grad():
        tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
        tr.quant_proj_V[0].copy_(torch.eye(tr.d_t))
        tr.quant_aux_proj_K[0].zero_()
        tr.quant_aux_proj_V[0].zero_()
        tr.quant_bias_K[0].zero_()
        tr.quant_bias_V[0].zero_()
        tr.quant_query_resid_K_left[0].zero_()
        tr.quant_query_resid_K_right[0].zero_()
        tr.quant_query_aux_resid_K_left[0].zero_()
        tr.quant_query_aux_resid_K_right[0].zero_()
        tr.quant_query_resid_V_left[0].zero_()
        tr.quant_query_resid_V_right[0].zero_()
        tr.quant_query_aux_resid_V_left[0].zero_()
        tr.quant_query_aux_resid_V_right[0].zero_()
        tr.quant_query_module_slots[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_q[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_k[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_v[0].copy_(torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_hidden[0].copy_(torch.tensor([[1.0]], dtype=torch.float32))
        tr.quant_query_module_K_out[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_V_out[0].copy_(torch.tensor([[0.5, 0.0, 0.0, 0.0]], dtype=torch.float32))

    K_base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
    V_base = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]], dtype=torch.float32)
    out_k, out_v = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_query_features=torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32),
    )

    assert out_k.shape == K_base.shape
    assert out_v.shape == V_base.shape
    assert not torch.allclose(out_k, K_base)
    assert not torch.allclose(out_v, V_base)


def test_fit_from_pairs_bridge_ridge_qk_module_adapter_populates_module_and_uses_teacher(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_module_adapter",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    calls: list[tuple[float, bool, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        base_prediction_k,
        base_prediction_v,
        residual_target_k,
        residual_target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                teacher_topk_log_probs is not None,
                teacher_topk_output_rows is not None,
            )
        )
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=residual_target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=residual_target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_adapter", fake_fit)

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

    assert calls == [(0.25, True, True)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))
    assert torch.allclose(tr.quant_query_module_q[0], torch.full_like(tr.quant_query_module_q[0], 2.0))
    assert torch.allclose(tr.quant_query_module_k[0], torch.full_like(tr.quant_query_module_k[0], 3.0))
    assert torch.allclose(tr.quant_query_module_v[0], torch.full_like(tr.quant_query_module_v[0], 4.0))
    assert torch.allclose(tr.quant_query_module_hidden[0], torch.full_like(tr.quant_query_module_hidden[0], 5.0))
    assert torch.allclose(tr.quant_query_module_K_out[0], torch.full_like(tr.quant_query_module_K_out[0], 6.0))
    assert torch.allclose(tr.quant_query_module_V_out[0], torch.full_like(tr.quant_query_module_V_out[0], 7.0))


def test_bridge_ridge_qk_module_replace_uses_module_output(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_module_replace",
        quantization_correction_rank=1,
    )
    with torch.no_grad():
        tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
        tr.quant_proj_V[0].copy_(torch.eye(tr.d_t))
        tr.quant_aux_proj_K[0].zero_()
        tr.quant_aux_proj_V[0].zero_()
        tr.quant_bias_K[0].zero_()
        tr.quant_bias_V[0].zero_()
        tr.quant_query_resid_K_left[0].zero_()
        tr.quant_query_resid_K_right[0].zero_()
        tr.quant_query_aux_resid_K_left[0].zero_()
        tr.quant_query_aux_resid_K_right[0].zero_()
        tr.quant_query_resid_V_left[0].zero_()
        tr.quant_query_resid_V_right[0].zero_()
        tr.quant_query_aux_resid_V_left[0].zero_()
        tr.quant_query_aux_resid_V_right[0].zero_()
        tr.quant_query_module_slots[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_q[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_k[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_v[0].copy_(torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_hidden[0].copy_(torch.tensor([[1.0]], dtype=torch.float32))
        tr.quant_query_module_K_out[0].copy_(torch.tensor([[3.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_V_out[0].copy_(torch.tensor([[4.0, 0.0, 0.0, 0.0]], dtype=torch.float32))

    K_base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
    V_base = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]], dtype=torch.float32)
    out_k, out_v = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_query_features=torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32),
    )

    assert out_k.shape == K_base.shape
    assert out_v.shape == V_base.shape
    assert not torch.allclose(out_k, K_base)
    assert not torch.allclose(out_v, V_base)
    assert out_k.abs().max().item() > K_base.abs().max().item()
    assert out_v.abs().max().item() > V_base.abs().max().item()


def test_fit_from_pairs_bridge_ridge_qk_module_replace_populates_module_and_uses_teacher(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    calls: list[tuple[float, bool, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                teacher_topk_log_probs is not None,
                teacher_topk_output_rows is not None,
            )
        )
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    assert calls == [(0.25, True, True)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))
    assert torch.allclose(tr.quant_query_module_q[0], torch.full_like(tr.quant_query_module_q[0], 2.0))
    assert torch.allclose(tr.quant_query_module_k[0], torch.full_like(tr.quant_query_module_k[0], 3.0))
    assert torch.allclose(tr.quant_query_module_v[0], torch.full_like(tr.quant_query_module_v[0], 4.0))
    assert torch.allclose(tr.quant_query_module_hidden[0], torch.full_like(tr.quant_query_module_hidden[0], 5.0))
    assert torch.allclose(tr.quant_query_module_K_out[0], torch.full_like(tr.quant_query_module_K_out[0], 6.0))
    assert torch.allclose(tr.quant_query_module_V_out[0], torch.full_like(tr.quant_query_module_V_out[0], 7.0))


def test_fit_from_pairs_bridge_ridge_qk_bytespan_module_replace_reuses_module_replace_fit(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_bytespan_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    calls: list[tuple[float, bool, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                teacher_topk_log_probs is not None,
                teacher_topk_output_rows is not None,
            )
        )
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    assert calls == [(0.25, True, True)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))
    assert torch.allclose(tr.quant_query_module_q[0], torch.full_like(tr.quant_query_module_q[0], 2.0))
    assert torch.allclose(tr.quant_query_module_k[0], torch.full_like(tr.quant_query_module_k[0], 3.0))
    assert torch.allclose(tr.quant_query_module_v[0], torch.full_like(tr.quant_query_module_v[0], 4.0))
    assert torch.allclose(tr.quant_query_module_hidden[0], torch.full_like(tr.quant_query_module_hidden[0], 5.0))
    assert torch.allclose(tr.quant_query_module_K_out[0], torch.full_like(tr.quant_query_module_K_out[0], 6.0))
    assert torch.allclose(tr.quant_query_module_V_out[0], torch.full_like(tr.quant_query_module_V_out[0], 7.0))


def test_fit_from_pairs_bridge_ridge_qk_spanalign_module_replace_reuses_module_replace_fit(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_spanalign_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    calls: list[tuple[float, bool, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                teacher_topk_log_probs is not None,
                teacher_topk_output_rows is not None,
            )
        )
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    assert calls == [(0.25, True, True)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))
    assert torch.allclose(tr.quant_query_module_q[0], torch.full_like(tr.quant_query_module_q[0], 2.0))
    assert torch.allclose(tr.quant_query_module_k[0], torch.full_like(tr.quant_query_module_k[0], 3.0))
    assert torch.allclose(tr.quant_query_module_v[0], torch.full_like(tr.quant_query_module_v[0], 4.0))
    assert torch.allclose(tr.quant_query_module_hidden[0], torch.full_like(tr.quant_query_module_hidden[0], 5.0))
    assert torch.allclose(tr.quant_query_module_K_out[0], torch.full_like(tr.quant_query_module_K_out[0], 6.0))
    assert torch.allclose(tr.quant_query_module_V_out[0], torch.full_like(tr.quant_query_module_V_out[0], 7.0))


def test_fit_from_pairs_bridge_ridge_qk_ctxalign_module_replace_reuses_module_replace_fit(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_ctxalign_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    calls: list[tuple[float, bool, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                teacher_topk_log_probs is not None,
                teacher_topk_output_rows is not None,
            )
        )
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    assert calls == [(0.25, True, True)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))
    assert torch.allclose(tr.quant_query_module_q[0], torch.full_like(tr.quant_query_module_q[0], 2.0))
    assert torch.allclose(tr.quant_query_module_k[0], torch.full_like(tr.quant_query_module_k[0], 3.0))
    assert torch.allclose(tr.quant_query_module_v[0], torch.full_like(tr.quant_query_module_v[0], 4.0))
    assert torch.allclose(tr.quant_query_module_hidden[0], torch.full_like(tr.quant_query_module_hidden[0], 5.0))
    assert torch.allclose(tr.quant_query_module_K_out[0], torch.full_like(tr.quant_query_module_K_out[0], 6.0))
    assert torch.allclose(tr.quant_query_module_V_out[0], torch.full_like(tr.quant_query_module_V_out[0], 7.0))


def test_fit_from_pairs_bridge_ridge_qk_dynalign_module_replace_reuses_module_replace_fit(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    calls: list[tuple[float, bool, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                teacher_topk_log_probs is not None,
                teacher_topk_output_rows is not None,
            )
        )
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    assert calls == [(0.25, True, True)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))
    assert torch.allclose(tr.quant_query_module_q[0], torch.full_like(tr.quant_query_module_q[0], 2.0))
    assert torch.allclose(tr.quant_query_module_k[0], torch.full_like(tr.quant_query_module_k[0], 3.0))
    assert torch.allclose(tr.quant_query_module_v[0], torch.full_like(tr.quant_query_module_v[0], 4.0))
    assert torch.allclose(tr.quant_query_module_hidden[0], torch.full_like(tr.quant_query_module_hidden[0], 5.0))
    assert torch.allclose(tr.quant_query_module_K_out[0], torch.full_like(tr.quant_query_module_K_out[0], 6.0))
    assert torch.allclose(tr.quant_query_module_V_out[0], torch.full_like(tr.quant_query_module_V_out[0], 7.0))


def test_fit_from_pairs_bridge_ridge_qk_dynalign_preserve_module_replace_projects_tail(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_preserve_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    target_norms: list[tuple[float, float]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        **kwargs,
    ):
        del quantized_k, predicted_k, quantized_v, predicted_v, query_features, kwargs
        target_norms.append((float(target_k.norm().item()), float(target_v.norm().item())))
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    raw_k = tr._rotate_and_flatten(tgt_kvs[0][0], tr.R_t).reshape(-1, tr.d_t)
    raw_v = tr._rotate_and_flatten(tgt_kvs[0][1], tr.R_t).reshape(-1, tr.d_t)

    tr.fit_from_pairs(src_kvs, tgt_kvs)

    assert len(target_norms) == 1
    tail_k_norm, tail_v_norm = target_norms[0]
    assert tail_k_norm < float(raw_k.norm().item())
    assert tail_v_norm < float(raw_v.norm().item())
    preserve_k = tr.quant_preserve_proj_K[0]
    preserve_v = tr.quant_preserve_proj_V[0]
    assert preserve_k.abs().sum() > 0
    assert preserve_v.abs().sum() > 0
    assert torch.allclose(preserve_k, preserve_k.T, atol=1e-5)
    assert torch.allclose(preserve_v, preserve_v.T, atol=1e-5)


def test_fit_from_pairs_bridge_ridge_qk_dynalign_eigenspace_module_replace_projects_head(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_eigenspace_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    target_norms: list[tuple[float, float]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        **kwargs,
    ):
        del quantized_k, predicted_k, quantized_v, predicted_v, query_features, kwargs
        target_norms.append((float(target_k.norm().item()), float(target_v.norm().item())))
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    raw_k = tr._rotate_and_flatten(tgt_kvs[0][0], tr.R_t).reshape(-1, tr.d_t)
    raw_v = tr._rotate_and_flatten(tgt_kvs[0][1], tr.R_t).reshape(-1, tr.d_t)

    tr.fit_from_pairs(src_kvs, tgt_kvs)

    assert len(target_norms) == 1
    head_k_norm, head_v_norm = target_norms[0]
    assert 0.0 < head_k_norm < float(raw_k.norm().item())
    assert 0.0 < head_v_norm < float(raw_v.norm().item())
    eig_k = tr.quant_preserve_proj_K[0]
    eig_v = tr.quant_preserve_proj_V[0]
    assert eig_k.abs().sum() > 0
    assert eig_v.abs().sum() > 0
    assert torch.allclose(eig_k, eig_k.T, atol=1e-5)
    assert torch.allclose(eig_v, eig_v.T, atol=1e-5)


def test_fit_from_pairs_bridge_ridge_qk_dynalign_saliency_module_replace_passes_feature_weights(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_saliency_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    seen: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        feature_weights_k=None,
        feature_weights_v=None,
        **kwargs,
    ):
        del self, quantized_k, predicted_k, quantized_v, predicted_v, query_features, target_k, target_v, rank, kwargs
        seen.append((feature_weights_k, feature_weights_v))
        return (
            torch.ones((tr.config.bridge_bank_size, tr.d_t), dtype=torch.float32),
            torch.ones((tr.d_t, 1), dtype=torch.float32),
            torch.ones((tr.d_t, 1), dtype=torch.float32),
            torch.ones((tr.d_t, 1), dtype=torch.float32),
            torch.ones((1, 1), dtype=torch.float32),
            torch.ones((1, tr.d_t), dtype=torch.float32),
            torch.ones((1, tr.d_t), dtype=torch.float32),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    assert len(seen) == 1
    fwk, fwv = seen[0]
    assert fwk is not None
    assert fwv is not None
    assert fwk.shape == (tr.d_t,)
    assert fwv.shape == (tr.d_t,)
    assert abs(float(fwk.mean().item()) - 1.0) < 1e-5
    assert abs(float(fwv.mean().item()) - 1.0) < 1e-5


def test_fit_from_pairs_bridge_ridge_qk_dynalign_saliency_preserve_module_replace_projects_salient_tail(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_saliency_preserve_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    target_norms: list[tuple[float, float]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        **kwargs,
    ):
        del quantized_k, predicted_k, quantized_v, predicted_v, query_features, kwargs
        target_norms.append((float(target_k.norm().item()), float(target_v.norm().item())))
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    raw_k = tr._rotate_and_flatten(tgt_kvs[0][0], tr.R_t).reshape(-1, tr.d_t)
    raw_v = tr._rotate_and_flatten(tgt_kvs[0][1], tr.R_t).reshape(-1, tr.d_t)

    tr.fit_from_pairs(src_kvs, tgt_kvs)

    assert len(target_norms) == 1
    tail_k_norm, tail_v_norm = target_norms[0]
    assert tail_k_norm < float(raw_k.norm().item())
    assert tail_v_norm < float(raw_v.norm().item())
    preserve_k = tr.quant_preserve_proj_K[0]
    preserve_v = tr.quant_preserve_proj_V[0]
    assert preserve_k.abs().sum() > 0
    assert preserve_v.abs().sum() > 0
    assert torch.allclose(preserve_k, torch.diag(torch.diagonal(preserve_k)), atol=1e-5)
    assert torch.allclose(preserve_v, torch.diag(torch.diagonal(preserve_v)), atol=1e-5)


def test_fit_from_pairs_bridge_ridge_qk_dynalign_anchor_tail_module_replace_stores_v_only_salient_anchor_mask(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_anchor_tail_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    target_norms: list[tuple[float, float]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        **kwargs,
    ):
        del quantized_k, predicted_k, quantized_v, predicted_v, query_features, kwargs
        target_norms.append((float(target_k.norm().item()), float(target_v.norm().item())))
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    raw_k = tr._rotate_and_flatten(tgt_kvs[0][0], tr.R_t).reshape(-1, tr.d_t)
    raw_v = tr._rotate_and_flatten(tgt_kvs[0][1], tr.R_t).reshape(-1, tr.d_t)

    tr.fit_from_pairs(src_kvs, tgt_kvs)

    assert len(target_norms) == 1
    fit_k_norm, fit_v_norm = target_norms[0]
    assert abs(fit_k_norm - float(raw_k.norm().item())) < 1e-5
    assert abs(fit_v_norm - float(raw_v.norm().item())) < 1e-5
    preserve_k = tr.quant_preserve_proj_K[0]
    preserve_v = tr.quant_preserve_proj_V[0]
    assert torch.count_nonzero(preserve_k) == 0
    assert preserve_v.abs().sum() > 0
    assert torch.allclose(preserve_k, torch.diag(torch.diagonal(preserve_k)), atol=1e-5)
    assert torch.allclose(preserve_v, torch.diag(torch.diagonal(preserve_v)), atol=1e-5)


def test_fit_outlier_escrow_projector_selects_outlier_and_error_channels(monkeypatch) -> None:
    tr = _make_identity_translator(monkeypatch)
    target = torch.tensor(
        [
            [50.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    base = target.clone()
    base[:, 2] = 80.0
    query_features = torch.zeros_like(target)

    projector = tr._fit_outlier_escrow_projector(target, base, query_features, rank=2)

    diag = torch.diagonal(projector)
    assert torch.count_nonzero(diag) == 2
    assert diag[0] == 1.0
    assert diag[2] == 1.0


def test_fit_from_pairs_bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace_stores_layer8_v_mask(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        tgt_layers=9,
        quantization_correction="bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32) for _ in range(9)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    calls = 0

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        **kwargs,
    ):
        nonlocal calls
        del quantized_k, predicted_k, quantized_v, predicted_v, query_features, target_k, target_v, kwargs
        calls += 1
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=torch.float32),
            torch.full((self.d_t, rank), 2.0, dtype=torch.float32),
            torch.full((self.d_t, rank), 3.0, dtype=torch.float32),
            torch.full((self.d_t, rank), 4.0, dtype=torch.float32),
            torch.full((rank, rank), 5.0, dtype=torch.float32),
            torch.full((rank, self.d_t), 6.0, dtype=torch.float32),
            torch.full((rank, self.d_t), 7.0, dtype=torch.float32),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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
    tgt_kvs = [(base * 1.5, base * 2.0) for _ in range(9)]

    tr.fit_from_pairs(src_kvs, tgt_kvs)

    assert calls == 9
    assert torch.count_nonzero(tr.quant_preserve_proj_V[8]) == 1
    assert torch.count_nonzero(tr.quant_preserve_proj_K[8]) == 0
    for layer_idx in range(8):
        assert torch.count_nonzero(tr.quant_preserve_proj_V[layer_idx]) == 0


def test_quantize_tail_with_preserve_keeps_anchor_and_quantizes_tail(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_anchor_tail_module_replace",
        quantization_correction_rank=1,
    )
    tr.quantizer = SimpleNamespace(
        quantize_dequantize=lambda x: x + 0.25 * (x != 0).to(dtype=x.dtype),
    )
    values = torch.tensor([[2.0, 0.7, 1.5, 0.0]], dtype=torch.float32)
    preserve = torch.diag(torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32))
    out = tr._quantize_tail_with_preserve(values, preserve)
    tail_only = values @ (torch.eye(tr.d_t, dtype=values.dtype) - preserve)
    expected = values @ preserve + tr.quantizer.quantize_dequantize(tail_only)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(out[..., 0], values[..., 0], atol=1e-6)
    assert torch.allclose(out[..., 2], values[..., 2], atol=1e-6)


def test_bridge_ridge_qk_dynalign_anchor_tail_module_replace_applies_tail_quantization_only_to_v(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_anchor_tail_module_replace",
        quantization_correction_rank=1,
    )
    with torch.no_grad():
        tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
        tr.quant_proj_V[0].copy_(torch.eye(tr.d_t))
        tr.quant_aux_proj_K[0].zero_()
        tr.quant_aux_proj_V[0].zero_()
        tr.quant_bias_K[0].zero_()
        tr.quant_bias_V[0].zero_()
        tr.quant_query_resid_K_left[0].zero_()
        tr.quant_query_resid_K_right[0].zero_()
        tr.quant_query_aux_resid_K_left[0].zero_()
        tr.quant_query_aux_resid_K_right[0].zero_()
        tr.quant_query_resid_V_left[0].zero_()
        tr.quant_query_resid_V_right[0].zero_()
        tr.quant_query_aux_resid_V_left[0].zero_()
        tr.quant_query_aux_resid_V_right[0].zero_()
        tr.quant_query_module_slots[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_q[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_k[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_v[0].copy_(torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_hidden[0].copy_(torch.tensor([[1.0]], dtype=torch.float32))
        tr.quant_query_module_K_out[0].copy_(torch.tensor([[3.0, 1.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_V_out[0].copy_(torch.tensor([[4.0, 1.5, 1.0, 0.0]], dtype=torch.float32))
        tr.quant_preserve_proj_K[0].copy_(torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)))
        tr.quant_preserve_proj_V[0].copy_(torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)))

    K_base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
    V_base = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]], dtype=torch.float32)
    qfeat = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)

    shifted_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def shifted_quantize_tail(self, values, preserve_proj):
        shifted_calls.append((values.detach().clone(), preserve_proj.detach().clone()))
        return values + 0.75

    monkeypatch.setattr(RotAlignKVTranslator, "_quantize_tail_with_preserve", shifted_quantize_tail)
    out_k_shifted, out_v_shifted = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_query_features=qfeat,
    )

    def identity_quantize_tail(self, values, preserve_proj):
        del self, preserve_proj
        return values

    monkeypatch.setattr(RotAlignKVTranslator, "_quantize_tail_with_preserve", identity_quantize_tail)
    out_k_plain, out_v_plain = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_query_features=qfeat,
    )

    base_v_flat = tr._rotate_and_flatten(V_base, tr.R_t).reshape(-1, tr.d_t)
    out_v_plain_flat = tr._rotate_and_flatten(out_v_plain, tr.R_t).reshape(-1, tr.d_t)

    assert len(shifted_calls) == 1
    assert torch.allclose(shifted_calls[0][0], out_v_plain_flat - base_v_flat, atol=1e-6)
    assert torch.allclose(shifted_calls[0][1], tr.quant_preserve_proj_V[0], atol=1e-6)
    assert torch.allclose(out_k_shifted, out_k_plain, atol=1e-6)
    assert not torch.allclose(out_v_shifted, out_v_plain)


def test_bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace_quantizes_only_layer8_v_tail(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        tgt_layers=9,
        quantization_correction="bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace",
        quantization_correction_rank=1,
    )
    with torch.no_grad():
        tr.quant_proj_K[8].copy_(torch.eye(tr.d_t))
        tr.quant_proj_V[8].copy_(torch.eye(tr.d_t))
        tr.quant_aux_proj_K[8].zero_()
        tr.quant_aux_proj_V[8].zero_()
        tr.quant_bias_K[8].zero_()
        tr.quant_bias_V[8].zero_()
        tr.quant_query_resid_K_left[8].zero_()
        tr.quant_query_resid_K_right[8].zero_()
        tr.quant_query_aux_resid_K_left[8].zero_()
        tr.quant_query_aux_resid_K_right[8].zero_()
        tr.quant_query_resid_V_left[8].zero_()
        tr.quant_query_resid_V_right[8].zero_()
        tr.quant_query_aux_resid_V_left[8].zero_()
        tr.quant_query_aux_resid_V_right[8].zero_()
        tr.quant_query_module_slots[8].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_q[8].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_k[8].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_v[8].copy_(torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_hidden[8].copy_(torch.tensor([[1.0]], dtype=torch.float32))
        tr.quant_query_module_K_out[8].copy_(torch.tensor([[3.0, 1.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_V_out[8].copy_(torch.tensor([[4.0, 1.5, 1.0, 0.0]], dtype=torch.float32))
        tr.quant_preserve_proj_V[8].copy_(torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)))

    K_base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
    V_base = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]], dtype=torch.float32)
    qfeat = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)

    shifted_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def shifted_quantize_tail(self, values, preserve_proj):
        shifted_calls.append((values.detach().clone(), preserve_proj.detach().clone()))
        return values + 0.75

    monkeypatch.setattr(RotAlignKVTranslator, "_quantize_tail_with_preserve", shifted_quantize_tail)
    tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=7,
        quantize=True,
        runtime_query_features=qfeat,
    )
    assert shifted_calls == []

    tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=8,
        quantize=True,
        runtime_query_features=qfeat,
    )

    assert len(shifted_calls) == 1
    assert torch.allclose(shifted_calls[0][1], tr.quant_preserve_proj_V[8], atol=1e-6)


def test_fit_from_pairs_bridge_ridge_qk_dynalign_routed_module_replace_stores_route_gates(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_routed_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        **kwargs,
    ):
        del self, quantized_k, predicted_k, quantized_v, predicted_v, query_features, target_k, target_v, rank, kwargs
        return (
            torch.ones((tr.config.bridge_bank_size, tr.d_t), dtype=torch.float32),
            torch.ones((tr.d_t, 1), dtype=torch.float32),
            torch.full((tr.d_t, 1), 2.0, dtype=torch.float32),
            torch.full((tr.d_t, 1), 3.0, dtype=torch.float32),
            torch.full((1, 1), 4.0, dtype=torch.float32),
            torch.full((1, tr.d_t), 5.0, dtype=torch.float32),
            torch.full((1, tr.d_t), 6.0, dtype=torch.float32),
        )

    def fake_predict(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        slot_tokens,
        q_proj,
        k_proj,
        v_proj,
        hidden_proj,
        out_k,
        out_v,
    ):
        del self, quantized_k, predicted_k, quantized_v, predicted_v, query_features, slot_tokens, q_proj, k_proj, v_proj, hidden_proj, out_k, out_v
        return (
            torch.full((4, tr.d_t), 2.0, dtype=torch.float32),
            torch.full((4, tr.d_t), 3.0, dtype=torch.float32),
        )

    seen = []

    def fake_route_gate(
        self,
        query_features,
        base_prediction,
        module_prediction,
        target,
        *,
        steps=100,
        lr=5e-2,
    ):
        del self, steps, lr
        seen.append((query_features.shape, base_prediction.shape, module_prediction.shape, target.shape))
        return (
            torch.full((tr.d_t, 1), 7.0, dtype=torch.float32),
            torch.full((1,), -0.25, dtype=torch.float32),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)
    monkeypatch.setattr(RotAlignKVTranslator, "_predict_bridge_query_module_replace", fake_predict)
    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_route_gate", fake_route_gate)

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

    assert len(seen) == 2
    assert torch.allclose(tr.quant_query_route_K[0], torch.full_like(tr.quant_query_route_K[0], 7.0))
    assert torch.allclose(tr.quant_query_route_V[0], torch.full_like(tr.quant_query_route_V[0], 7.0))
    assert torch.allclose(tr.quant_query_route_K_bias[0], torch.full_like(tr.quant_query_route_K_bias[0], -0.25))
    assert torch.allclose(tr.quant_query_route_V_bias[0], torch.full_like(tr.quant_query_route_V_bias[0], -0.25))


def test_fit_from_pairs_bridge_ridge_qk_dynalign_value_routed_module_replace_stores_only_v_route_gates(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_value_routed_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        **kwargs,
    ):
        del self, quantized_k, predicted_k, quantized_v, predicted_v, query_features, target_k, target_v, rank, kwargs
        return (
            torch.ones((tr.config.bridge_bank_size, tr.d_t), dtype=torch.float32),
            torch.ones((tr.d_t, 1), dtype=torch.float32),
            torch.full((tr.d_t, 1), 2.0, dtype=torch.float32),
            torch.full((tr.d_t, 1), 3.0, dtype=torch.float32),
            torch.full((1, 1), 4.0, dtype=torch.float32),
            torch.full((1, tr.d_t), 5.0, dtype=torch.float32),
            torch.full((1, tr.d_t), 6.0, dtype=torch.float32),
        )

    def fake_predict(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        slot_tokens,
        q_proj,
        k_proj,
        v_proj,
        hidden_proj,
        out_k,
        out_v,
    ):
        del self, quantized_k, predicted_k, quantized_v, predicted_v, query_features, slot_tokens, q_proj, k_proj, v_proj, hidden_proj, out_k, out_v
        return (
            torch.full((4, tr.d_t), 2.0, dtype=torch.float32),
            torch.full((4, tr.d_t), 3.0, dtype=torch.float32),
        )

    seen = []

    def fake_route_gate(
        self,
        query_features,
        base_prediction,
        module_prediction,
        target,
        *,
        steps=100,
        lr=5e-2,
    ):
        del self, steps, lr
        seen.append((query_features.shape, base_prediction.shape, module_prediction.shape, target.shape))
        return (
            torch.full((tr.d_t, 1), 7.0, dtype=torch.float32),
            torch.full((1,), -0.25, dtype=torch.float32),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)
    monkeypatch.setattr(RotAlignKVTranslator, "_predict_bridge_query_module_replace", fake_predict)
    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_route_gate", fake_route_gate)

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

    assert len(seen) == 1
    assert torch.allclose(tr.quant_query_route_K[0], torch.zeros_like(tr.quant_query_route_K[0]))
    assert torch.allclose(tr.quant_query_route_K_bias[0], torch.zeros_like(tr.quant_query_route_K_bias[0]))
    assert torch.allclose(tr.quant_query_route_V[0], torch.full_like(tr.quant_query_route_V[0], 7.0))
    assert torch.allclose(tr.quant_query_route_V_bias[0], torch.full_like(tr.quant_query_route_V_bias[0], -0.25))


def test_bridge_ridge_qk_dynalign_value_bank_module_replace_routes_value_bank_only(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_value_bank_module_replace",
        quantization_correction_rank=1,
        bridge_bank_size=2,
        bridge_bank_temperature=40.0,
        transport_template_bins=4,
    )
    with torch.no_grad():
        tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
        tr.quant_proj_V[0].copy_(torch.eye(tr.d_t))
        tr.quant_aux_proj_K[0].zero_()
        tr.quant_aux_proj_V[0].zero_()
        tr.quant_bias_K[0].zero_()
        tr.quant_bias_V[0].zero_()
        tr.quant_query_module_slots[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_q[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_k[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_v[0].copy_(torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_hidden[0].copy_(torch.tensor([[1.0]], dtype=torch.float32))
        tr.quant_query_module_K_out[0].copy_(torch.tensor([[3.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_V_out[0].copy_(torch.tensor([[4.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.bridge_bank_templates[0].copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        tr.bridge_bank_priors[0].copy_(torch.tensor([0.5, 0.5], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_left[0].data[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_right[0].data[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_left[0].data[1].copy_(torch.tensor([[2.0], [0.0], [2.0], [0.0]], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_right[0].data[1].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.bridge_bank_query_aux_resid_V_left[0].zero_()
        tr.bridge_bank_query_aux_resid_V_right[0].zero_()
        tr.bridge_bank_query_resid_K_left[0].zero_()
        tr.bridge_bank_query_resid_K_right[0].zero_()
        tr.bridge_bank_query_aux_resid_K_left[0].zero_()
        tr.bridge_bank_query_aux_resid_K_right[0].zero_()
        tr.bridge_bank_proj_K_left[0].zero_()
        tr.bridge_bank_proj_K_right[0].zero_()
        tr.bridge_bank_aux_proj_K_left[0].zero_()
        tr.bridge_bank_aux_proj_K_right[0].zero_()
        tr.bridge_bank_proj_V_left[0].zero_()
        tr.bridge_bank_proj_V_right[0].zero_()
        tr.bridge_bank_aux_proj_V_left[0].zero_()
        tr.bridge_bank_aux_proj_V_right[0].zero_()
        tr.bridge_bank_bias_K[0].zero_()
        tr.bridge_bank_bias_V[0].zero_()

    K_base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
    V_base = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]], dtype=torch.float32)
    qfeat = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)

    K_match, V_match = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_attention_profile=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        runtime_query_features=qfeat,
    )
    K_other, V_other = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_attention_profile=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        runtime_query_features=qfeat,
    )

    assert torch.allclose(K_match, K_other, atol=1e-2, rtol=1e-2)
    assert not torch.allclose(V_match, V_other)
    assert V_other.abs().max().item() > V_match.abs().max().item()


def test_fit_from_pairs_bridge_ridge_qk_dynalign_value_bank_module_replace_populates_only_v_bank(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_value_bank_module_replace",
        quantization_correction_rank=1,
        bridge_bank_size=2,
        transport_template_bins=4,
    )
    tr.set_bridge_sample_query_features([torch.ones(16, tr.d_t, dtype=torch.float32)])
    tr._bridge_prompt_cluster_labels = [torch.tensor([0, 1], dtype=torch.long)]
    tr._bridge_sample_prompt_ids = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)
    tr.set_bridge_prediction_teacher(
        torch.full((16, 3), -1.0, dtype=torch.float32),
        torch.ones(16, 3, tr.d_t, dtype=torch.float32),
    )

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        **kwargs,
    ):
        del self, quantized_k, predicted_k, quantized_v, predicted_v, query_features, target_k, target_v, rank, kwargs
        return (
            torch.ones((tr.config.bridge_bank_size, tr.d_t), dtype=torch.float32),
            torch.ones((tr.d_t, 1), dtype=torch.float32),
            torch.full((tr.d_t, 1), 2.0, dtype=torch.float32),
            torch.full((tr.d_t, 1), 3.0, dtype=torch.float32),
            torch.full((1, 1), 4.0, dtype=torch.float32),
            torch.full((1, tr.d_t), 5.0, dtype=torch.float32),
            torch.full((1, tr.d_t), 6.0, dtype=torch.float32),
        )

    def fake_predict(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        slot_tokens,
        q_proj,
        k_proj,
        v_proj,
        hidden_proj,
        out_k,
        out_v,
    ):
        del self, quantized_k, predicted_k, quantized_v, predicted_v, query_features, slot_tokens, q_proj, k_proj, v_proj, hidden_proj, out_k, out_v
        return (
            torch.full((16, tr.d_t), 2.0, dtype=torch.float32),
            torch.full((16, tr.d_t), 3.0, dtype=torch.float32),
        )

    calls = []

    def fake_resid(
        self,
        quantized,
        predicted,
        query_features,
        base_prediction,
        residual_target,
        *,
        rank,
        **kwargs,
    ):
        del self, quantized, predicted, query_features, base_prediction, residual_target, rank, kwargs
        fill = float(len(calls) + 1)
        calls.append(fill)
        left = torch.full((tr.d_t, 1), fill, dtype=torch.float32)
        right = torch.full((1, tr.d_t), fill, dtype=torch.float32)
        aux_left = torch.full((tr.d_t, 1), fill + 10.0, dtype=torch.float32)
        aux_right = torch.full((1, tr.d_t), fill + 10.0, dtype=torch.float32)
        return left, right, aux_left, aux_right

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)
    monkeypatch.setattr(RotAlignKVTranslator, "_predict_bridge_query_module_replace", fake_predict)
    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_residual_adapter", fake_resid)

    base = torch.arange(64, dtype=torch.float32).view(8, 2, 2, 2)
    src_kvs = [(base, base + 0.5)]
    tgt_kvs = [(base * 1.5, base * 2.0)]

    tr.fit_from_pairs(src_kvs, tgt_kvs)

    assert len(calls) == 3
    assert torch.allclose(tr.bridge_bank_query_resid_K_left[0], torch.zeros_like(tr.bridge_bank_query_resid_K_left[0]))
    assert torch.allclose(tr.bridge_bank_query_resid_K_right[0], torch.zeros_like(tr.bridge_bank_query_resid_K_right[0]))
    assert torch.allclose(tr.bridge_bank_query_aux_resid_K_left[0], torch.zeros_like(tr.bridge_bank_query_aux_resid_K_left[0]))
    assert torch.allclose(tr.bridge_bank_query_aux_resid_K_right[0], torch.zeros_like(tr.bridge_bank_query_aux_resid_K_right[0]))
    assert torch.allclose(tr.bridge_bank_query_resid_V_left[0][0], torch.full_like(tr.bridge_bank_query_resid_V_left[0][0], 2.0))
    assert torch.allclose(tr.bridge_bank_query_resid_V_left[0][1], torch.full_like(tr.bridge_bank_query_resid_V_left[0][1], 3.0))
    assert torch.allclose(tr.bridge_bank_proj_V_left[0], torch.zeros_like(tr.bridge_bank_proj_V_left[0]))
    assert torch.allclose(tr.bridge_bank_bias_V[0], torch.zeros_like(tr.bridge_bank_bias_V[0]))


def test_bridge_ridge_qk_dynalign_value_query_bank_module_replace_routes_from_query_centroids(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_value_query_bank_module_replace",
        quantization_correction_rank=1,
        bridge_bank_size=2,
        bridge_bank_temperature=40.0,
    )
    with torch.no_grad():
        tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
        tr.quant_proj_V[0].copy_(torch.eye(tr.d_t))
        tr.quant_aux_proj_K[0].zero_()
        tr.quant_aux_proj_V[0].zero_()
        tr.quant_bias_K[0].zero_()
        tr.quant_bias_V[0].zero_()
        tr.quant_query_module_slots[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_q[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_k[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_v[0].copy_(torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_hidden[0].copy_(torch.tensor([[1.0]], dtype=torch.float32))
        tr.quant_query_module_K_out[0].copy_(torch.tensor([[3.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_V_out[0].copy_(torch.tensor([[4.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.bridge_bank_query_centroids[0].copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )
        tr.bridge_bank_priors[0].copy_(torch.tensor([0.5, 0.5], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_left[0].data[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_right[0].data[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_left[0].data[1].copy_(torch.tensor([[2.0], [0.0], [2.0], [0.0]], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_right[0].data[1].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.bridge_bank_query_aux_resid_V_left[0].zero_()
        tr.bridge_bank_query_aux_resid_V_right[0].zero_()
        tr.bridge_bank_query_resid_K_left[0].zero_()
        tr.bridge_bank_query_resid_K_right[0].zero_()
        tr.bridge_bank_query_aux_resid_K_left[0].zero_()
        tr.bridge_bank_query_aux_resid_K_right[0].zero_()
        tr.bridge_bank_proj_K_left[0].zero_()
        tr.bridge_bank_proj_K_right[0].zero_()
        tr.bridge_bank_aux_proj_K_left[0].zero_()
        tr.bridge_bank_aux_proj_K_right[0].zero_()
        tr.bridge_bank_proj_V_left[0].zero_()
        tr.bridge_bank_proj_V_right[0].zero_()
        tr.bridge_bank_aux_proj_V_left[0].zero_()
        tr.bridge_bank_aux_proj_V_right[0].zero_()
        tr.bridge_bank_bias_K[0].zero_()
        tr.bridge_bank_bias_V[0].zero_()

    K_base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
    V_base = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]], dtype=torch.float32)
    qfeat_match = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)

    K_match, V_match = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_query_features=qfeat_match,
    )
    with torch.no_grad():
        tr.bridge_bank_query_centroids[0].copy_(
            torch.tensor(
                [
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
    K_other, V_other = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_query_features=qfeat_match,
    )

    assert torch.allclose(K_match, K_other, atol=1e-2, rtol=1e-2)
    assert not torch.allclose(V_match, V_other)
    assert V_other.abs().max().item() > V_match.abs().max().item()


def test_fit_from_pairs_bridge_ridge_qk_dynalign_value_query_bank_module_replace_populates_query_centroid_v_bank(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_value_query_bank_module_replace",
        quantization_correction_rank=1,
        bridge_bank_size=2,
    )
    tr.set_bridge_sample_query_features(
        [
            torch.cat(
                [
                    torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32).repeat(8, 1),
                    torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32).repeat(8, 1),
                ],
                dim=0,
            )
        ]
    )
    tr.set_bridge_prediction_teacher(
        torch.full((16, 3), -1.0, dtype=torch.float32),
        torch.ones(16, 3, tr.d_t, dtype=torch.float32),
    )

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        **kwargs,
    ):
        del self, quantized_k, predicted_k, quantized_v, predicted_v, query_features, target_k, target_v, rank, kwargs
        return (
            torch.ones((tr.config.bridge_bank_size, tr.d_t), dtype=torch.float32),
            torch.ones((tr.d_t, 1), dtype=torch.float32),
            torch.full((tr.d_t, 1), 2.0, dtype=torch.float32),
            torch.full((tr.d_t, 1), 3.0, dtype=torch.float32),
            torch.full((1, 1), 4.0, dtype=torch.float32),
            torch.full((1, tr.d_t), 5.0, dtype=torch.float32),
            torch.full((1, tr.d_t), 6.0, dtype=torch.float32),
        )

    def fake_predict(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        slot_tokens,
        q_proj,
        k_proj,
        v_proj,
        hidden_proj,
        out_k,
        out_v,
    ):
        del self, quantized_k, predicted_k, quantized_v, predicted_v, query_features, slot_tokens, q_proj, k_proj, v_proj, hidden_proj, out_k, out_v
        return (
            torch.full((16, tr.d_t), 2.0, dtype=torch.float32),
            torch.full((16, tr.d_t), 3.0, dtype=torch.float32),
        )

    calls = []

    def fake_cluster(
        self,
        query_bank,
        *,
        num_clusters,
    ):
        del self, query_bank, num_clusters
        return (
            torch.tensor(
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            torch.tensor([0.5, 0.5], dtype=torch.float32),
            torch.tensor([0] * 8 + [1] * 8, dtype=torch.long),
        )

    def fake_resid(
        self,
        quantized,
        predicted,
        query_features,
        base_prediction,
        residual_target,
        *,
        rank,
        **kwargs,
    ):
        del self, quantized, predicted, query_features, base_prediction, residual_target, rank, kwargs
        fill = float(len(calls) + 1)
        calls.append(fill)
        left = torch.full((tr.d_t, 1), fill, dtype=torch.float32)
        right = torch.full((1, tr.d_t), fill, dtype=torch.float32)
        aux_left = torch.full((tr.d_t, 1), fill + 10.0, dtype=torch.float32)
        aux_right = torch.full((1, tr.d_t), fill + 10.0, dtype=torch.float32)
        return left, right, aux_left, aux_right

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)
    monkeypatch.setattr(RotAlignKVTranslator, "_predict_bridge_query_module_replace", fake_predict)
    monkeypatch.setattr(RotAlignKVTranslator, "_cluster_query_feature_bank", fake_cluster)
    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_residual_adapter", fake_resid)

    base = torch.arange(64, dtype=torch.float32).view(8, 2, 2, 2)
    src_kvs = [(base, base + 0.5)]
    tgt_kvs = [(base * 1.5, base * 2.0)]

    tr.fit_from_pairs(src_kvs, tgt_kvs)

    assert len(calls) == 3
    assert torch.allclose(
        tr.bridge_bank_query_centroids[0],
        torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=tr.bridge_bank_query_centroids[0].dtype,
        ),
    )
    assert torch.allclose(tr.bridge_bank_templates[0], torch.zeros_like(tr.bridge_bank_templates[0]))
    assert torch.allclose(tr.bridge_bank_query_resid_K_left[0], torch.zeros_like(tr.bridge_bank_query_resid_K_left[0]))
    assert torch.allclose(tr.bridge_bank_query_resid_V_left[0][0], torch.full_like(tr.bridge_bank_query_resid_V_left[0][0], 2.0))
    assert torch.allclose(tr.bridge_bank_query_resid_V_left[0][1], torch.full_like(tr.bridge_bank_query_resid_V_left[0][1], 3.0))
    assert torch.allclose(tr.bridge_bank_proj_V_left[0], torch.zeros_like(tr.bridge_bank_proj_V_left[0]))
    assert torch.allclose(tr.bridge_bank_bias_V[0], torch.zeros_like(tr.bridge_bank_bias_V[0]))


def test_bridge_ridge_qk_dynalign_value_routed_bank_module_replace_routes_value_gate_and_sparse_bank(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_value_routed_bank_module_replace",
        quantization_correction_rank=1,
        bridge_bank_size=3,
        bridge_bank_temperature=40.0,
        transport_template_bins=4,
    )
    with torch.no_grad():
        tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
        tr.quant_proj_V[0].copy_(torch.eye(tr.d_t))
        tr.quant_aux_proj_K[0].zero_()
        tr.quant_aux_proj_V[0].zero_()
        tr.quant_bias_K[0].zero_()
        tr.quant_bias_V[0].zero_()
        tr.quant_query_module_slots[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_q[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_k[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_v[0].copy_(torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_hidden[0].copy_(torch.tensor([[1.0]], dtype=torch.float32))
        tr.quant_query_module_K_out[0].copy_(torch.tensor([[3.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_V_out[0].copy_(torch.tensor([[4.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_route_V[0].copy_(torch.tensor([[8.0], [0.0], [8.0], [0.0]], dtype=torch.float32))
        tr.quant_query_route_V_bias[0].copy_(torch.tensor([0.0], dtype=torch.float32))
        tr.bridge_bank_templates[0].copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        tr.bridge_bank_priors[0].copy_(torch.tensor([0.34, 0.33, 0.33], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_left[0].data[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_right[0].data[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_left[0].data[1].copy_(torch.tensor([[2.0], [0.0], [2.0], [0.0]], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_right[0].data[1].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_left[0].data[2].copy_(torch.tensor([[20.0], [0.0], [20.0], [0.0]], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_right[0].data[2].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.bridge_bank_query_aux_resid_V_left[0].zero_()
        tr.bridge_bank_query_aux_resid_V_right[0].zero_()
        tr.bridge_bank_query_resid_K_left[0].zero_()
        tr.bridge_bank_query_resid_K_right[0].zero_()
        tr.bridge_bank_query_aux_resid_K_left[0].zero_()
        tr.bridge_bank_query_aux_resid_K_right[0].zero_()
        tr.bridge_bank_proj_K_left[0].zero_()
        tr.bridge_bank_proj_K_right[0].zero_()
        tr.bridge_bank_aux_proj_K_left[0].zero_()
        tr.bridge_bank_aux_proj_K_right[0].zero_()
        tr.bridge_bank_proj_V_left[0].zero_()
        tr.bridge_bank_proj_V_right[0].zero_()
        tr.bridge_bank_aux_proj_V_left[0].zero_()
        tr.bridge_bank_aux_proj_V_right[0].zero_()
        tr.bridge_bank_bias_K[0].zero_()
        tr.bridge_bank_bias_V[0].zero_()

    K_base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
    V_base = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]], dtype=torch.float32)
    qfeat = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)

    K_match, V_match = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_attention_profile=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        runtime_query_features=qfeat,
    )
    K_other, V_other = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_attention_profile=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        runtime_query_features=qfeat,
    )

    assert torch.allclose(K_match, K_other, atol=1e-2, rtol=1e-2)
    assert not torch.allclose(V_match, V_other)
    assert V_other.abs().max().item() > V_match.abs().max().item()


def test_fit_from_pairs_bridge_ridge_qk_dynalign_value_routed_bank_module_replace_populates_route_and_v_bank(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_value_routed_bank_module_replace",
        quantization_correction_rank=1,
        bridge_bank_size=2,
        transport_template_bins=4,
    )
    tr.set_bridge_sample_query_features([torch.ones(16, tr.d_t, dtype=torch.float32)])
    tr._bridge_prompt_cluster_labels = [torch.tensor([0, 1], dtype=torch.long)]
    tr._bridge_sample_prompt_ids = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)
    tr.set_bridge_prediction_teacher(
        torch.full((16, 3), -1.0, dtype=torch.float32),
        torch.ones(16, 3, tr.d_t, dtype=torch.float32),
    )

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        **kwargs,
    ):
        del self, quantized_k, predicted_k, quantized_v, predicted_v, query_features, target_k, target_v, rank, kwargs
        return (
            torch.ones((tr.config.bridge_bank_size, tr.d_t), dtype=torch.float32),
            torch.ones((tr.d_t, 1), dtype=torch.float32),
            torch.full((tr.d_t, 1), 2.0, dtype=torch.float32),
            torch.full((tr.d_t, 1), 3.0, dtype=torch.float32),
            torch.full((1, 1), 4.0, dtype=torch.float32),
            torch.full((1, tr.d_t), 5.0, dtype=torch.float32),
            torch.full((1, tr.d_t), 6.0, dtype=torch.float32),
        )

    def fake_predict(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        slot_tokens,
        q_proj,
        k_proj,
        v_proj,
        hidden_proj,
        out_k,
        out_v,
    ):
        del self, quantized_k, predicted_k, quantized_v, predicted_v, query_features, slot_tokens, q_proj, k_proj, v_proj, hidden_proj, out_k, out_v
        return (
            torch.full((16, tr.d_t), 2.0, dtype=torch.float32),
            torch.full((16, tr.d_t), 3.0, dtype=torch.float32),
        )

    route_calls = []
    resid_calls = []

    def fake_route_gate(
        self,
        query_features,
        base_prediction,
        module_prediction,
        target,
        *,
        steps=100,
        lr=5e-2,
    ):
        del self, query_features, base_prediction, module_prediction, target, steps, lr
        route_calls.append(True)
        return (
            torch.full((tr.d_t, 1), 7.0, dtype=torch.float32),
            torch.full((1,), -0.25, dtype=torch.float32),
        )

    def fake_resid(
        self,
        quantized,
        predicted,
        query_features,
        base_prediction,
        residual_target,
        *,
        rank,
        **kwargs,
    ):
        del self, quantized, predicted, query_features, base_prediction, residual_target, rank, kwargs
        fill = float(len(resid_calls) + 1)
        resid_calls.append(fill)
        left = torch.full((tr.d_t, 1), fill, dtype=torch.float32)
        right = torch.full((1, tr.d_t), fill, dtype=torch.float32)
        aux_left = torch.full((tr.d_t, 1), fill + 10.0, dtype=torch.float32)
        aux_right = torch.full((1, tr.d_t), fill + 10.0, dtype=torch.float32)
        return left, right, aux_left, aux_right

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)
    monkeypatch.setattr(RotAlignKVTranslator, "_predict_bridge_query_module_replace", fake_predict)
    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_route_gate", fake_route_gate)
    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_residual_adapter", fake_resid)

    base = torch.arange(64, dtype=torch.float32).view(8, 2, 2, 2)
    src_kvs = [(base, base + 0.5)]
    tgt_kvs = [(base * 1.5, base * 2.0)]

    tr.fit_from_pairs(src_kvs, tgt_kvs)

    assert len(route_calls) == 1
    assert len(resid_calls) == 3
    assert torch.allclose(tr.quant_query_route_K[0], torch.zeros_like(tr.quant_query_route_K[0]))
    assert torch.allclose(tr.quant_query_route_K_bias[0], torch.zeros_like(tr.quant_query_route_K_bias[0]))
    assert torch.allclose(tr.quant_query_route_V[0], torch.full_like(tr.quant_query_route_V[0], 7.0))
    assert torch.allclose(tr.quant_query_route_V_bias[0], torch.full_like(tr.quant_query_route_V_bias[0], -0.25))
    assert torch.allclose(tr.bridge_bank_query_resid_K_left[0], torch.zeros_like(tr.bridge_bank_query_resid_K_left[0]))
    assert torch.allclose(tr.bridge_bank_query_resid_V_left[0][0], torch.full_like(tr.bridge_bank_query_resid_V_left[0][0], 2.0))
    assert torch.allclose(tr.bridge_bank_query_resid_V_left[0][1], torch.full_like(tr.bridge_bank_query_resid_V_left[0][1], 3.0))
    assert torch.allclose(tr.bridge_bank_proj_V_left[0], torch.zeros_like(tr.bridge_bank_proj_V_left[0]))
    assert torch.allclose(tr.bridge_bank_bias_V[0], torch.zeros_like(tr.bridge_bank_bias_V[0]))


def test_bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace_routes_only_v_sidecar(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace",
        quantization_correction_rank=1,
    )
    with torch.no_grad():
        tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
        tr.quant_proj_V[0].copy_(torch.eye(tr.d_t))
        tr.quant_aux_proj_K[0].zero_()
        tr.quant_aux_proj_V[0].zero_()
        tr.quant_bias_K[0].zero_()
        tr.quant_bias_V[0].zero_()
        tr.quant_query_module_slots[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_q[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_k[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_v[0].copy_(torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_hidden[0].copy_(torch.tensor([[1.0]], dtype=torch.float32))
        tr.quant_query_module_K_out[0].copy_(torch.tensor([[3.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_V_out[0].copy_(torch.tensor([[4.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_route_V[0].zero_()
        tr.quant_query_route_V_bias[0].copy_(torch.tensor([8.0], dtype=torch.float32))
        tr.quant_query_sidecar_route_V[0].copy_(torch.tensor([[8.0], [0.0], [8.0], [0.0]], dtype=torch.float32))
        tr.quant_query_sidecar_route_V_bias[0].copy_(torch.tensor([-8.0], dtype=torch.float32))
        tr.quant_query_resid_V_left[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_resid_V_right[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_aux_resid_V_left[0].zero_()
        tr.quant_query_aux_resid_V_right[0].zero_()
        tr.quant_query_resid_K_left[0].zero_()
        tr.quant_query_resid_K_right[0].zero_()
        tr.quant_query_aux_resid_K_left[0].zero_()
        tr.quant_query_aux_resid_K_right[0].zero_()

    K_base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
    V_base = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]], dtype=torch.float32)
    qfeat = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)

    K_accept, V_accept = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_query_features=qfeat,
    )
    with torch.no_grad():
        tr.quant_query_sidecar_route_V[0].zero_()
        tr.quant_query_sidecar_route_V_bias[0].fill_(-20.0)
    K_reject, V_reject = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_query_features=qfeat,
    )

    assert torch.allclose(K_accept, K_reject, atol=1e-2, rtol=1e-2)
    assert not torch.allclose(V_accept, V_reject)
    assert V_accept.abs().max().item() > V_reject.abs().max().item()


def test_fit_from_pairs_bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace_populates_v_sidecar(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_value_verifier_sidecar_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(16, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((16, 3), -1.0, dtype=torch.float32),
        torch.ones(16, 3, tr.d_t, dtype=torch.float32),
    )

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        **kwargs,
    ):
        del self, quantized_k, predicted_k, quantized_v, predicted_v, query_features, target_k, target_v, rank, kwargs
        return (
            torch.ones((tr.config.bridge_bank_size, tr.d_t), dtype=torch.float32),
            torch.ones((tr.d_t, 1), dtype=torch.float32),
            torch.full((tr.d_t, 1), 2.0, dtype=torch.float32),
            torch.full((tr.d_t, 1), 3.0, dtype=torch.float32),
            torch.full((1, 1), 4.0, dtype=torch.float32),
            torch.full((1, tr.d_t), 5.0, dtype=torch.float32),
            torch.full((1, tr.d_t), 6.0, dtype=torch.float32),
        )

    def fake_predict(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        slot_tokens,
        q_proj,
        k_proj,
        v_proj,
        hidden_proj,
        out_k,
        out_v,
    ):
        del self, quantized_k, predicted_k, quantized_v, predicted_v, query_features, slot_tokens, q_proj, k_proj, v_proj, hidden_proj, out_k, out_v
        return (
            torch.full((16, tr.d_t), 2.0, dtype=torch.float32),
            torch.full((16, tr.d_t), 3.0, dtype=torch.float32),
        )

    route_calls = []
    resid_calls = []

    def fake_route_gate(
        self,
        query_features,
        base_prediction,
        module_prediction,
        target,
        *,
        steps=100,
        lr=5e-2,
    ):
        del self, query_features, base_prediction, module_prediction, target, steps, lr
        fill = 7.0 + 2.0 * len(route_calls)
        route_calls.append(fill)
        return (
            torch.full((tr.d_t, 1), fill, dtype=torch.float32),
            torch.full((1,), -0.25 * (len(route_calls)), dtype=torch.float32),
        )

    def fake_resid(
        self,
        quantized,
        predicted,
        query_features,
        base_prediction,
        residual_target,
        *,
        rank,
        **kwargs,
    ):
        del self, quantized, predicted, query_features, base_prediction, residual_target, rank, kwargs
        resid_calls.append(True)
        left = torch.full((tr.d_t, 1), 1.5, dtype=torch.float32)
        right = torch.full((1, tr.d_t), 2.5, dtype=torch.float32)
        aux_left = torch.full((tr.d_t, 1), 11.5, dtype=torch.float32)
        aux_right = torch.full((1, tr.d_t), 12.5, dtype=torch.float32)
        return left, right, aux_left, aux_right

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)
    monkeypatch.setattr(RotAlignKVTranslator, "_predict_bridge_query_module_replace", fake_predict)
    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_route_gate", fake_route_gate)
    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_residual_adapter", fake_resid)

    base = torch.arange(64, dtype=torch.float32).view(8, 2, 2, 2)
    src_kvs = [(base, base + 0.5)]
    tgt_kvs = [(base * 1.5, base * 2.0)]

    tr.fit_from_pairs(src_kvs, tgt_kvs)

    assert len(route_calls) == 2
    assert len(resid_calls) == 1
    assert torch.allclose(tr.quant_query_route_K[0], torch.zeros_like(tr.quant_query_route_K[0]))
    assert torch.allclose(tr.quant_query_route_K_bias[0], torch.zeros_like(tr.quant_query_route_K_bias[0]))
    assert torch.allclose(tr.quant_query_route_V[0], torch.full_like(tr.quant_query_route_V[0], 7.0))
    assert torch.allclose(tr.quant_query_route_V_bias[0], torch.full_like(tr.quant_query_route_V_bias[0], -0.25))
    assert torch.allclose(tr.quant_query_sidecar_route_K[0], torch.zeros_like(tr.quant_query_sidecar_route_K[0]))
    assert torch.allclose(tr.quant_query_sidecar_route_K_bias[0], torch.zeros_like(tr.quant_query_sidecar_route_K_bias[0]))
    assert torch.allclose(tr.quant_query_sidecar_route_V[0], torch.full_like(tr.quant_query_sidecar_route_V[0], 9.0))
    assert torch.allclose(tr.quant_query_sidecar_route_V_bias[0], torch.full_like(tr.quant_query_sidecar_route_V_bias[0], -0.5))
    assert torch.allclose(tr.quant_query_resid_K_left[0], torch.zeros_like(tr.quant_query_resid_K_left[0]))
    assert torch.allclose(tr.quant_query_resid_K_right[0], torch.zeros_like(tr.quant_query_resid_K_right[0]))
    assert torch.allclose(tr.quant_query_resid_V_left[0], torch.full_like(tr.quant_query_resid_V_left[0], 1.5))
    assert torch.allclose(tr.quant_query_resid_V_right[0], torch.full_like(tr.quant_query_resid_V_right[0], 2.5))
    assert torch.allclose(tr.quant_query_aux_resid_V_left[0], torch.full_like(tr.quant_query_aux_resid_V_left[0], 11.5))
    assert torch.allclose(tr.quant_query_aux_resid_V_right[0], torch.full_like(tr.quant_query_aux_resid_V_right[0], 12.5))


def test_fit_from_pairs_bridge_ridge_qk_dynalign_ctxonly_module_replace_reuses_module_replace_fit(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_ctxonly_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    calls: list[tuple[float, bool, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                teacher_topk_log_probs is not None,
                teacher_topk_output_rows is not None,
            )
        )
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    assert calls == [(0.25, True, True)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))
    assert torch.allclose(tr.quant_query_module_q[0], torch.full_like(tr.quant_query_module_q[0], 2.0))
    assert torch.allclose(tr.quant_query_module_k[0], torch.full_like(tr.quant_query_module_k[0], 3.0))
    assert torch.allclose(tr.quant_query_module_v[0], torch.full_like(tr.quant_query_module_v[0], 4.0))
    assert torch.allclose(tr.quant_query_module_hidden[0], torch.full_like(tr.quant_query_module_hidden[0], 5.0))
    assert torch.allclose(tr.quant_query_module_K_out[0], torch.full_like(tr.quant_query_module_K_out[0], 6.0))
    assert torch.allclose(tr.quant_query_module_V_out[0], torch.full_like(tr.quant_query_module_V_out[0], 7.0))


def test_fit_from_pairs_bridge_ridge_qk_dynalign_query_resampler_guards_nonfinite_fit_tensors(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_query_resampler_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    alignment_calls = 0
    module_calls = 0

    def fake_alignment(
        self,
        X,
        Y,
        *,
        method,
        lam,
        rank=None,
        protected_output_mask=None,
        protected_lam=None,
    ):
        nonlocal alignment_calls
        alignment_calls += 1
        if alignment_calls == 2:
            return torch.full((X.shape[1], Y.shape[1]), float("nan"), dtype=Y.dtype)
        return torch.eye(X.shape[1], Y.shape[1], dtype=Y.dtype)

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        nonlocal module_calls
        module_calls += 1
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), float("nan"), dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_alignment_with_protected_outputs", fake_alignment)
    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    tr.fit_from_pairs([(base, base + 0.5)], [(base * 1.5, base * 2.0)])

    assert alignment_calls == 2
    assert module_calls == 1
    assert torch.isfinite(tr.W_V[0]).all()
    assert torch.allclose(tr.W_V[0], torch.zeros_like(tr.W_V[0]))
    assert torch.allclose(tr.quant_query_module_K_out[0], torch.full_like(tr.quant_query_module_K_out[0], 6.0))
    assert torch.isfinite(tr.quant_query_module_V_out[0]).all()
    assert torch.allclose(tr.quant_query_module_V_out[0], torch.zeros_like(tr.quant_query_module_V_out[0]))


def test_fit_from_pairs_bridge_ridge_qk_dynalign_query_innovation_resampler_fits_residual_targets(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_query_innovation_resampler_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    captured: dict[str, torch.Tensor] = {}

    def fake_bridge_ridge(self, quantized, predicted, target, *, lam, sample_weights=None):
        del quantized, predicted, target, lam, sample_weights
        return (
            torch.zeros(self.d_t, self.d_t, dtype=torch.float32),
            torch.zeros(self.d_t, self.d_t, dtype=torch.float32),
            torch.full((self.d_t,), 0.25, dtype=torch.float32),
        )

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        del quantized_k, predicted_k, quantized_v, predicted_v, query_features
        del prediction_distill_weight, teacher_topk_log_probs, teacher_topk_output_rows, kwargs
        captured["target_k"] = target_k.detach().clone()
        captured["target_v"] = target_v.detach().clone()
        return (
            torch.zeros(self.config.bridge_bank_size, self.d_t, dtype=target_k.dtype),
            torch.zeros(self.d_t, rank, dtype=target_k.dtype),
            torch.zeros(self.d_t, rank, dtype=target_k.dtype),
            torch.zeros(self.d_t, rank, dtype=target_k.dtype),
            torch.zeros(rank, rank, dtype=target_k.dtype),
            torch.zeros(rank, self.d_t, dtype=target_k.dtype),
            torch.zeros(rank, self.d_t, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_ridge_correction", fake_bridge_ridge)
    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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
    tgt_k = base * 1.5
    tgt_v = base * 2.0

    tr.fit_from_pairs([(base, base + 0.5)], [(tgt_k, tgt_v)])

    expected_k = tr._rotate_and_flatten(tgt_k, tr.R_t).reshape(-1, tr.d_t) - 0.25
    expected_v = tr._rotate_and_flatten(tgt_v, tr.R_t).reshape(-1, tr.d_t) - 0.25
    assert torch.allclose(captured["target_k"], expected_k)
    assert torch.allclose(captured["target_v"], expected_v)


def test_fit_from_pairs_bridge_ridge_qk_dynalign_query_innovation_resampler_forwards_sample_weights(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_query_innovation_resampler_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )
    expected_weights = torch.tensor([1.0, 4.0, 1.0, 4.0], dtype=torch.float32)
    tr.set_bridge_sample_weights([expected_weights])

    base_ridge_weighted: list[bool] = []
    captured: dict[str, torch.Tensor] = {}

    def fake_bridge_ridge(self, quantized, predicted, target, *, lam, sample_weights=None):
        del quantized, predicted, target, lam
        base_ridge_weighted.append(sample_weights is not None)
        return (
            torch.zeros(self.d_t, self.d_t, dtype=torch.float32),
            torch.zeros(self.d_t, self.d_t, dtype=torch.float32),
            torch.full((self.d_t,), 0.25, dtype=torch.float32),
        )

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        sample_weights=None,
        **kwargs,
    ):
        del quantized_k, predicted_k, quantized_v, predicted_v, query_features
        del target_k, target_v, rank, kwargs
        assert sample_weights is not None
        captured["sample_weights"] = sample_weights.detach().cpu().clone()
        return (
            torch.zeros(self.config.bridge_bank_size, self.d_t, dtype=torch.float32),
            torch.zeros(self.d_t, 1, dtype=torch.float32),
            torch.zeros(self.d_t, 1, dtype=torch.float32),
            torch.zeros(self.d_t, 1, dtype=torch.float32),
            torch.zeros(1, 1, dtype=torch.float32),
            torch.zeros(1, self.d_t, dtype=torch.float32),
            torch.zeros(1, self.d_t, dtype=torch.float32),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_ridge_correction", fake_bridge_ridge)
    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    tr.fit_from_pairs([(base, base + 0.5)], [(base * 1.5, base * 2.0)])

    assert base_ridge_weighted == [False, False]
    assert torch.allclose(captured["sample_weights"], expected_weights)


def test_dynalign_query_innovation_resampler_fuse_adds_bounded_residual(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_query_innovation_resampler_replace",
        quantization_correction_rank=1,
    )
    tr.set_fixed_gates(0.25)

    K_t = torch.ones(1, 2, 1, 2, dtype=torch.float32)
    V_t = torch.full((1, 2, 1, 2), 2.0, dtype=torch.float32)
    K_delta = torch.full_like(K_t, 10.0)
    V_delta = torch.full_like(V_t, 10.0)

    out_k, out_v = tr.fuse_layer(K_t, V_t, K_delta, V_delta, tgt_layer_idx=0)

    assert torch.allclose(out_k, torch.full_like(K_t, 1.0625), atol=1e-6)
    assert torch.allclose(out_v, torch.full_like(V_t, 2.125), atol=1e-6)


def test_dynalign_query_resampler_allows_zero_bridge_bank(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_query_resampler_replace",
        quantization_correction_rank=1,
        bridge_bank_size=0,
    )

    assert tr.quant_query_module_slots[0].shape == (0, tr.d_t)

    base = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 1.0, 0.0],
            [1.5, 0.0, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    fitted = tr._fit_bridge_query_module_replace(
        base,
        base + 0.1,
        base + 0.2,
        base + 0.3,
        torch.ones_like(base),
        base + 0.4,
        base + 0.5,
        rank=1,
        steps=1,
    )

    assert fitted[0].shape == (0, tr.d_t)
    assert all(torch.isfinite(tensor).all().item() for tensor in fitted)

    K_base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
    V_base = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]], dtype=torch.float32)
    out_k, out_v = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_query_features=torch.ones(1, 1, tr.d_t, dtype=torch.float32),
    )

    assert out_k.shape == K_base.shape
    assert out_v.shape == V_base.shape
    assert torch.isfinite(out_k).all()
    assert torch.isfinite(out_v).all()


def test_dynalign_query_innovation_resampler_allows_zero_bridge_bank(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_query_innovation_resampler_replace",
        quantization_correction_rank=1,
        bridge_bank_size=0,
    )

    assert tr.quant_query_module_slots[0].shape == (0, tr.d_t)

    base = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 1.0, 0.0],
            [1.5, 0.0, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    fitted = tr._fit_bridge_query_module_replace(
        base,
        base + 0.1,
        base + 0.2,
        base + 0.3,
        torch.ones_like(base),
        base + 0.4,
        base + 0.5,
        rank=1,
        steps=1,
    )

    assert fitted[0].shape == (0, tr.d_t)
    assert all(torch.isfinite(tensor).all().item() for tensor in fitted)


def test_fit_from_pairs_bridge_ridge_qk_dynalign_dwakd_module_replace_uses_dynamic_teacher_and_weights(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_dwakd_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )
    tr.set_bridge_sample_weights([torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)])

    calls: list[tuple[float, float, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        dynamic_prediction_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        sample_weights=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                float(dynamic_prediction_weight),
                sample_weights is not None,
            )
        )
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    assert calls == [(0.25, 0.25, True)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))
    assert torch.allclose(tr.quant_query_module_q[0], torch.full_like(tr.quant_query_module_q[0], 2.0))
    assert torch.allclose(tr.quant_query_module_k[0], torch.full_like(tr.quant_query_module_k[0], 3.0))
    assert torch.allclose(tr.quant_query_module_v[0], torch.full_like(tr.quant_query_module_v[0], 4.0))
    assert torch.allclose(tr.quant_query_module_hidden[0], torch.full_like(tr.quant_query_module_hidden[0], 5.0))
    assert torch.allclose(tr.quant_query_module_K_out[0], torch.full_like(tr.quant_query_module_K_out[0], 6.0))
    assert torch.allclose(tr.quant_query_module_V_out[0], torch.full_like(tr.quant_query_module_V_out[0], 7.0))


def test_fit_from_pairs_bridge_ridge_qk_dynalign_likelihood_module_replace_uses_dynamic_teacher_and_weights(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_likelihood_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )
    tr.set_bridge_sample_weights([torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)])

    calls: list[tuple[float, float, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        dynamic_prediction_weight=0.0,
        sample_weights=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                float(dynamic_prediction_weight),
                sample_weights is not None,
            )
        )
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    assert calls == [(0.25, 0.25, True)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))
    assert torch.allclose(tr.quant_query_module_q[0], torch.full_like(tr.quant_query_module_q[0], 2.0))
    assert torch.allclose(tr.quant_query_module_k[0], torch.full_like(tr.quant_query_module_k[0], 3.0))
    assert torch.allclose(tr.quant_query_module_v[0], torch.full_like(tr.quant_query_module_v[0], 4.0))
    assert torch.allclose(tr.quant_query_module_hidden[0], torch.full_like(tr.quant_query_module_hidden[0], 5.0))
    assert torch.allclose(tr.quant_query_module_K_out[0], torch.full_like(tr.quant_query_module_K_out[0], 6.0))
    assert torch.allclose(tr.quant_query_module_V_out[0], torch.full_like(tr.quant_query_module_V_out[0], 7.0))


def test_fit_from_pairs_bridge_ridge_qk_dynalign_spanalm_module_replace_uses_dynamic_teacher_and_weights(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_spanalm_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )
    tr.set_bridge_sample_weights([torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)])

    calls: list[tuple[float, float, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        dynamic_prediction_weight=0.0,
        sample_weights=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                float(dynamic_prediction_weight),
                sample_weights is not None,
            )
        )
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    assert calls == [(0.25, 0.25, True)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))
    assert torch.allclose(tr.quant_query_module_q[0], torch.full_like(tr.quant_query_module_q[0], 2.0))
    assert torch.allclose(tr.quant_query_module_k[0], torch.full_like(tr.quant_query_module_k[0], 3.0))
    assert torch.allclose(tr.quant_query_module_v[0], torch.full_like(tr.quant_query_module_v[0], 4.0))
    assert torch.allclose(tr.quant_query_module_hidden[0], torch.full_like(tr.quant_query_module_hidden[0], 5.0))
    assert torch.allclose(tr.quant_query_module_K_out[0], torch.full_like(tr.quant_query_module_K_out[0], 6.0))
    assert torch.allclose(tr.quant_query_module_V_out[0], torch.full_like(tr.quant_query_module_V_out[0], 7.0))


def test_fit_from_pairs_bridge_ridge_qk_dynalign_prefdist_module_replace_uses_pairwise_preference_loss(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_prefdist_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )
    tr.set_bridge_sample_weights([torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)])

    calls: list[tuple[float, float, bool, float]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        dynamic_prediction_weight=0.0,
        sample_weights=None,
        span_preference_weight=0.0,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                float(dynamic_prediction_weight),
                sample_weights is not None,
                float(span_preference_weight),
            )
        )
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    assert calls == [(0.25, 0.25, True, 0.25)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))
    assert torch.allclose(tr.quant_query_module_q[0], torch.full_like(tr.quant_query_module_q[0], 2.0))
    assert torch.allclose(tr.quant_query_module_k[0], torch.full_like(tr.quant_query_module_k[0], 3.0))
    assert torch.allclose(tr.quant_query_module_v[0], torch.full_like(tr.quant_query_module_v[0], 4.0))
    assert torch.allclose(tr.quant_query_module_hidden[0], torch.full_like(tr.quant_query_module_hidden[0], 5.0))
    assert torch.allclose(tr.quant_query_module_K_out[0], torch.full_like(tr.quant_query_module_K_out[0], 6.0))
    assert torch.allclose(tr.quant_query_module_V_out[0], torch.full_like(tr.quant_query_module_V_out[0], 7.0))


def test_fit_from_pairs_bridge_ridge_qk_dynalign_dwainteract_module_replace_stacks_dynamic_weights_and_prompt_ids(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_dwainteract_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )
    tr.set_bridge_sample_weights([torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)])
    tr.set_bridge_sample_prompt_ids(torch.tensor([0, 0, 1, 1], dtype=torch.long))

    calls: list[tuple[float, bool, float, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        dynamic_prediction_weight=0.0,
        sample_weights=None,
        interaction_distill_weight=0.0,
        sample_prompt_ids=None,
        **kwargs,
    ):
        calls.append(
            (
                float(dynamic_prediction_weight),
                sample_weights is not None,
                float(interaction_distill_weight),
                sample_prompt_ids is not None,
            )
        )
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    assert calls == [(0.25, True, 0.25, True)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))
    assert torch.allclose(tr.quant_query_module_q[0], torch.full_like(tr.quant_query_module_q[0], 2.0))
    assert torch.allclose(tr.quant_query_module_k[0], torch.full_like(tr.quant_query_module_k[0], 3.0))
    assert torch.allclose(tr.quant_query_module_v[0], torch.full_like(tr.quant_query_module_v[0], 4.0))
    assert torch.allclose(tr.quant_query_module_hidden[0], torch.full_like(tr.quant_query_module_hidden[0], 5.0))
    assert torch.allclose(tr.quant_query_module_K_out[0], torch.full_like(tr.quant_query_module_K_out[0], 6.0))
    assert torch.allclose(tr.quant_query_module_V_out[0], torch.full_like(tr.quant_query_module_V_out[0], 7.0))


def test_fit_from_pairs_bridge_ridge_qk_dynalign_interact_module_replace_uses_prompt_ids(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dynalign_interact_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )
    tr.set_bridge_sample_prompt_ids(torch.tensor([0, 0, 1, 1], dtype=torch.long))

    calls: list[tuple[float, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        interaction_distill_weight=0.0,
        sample_prompt_ids=None,
        **kwargs,
    ):
        calls.append((float(interaction_distill_weight), sample_prompt_ids is not None))
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    assert calls == [(0.25, True)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))


def test_fit_from_pairs_bridge_ridge_qk_dpalign_module_replace_reuses_module_replace_fit(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_dpalign_module_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    calls: list[tuple[float, bool, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                teacher_topk_log_probs is not None,
                teacher_topk_output_rows is not None,
            )
        )
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 7.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_module_replace", fake_fit)

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

    assert calls == [(0.25, True, True)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))
    assert torch.allclose(tr.quant_query_module_q[0], torch.full_like(tr.quant_query_module_q[0], 2.0))
    assert torch.allclose(tr.quant_query_module_k[0], torch.full_like(tr.quant_query_module_k[0], 3.0))
    assert torch.allclose(tr.quant_query_module_v[0], torch.full_like(tr.quant_query_module_v[0], 4.0))
    assert torch.allclose(tr.quant_query_module_hidden[0], torch.full_like(tr.quant_query_module_hidden[0], 5.0))
    assert torch.allclose(tr.quant_query_module_K_out[0], torch.full_like(tr.quant_query_module_K_out[0], 6.0))
    assert torch.allclose(tr.quant_query_module_V_out[0], torch.full_like(tr.quant_query_module_V_out[0], 7.0))


def test_bridge_ridge_qk_tokenbasis_replace_uses_token_basis_output(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_tokenbasis_replace",
        quantization_correction_rank=1,
    )
    with torch.no_grad():
        tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
        tr.quant_proj_V[0].copy_(torch.eye(tr.d_t))
        tr.quant_aux_proj_K[0].zero_()
        tr.quant_aux_proj_V[0].zero_()
        tr.quant_bias_K[0].zero_()
        tr.quant_bias_V[0].zero_()
        tr.quant_query_resid_K_left[0].zero_()
        tr.quant_query_resid_K_right[0].zero_()
        tr.quant_query_aux_resid_K_left[0].zero_()
        tr.quant_query_aux_resid_K_right[0].zero_()
        tr.quant_query_resid_V_left[0].zero_()
        tr.quant_query_resid_V_right[0].zero_()
        tr.quant_query_aux_resid_V_left[0].zero_()
        tr.quant_query_aux_resid_V_right[0].zero_()
        tr.quant_query_module_slots[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_module_q[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_k[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_v[0].copy_(torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32))
        tr.quant_query_module_hidden[0].copy_(torch.tensor([[1.0]], dtype=torch.float32))
        tr.quant_query_token_basis[0].copy_(torch.tensor([[3.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.quant_query_token_K_coeff[0].copy_(torch.tensor([[1.0]], dtype=torch.float32))
        tr.quant_query_token_V_coeff[0].copy_(torch.tensor([[2.0]], dtype=torch.float32))

    K_base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
    V_base = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]], dtype=torch.float32)
    out_k, out_v = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_query_features=torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32),
    )

    assert out_k.shape == K_base.shape
    assert out_v.shape == V_base.shape
    assert not torch.allclose(out_k, K_base)
    assert not torch.allclose(out_v, V_base)
    assert out_v.abs().max().item() > out_k.abs().max().item()


def test_fit_from_pairs_bridge_ridge_qk_tokenbasis_replace_populates_basis_and_uses_teacher(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_tokenbasis_replace",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    calls: list[tuple[float, bool, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        target_k,
        target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                teacher_topk_log_probs is not None,
                teacher_topk_output_rows is not None,
            )
        )
        return (
            torch.full((self.config.bridge_bank_size, self.d_t), 1.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 3.0, dtype=target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=target_k.dtype),
            torch.full((rank, rank), 5.0, dtype=target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=target_k.dtype),
            torch.full((rank, rank), 7.0, dtype=target_k.dtype),
            torch.full((rank, rank), 8.0, dtype=target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_tokenbasis_replace", fake_fit)

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

    assert calls == [(0.25, True, True)]
    assert torch.allclose(tr.quant_query_module_slots[0], torch.ones_like(tr.quant_query_module_slots[0]))
    assert torch.allclose(tr.quant_query_module_q[0], torch.full_like(tr.quant_query_module_q[0], 2.0))
    assert torch.allclose(tr.quant_query_module_k[0], torch.full_like(tr.quant_query_module_k[0], 3.0))
    assert torch.allclose(tr.quant_query_module_v[0], torch.full_like(tr.quant_query_module_v[0], 4.0))
    assert torch.allclose(tr.quant_query_module_hidden[0], torch.full_like(tr.quant_query_module_hidden[0], 5.0))
    assert torch.allclose(tr.quant_query_token_basis[0], torch.full_like(tr.quant_query_token_basis[0], 6.0))
    assert torch.allclose(tr.quant_query_token_K_coeff[0], torch.full_like(tr.quant_query_token_K_coeff[0], 7.0))
    assert torch.allclose(tr.quant_query_token_V_coeff[0], torch.full_like(tr.quant_query_token_V_coeff[0], 8.0))


def test_fit_from_pairs_bridge_ridge_qk_asym_predkl_adapter_passes_teacher(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_asym_predkl_adapter",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    calls: list[tuple[float, bool, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        base_prediction_k,
        base_prediction_v,
        residual_target_k,
        residual_target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                teacher_topk_log_probs is not None,
                teacher_topk_output_rows is not None,
            )
        )
        return (
            torch.full((self.d_t, rank), 3.0, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 5.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=residual_target_v.dtype),
            torch.full((self.d_t, rank), 1.0, dtype=residual_target_k.dtype),
            torch.ones(rank, self.d_t, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 1.5, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 1.5, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=residual_target_v.dtype),
            torch.full((rank, self.d_t), 2.0, dtype=residual_target_v.dtype),
            torch.full((self.d_t, rank), 2.5, dtype=residual_target_v.dtype),
            torch.full((rank, self.d_t), 2.5, dtype=residual_target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_shared_residual_adapter", fake_fit)

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

    assert calls == [(0.25, True, True)]
    assert torch.allclose(tr.quant_query_shared_left[0], torch.full_like(tr.quant_query_shared_left[0], 3.0))
    assert torch.allclose(tr.quant_query_shared_K_right[0], torch.full_like(tr.quant_query_shared_K_right[0], 5.0))
    assert torch.allclose(tr.quant_query_shared_V_right[0], torch.full_like(tr.quant_query_shared_V_right[0], 6.0))
    assert torch.allclose(tr.quant_query_resid_K_left[0], torch.ones_like(tr.quant_query_resid_K_left[0]))
    assert torch.allclose(tr.quant_query_resid_V_left[0], torch.full_like(tr.quant_query_resid_V_left[0], 2.0))


def test_fit_from_pairs_bridge_ridge_qk_asym_dynmap_adapter_passes_dynamic_teacher(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_asym_dynmap_adapter",
        quantization_correction_rank=1,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )

    calls: list[tuple[float, float, bool, bool]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        base_prediction_k,
        base_prediction_v,
        residual_target_k,
        residual_target_v,
        *,
        rank,
        prediction_distill_weight=0.0,
        dynamic_prediction_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                float(dynamic_prediction_weight),
                teacher_topk_log_probs is not None,
                teacher_topk_output_rows is not None,
            )
        )
        return (
            torch.full((self.d_t, rank), 3.0, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 4.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 5.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 6.0, dtype=residual_target_v.dtype),
            torch.full((self.d_t, rank), 1.0, dtype=residual_target_k.dtype),
            torch.ones(rank, self.d_t, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 1.5, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 1.5, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 2.0, dtype=residual_target_v.dtype),
            torch.full((rank, self.d_t), 2.0, dtype=residual_target_v.dtype),
            torch.full((self.d_t, rank), 2.5, dtype=residual_target_v.dtype),
            torch.full((rank, self.d_t), 2.5, dtype=residual_target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_shared_residual_adapter", fake_fit)

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

    assert calls == [(0.0, 0.25, True, True)]
    assert torch.allclose(tr.quant_query_shared_left[0], torch.full_like(tr.quant_query_shared_left[0], 3.0))
    assert torch.allclose(tr.quant_query_shared_K_right[0], torch.full_like(tr.quant_query_shared_K_right[0], 5.0))
    assert torch.allclose(tr.quant_query_shared_V_right[0], torch.full_like(tr.quant_query_shared_V_right[0], 6.0))
    assert torch.allclose(tr.quant_query_resid_K_left[0], torch.ones_like(tr.quant_query_resid_K_left[0]))
    assert torch.allclose(tr.quant_query_resid_V_left[0], torch.full_like(tr.quant_query_resid_V_left[0], 2.0))


def test_bridge_ridge_qk_sae_adapter_uses_sparse_shared_codes(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_sae_adapter",
        quantization_correction_rank=2,
    )
    with torch.no_grad():
        tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
        tr.quant_proj_V[0].copy_(torch.eye(tr.d_t))
        tr.quant_aux_proj_K[0].zero_()
        tr.quant_aux_proj_V[0].zero_()
        tr.quant_bias_K[0].zero_()
        tr.quant_bias_V[0].zero_()
        tr.quant_query_resid_K_left[0].zero_()
        tr.quant_query_resid_K_right[0].zero_()
        tr.quant_query_aux_resid_K_left[0].zero_()
        tr.quant_query_aux_resid_K_right[0].zero_()
        tr.quant_query_resid_V_left[0].zero_()
        tr.quant_query_resid_V_right[0].zero_()
        tr.quant_query_aux_resid_V_left[0].zero_()
        tr.quant_query_aux_resid_V_right[0].zero_()
        tr.quant_query_sparse_left[0].copy_(
            torch.tensor(
                [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                dtype=torch.float32,
            )
        )
        tr.quant_query_sparse_aux_left[0].zero_()
        tr.quant_query_sparse_K_right[0].copy_(
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
            )
        )
        tr.quant_query_sparse_V_right[0].copy_(
            torch.tensor(
                [[0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
            )
        )

    K_base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
    V_base = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]], dtype=torch.float32)
    out_k, out_v = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_query_features=torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32),
    )

    assert out_k.shape == K_base.shape
    assert out_v.shape == V_base.shape
    assert not torch.allclose(out_k, K_base)
    assert not torch.allclose(out_v, V_base)


def test_fit_from_pairs_bridge_ridge_qk_sae_adapter_populates_sparse_bridge(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_sae_adapter",
        quantization_correction_rank=2,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])

    calls: list[int] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        base_prediction_k,
        base_prediction_v,
        residual_target_k,
        residual_target_v,
        *,
        rank,
        **kwargs,
    ):
        calls.append(rank)
        return (
            torch.full((self.d_t, rank), 7.0, dtype=residual_target_k.dtype),
            torch.full((self.d_t, rank), 8.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 9.0, dtype=residual_target_k.dtype),
            torch.full((rank, self.d_t), 10.0, dtype=residual_target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_sparse_adapter", fake_fit)

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

    assert calls == [2]
    assert torch.allclose(tr.quant_query_sparse_left[0], torch.full_like(tr.quant_query_sparse_left[0], 7.0))
    assert torch.allclose(tr.quant_query_sparse_V_right[0], torch.full_like(tr.quant_query_sparse_V_right[0], 10.0))


def test_bridge_ridge_qk_generated_adapter_uses_dynamic_mixture(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_generated_adapter",
        quantization_correction_rank=1,
        bridge_bank_size=2,
        bridge_bank_temperature=1.0,
    )
    with torch.no_grad():
        tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
        tr.quant_proj_V[0].copy_(torch.eye(tr.d_t))
        tr.quant_aux_proj_K[0].zero_()
        tr.quant_aux_proj_V[0].zero_()
        tr.quant_bias_K[0].zero_()
        tr.quant_bias_V[0].zero_()
        tr.quant_query_resid_K_left[0].zero_()
        tr.quant_query_resid_K_right[0].zero_()
        tr.quant_query_aux_resid_K_left[0].zero_()
        tr.quant_query_aux_resid_K_right[0].zero_()
        tr.quant_query_resid_V_left[0].zero_()
        tr.quant_query_resid_V_right[0].zero_()
        tr.quant_query_aux_resid_V_left[0].zero_()
        tr.quant_query_aux_resid_V_right[0].zero_()
        tr.quant_query_hyper_left[0].copy_(
            torch.tensor(
                [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                dtype=torch.float32,
            )
        )
        tr.quant_query_hyper_aux_left[0].zero_()
        tr.bridge_bank_query_resid_K_left[0][0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
        tr.bridge_bank_query_resid_K_right[0][0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.bridge_bank_query_aux_resid_K_left[0].zero_()
        tr.bridge_bank_query_aux_resid_K_right[0].zero_()
        tr.bridge_bank_query_resid_V_left[0][0].copy_(torch.tensor([[0.5], [0.0], [0.5], [0.0]], dtype=torch.float32))
        tr.bridge_bank_query_resid_V_right[0][0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
        tr.bridge_bank_query_aux_resid_V_left[0].zero_()
        tr.bridge_bank_query_aux_resid_V_right[0].zero_()

    K_base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
    V_base = torch.tensor([[[[2.0, 0.0]], [[2.0, 0.0]]]], dtype=torch.float32)
    out_k, out_v = tr.translate_layer(
        K_base,
        V_base,
        tgt_layer_idx=0,
        quantize=True,
        runtime_query_features=torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32),
    )

    assert out_k.shape == K_base.shape
    assert out_v.shape == V_base.shape
    assert not torch.allclose(out_k, K_base)
    assert not torch.allclose(out_v, V_base)


def test_fit_from_pairs_bridge_ridge_qk_generated_adapter_populates_bank_and_hyper(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_generated_adapter",
        quantization_correction_rank=2,
        bridge_bank_size=3,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])

    calls: list[tuple[int, int]] = []

    def fake_fit(
        self,
        quantized_k,
        predicted_k,
        quantized_v,
        predicted_v,
        query_features,
        base_prediction_k,
        base_prediction_v,
        residual_target_k,
        residual_target_v,
        *,
        rank,
        experts,
        **kwargs,
    ):
        calls.append((rank, experts))
        return (
            torch.full((self.d_t, experts), 11.0, dtype=residual_target_k.dtype),
            torch.full((self.d_t, experts), 12.0, dtype=residual_target_k.dtype),
            torch.full((experts, self.d_t, rank), 13.0, dtype=residual_target_k.dtype),
            torch.full((experts, rank, self.d_t), 14.0, dtype=residual_target_k.dtype),
            torch.full((experts, self.d_t, rank), 15.0, dtype=residual_target_k.dtype),
            torch.full((experts, rank, self.d_t), 16.0, dtype=residual_target_k.dtype),
            torch.full((experts, self.d_t, rank), 17.0, dtype=residual_target_v.dtype),
            torch.full((experts, rank, self.d_t), 18.0, dtype=residual_target_v.dtype),
            torch.full((experts, self.d_t, rank), 19.0, dtype=residual_target_v.dtype),
            torch.full((experts, rank, self.d_t), 20.0, dtype=residual_target_v.dtype),
        )

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_generated_adapter", fake_fit)

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

    assert calls == [(2, 3)]
    assert torch.allclose(tr.quant_query_hyper_left[0], torch.full_like(tr.quant_query_hyper_left[0], 11.0))
    assert torch.allclose(tr.bridge_bank_query_resid_K_left[0], torch.full_like(tr.bridge_bank_query_resid_K_left[0], 13.0))
    assert torch.allclose(tr.bridge_bank_query_aux_resid_V_right[0], torch.full_like(tr.bridge_bank_query_aux_resid_V_right[0], 20.0))


def test_bridge_low_rank_bank_selects_runtime_matched_expert(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_low_rank_bank",
        quantization_correction_rank=1,
        bridge_bank_size=2,
        bridge_bank_temperature=40.0,
        transport_template_bins=4,
    )
    with torch.no_grad():
        tr.bridge_bank_templates[0].copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        tr.bridge_bank_priors[0].copy_(torch.tensor([0.5, 0.5], dtype=torch.float32))
        tr.bridge_bank_bias_K[0].data[0].fill_(1.0)
        tr.bridge_bank_bias_K[0].data[1].fill_(3.0)
        tr.bridge_bank_bias_V[0].data[0].fill_(2.0)
        tr.bridge_bank_bias_V[0].data[1].fill_(4.0)

    zeros = torch.zeros(1, 2, 1, 2, dtype=torch.float32)
    K_match, V_match = tr.translate_layer(
        zeros,
        zeros,
        tgt_layer_idx=0,
        quantize=True,
        runtime_attention_profile=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    K_other, V_other = tr.translate_layer(
        zeros,
        zeros,
        tgt_layer_idx=0,
        quantize=True,
        runtime_attention_profile=torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )

    assert torch.allclose(K_match, torch.ones_like(K_match), atol=1e-2, rtol=1e-2)
    assert torch.allclose(V_match, 2.0 * torch.ones_like(V_match), atol=1e-2, rtol=1e-2)
    assert torch.allclose(K_other, 3.0 * torch.ones_like(K_other), atol=1e-2, rtol=1e-2)
    assert torch.allclose(V_other, 4.0 * torch.ones_like(V_other), atol=1e-2, rtol=1e-2)


def test_bridge_ridge_residual_bank_adds_runtime_selected_residual(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_residual_bank",
        quantization_correction_rank=1,
        bridge_bank_size=2,
        bridge_bank_temperature=40.0,
        transport_template_bins=4,
    )
    with torch.no_grad():
        tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
        tr.quant_proj_V[0].copy_(torch.eye(tr.d_t))
        tr.quant_aux_proj_K[0].zero_()
        tr.quant_aux_proj_V[0].zero_()
        tr.quant_bias_K[0].fill_(5.0)
        tr.quant_bias_V[0].fill_(7.0)
        tr.bridge_bank_templates[0].copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        tr.bridge_bank_priors[0].copy_(torch.tensor([0.5, 0.5], dtype=torch.float32))
        tr.bridge_bank_bias_K[0].data[0].fill_(1.0)
        tr.bridge_bank_bias_K[0].data[1].fill_(3.0)
        tr.bridge_bank_bias_V[0].data[0].fill_(2.0)
        tr.bridge_bank_bias_V[0].data[1].fill_(4.0)

    zeros = torch.zeros(1, 2, 1, 2, dtype=torch.float32)
    K_match, V_match = tr.translate_layer(
        zeros,
        zeros,
        tgt_layer_idx=0,
        quantize=True,
        runtime_attention_profile=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    K_other, V_other = tr.translate_layer(
        zeros,
        zeros,
        tgt_layer_idx=0,
        quantize=True,
        runtime_attention_profile=torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )

    assert torch.allclose(K_match, 6.0 * torch.ones_like(K_match), atol=1e-2, rtol=1e-2)
    assert torch.allclose(V_match, 9.0 * torch.ones_like(V_match), atol=1e-2, rtol=1e-2)
    assert torch.allclose(K_other, 8.0 * torch.ones_like(K_other), atol=1e-2, rtol=1e-2)
    assert torch.allclose(V_other, 11.0 * torch.ones_like(V_other), atol=1e-2, rtol=1e-2)


def test_bridge_ridge_qk_bank_routes_query_conditioned_experts(monkeypatch) -> None:
    for mode in ("bridge_ridge_qk_cab_bank", "bridge_ridge_qk_predkl_bank"):
        tr = _make_identity_translator(
            monkeypatch,
            quantization_correction=mode,
            quantization_correction_rank=1,
            bridge_bank_size=2,
            bridge_bank_temperature=40.0,
            transport_template_bins=4,
        )
        with torch.no_grad():
            tr.quant_proj_K[0].copy_(torch.eye(tr.d_t))
            tr.quant_proj_V[0].copy_(torch.eye(tr.d_t))
            tr.quant_aux_proj_K[0].zero_()
            tr.quant_aux_proj_V[0].zero_()
            tr.quant_bias_K[0].fill_(5.0)
            tr.quant_bias_V[0].fill_(7.0)
            tr.bridge_bank_templates[0].copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            )
            tr.bridge_bank_priors[0].copy_(torch.tensor([0.5, 0.5], dtype=torch.float32))
            tr.bridge_bank_query_resid_K_left[0].data[0].copy_(torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32))
            tr.bridge_bank_query_resid_K_right[0].data[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
            tr.bridge_bank_query_resid_K_left[0].data[1].copy_(torch.tensor([[3.0], [0.0], [3.0], [0.0]], dtype=torch.float32))
            tr.bridge_bank_query_resid_K_right[0].data[1].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
            tr.bridge_bank_query_resid_V_left[0].data[0].copy_(torch.tensor([[2.0], [0.0], [2.0], [0.0]], dtype=torch.float32))
            tr.bridge_bank_query_resid_V_right[0].data[0].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
            tr.bridge_bank_query_resid_V_left[0].data[1].copy_(torch.tensor([[4.0], [0.0], [4.0], [0.0]], dtype=torch.float32))
            tr.bridge_bank_query_resid_V_right[0].data[1].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))

        base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32)
        qfeat = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)
        K_match, V_match = tr.translate_layer(
            base,
            torch.zeros_like(base),
            tgt_layer_idx=0,
            quantize=True,
            runtime_attention_profile=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            runtime_query_features=qfeat,
        )
        K_other, V_other = tr.translate_layer(
            base,
            torch.zeros_like(base),
            tgt_layer_idx=0,
            quantize=True,
            runtime_attention_profile=torch.tensor([0.0, 1.0, 0.0, 0.0]),
            runtime_query_features=qfeat,
        )

        assert torch.allclose(K_match, torch.tensor([[[[8.0, 5.0]], [[6.0, 5.0]]]]), atol=1e-2, rtol=1e-2)
        assert torch.allclose(V_match, 7.0 * torch.ones_like(V_match), atol=1e-2, rtol=1e-2)
        assert torch.allclose(K_other, torch.tensor([[[[12.0, 5.0]], [[6.0, 5.0]]]]), atol=1e-2, rtol=1e-2)
        assert torch.allclose(V_other, 7.0 * torch.ones_like(V_other), atol=1e-2, rtol=1e-2)


def test_fit_from_pairs_bridge_ridge_qk_predkl_bank_passes_teacher_to_global_bank_fit(monkeypatch) -> None:
    tr = _make_identity_translator(
        monkeypatch,
        quantization_correction="bridge_ridge_qk_predkl_bank",
        quantization_correction_rank=1,
        bridge_bank_size=2,
        transport_template_bins=4,
    )
    tr.set_bridge_sample_query_features([torch.ones(4, tr.d_t, dtype=torch.float32)])
    tr.set_bridge_prediction_teacher(
        torch.full((4, 3), -1.0, dtype=torch.float32),
        torch.ones(4, 3, tr.d_t, dtype=torch.float32),
    )
    tr._bridge_prompt_cluster_labels = [torch.tensor([0, 1], dtype=torch.long)]
    tr._bridge_sample_prompt_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    calls: list[tuple[float, bool, bool]] = []

    def fake_fit(
        self,
        quantized,
        predicted,
        query_features,
        base_prediction,
        residual_target,
        *,
        rank,
        prediction_distill_weight=0.0,
        teacher_topk_log_probs=None,
        teacher_topk_output_rows=None,
        **kwargs,
    ):
        calls.append(
            (
                float(prediction_distill_weight),
                teacher_topk_log_probs is not None,
                teacher_topk_output_rows is not None,
            )
        )
        fill = float(len(calls))
        left = torch.full((self.d_t, rank), fill, dtype=residual_target.dtype)
        right = torch.zeros(rank, self.d_t, dtype=residual_target.dtype)
        right[0, 0] = fill
        aux_left = torch.zeros_like(left)
        aux_right = torch.zeros_like(right)
        return left, right, aux_left, aux_right

    monkeypatch.setattr(RotAlignKVTranslator, "_fit_bridge_query_residual_adapter", fake_fit)

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

    assert calls == [(0.25, True, True), (0.25, True, True)]
    assert torch.allclose(tr.bridge_bank_query_resid_K_left[0][0], torch.ones_like(tr.bridge_bank_query_resid_K_left[0][0]))
    assert torch.allclose(tr.bridge_bank_query_resid_V_left[0][0], torch.full_like(tr.bridge_bank_query_resid_V_left[0][0], 2.0))


def test_fit_from_pairs_low_rank_correction_reduces_quantized_error(monkeypatch) -> None:
    monkeypatch.setattr(translator_mod, "GaussianQuantizer", _OffsetQuantizer)
    monkeypatch.setattr(translator_mod, "make_rotation", lambda d, **_: torch.eye(d))

    cfg_none = translator_mod.TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=1,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=1,
        quantization_correction="none",
        ridge_lambda=1e-4,
    )
    cfg_low_rank = translator_mod.TranslatorConfig(
        src_head_dim=2,
        src_num_heads=2,
        num_src_layers=1,
        tgt_head_dim=2,
        tgt_num_heads=2,
        num_tgt_layers=1,
        quantization_correction="low_rank",
        quantization_correction_rank=1,
        ridge_lambda=1e-4,
    )
    tr_none = RotAlignKVTranslator(cfg_none)
    tr_low_rank = RotAlignKVTranslator(cfg_low_rank)

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
    tr_low_rank.fit_from_pairs(src_kvs, tgt_kvs)

    K_none, _ = tr_none.translate_layer(base, base + 0.5, tgt_layer_idx=0, quantize=True)
    K_low_rank, _ = tr_low_rank.translate_layer(base, base + 0.5, tgt_layer_idx=0, quantize=True)
    target = base * 1.5

    err_none = (K_none - target).pow(2).mean()
    err_low_rank = (K_low_rank - target).pow(2).mean()

    assert err_low_rank < err_none
    assert not torch.allclose(tr_low_rank.quant_proj_K[0], torch.eye(tr_low_rank.d_t))


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
