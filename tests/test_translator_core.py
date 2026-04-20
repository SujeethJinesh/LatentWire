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
