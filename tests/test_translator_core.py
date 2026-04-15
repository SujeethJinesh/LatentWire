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


def _make_identity_translator(monkeypatch, *, src_layers: int = 1, tgt_layers: int = 1) -> RotAlignKVTranslator:
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
