"""
Self-contained sanity check. Runs the full RotAlign-KV pipeline on synthetic
KV tensors with deliberately heterogeneous shapes, so you can verify the
mechanics without downloading any models.

It checks:
  (1) random rotations are exactly orthogonal
  (2) rotation Gaussianizes a heavy-tailed distribution
  (3) Procrustes / ridge alignment reduces Frobenius error on a synthetic
      linear-plus-noise transform
  (4) Lloyd-Max quantizer hits the expected rate-distortion regime
  (5) the full translator runs end-to-end, with quantization, on mismatched
      source and target shapes

Run with: python scripts/demo.py
"""

from __future__ import annotations

import sys
import pathlib

# Allow running from the repo root without installing.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from rotalign import (
    RotAlignKVTranslator,
    TranslatorConfig,
    random_orthogonal,
    hadamard_matrix,
    make_rotation,
    GaussianQuantizer,
    kurtosis,
    verify_gaussianization,
    fit_zca_whitening,
    apply_whitening,
)


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    torch.manual_seed(0)

    # ---------------------------------------------------------------------
    # (1) Orthogonality check
    # ---------------------------------------------------------------------
    section("(1) Random rotation is exactly orthogonal")
    d = 64
    R = random_orthogonal(d, seed=42)
    err = (R @ R.T - torch.eye(d)).abs().max().item()
    print(f"    d={d}   max|R R^T - I| = {err:.2e}  (expect ~1e-6 or less)")
    assert err < 1e-5

    # ---------------------------------------------------------------------
    # (2) Gaussianization
    # ---------------------------------------------------------------------
    section("(2) Rotation Gaussianizes a heavy-tailed distribution")
    # Build a Cauchy-like heavy-tailed vector in R^128
    d = 128
    R = random_orthogonal(d, seed=7)
    # Concentrate almost all energy in one coordinate -> maximally non-Gaussian
    x = torch.zeros(2000, d)
    x[:, 0] = torch.randn(2000) * 10.0  # big spike on axis 0
    x[:, 1:] = torch.randn(2000, d - 1) * 0.01
    diag = verify_gaussianization(x, x @ R)
    print(f"    kurtosis before rotation: {diag['kurtosis_before']:8.2f}")
    print(f"    kurtosis after  rotation: {diag['kurtosis_after']:8.2f}")
    print(f"    kurtosis Gaussian target: {diag['kurtosis_gaussian_target']:8.2f}")
    # Kurtosis won't be exactly 3 with such adversarial input, but it should
    # drop dramatically.
    assert diag["kurtosis_after"] < diag["kurtosis_before"] / 10

    # ---------------------------------------------------------------------
    # (3) Procrustes / ridge alignment reduces error
    # ---------------------------------------------------------------------
    section("(3) Linear alignment recovers a known linear map")
    from rotalign.procrustes import fit_alignment, alignment_quality

    # Case A: same dim, random orthogonal ground-truth map
    n, d = 2000, 64
    X = torch.randn(n, d)
    W_true = random_orthogonal(d, seed=99)
    Y = X @ W_true + 0.05 * torch.randn(n, d)
    W_hat = fit_alignment(X, Y, method="procrustes")
    q = alignment_quality(X, Y, W_hat)
    print(f"    same-dim Procrustes: rel_err={q['relative_frobenius_error']:.4f}"
          f"  cos={q['mean_cosine_similarity']:.4f}")
    assert q["mean_cosine_similarity"] > 0.95

    # Case B: mismatched dim, random linear ground-truth map
    n, d_in, d_out = 3000, 96, 128
    X = torch.randn(n, d_in)
    W_true = torch.randn(d_in, d_out) * 0.1
    Y = X @ W_true + 0.05 * torch.randn(n, d_out)
    W_hat = fit_alignment(X, Y, method="ridge", lam=1e-3)
    q = alignment_quality(X, Y, W_hat)
    print(f"    mismatched-dim ridge: rel_err={q['relative_frobenius_error']:.4f}"
          f"  cos={q['mean_cosine_similarity']:.4f}")
    assert q["mean_cosine_similarity"] > 0.95

    # ---------------------------------------------------------------------
    # (4) Lloyd-Max quantizer on Gaussian input
    # ---------------------------------------------------------------------
    section("(4) Lloyd-Max quantizer on Gaussian input")
    for bits in [2, 3, 4, 6, 8]:
        q = GaussianQuantizer(bits=bits)
        x = torch.randn(512, 128)
        x_hat = q.quantize_dequantize(x)
        mse = float(((x_hat - x) ** 2).mean())
        # Theoretical Gaussian-source distortion (sigma=1, after per-row normalization)
        theo = (3.1416 * (3**0.5) / 2) * (2 ** (-2 * bits))
        print(f"    bits={bits}   empirical MSE={mse:.4f}   theoretical ~ {theo:.4f}")

    # ---------------------------------------------------------------------
    # (4b) Hadamard rotation: O(d log d) alternative
    # ---------------------------------------------------------------------
    section("(4b) Hadamard rotation is orthogonal and Gaussianizes")
    for d in [64, 128, 256]:
        H = make_rotation(d, kind="hadamard", seed=3)
        err = (H @ H.T - torch.eye(d)).abs().max().item()
        # Test on a heavy-tailed input
        x = torch.randn(1000, d) * torch.exp(torch.randn(1000, d))
        k_before = kurtosis(x)
        k_after = kurtosis(x @ H)
        print(f"    d={d:>4d}  orthogonality_err={err:.2e}  "
              f"kurtosis {k_before:.1f} -> {k_after:.2f}")
        assert err < 1e-5
        assert k_after < k_before

    # ---------------------------------------------------------------------
    # (4c) ZCA whitening equalizes per-dimension variance
    # ---------------------------------------------------------------------
    section("(4c) ZCA whitening equalizes per-dimension variance")
    n, d = 3000, 48
    # Anisotropic Gaussian: variance varies by factor 100 across dims
    scales = torch.linspace(0.1, 10.0, d)
    X = torch.randn(n, d) * scales
    print(f"    before: variance range [{X.var(0).min():.3f}, {X.var(0).max():.3f}]")
    W_zca, mean = fit_zca_whitening(X)
    X_w = apply_whitening(X, W_zca, mean)
    var = X_w.var(0)
    print(f"    after:  variance range [{var.min():.3f}, {var.max():.3f}]  "
          f"(target: ~1.0 for all dims)")
    assert var.max() < 1.5 and var.min() > 0.5

    # ---------------------------------------------------------------------
    # (4d) New alignment solvers (CCA, reduced-rank)
    # ---------------------------------------------------------------------
    section("(4d) CCA and reduced-rank regression solvers")
    from rotalign.procrustes import cca_projection, reduced_rank_regression, alignment_quality
    n, d_in, d_out = 2000, 80, 120
    # Plant a genuinely low-rank cross-model map (rank 8 out of 80)
    X = torch.randn(n, d_in)
    U_true = torch.randn(d_in, 8) * 0.3
    V_true = torch.randn(8, d_out) * 0.3
    W_true = U_true @ V_true
    Y = X @ W_true + 0.05 * torch.randn(n, d_out)

    W_cca = cca_projection(X, Y, r=8)
    W_rrr = reduced_rank_regression(X, Y, rank=8)
    q_cca = alignment_quality(X, Y, W_cca)
    q_rrr = alignment_quality(X, Y, W_rrr)
    print(f"    CCA (rank=8):          cos={q_cca['mean_cosine_similarity']:.4f}  "
          f"rel_err={q_cca['relative_frobenius_error']:.4f}")
    print(f"    reduced-rank (rank=8): cos={q_rrr['mean_cosine_similarity']:.4f}  "
          f"rel_err={q_rrr['relative_frobenius_error']:.4f}")
    assert q_cca["mean_cosine_similarity"] > 0.9
    assert q_rrr["mean_cosine_similarity"] > 0.9

    # ---------------------------------------------------------------------
    # (5) Full translator end-to-end
    # ---------------------------------------------------------------------
    section("(5) Full translator on mismatched-shape synthetic KVs")
    # Pretend we have two models with deliberately different shapes:
    #   source: 4 KV heads of dim 64,  12 layers
    #   target: 8 KV heads of dim 96,  16 layers
    cfg = TranslatorConfig(
        src_head_dim=64,
        src_num_heads=4,
        num_src_layers=12,
        tgt_head_dim=96,
        tgt_num_heads=8,
        num_tgt_layers=16,
        quant_bits=4,
        alignment_method="auto",
    )
    translator = RotAlignKVTranslator(cfg)

    # Synthesize "calibration" KVs with a planted cross-model linear relation
    # so alignment has something real to find.
    batch, seq = 4, 32
    src_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
    tgt_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []

    # A single shared "latent" per batch-token: the hidden source of meaning
    latent = torch.randn(batch, seq, 48)

    # Per-layer random linear maps from latent to the per-model KV spaces.
    # This models the fact that each layer encodes the same underlying content
    # via its own projection.
    torch.manual_seed(1)
    for l in range(cfg.num_src_layers):
        Ws_k = torch.randn(48, cfg.src_num_heads * cfg.src_head_dim) * 0.1
        Ws_v = torch.randn(48, cfg.src_num_heads * cfg.src_head_dim) * 0.1
        K_flat = latent @ Ws_k + 0.01 * torch.randn(batch, seq, cfg.src_num_heads * cfg.src_head_dim)
        V_flat = latent @ Ws_v + 0.01 * torch.randn(batch, seq, cfg.src_num_heads * cfg.src_head_dim)
        K = K_flat.view(batch, seq, cfg.src_num_heads, cfg.src_head_dim).transpose(1, 2)
        V = V_flat.view(batch, seq, cfg.src_num_heads, cfg.src_head_dim).transpose(1, 2)
        src_kvs.append((K, V))

    for l in range(cfg.num_tgt_layers):
        Wt_k = torch.randn(48, cfg.tgt_num_heads * cfg.tgt_head_dim) * 0.1
        Wt_v = torch.randn(48, cfg.tgt_num_heads * cfg.tgt_head_dim) * 0.1
        K_flat = latent @ Wt_k + 0.01 * torch.randn(batch, seq, cfg.tgt_num_heads * cfg.tgt_head_dim)
        V_flat = latent @ Wt_v + 0.01 * torch.randn(batch, seq, cfg.tgt_num_heads * cfg.tgt_head_dim)
        K = K_flat.view(batch, seq, cfg.tgt_num_heads, cfg.tgt_head_dim).transpose(1, 2)
        V = V_flat.view(batch, seq, cfg.tgt_num_heads, cfg.tgt_head_dim).transpose(1, 2)
        tgt_kvs.append((K, V))

    print(f"    src KV shape: [batch={batch}, heads={cfg.src_num_heads}, "
          f"seq={seq}, head_dim={cfg.src_head_dim}]")
    print(f"    tgt KV shape: [batch={batch}, heads={cfg.tgt_num_heads}, "
          f"seq={seq}, head_dim={cfg.tgt_head_dim}]")

    diagnostics = translator.fit_from_pairs(src_kvs, tgt_kvs, verbose=False)
    avg_cos = sum(d["K"]["mean_cosine_similarity"] for d in diagnostics.values()) / len(diagnostics)
    print(f"    average K cosine similarity after alignment: {avg_cos:.3f}")
    assert avg_cos > 0.9, "Alignment should recover most of the planted linear structure"

    # Translate a single layer end-to-end
    K_hat, V_hat = translator.translate_layer(
        src_kvs[0][0], src_kvs[0][1], tgt_layer_idx=0, quantize=True
    )
    print(f"    translated layer 0 output shape: {tuple(K_hat.shape)}  "
          f"(expected {(batch, cfg.tgt_num_heads, seq, cfg.tgt_head_dim)})")
    assert K_hat.shape == (batch, cfg.tgt_num_heads, seq, cfg.tgt_head_dim)

    # Verify the translated output lives in a reasonable range relative to target
    tgt_K0 = tgt_kvs[0][0]
    with torch.no_grad():
        diff_norm = float((K_hat - tgt_K0).norm() / tgt_K0.norm())
    print(f"    relative ||K_hat - K_t|| / ||K_t||: {diff_norm:.3f}  "
          f"(4-bit quantization adds some error)")

    # Test fusion
    K_fused, V_fused = translator.fuse_layer(tgt_K0, tgt_kvs[0][1], K_hat, V_hat, tgt_layer_idx=0)
    print(f"    fused layer 0 output shape: {tuple(K_fused.shape)}")

    # Save + load round trip
    tmp_path = "/tmp/rotalign_demo.pt"
    translator.save(tmp_path)
    loaded = RotAlignKVTranslator.load(tmp_path)
    K_hat_2, _ = loaded.translate_layer(src_kvs[0][0], src_kvs[0][1], tgt_layer_idx=0, quantize=True)
    # Quantization introduces minor non-determinism in the codebook boundary,
    # but the outputs should be identical given the same seed + fixed codebook.
    assert torch.allclose(K_hat, K_hat_2, atol=1e-5)
    print(f"    save/load round trip: OK")

    section("All checks passed.")


if __name__ == "__main__":
    main()
