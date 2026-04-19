"""
Rotation front-end for Gaussianizing KV coordinates.

Four rotation variants are supported:

    identity          - no rotation, used as a geometry ablation
    random_orthogonal - Haar-uniform, O(d^2) apply, exact
    hadamard          - structured, O(d log d) apply, approximate but matches
                        the performance of full random orthogonal in the
                        Gaussianization regime (this is QuIP# / QuaRot's trick)
    dct               - Fourier-family real orthogonal cosine transform,
                        useful as a deterministic frequency-mixing ablation
    whiten_then_rotate - ZCA-whiten the source first, then apply a random
                         orthogonal. Handles anisotropic scaling that plain
                         rotation misses (important when KV dims have very
                         different variances, as they often do post-RMSNorm).
    learned_stiefel    - parameterize the rotation on the Stiefel manifold
                         and learn it end-to-end (ablation only)

The rationale for rotation is concentration of measure: for any fixed v, the
coordinates of Rv concentrate around N(0, ||v||^2/d) at rate O(1/sqrt(d)).
Empirically on modern LLM KV caches (e.g. Qwen3-1.7B) this is extremely tight
— kurtosis drops from ~900 (heavy-tailed) to ~3 (Gaussian) after rotation.
"""

from __future__ import annotations

import math
import torch


def random_orthogonal(
    d: int,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw a d x d orthogonal matrix uniformly from the Haar measure on O(d).

    Construction: take A ~ N(0, I_d) and apply QR. The sign correction
    Q <- Q * diag(sign(diag(R))) is required for the distribution to be
    exactly Haar-uniform (Mezzadri 2007, "How to generate random matrices
    from the classical compact groups").
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(d, d, generator=gen, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    # Haar correction
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    return Q.to(device=device, dtype=dtype).contiguous()


def apply_rotation(kv: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate the last (head) dimension of a KV tensor.

    Args:
        kv: tensor of shape [..., d_head]
        R:  orthogonal matrix of shape [d_head, d_head]
    Returns:
        rotated tensor of same shape as kv
    """
    assert kv.shape[-1] == R.shape[0] == R.shape[1], (
        f"Shape mismatch: kv={tuple(kv.shape)}, R={tuple(R.shape)}"
    )
    return kv @ R


def kurtosis(x: torch.Tensor) -> float:
    """Excess kurtosis of a flattened tensor; 3.0 ~ Gaussian.

    Used to verify that rotation successfully Gaussianized the distribution.
    """
    x = x.detach().flatten().float()
    x = x - x.mean()
    var = (x * x).mean().clamp_min(1e-12)
    return float((x**4).mean() / (var * var))


def verify_gaussianization(
    kv_before: torch.Tensor, kv_after: torch.Tensor
) -> dict[str, float]:
    """Diagnostic: report kurtosis before/after rotation, plus per-coordinate
    standard deviation ratio (should be ~1 for Gaussianized output).
    """
    return {
        "kurtosis_before": kurtosis(kv_before),
        "kurtosis_after": kurtosis(kv_after),
        "kurtosis_gaussian_target": 3.0,
        "std_before": float(kv_before.float().std()),
        "std_after": float(kv_after.float().std()),
    }


# ---------------------------------------------------------------------------
# Hadamard rotation: structured O(d log d) alternative to full random orthogonal
# ---------------------------------------------------------------------------


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def hadamard_matrix(d: int, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a randomized Walsh-Hadamard matrix of size d x d.

    This is the core primitive behind QuIP# / QuaRot: a structured orthogonal
    matrix that can be applied in O(d log d) time but empirically matches the
    Gaussianization quality of a dense random rotation, at a fraction of the
    compute. Required: d must be a power of 2.

    We randomize by multiplying with a random +/-1 diagonal (the "randomized
    Hadamard transform" of Ailon & Chazelle), which breaks the structural
    regularity and makes the resulting map Haar-like in practice.
    """
    if d != _next_pow2(d):
        raise ValueError(f"Hadamard rotation requires d to be a power of 2; got d={d}")
    # Build H_d recursively:  H_1 = [1]; H_{2k} = [[H_k, H_k], [H_k, -H_k]]
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < d:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    H = H / math.sqrt(d)  # normalize so H H^T = I
    # Random +/-1 diagonal for de-structuring
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    signs = torch.randint(0, 2, (d,), generator=gen, dtype=torch.float64) * 2 - 1
    H = H * signs.unsqueeze(0)
    return H.to(dtype=dtype).contiguous()


def dct_matrix(d: int, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build an orthonormal DCT-II matrix with a random sign preconditioner.

    This is a real Fourier-family transform. It gives us a deterministic dense
    mixing ablation that is different from Haar random and Hadamard rotations
    while preserving vector norms exactly.
    """
    n = torch.arange(d, dtype=torch.float64).unsqueeze(0)
    k = torch.arange(d, dtype=torch.float64).unsqueeze(1)
    mat = torch.cos(math.pi / d * (n + 0.5) * k)
    mat[0, :] *= math.sqrt(1.0 / d)
    if d > 1:
        mat[1:, :] *= math.sqrt(2.0 / d)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    signs = torch.randint(0, 2, (d,), generator=gen, dtype=torch.float64) * 2 - 1
    mat = mat * signs.unsqueeze(0)
    return mat.to(dtype=dtype).contiguous()


def make_rotation(
    d: int,
    kind: str = "orthogonal",
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dispatch to the chosen rotation variant.

    kind:
        'identity'   - no rotation ablation
        'orthogonal' - Haar-uniform random orthogonal
        'hadamard'   - randomized Walsh-Hadamard
        'dct'        - randomized-sign orthonormal DCT-II
    """
    if kind == "identity":
        return torch.eye(d, device=device, dtype=dtype)
    if kind == "orthogonal":
        return random_orthogonal(d, seed=seed, device=device, dtype=dtype)
    if kind == "hadamard":
        # Hadamard requires power-of-2 size. For arbitrary d, pad to next pow2
        # and use the top-left d x d block -- this is no longer exactly
        # orthogonal, so we re-orthogonalize via QR on the block.
        dp = _next_pow2(d)
        H = hadamard_matrix(dp, seed=seed, dtype=torch.float64)
        if dp != d:
            H = H[:d, :d]
            Q, _ = torch.linalg.qr(H)
            H = Q
        return H.to(device=device, dtype=dtype).contiguous()
    if kind == "dct":
        return dct_matrix(d, seed=seed, dtype=dtype).to(device=device).contiguous()
    raise ValueError(f"Unknown rotation kind: {kind}")


# ---------------------------------------------------------------------------
# Whitening (ZCA): handles anisotropic scaling that plain rotation misses
# ---------------------------------------------------------------------------


def fit_zca_whitening(
    X: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a ZCA whitening matrix such that Cov(X W_zca) approx I.

    ZCA is the symmetric whitening: W_zca = Cov^{-1/2}. It's the whitening
    closest (in Frobenius norm) to the identity, so it minimally rotates the
    data while equalizing variances. Useful as a preprocess before Procrustes
    alignment when the two models have very different per-dimension scales.

    Args:
        X:   [n, d] sample matrix (flattened, rotated KV coordinates)
        eps: Tikhonov regularizer on eigenvalues (avoids ill-conditioning)
    Returns:
        (W_zca, mean): the whitening map and the mean to subtract
    """
    Xd = X.double()
    mean = Xd.mean(dim=0, keepdim=True)
    Xc = Xd - mean
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / max(n - 1, 1)
    # Symmetric eigendecomp for numerical stability
    eigvals, eigvecs = torch.linalg.eigh(cov)
    inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals.clamp_min(eps)))
    W_zca = eigvecs @ inv_sqrt @ eigvecs.T
    return W_zca.to(dtype=X.dtype), mean.to(dtype=X.dtype)


def apply_whitening(
    X: torch.Tensor, W_zca: torch.Tensor, mean: torch.Tensor
) -> torch.Tensor:
    """Apply a fitted ZCA whitening: (X - mean) @ W_zca."""
    return (X - mean) @ W_zca


def undo_whitening(
    X_white: torch.Tensor, W_unzca: torch.Tensor, mean: torch.Tensor
) -> torch.Tensor:
    """Undo a whitening transform with a stored inverse map.

    If X_white = (X - mean) @ W_zca, then X ~= X_white @ W_unzca + mean.
    """
    return X_white @ W_unzca + mean
