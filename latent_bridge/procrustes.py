"""
Closed-form linear alignment between two sets of rotated KV coordinates.

Five solver options:

    identity         - rectangular identity map / no alignment ablation
    procrustes       - orthogonal Procrustes (Case: d_s = d_t, near-identity scaling)
    ridge            - Tikhonov-regularized least squares (general case)
    cca              - canonical correlation analysis (maximally correlated subspaces)
    reduced_rank     - rank-r constrained regression (when true map is low-rank)
    procrustes_rand  - randomized SVD Procrustes (scales to 70B-class models)

All five are closed-form SVD / eigendecomp solves — no gradient descent.
"""

from __future__ import annotations

import torch


def identity_projection(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Return the closest thing to an identity map under dimension mismatch.

    This is an ablation baseline, not a fitted solver. If the source and target
    dimensions differ, keep the leading diagonal and zero the rest.
    """
    d_in, d_out = X.shape[1], Y.shape[1]
    W = torch.zeros(d_in, d_out, dtype=X.dtype, device=X.device)
    diag = min(d_in, d_out)
    W[:diag, :diag] = torch.eye(diag, dtype=X.dtype, device=X.device)
    return W


def orthogonal_procrustes(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Solve W* = argmin ||X W - Y||_F  subject to W^T W = I.

    Closed form:  W* = U V^T  where U S V^T = SVD(X^T Y).

    Args:
        X: [n, d] source-side samples (rotated & flattened across heads)
        Y: [n, d] target-side samples (rotated & flattened across heads)
    Returns:
        W: [d, d] orthogonal matrix mapping X -> Y
    """
    assert X.shape[1] == Y.shape[1], (
        "Procrustes requires matching dimensions; "
        f"got X: {tuple(X.shape)}, Y: {tuple(Y.shape)}. "
        "Use ridge_projection for mismatched dims."
    )
    # Work in float64 for numerical stability, then cast back.
    Xd = X.double()
    Yd = Y.double()
    M = Xd.T @ Yd  # [d, d]
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vt
    return W.to(dtype=X.dtype)


def ridge_projection(
    X: torch.Tensor, Y: torch.Tensor, lam: float = 1e-3
) -> torch.Tensor:
    """Solve W* = argmin ||X W - Y||_F^2 + lam ||W||_F^2.

    Closed form:  W* = (X^T X + lam I)^{-1} X^T Y.

    Works for mismatched dimensions (d_in != d_out).

    Args:
        X:   [n, d_in]
        Y:   [n, d_out]
        lam: ridge regularization (small positive number)
    Returns:
        W: [d_in, d_out]
    """
    Xd = X.double()
    Yd = Y.double()
    d_in = Xd.shape[1]
    A = Xd.T @ Xd + lam * torch.eye(d_in, dtype=torch.float64, device=Xd.device)
    B = Xd.T @ Yd
    W = torch.linalg.solve(A, B)
    return W.to(dtype=X.dtype)


def cca_projection(
    X: torch.Tensor, Y: torch.Tensor, r: int | None = None, eps: float = 1e-4
) -> torch.Tensor:
    """Canonical Correlation Analysis: find the linear map from X's space to
    Y's space via their maximally correlated subspaces.

    CCA solves for projections A, B that maximize corr(X A, Y B). The resulting
    alignment matrix is  W = A @ B^+  where B^+ is the pseudoinverse. CCA is
    rotation/scale invariant, so it's a natural fit after Gaussianization when
    you don't know the right dimensionality of the cross-model map.

    Args:
        X:   [n, d_in]
        Y:   [n, d_out]
        r:   rank of the canonical subspace (default min(d_in, d_out))
        eps: regularizer added to covariances for numerical stability
    Returns:
        W: [d_in, d_out]
    """
    Xd = X.double()
    Yd = Y.double()
    n = Xd.shape[0]
    d_in, d_out = Xd.shape[1], Yd.shape[1]
    if r is None:
        r = min(d_in, d_out)

    Xc = Xd - Xd.mean(0, keepdim=True)
    Yc = Yd - Yd.mean(0, keepdim=True)

    Cxx = (Xc.T @ Xc) / (n - 1) + eps * torch.eye(d_in, dtype=torch.float64, device=Xd.device)
    Cyy = (Yc.T @ Yc) / (n - 1) + eps * torch.eye(d_out, dtype=torch.float64, device=Xd.device)
    Cxy = (Xc.T @ Yc) / (n - 1)

    # Symmetric inverse square roots via eigendecomposition
    def inv_sqrt(M: torch.Tensor) -> torch.Tensor:
        vals, vecs = torch.linalg.eigh(M)
        return vecs @ torch.diag(1.0 / torch.sqrt(vals.clamp_min(eps))) @ vecs.T

    Cxx_inv_half = inv_sqrt(Cxx)
    Cyy_inv_half = inv_sqrt(Cyy)
    T = Cxx_inv_half @ Cxy @ Cyy_inv_half
    U, S, Vt = torch.linalg.svd(T, full_matrices=False)
    # Keep top-r canonical directions
    U = U[:, :r]
    Vt = Vt[:r, :]
    A = Cxx_inv_half @ U  # [d_in, r]
    B = Cyy_inv_half @ Vt.T  # [d_out, r]
    # Compose into a direct X -> Y map
    # Y_hat = X @ A @ diag(S) @ B^T
    W = A @ torch.diag(S[:r]) @ B.T  # [d_in, d_out]
    return W.to(dtype=X.dtype)


def reduced_rank_regression(
    X: torch.Tensor, Y: torch.Tensor, rank: int, lam: float = 1e-3
) -> torch.Tensor:
    """Ridge regression constrained to rank-r solutions.

    Useful when you suspect the true cross-model map is low-rank (i.e., the
    "shared semantic content" lives in a small subspace). Yields a factorized
    W = U V with U in R^{d_in x r}, V in R^{r x d_out}, which is both a
    regularizer and a compression (2 d r params instead of d^2).

    Closed form:
        W_ridge = (X^T X + lam I)^{-1} X^T Y
        Y_hat = X W_ridge
        SVD of Y_hat = U S V^T;  take top r components
        W_rrr = W_ridge @ V[:, :r] @ V[:, :r]^T
    """
    Xd = X.double()
    Yd = Y.double()
    d_in = Xd.shape[1]
    A = Xd.T @ Xd + lam * torch.eye(d_in, dtype=torch.float64, device=Xd.device)
    W_ridge = torch.linalg.solve(A, Xd.T @ Yd)
    Y_hat = Xd @ W_ridge
    U, S, Vt = torch.linalg.svd(Y_hat, full_matrices=False)
    V_r = Vt[:rank].T  # [d_out, r]
    W_rrr = W_ridge @ V_r @ V_r.T  # [d_in, d_out]
    return W_rrr.to(dtype=X.dtype)


def orthogonal_procrustes_randomized(
    X: torch.Tensor, Y: torch.Tensor, oversample: int = 10, n_iter: int = 2
) -> torch.Tensor:
    """Orthogonal Procrustes via randomized SVD. Scales to very large d.

    The standard Procrustes computes SVD of X^T Y at cost O(d^3). For 70B-class
    models with d_head up to 256 flattened across 64 heads -> d=16384, this is
    the bottleneck. Randomized SVD (Halko et al. 2011) computes the same SVD
    in O(d^2 * (r + oversample)) by projecting onto a small random subspace
    and refining with power iteration.

    Since orthogonal Procrustes uses the *full* left/right singular vectors,
    we set r = d and treat `oversample` as the slack.
    """
    Xd = X.double()
    Yd = Y.double()
    M = Xd.T @ Yd  # [d, d]
    d = M.shape[0]
    r = d + oversample
    # Randomized range finder
    gen = torch.Generator(device="cpu").manual_seed(0)
    Omega = torch.randn(d, r, generator=gen, dtype=torch.float64)
    Z = M @ Omega
    for _ in range(n_iter):
        Z = M @ (M.T @ Z)
    Q, _ = torch.linalg.qr(Z)
    B = Q.T @ M  # [r, d]
    Uh, _, Vt = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Uh  # [d, r]
    U = U[:, :d]
    W = U @ Vt[:d]
    return W.to(dtype=X.dtype)


def fit_alignment(
    X: torch.Tensor,
    Y: torch.Tensor,
    method: str = "auto",
    lam: float = 1e-3,
    rank: int | None = None,
) -> torch.Tensor:
    """Dispatch to the chosen alignment solver.

    method:
        'auto'            - Procrustes if dims match, else ridge
        'identity'        - rectangular identity map, no learned alignment
        'procrustes'      - orthogonal Procrustes (requires d_in = d_out)
        'procrustes_rand' - randomized-SVD Procrustes (same, but scales to large d)
        'ridge'           - Tikhonov-regularized least squares
        'cca'             - canonical correlation analysis
        'reduced_rank'    - rank-r regression (requires `rank`)
    """
    if method == "auto":
        method = "procrustes" if X.shape[1] == Y.shape[1] else "ridge"
    if method == "identity":
        return identity_projection(X, Y)
    if method == "procrustes":
        return orthogonal_procrustes(X, Y)
    if method == "procrustes_rand":
        return orthogonal_procrustes_randomized(X, Y)
    if method == "ridge":
        return ridge_projection(X, Y, lam=lam)
    if method == "cca":
        return cca_projection(X, Y, r=rank)
    if method == "reduced_rank":
        if rank is None:
            raise ValueError("reduced_rank requires `rank` to be specified")
        return reduced_rank_regression(X, Y, rank=rank, lam=lam)
    raise ValueError(f"Unknown alignment method: {method}")


def alignment_quality(
    X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor
) -> dict[str, float]:
    """Diagnostic: report relative Frobenius error and mean cosine similarity
    between the projected source (X W) and the target Y.
    """
    Y_hat = X @ W
    err = (Y_hat - Y).norm()
    rel_err = float(err / (Y.norm() + 1e-12))
    # Cosine per row, then mean
    yhn = torch.nn.functional.normalize(Y_hat.float(), dim=-1)
    yn = torch.nn.functional.normalize(Y.float(), dim=-1)
    cos = (yhn * yn).sum(dim=-1).mean()
    return {
        "relative_frobenius_error": rel_err,
        "mean_cosine_similarity": float(cos),
        "n_samples": int(X.shape[0]),
    }
