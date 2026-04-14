"""
Scalar Lloyd-Max quantizer optimized for a standard Gaussian source.

After §2.1's rotation, aligned KV coordinates are approximately Gaussian,
so a scalar quantizer whose codebook is optimized for N(0, 1) is near-optimal.
Theoretical distortion for b bits on Gaussian input:

    D(b) ~ (pi sqrt(3) / 2) * sigma^2 * 2^(-2b)

which is within ~1.53 dB of the Shannon rate-distortion lower bound.

Usage:
    q = GaussianQuantizer(bits=4)
    idx, sigma = q.encode(x)            # compressed payload
    x_hat     = q.decode(idx, sigma)    # reconstructed tensor
"""

from __future__ import annotations

import numpy as np
import torch


def lloyd_max_gaussian(
    bits: int, n_iter: int = 100, n_samples: int = 50_000, seed: int = 0
) -> torch.Tensor:
    """Compute the Lloyd-Max codebook for N(0, 1) at the given bit rate.

    Alternates between (1) nearest-centroid assignment for a large Gaussian
    sample, and (2) centroid update to the mean of each Voronoi cell. Since
    the centroids are strictly monotone on a 1-D input, we use searchsorted
    instead of materializing an [n_samples, 2^bits] distance matrix — this
    avoids OOM for large bit rates.

    Returns: tensor of shape [2^bits] containing sorted centroids.
    """
    rng = np.random.default_rng(seed)
    samples = np.sort(rng.standard_normal(n_samples))
    k = 2**bits
    # Initialize at equally spaced quantiles (avoids dead cells)
    qs = np.linspace(0, 1, k + 2)[1:-1]
    centroids = np.quantile(samples, qs).astype(np.float64)
    for _ in range(n_iter):
        # Midpoints between adjacent centroids partition the real line into
        # Voronoi cells. Use searchsorted (O(n log k)) instead of an O(n k)
        # distance matrix.
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        assign = np.searchsorted(boundaries, samples)  # values in [0, k)
        new = np.empty_like(centroids)
        # np.bincount gives sums and counts per cell
        counts = np.bincount(assign, minlength=k)
        sums = np.bincount(assign, weights=samples, minlength=k)
        nonempty = counts > 0
        new[nonempty] = sums[nonempty] / counts[nonempty]
        new[~nonempty] = centroids[~nonempty]  # keep empty cells in place
        if np.allclose(centroids, new, atol=1e-8):
            centroids = new
            break
        centroids = new
    centroids.sort()
    return torch.from_numpy(centroids.astype(np.float32))


class GaussianQuantizer:
    """Scalar Lloyd-Max quantizer for Gaussian-distributed tensors.

    Each row is normalized by its own std deviation before quantization,
    and the std is transmitted alongside the code indices. The per-row std
    adds 32 bits of overhead per row but lets each sequence position have
    its own dynamic range.
    """

    def __init__(self, bits: int = 4, n_iter: int = 100):
        self.bits = bits
        self.codebook = lloyd_max_gaussian(bits, n_iter=n_iter)  # [2^bits]
        self._codebook_device: torch.device | None = None

    def to(
        self, device: str | torch.device, dtype: torch.dtype = torch.float32
    ) -> "GaussianQuantizer":
        """Move codebook to the device/dtype of the tensors it will quantize."""
        self.codebook = self.codebook.to(device=device, dtype=dtype)
        self._codebook_device = torch.device(device)
        return self

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode x to (indices, per-row sigma).

        Args:
            x: arbitrary tensor of shape [..., d]; last dim is the quantized axis
        Returns:
            indices: LongTensor of same shape as x (values in [0, 2^bits))
            sigma:   Tensor of shape [..., 1], the per-row std used for normalization
        """
        if self.codebook.device != x.device or self.codebook.dtype != x.dtype:
            self.to(x.device, x.dtype)
        sigma = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x_norm = x / sigma
        # Because the codebook is sorted and 1-D, nearest-centroid lookup
        # reduces to a binary search over midpoints between adjacent centroids.
        # This is O(n log k) instead of O(n k) and has O(n) memory.
        midpoints = (self.codebook[:-1] + self.codebook[1:]) / 2.0  # [2^b - 1]
        indices = torch.searchsorted(midpoints, x_norm.contiguous())
        return indices, sigma

    def decode(
        self, indices: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Inverse of encode: reconstruct from indices and per-row sigma."""
        if self.codebook.device != indices.device:
            self.codebook = self.codebook.to(device=indices.device)
        return self.codebook[indices] * sigma

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Round-trip x through the quantizer (returns reconstruction).

        Useful for straight-through training where you want the quantization
        error to show up in the forward pass but want gradients to flow as if
        it were identity.
        """
        idx, sigma = self.encode(x)
        return self.decode(idx, sigma)

    def distortion(self, x: torch.Tensor) -> float:
        """Report the empirical MSE between x and its round-trip reconstruction."""
        x_hat = self.quantize_dequantize(x)
        return float(((x_hat - x) ** 2).mean())
