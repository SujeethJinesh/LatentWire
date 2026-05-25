"""Small analysis helpers for spectral and component summaries."""

from __future__ import annotations

import numpy as np


def spectral_entropy(signal: np.ndarray) -> float:
    """Return normalized spectral entropy for a one-dimensional signal."""

    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("signal must be one-dimensional with at least two values")
    power = np.abs(np.fft.rfft(arr - arr.mean())) ** 2
    if np.allclose(power.sum(), 0.0):
        return 0.0
    probs = power / power.sum()
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return entropy / float(np.log(probs.size))


def autocorrelation_length(signal: np.ndarray, threshold: float = 0.5) -> int:
    """Return first lag where normalized autocorrelation drops below threshold."""

    arr = np.asarray(signal, dtype=np.float64)
    centered = arr - arr.mean()
    denom = float(np.dot(centered, centered))
    if denom == 0.0:
        return 0
    for lag in range(1, arr.size):
        corr = float(np.dot(centered[:-lag], centered[lag:]) / denom)
        if corr < threshold:
            return lag
    return arr.size - 1
