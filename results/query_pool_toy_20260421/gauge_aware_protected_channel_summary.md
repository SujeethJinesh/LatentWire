# Gauge-Aware Protected-Channel Toy Summary

Date: 2026-04-21

Artifact:

- `query_pool_gauge_aware_protected_channel_vs_topk.json`
- `query_pool_gauge_aware_protected_channel_vs_topk.md`

## Result

| Scenario | Residual codebook | Fixed protected residual | Gauge-aware protected residual | Gauge orthogonality error | Gauge protected energy | Gauge top margin |
|---|---:|---:|---:|---:|---:|---:|
| aligned | 0.5417 | 0.5938 | 0.5469 | 1.22e-07 | 0.101 | 0.056 |
| rotated | 0.6406 | 0.5729 | 0.5677 | 1.16e-07 | 0.102 | 0.068 |
| outlier | 0.5677 | 0.6198 | 0.5677 | 1.15e-07 | 0.327 | 2.393 |
| slot_permuted | 0.5417 | 0.4948 | 0.5260 | 1.34e-07 | 0.102 | 0.081 |

## Interpretation

Gauge-aware protected channels do not produce a new positive toy method. They
do produce a useful blocker:

- Orthogonal calibration is numerically clean, so failure is not from an
  unstable basis.
- PCA/covariance protection is not enough for rotated settings; it selects
  high-variance directions, not necessarily task-relevant communication
  directions.
- The method repairs part of the slot-permutation damage, which suggests that
  canonicalization helps only when the mismatch is mostly coordinate ordering.
- The outlier case shows a large protected-energy fraction and top margin, but
  fixed protection is still better on task accuracy, so preserving outlier
  energy is not sufficient by itself.

## Next Diagnostic

The next protected-channel variant should be signal-aware:

- supervised Procrustes basis from query-to-correct-slot pairs;
- learned orthogonal basis with an explicit orthogonality penalty;
- protected mask selected by answer-loss or reconstruction gradient, not by
  covariance alone;
- seed repeats to separate stable mechanism from lucky codebook initialization.
