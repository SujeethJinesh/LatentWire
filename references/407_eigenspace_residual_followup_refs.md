# Eigenspace Residual Follow-Up Refs (2026-04-21)

Purpose: capture the strongest exact references and next experiments for the
eigenspace-aware residual branch on top of the live `dynalign` lane.

## Strongest Sources

1. EoRA
   Link: https://arxiv.org/abs/2410.21271
   Why it matters: clearest template for repairing error in the activation
   eigenspace instead of the raw coordinate basis.

2. ResQ
   Link: https://arxiv.org/abs/2412.14363
   Why it matters: preserve a dominant activation subspace and repair the rest
   with a residual branch.

3. Low-Rank Correction for Quantized LLMs
   Link: https://openreview.net/forum?id=FA3iYp1y6z
   Why it matters: direct low-rank error-correction reference for frozen
   baselines.

4. PiSSA
   Link: https://arxiv.org/abs/2404.02948
   Why it matters: principal singular directions are a stronger initialization
   than raw random low-rank factors.

5. QuaRot
   Link: https://openreview.net/forum?id=dfqsW38v1X
   Why it matters: supports the read that geometry and rotation matter before
   low-rank correction is applied.

6. ESPACE
   Link: https://openreview.net/forum?id=HAcaANQNMK
   Why it matters: activation-centric dimensionality reduction is a natural
   bridge to eigenspace-constrained repair.

7. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: best recent preserve-core / repair-tail reference if the
   eigenspace branch needs a second split after the first pass.

8. FLAT-LLM
   Link: https://arxiv.org/abs/2505.23966
   Why it matters: per-head or per-layer PCA rank budgeting is the strongest
   follow-on if one global residual rank is too crude.

9. SERQ
   Link: https://openreview.net/forum?id=nFjj8NEBqv
   Why it matters: strongest saliency-aware residual reconstruction reference
   if the pure eigenspace branch only ties the current live row.

## Exact Next Ablations

1. `dynalign_eigenspace_module_replace_residrank16`
   Why now: nearest test of whether constraining the live residual to the
   dominant target eigenspace keeps or improves the current `0.1250` row.

2. `dynalign_module_replace + saliency-gated residual`
   Why now: strongest next follow-up if the plain eigenspace row only ties.

3. `tokenbasis_replace + eigenspace residual`
   Why now: matched control for whether the lift is geometry-specific or
   truly basis-specific.

## Interpretable Telemetry

- eigenspace rank
- explained variance kept
- projected-target norm / full-target norm
- repaired MSE vs raw bridge MSE
- exact match and numeric extraction coverage
- calibration time and latency overhead

## Current Read

- The current live same-pair row is still `dynalign_module_replace_residrank16
  = 0.1250`.
- Adaptive canonicalization is safe but non-additive, and raw-basis
  preserve-core splitting is negative.
- The first dominant-eigenspace follow-up is also negative:
  `dynalign_eigenspace_module_replace_residrank16 = 0.0312`, full numeric
  extraction coverage, `0/32` wins, `1/32` loss vs target.
- So the next exact residual-side question is no longer “does naive
  eigenspace projection help”; it is whether saliency-aware or
  learned-importance residual repair can preserve the live dynalign lift.
