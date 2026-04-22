# Saliency Residual Follow-Up Refs (2026-04-21)

Purpose: capture the strongest references and exact next ablations for the
saliency-aware residual branch on top of the live `dynalign` lane.

## Strongest Sources

1. SERQ
   Link: https://openreview.net/forum?id=nFjj8NEBqv
   Why it matters: strongest direct reference for saliency-aware low-rank error
   reconstruction rather than plain geometric projection.

2. AWQ
   Link: https://arxiv.org/abs/2306.00978
   Why it matters: protect activation-salient directions before compressing the
   rest.

3. LQER
   Link: https://proceedings.mlr.press/v235/zhang24j.html
   Why it matters: clearest low-rank residual reconstruction reference for
   quantization error that is conditioned on activation structure.

4. QERA
   Link: https://arxiv.org/abs/2410.06040
   Why it matters: analytic error-reconstruction framing for a residual bridge.

5. EoRA
   Link: https://arxiv.org/abs/2410.21271
   Why it matters: strongest eigenspace repair baseline, useful now mainly as a
   negative control because the naive eigenspace branch already regressed.

6. ResQ
   Link: https://arxiv.org/abs/2412.14363
   Why it matters: preserve a dominant activation subspace and repair the rest,
   but with mixed precision and importance awareness rather than a raw split.

7. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: strongest preserve-core / repair-tail reference if the first
   saliency-weighted branch only ties the live row.

## Exact Next Ablations

1. `dynalign_saliency_module_replace_residrank16`
   Why now: nearest learned-importance repair branch after the negative
   preserve-core and eigenspace rows.

2. `dynalign_saliency_module_replace_residrank{4,8,16}`
   Why now: isolate whether the live contract is rank-sensitive or weighting-
   sensitive.

3. `tokenbasis + saliency residual`
   Why now: matched control for whether saliency helps the weaker basis too.

4. `dynalign + saliency-preserve + residual tail`
   Why now: only if the plain saliency-weighted fit preserves the live row.

## Interpretable Telemetry

- saliency weight entropy
- top-k saliency mass
- correlation between saliency weights and absolute bridge error
- repaired MSE vs raw bridge MSE
- exact match and numeric extraction coverage
- wins/losses vs target on the frozen contract

## Current Read

- `dynalign_module_replace_residrank16 = 0.1250` is still the only live real
  same-pair row.
- `tokenbasis_replace_residrank16 = 0.0625`,
  `dynalign_preserve_module_replace_residrank16 = 0.0625`, and
  `dynalign_eigenspace_module_replace_residrank16 = 0.0312` are the current
  matched negative controls around that row.
- The first saliency-weighted branch is also negative:
  `dynalign_saliency_module_replace_residrank16 = 0.0312`, full numeric
  extraction coverage, `0/32` wins, `1/32` loss vs target.
- The next serious residual-side question is no longer “does one-shot
  importance weighting help”; it is whether learned-importance preserve-plus-
  tail repair, routed residual repair, or codebook-style repair can preserve
  the live dynalign lift.
