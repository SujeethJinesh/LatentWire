# Residual Rank Sweep Next Refs (2026-04-21)

Purpose: pin down the exact residual-correction branch that should follow the
validated frozen GSM8K32 baseline harness. The live same-pair ceiling remains
`0.0938`, so the next real benchmark lift has to come from explicit residual
repair on top of the output-aware alignment lane, not more teacher-side
complexity.

## Highest-Signal References

1. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: preserve dominant bridge directions first, then spend the
   remaining rank budget only on the hard residual.

2. Rethinking Residual Errors in Compensation-based LLM Quantization
   Link: https://arxiv.org/abs/2604.07955
   Why it matters: fit residual correction against the true full target
   behavior, not against already-compensated intermediates.

3. ResQ
   Link: https://openreview.net/forum?id=sROOmwTHyW
   Why it matters: the closest direct precedent for adding a low-rank
   residual branch on top of a compressed or partially aligned main path.

4. R2Q
   Link: https://arxiv.org/abs/2511.21736
   Why it matters: staged residual refinement is a stronger design than one
   monolithic correction if the first rank-limited branch saturates.

5. QuaRot
   Link: https://openreview.net/forum?id=dfqsW38v1X
   Why it matters: rotate into a compression-friendly basis before applying
   residual repair; relevant if the remaining bridge error is outlier-shaped.

6. Residual Vector Quantization for KV Cache Compression in Large Language Models
   Link: https://arxiv.org/abs/2410.15704
   Why it matters: suggests a residual codebook stack as the next fallback if
   low-rank repair alone does not beat the contract ceiling.

## Concrete Repo-Side Ablations

1. `dynalign_module_replace + residual rank 16`
   Why now: highest-value direct continuation of the exact same-pair real lane.

2. `tokenbasis_replace + residual rank 16`
   Why now: required control for deciding whether the gain is truly
   output-aware or merely basis-regularized.

3. Preserve-then-quantize residual split on the better of those two
   Why now: best literature-backed next move if plain rank-16 repair only ties
   the `0.0938` ceiling.

## Current Read

- The residual sweep harness is now validated on the frozen GSM8K32 contract:
  reused rank-8 `dynalign_module_replace` and `tokenbasis_replace` both
  reproduce `0.0938` with full numeric extraction coverage and exact example
  parity.
- Do not claim a residual-repair gain yet; the expensive rank-16 recalibration
  has not finished.
- The exact next benchmark action is therefore:
  `dynalign_module_replace_residrank16` and
  `tokenbasis_replace_residrank16` on the same frozen contract.
