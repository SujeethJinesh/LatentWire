# Gauge Wrapper Real-Lane Refs (2026-04-21)

Purpose: narrow the symmetry-side branch to the pieces that can actually sit on
top of the current real same-pair benchmark lane (`dynalign_module_replace` /
`tokenbasis_replace`) instead of replacing it.

## Highest-Signal References

1. Complete Characterization of Gauge Symmetries in Transformer Architectures
   Link: https://openreview.net/forum?id=KrkbYbK0cH
   Why it matters: gives a principled head-wise gauge fix and permutation-aware
   canonical form before fitting any bridge.

2. Maximal Gauge Symmetry in Transformer Architectures
   Link: https://openreview.net/forum?id=K1df8mmncF
   Why it matters: justifies why Qwen-style models still admit nontrivial
   quotient structure even after naive architectural matching.

3. Gauge Fiber Bundle Geometry of Transformers
   Link: https://openreview.net/forum?id=sPCLRX1yOY
   Why it matters: strongest argument for treating canonicalization as a choice
   of representative on the quotient, not a cosmetic preprocessing trick.

4. RECON
   Link: https://openreview.net/forum?id=bpWzTPDybh
   Why it matters: the best current inspiration for adaptive, data-dependent
   canonicalization instead of one fixed global gauge.

5. A Canonicalization Perspective on Invariant and Equivariant Learning
   Link: https://arxiv.org/abs/2405.18378
   Why it matters: frames canonicalization as an ablation family with a clean
   stability/optimality argument rather than a one-off heuristic.

6. GaugeKV
   Link: https://openreview.net/forum?id=rSxYPLzyBu
   Why it matters: strongest direct example of “canonicalize first, then
   compress/manipulate K/V space.”

## Concrete Repo-Side Ablations

1. Gauge-fix wrapper before `dynalign_module_replace`
   Why now: smallest real-lane test of whether unresolved head/frame mismatch
   is still capping the `0.0938` row.

2. Gauge-fix wrapper before `tokenbasis_replace`
   Why now: matched control for whether the benefit is specific to dynalign or
   generic to the token-grounded basis lane.

3. Adaptive canonicalization selection on the calibration slice
   Why now: strongest fallback if a single static gauge wrapper does not move
   the same-pair contract.

## Current Read

- Gauge/canonicalization should be treated as a wrapper or initializer on top
  of the real output-aware alignment lane, not as a replacement for it.
- The residual branch still has priority because it is the most direct route to
  moving the real same-pair ceiling.
- If the rank-16 residual branch fails to beat `0.0938`, the next live
  benchmark move should be `dynalign/tokenbasis + gauge wrapper`, not another
  dynalign-teacher variant.
