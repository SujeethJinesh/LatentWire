# Adaptive Canonicalization / Gauge-Invariant Matching (2026-04-21)

Purpose: capture the most relevant fresh references for the next symmetry-side
branch after the frozen GSM8K32 checkpoint sweep. The current read is that the
best real-model proxies still come from stronger output-aware alignment, so the
next structural upgrade should test whether per-example orbit choice and
gauge-invariant matching help the shared-basis lane survive on real contracts.

## Highest-Signal References

1. RECON
   Link: https://openreview.net/forum?id=bpWzTPDybh
   Why it matters: introduces plug-in test-time canonicalization instead of one
   fixed global representative, which is the cleanest challenge to our current
   global GPA-style symmetry handling.

2. Adaptive Canonicalization
   Link: https://arxiv.org/abs/2509.24886
   Why it matters: makes the canonical representative input-dependent, which is
   the exact lateral move missing from our current quotient-aware but still
   mostly global alignment story.

3. Probabilistic Symmetry Breaking
   Link: https://arxiv.org/abs/2503.21985
   Why it matters: suggests that a stochastic or soft orbit choice can be more
   stable than a single hard canonicalizer, which may matter when our route
   choice and canonical choice interact.

4. Bispectral Invariants for Transformers
   Link: https://openreview.net/forum?id=QxVvKboznV
   Why it matters: provides higher-order symmetry invariants that go beyond the
   pairwise geometry of Procrustes-style matching.

5. Curvature Meets Bispectrum
   Link: https://openreview.net/forum?id=pcqyhDvG0i
   Why it matters: sharpens the gauge-invariant matching story with a more
   informative invariant score than the current low-order alignment geometry.

6. GaugeKV
   Link: https://openreview.net/forum?id=rSxYPLzyBu
   Why it matters: directly connects gauge-compatible structure to cache-space
   manipulation, which is closer to our communication setting than abstract
   representation similarity alone.

## Non-Redundant Ablations

1. Per-example adaptive canonicalization + quotient-aware matching
   Why now: tests whether the remaining real-model gap is partly due to using
   one global orbit representative rather than choosing a representative per
   prompt or per aligned example.

2. Gauge-invariant head / subspace matching score after canonicalization
   Why now: lets us test whether the current head-matching logic is leaving
   recoverable accuracy on the table even before adding any new interface
   component.

## Recommended Next Experiment

Run `adaptive_canonicalization + gauge_invariant_matching` first on the
held-out-family toy, then port the smallest viable version on top of the
current best real same-pair proxy (`dynalign_module_replace`) for the frozen
GSM8K32 contract.
