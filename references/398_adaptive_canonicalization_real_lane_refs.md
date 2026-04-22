# Adaptive Canonicalization Real-Lane Refs (2026-04-21)

Purpose: bank the highest-signal references and exact experiment ideas for
adaptive canonicalization as a wrapper around the only live same-pair real lane,
`dynalign_module_replace + rank16 residual`.

## Strongest Sources

1. Complete Characterization of Gauge Symmetries in Transformer Architectures
   Link: https://openreview.net/forum?id=KrkbYbK0cH
   Why it matters: strongest transformer-specific reference for what symmetry
   group a wrapper should actually remove before bridge fitting.

2. Maximal Gauge Symmetry in Transformer Architectures
   Link: https://openreview.net/forum?id=K1df8mmncF
   Why it matters: extends the symmetry characterization to modern transformer
   variants and clarifies the legal gauge-fix space.

3. Gauge Fiber Bundle Geometry of Transformers
   Link: https://openreview.net/forum?id=27f8abe007d4895e12e7c655bb59815096d575d4
   Why it matters: strongest quotient-manifold view for separating gauge motion
   from function-changing motion.

4. Curvature Meets Bispectrum
   Link: https://openreview.net/forum?id=pcqyhDvG0i
   Why it matters: suggests invariant diagnostics that can tell whether a
   wrapper is only moving in symmetry directions.

5. Bispectral Invariants for Transformers
   Link: https://openreview.net/forum?id=QxVvKboznV
   Why it matters: strongest source for comparing post-canonicalized
   representations with invariant summaries instead of raw coordinates.

6. RECON
   Link: https://openreview.net/forum?id=bpWzTPDybh
   Why it matters: best practical template for explicit canonical orientation
   normalization when one fixed gauge is too brittle.

7. Adaptive Canonicalization with Application to Invariant Anisotropic Geometric Networks
   Link: https://arxiv.org/abs/2509.24886
   Why it matters: strongest direct support for choosing the canonical frame
   adaptively from calibration data rather than fixing it once globally.

8. Multi-Way Representation Alignment
   Link: https://arxiv.org/abs/2602.06205
   Why it matters: strongest multi-way GPA reference for shared canonical hubs
   before any sparse bridge is fit.

## Exact Next Ablations

1. `dynalign_module_replace_residrank16 + grouped_adaptive_canonical_transport`
   Why now: nearest real-lane test of whether adaptive canonicalization helps
   the surviving same-pair branch instead of collapsing it like fixed wrappers.

2. `tokenbasis_replace_residrank16 + grouped_adaptive_canonical_transport`
   Why now: matched negative control that tells us whether any lift is
   dynalign-specific or actually canonicalization-driven.

3. Adaptive canonicalization on the held-out-family toy shared-basis lane
   Why now: strongest cross-check for whether the symmetry fix helps only the
   real same-pair proxy or also the low-shot shared-basis story.

## Interpretable Telemetry

- `raw_alignment_score`
- `quotient_gap`
- `task_score`
- transport-plan entropy and head-match agreement
- canonical frame identifier or score chosen on calibration data
- numeric extraction coverage on frozen GSM8K32

## Current Read

- Pure canonicalization is unlikely to be enough by itself on the current Qwen
  same-pair contract.
- Adaptive canonicalization is still worth testing as a wrapper around the live
  residual lane.
- If adaptive canonicalization fails on the frozen contract, the next real
  branch should move to eigenspace-aware residual repair, not another fixed
  gauge sweep.
