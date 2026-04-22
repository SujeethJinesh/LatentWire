# Preserve-Anchor Tail Real-Lane Refs (2026-04-22)

Purpose: keep the preserve-anchor lane alive only in forms that change the
importance estimator or the tail family after the negative saliency-preserve
same-pair row.

## Strongest Sources

1. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: explicit preserve-core versus repaired-tail allocation.

2. ResQ
   Link: https://openreview.net/forum?id=4qIP1sXcR1
   Why it matters: preserve dominant directions, repair only what remains.

3. QERA
   Link: https://openreview.net/forum?id=LB5cKhgOTu
   Why it matters: the cleanest analytical framing for low-rank tail repair.

4. GuidedQuant
   Link: https://arxiv.org/abs/2505.07004
   Why it matters: end-loss-guided importance is a stronger next proxy than
   one-shot saliency weighting.

5. SERQ
   Link: https://openreview.net/forum?id=nFjj8NEBqv
   Why it matters: shared compensation is a better next step than repeating
   dense per-layer tail repair.

6. GlowQ
   Link: https://openreview.net/forum?id=kVojSLUcvS
   Why it matters: grouped/shared repair may beat paying a full tail fix
   everywhere.

7. AQLM
   Link: https://arxiv.org/abs/2401.06118
   Why it matters: best direct codebook-tail reference if dense tails keep
   saturating.

8. TurboQuant
   Link: https://openreview.net/forum?id=tO3ASKZlok
   Why it matters: modern vector-codebook fallback for the repaired tail.

## Exact Next Ablations

1. fixed preserved budget, swap preserve-mask estimator
2. fixed preserved anchors, compare dense tail repair vs shared/group tail vs
   codebook tail

## Interpretable Telemetry

- preserved saliency mass
- preserved-rank fraction
- tail residual norm before/after repair
- codebook perplexity and dead-code rate
- bytes and latency versus plain `dynalign_module_replace_residrank16`
