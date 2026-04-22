# Importance Preserve-Tail Follow-Up Refs (2026-04-21)

Purpose: keep the preserve-plus-tail lane alive only in the forms that actually
change the importance proxy or the tail family after the negative saliency and
saliency-preserve rows.

## Strongest Sources

1. AWQ
   Link: https://arxiv.org/abs/2306.00978
   Why it matters: activation-aware selective preservation is still the cleanest
   baseline importance proxy.

2. LQER
   Link: https://proceedings.mlr.press/v235/zhang24j.html
   Why it matters: low-rank error repair gives a closed-form residual baseline
   for preserved-core versus tail tradeoffs.

3. QERA
   Link: https://openreview.net/forum?id=LB5cKhgOTu
   Why it matters: connects quantization error directly to low-rank repair
   allocation decisions.

4. GuidedQuant
   Link: https://arxiv.org/abs/2505.07004
   Why it matters: end-loss-guided importance is a stronger next proxy than the
   current one-shot saliency weighting.

5. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: explicit preserve-core / quantize-tail allocation is the
   right direct precedent for the current negative branch.

6. ResQ
   Link: https://arxiv.org/abs/2412.14363
   Why it matters: preserve a dominant subspace, quantize the tail, and repair
   only the residual that remains.

7. SERQ
   Link: https://openreview.net/forum?id=nFjj8NEBqv
   Why it matters: saliency-aware single compensation matrix is a useful shared
   alternative to per-layer dense repair.

8. GlowQ
   Link: https://openreview.net/forum?id=kVojSLUcvS
   Why it matters: shared or grouped repair may work better than paying a full
   dense tail fix everywhere.

## Exact Next Ablations

1. swap the preserve-mask estimator while keeping the same preserved budget
2. keep the preserved anchors fixed, then compare dense tail repair vs shared
   group repair vs codebook tail repair

## Current Read

- The negative saliency-preserve row does not kill the preserve-plus-tail lane.
- It only rules out the current importance proxy plus the current dense tail
  family on the frozen GSM8K32 contract.
