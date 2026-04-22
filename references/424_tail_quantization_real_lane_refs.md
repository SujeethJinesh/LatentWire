# Tail Quantization Real-Lane Refs (2026-04-22)

Purpose: keep the preserve-anchor/codebook-tail backup branch grounded in the
 strongest recent compression and residual-repair papers rather than generic
 quantization folklore.

## Strongest Sources

1. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: closest match to the preserve-core, compress-tail idea for
   the live residual lane.

2. ResQ
   Link: https://openreview.net/forum?id=sROOmwTHyW
   Why it matters: strong reference for keeping a protected subspace in higher
   precision while quantizing the remainder.

3. GuidedQuant
   Link: https://arxiv.org/abs/2505.07004
   Why it matters: best current source for importance-aware ranking of what to
   preserve before compressing the tail.

4. GPTAQ
   Link: https://arxiv.org/abs/2504.02692
   Why it matters: useful if tail repair should match exact layer outputs
   instead of only local matrix error.

5. SERQ
   Link: https://openreview.net/forum?id=nFjj8NEBqv
   Why it matters: strongest recent saliency-aware reconstruction pointer for
   a preserved-anchor plus compensated-tail design.

6. AQLM
   Link: https://arxiv.org/abs/2401.06118
   Why it matters: best codebook-tail precedent for extremely compact additive
   quantization.

7. AWQ
   Link: https://arxiv.org/abs/2306.00978
   Why it matters: still the cleanest activation-sensitivity baseline for what
   to protect exactly.

## Exact Next Ablations

1. fixed-anchor + scalar tail, no repair
2. fixed-anchor + mixed-bit tail
3. fixed-anchor + additive codebook tail + rank-4 tail repair

## Interpretable Telemetry

- preserve rank
- saliency mass preserved
- average tail bit budget
- tail reconstruction MSE
- codebook usage/perplexity
- residual rank and bytes
- per-example win/loss/tie versus target

## Current Read

- The preserve-core family is still a backup, not the live same-pair lane.
- If routed/value-bank repair fails to move the GSM8K32 contract above
  `0.1250`, this is the cleanest next non-dense family to test.
