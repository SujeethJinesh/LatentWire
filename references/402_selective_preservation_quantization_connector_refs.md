# Selective Preservation Quantization Connector Refs (2026-04-21)

Purpose: capture compression-inspired connector ideas where a small salient
subset is preserved exactly and the rest is compressed or repaired.

## Strongest Sources

1. AWQ
   Link: https://arxiv.org/abs/2306.00978
   Why it matters: clean selective-preservation baseline for protecting a small
   set of salient channels while quantizing the rest.

2. No Token Left Behind
   Link: https://arxiv.org/abs/2402.18096
   Why it matters: strongest token-aware mixed-precision analogue for keeping
   informative bridge tokens at higher fidelity.

3. QTIP
   Link: https://arxiv.org/abs/2406.11235
   Why it matters: structured incoherence transforms plus codebook-style
   quantization suggest a rotate-then-compress connector design.

4. HIGGS
   Link: https://aclanthology.org/2025.naacl-long.543/
   Why it matters: calibration-light grids after simple orthogonal transforms
   are a useful stabilization template if bridge calibration is brittle.

5. TurboQuant
   Link: https://arxiv.org/abs/2504.19874
   Why it matters: strongest recent residual vector-codebook idea for a coarse
   connector plus a compact residual channel.

6. FastKV
   Link: https://arxiv.org/abs/2502.01068
   Why it matters: strongest token-selective propagation reference for
   cross-model communication under a budget.

7. KVzip
   Link: https://arxiv.org/abs/2505.23416
   Why it matters: compressed caches should still reconstruct reusable context,
   not just satisfy one query.

## Exact Next Ablations

1. AWQ-style connector protection on the live dynalign residual lane
   Why now: lowest-cost selective-preservation control against plain rank16
   repair.

2. TurboQuant-style residual codebook on the bridge tail
   Why now: strongest codebook follow-up if low-rank repair saturates.

3. Token-aware mixed-precision routing for bridge tokens
   Why now: strongest budgeted connector experiment if we widen to mismatched
   or long-context settings.

## Interpretable Telemetry

- preserved channel/head/token mask
- saliency scores used for selection
- compression ratio by layer and token type
- residual energy before and after compression
- selection overlap across ablations

## Current Read

- The most useful compression lesson for LatentWire is selective preservation
  plus residual correction, not blanket compression.
- These ideas are strongest as additives on a stable live lane, not as a
  replacement for alignment itself.
