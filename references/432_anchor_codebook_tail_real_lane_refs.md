# Anchor Codebook Tail Real-Lane Refs (2026-04-22)

Purpose: keep the backup branch centered on preserving the live routed core
 while only compressing or repairing the residual tail.

## Strongest Sources

1. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: the clearest preserve-core plus repair-tail template.

2. ResQ
   Link: https://arxiv.org/abs/2412.14363
   Why it matters: additive error correction rather than blind compression.

3. SERQ
   Link: https://arxiv.org/abs/2603.08185
   Why it matters: selective reconstruction over a residual code path.

4. AWQ
   Link: https://arxiv.org/abs/2306.00978
   Why it matters: importance-aware preservation when the full space is too costly.

5. AQLM
   Link: https://arxiv.org/abs/2401.06118
   Why it matters: compact codebook design for high-value residual channels.

6. TurboQuant
   Link: https://arxiv.org/abs/2504.19874
   Why it matters: good inspiration for mixing preservation with aggressive compression.

## Exact Next Ablations

1. anchor-preserving codebook tail on top of `dynalign_module_replace_residrank16`
2. preserve-topk plus quantized-tail with learned importance proxy
3. codebook-tail only on `V`, keeping `K` on the live dynalign path

## Minimal Telemetry

- anchor coverage
- codebook usage histogram
- tail-only bytes
- exact accuracy and coverage
- win/tie/loss vs target

## Current Read

- If verifier-gated sidecars do not beat the frozen contract, this is the
  cleanest non-routing backup that still matches the paper story.
