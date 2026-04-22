# Selective Preservation Quantization Follow-Up Refs (2026-04-21)

Purpose: capture the strongest quantization-inspired selective-preservation
ideas that can map directly to latent-bridge repair.

## Strongest Sources

1. AWQ
   Link: https://arxiv.org/abs/2306.00978
   Why it matters: protect activation-salient channels first.

2. AQLM
   Link: https://arxiv.org/abs/2401.06118
   Why it matters: additive codebooks are the cleanest fallback if low-rank
   repair saturates.

3. LQER
   Link: https://proceedings.mlr.press/v235/zhang24j.html
   Why it matters: best direct low-rank quantization-error reconstruction
   template.

4. EoRA
   Link: https://arxiv.org/abs/2410.21271
   Why it matters: strongest eigenspace-aware repair source for the live lane.

5. ResQ
   Link: https://arxiv.org/abs/2412.14363
   Why it matters: preserve top-variance subspace and repair the rest.

6. QERA
   Link: https://arxiv.org/abs/2410.06040
   Why it matters: analytic error reconstruction framing is useful for
   justifying a residual bridge.

7. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: strongest preserve-core / repair-tail support.

8. EXL2 / ExLlamaV2
   Link: https://github.com/turboderp-org/exllamav2
   Why it matters: practical mixed-budget allocation reference for per-layer or
   per-route repair budgeting.

9. CommVQ
   Link: https://arxiv.org/abs/2506.18879
   Why it matters: strongest codebook-style fallback if low-rank repair is not
   enough.

## Exact Next Ablations

1. `dynalign + eigenspace residual`
2. `dynalign + saliency-gated residual`
3. `tokenbasis + preserve-core / tail-repair control`
4. codebook residual only after low-rank repair saturates

## Current Read

- The quantization literature keeps pointing to the same pattern: preserve the
  sensitive subspace first, then spend a small residual budget on the error
  tail.
- The first raw-basis preserve-core branch is negative, so the next preserve
  attempt should be eigenspace- or saliency-aware.
