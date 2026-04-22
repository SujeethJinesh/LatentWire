# Preserve-Tail Quantization Follow-Up Refs (2026-04-21)

Purpose: capture the strongest preserve-dominant-subspace plus low-rank-tail
references for the live `dynalign + residual` lane, now that naive preserve,
eigenspace, and one-shot saliency weighting have all failed on the frozen
GSM8K32 contract.

## Strongest Sources

1. AWQ
   Link: https://arxiv.org/abs/2306.00978
   Why it matters: activation-aware preservation of a tiny salient channel set
   before compressing the rest.

2. LQER
   Link: https://proceedings.mlr.press/v235/zhang24j.html
   Why it matters: low-rank quantization-error reconstruction with explicit
   activation-aware scaling.

3. QERA
   Link: https://openreview.net/forum?id=LB5cKhgOTu
   Why it matters: analytic framing for how much low-rank repair budget should
   be spent on tail error.

4. ResQ
   Link: https://openreview.net/forum?id=4qIP1sXcR1
   Why it matters: dominant-subspace preservation plus low-rank residuals is
   the closest structural precursor to a stronger LatentWire tail repair lane.

5. Preserve-Then-Quantize
   Link: https://openreview.net/forum?id=cHUXlmsSXK
   Why it matters: preserve-first then quantize-tail is the cleanest direct
   reference for the current preserve-plus-tail hypothesis.

6. SERQ
   Link: https://openreview.net/forum?id=nFjj8NEBqv
   Why it matters: saliency-aware low-rank compensation without a naive flat
   weighting pass.

7. GlowQ
   Link: https://arxiv.org/abs/2603.25385
   Why it matters: group-shared low-rank factors if per-layer repair proves too
   brittle or too expensive.

## Exact Next Ablations

1. `dynalign + learned preserve-plus-tail residual`
   Why now: the preserve idea still has structural support, but the current
   hard split is too naive.

2. `dynalign + group-shared preserve-tail residual`
   Why now: test whether the live lift is too fragile to support layer-local
   repair without sharing.

3. `dynalign + saliency-preserve codebook tail`
   Why now: if a pure low-rank tail still fails, move the tail codec from
   dense residuals to a discrete residual bank.

## Interpretable Telemetry

- preserved-rank fraction
- saliency mass captured by the preserved set
- tail residual norm before and after repair
- preserve-vs-tail error decomposition
- coverage, wins/losses vs target, and latency

## Current Read

- `dynalign_module_replace_residrank16 = 0.1250` is still the only live real
  same-pair row.
- `dynalign_preserve_module_replace_residrank16 = 0.0625`,
  `dynalign_eigenspace_module_replace_residrank16 = 0.0312`, and
  `dynalign_saliency_module_replace_residrank16 = 0.0312` show that simple
  preserve, eigenspace, and flat saliency weighting do not preserve that lift.
- `dynalign_saliency_preserve_module_replace_residrank16 = 0.0625` also fails
  to preserve the lift: it restores full coverage, but only ties target with
  `1/32` win and `1/32` loss vs target.
- The next preserve-side question is not whether preservation matters, but how
  to allocate budget between preserved structure and repaired tail.
