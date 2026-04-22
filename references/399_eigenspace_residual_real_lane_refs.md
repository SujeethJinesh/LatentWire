# Eigenspace Residual Real-Lane Refs (2026-04-21)

Purpose: capture the strongest residual-repair references and exact next
implementation ideas after `dynalign_module_replace + rank16 residual` became
the first real same-pair row to reach the current frozen smoke bar.

## Strongest Sources

1. LQER
   Link: https://arxiv.org/abs/2402.02446
   Why it matters: clean low-rank quantization-error reconstruction baseline
   and closest direct template for bridge-side residual repair.

2. QERA
   Link: https://arxiv.org/abs/2410.06040
   Why it matters: analytical residual initialization from error statistics is
   a strong alternative to plain SVD.

3. EoRA
   Link: https://arxiv.org/abs/2410.21271
   Why it matters: strongest eigenspace-aware correction reference for pushing
   residual repair into activation-dominant directions instead of raw space.

4. ResQ
   Link: https://arxiv.org/abs/2412.14363
   Why it matters: preserve a dominant subspace and repair only the tail.

5. LoPRo
   Link: https://arxiv.org/abs/2601.19675
   Why it matters: residuals become easier to repair after a structured
   rotation/permutation.

6. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: supports preserve-core, repair-tail splits instead of one
   monolithic low-rank branch.

7. SERQ
   Link: https://arxiv.org/abs/2603.08185
   Why it matters: saliency-aware residual repair is a strong fallback if plain
   rank-limited repair saturates.

8. LoRaQ
   Link: https://arxiv.org/abs/2604.18117
   Why it matters: optimized low-rank approximation offers a cleaner
   adapter-residual hybrid if we need a deployable branch.

## Exact Next Ablations

1. EoRA-style eigenspace residual on `dynalign_module_replace`
   Sweep: `r in {4, 8, 16}`
   Why now: strongest exact follow-up if adaptive canonicalization does not
   move the frozen contract.

2. Preserve-then-quantize split on `tokenbasis_replace`
   Why now: best control for whether the lift comes from residual repair alone
   or from spending the residual budget on the right basis.

3. Saliency-aware residual on `dynalign_module_replace`
   Why now: strongest low-cost backup if eigenspace repair is too brittle.

## Interpretable Telemetry

- per-layer reconstruction error before and after repair
- explained variance captured by the preserved subspace
- residual norm reduction by example
- numeric extraction coverage and exact-answer match
- runtime and memory overhead

## Current Read

- Residual repair is real on the live dynalign lane but not generic across
  bases.
- The next exact proof step is eigenspace-aware or saliency-aware repair, not
  more teacher-side alignment variants.
