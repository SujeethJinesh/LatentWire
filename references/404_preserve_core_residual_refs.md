# Preserve-Core Residual Refs (2026-04-21)

Purpose: capture the strongest sources and exact next experiments for the
preserve-core / repair-tail branch on top of the live `dynalign` lane.

## Strongest Sources

1. AWQ
   Link: https://arxiv.org/abs/2306.00978
   Why it matters: strongest selective-preservation reference for protecting a
   small activation-salient core while quantizing the rest.

2. AQLM
   Link: https://arxiv.org/abs/2401.06118
   Why it matters: additive codebooks are the cleanest fallback if plain
   low-rank tail repair saturates.

3. APTQ
   Link: https://arxiv.org/abs/2402.14866
   Why it matters: mixed precision guided by attention outputs is the best
   reference for budgeted bridge-side preservation.

4. EoRA
   Link: https://arxiv.org/abs/2410.21271
   Why it matters: strongest eigenspace-aware residual repair reference for the
   live same-pair lane.

5. ResQ
   Link: https://arxiv.org/abs/2412.14363
   Why it matters: preserve a dominant activation subspace and repair only the
   tail with a residual branch.

6. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: strongest direct support for splitting limited rank budget
   between preserved core and repaired tail.

7. CommVQ
   Link: https://arxiv.org/abs/2506.18879
   Why it matters: strongest codebook-style fallback if low-rank tail repair
   alone is not enough.

## Exact Next Ablations

1. `dynalign_preserve_module_replace_residrank16`
   Why now: nearest preserve-core / repair-tail additive branch on the live
   real same-pair benchmark lane.

2. EoRA-style eigenspace residual on `dynalign_module_replace`
   Why now: strongest next residual formulation if preserve-core only ties the
   current row.

3. Preserve-core on `tokenbasis_replace`
   Why now: matched control for whether selective preservation itself is enough
   or still basis-dependent.

## Interpretable Telemetry

- preserved-subspace rank
- preserved-subspace energy / explained variance
- tail residual norm before and after repair
- exact match and numeric extraction coverage
- latency and memory overhead

## Current Read

- The most useful compression lesson for LatentWire is now preserve-core plus
  repair-tail, not blanket residual complexity.
- The first raw-basis preserve-core split on the frozen same-pair contract is
  negative: `dynalign_preserve_module_replace_residrank16 = 0.0625`, full
  coverage, `0/32` wins over target.
- So the next preserve-side attempt should be eigenspace-aware or
  saliency-aware, not another simple top-subspace split in the raw basis.
