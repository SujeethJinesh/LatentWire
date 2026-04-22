# Rank16 Residual / Gauge / Benchmark Follow-Up Refs (2026-04-21)

Purpose: capture the highest-signal references and exact next ablations after
the first real same-pair lift on the frozen GSM8K32 contract
(`dynalign_module_replace_residrank16 = 0.1250`).

## Residual Repair

1. LQER
   Link: https://proceedings.mlr.press/v235/zhang24j.html
   Why it matters: activation-induced scaling plus low-rank error
   reconstruction is the closest direct template for bridge residual repair.

2. EoRA
   Link: https://arxiv.org/abs/2410.21271
   Why it matters: pushes residual correction into an activation eigenspace
   instead of a raw hidden-state basis.

3. ResQ
   Link: https://arxiv.org/abs/2412.14363
   Why it matters: preserve a dominant subspace and repair only the hard tail;
   this is a strong blueprint for the current same-pair lane.

4. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: preserve top shared directions first, then spend low-rank
   budget on the residual only.

5. GEAR
   Link: https://arxiv.org/abs/2403.05527
   Why it matters: separates global low-rank repair from sparse outlier fixes,
   which is a good fallback if plain rank-limited repair saturates.

## Gauge / Canonicalization Wrappers

1. Complete Characterization of Gauge Symmetries in Transformer Architectures
   Link: https://openreview.net/forum?id=KrkbYbK0cH
   Why it matters: gives the transformer-specific gauge structure that should
   be fixed before bridge fitting.

2. Bispectral Invariants for Transformers
   Link: https://openreview.net/forum?id=QxVvKboznV
   Why it matters: strongest reference for removing continuous gauge freedom
   before matching the remaining invariant structure.

3. RECON
   Link: https://openreview.net/forum?id=bpWzTPDybh
   Why it matters: practical canonical-orientation normalization that can be
   wrapped around an existing lane.

4. Adaptive Canonicalization
   Link: https://arxiv.org/abs/2509.24886
   Why it matters: strongest motivation for prompt- or input-dependent
   canonicalization instead of one fixed global gauge.

5. Model Stitching by Invariance-aware Functional Latent Alignment
   Link: https://openreview.net/forum?id=hJvcbkf2nO
   Why it matters: reframes alignment as functional compatibility rather than
   geometry-only fit.

## Benchmark Contract / Competitor Norms

1. C2C
   Link: https://arxiv.org/abs/2510.03215
   Why it matters: live external bar on the frozen same-pair contract and the
   clearest direct semantic cache-communication comparator.

2. KVComm
   Link: https://arxiv.org/abs/2510.03346
   Why it matters: strongest selective KV-sharing reference; suggests explicit
   KV-fraction or layer-fraction budget matching.

3. Let Models Speak Ciphers
   Link: https://arxiv.org/abs/2310.06272
   Why it matters: clean non-text embedding-channel baseline for comparison
   against latent communication rows.

4. Augmenting Multi-Agent Communication with State Delta Trajectory
   Link: https://aclanthology.org/2025.emnlp-main.518.pdf
   Why it matters: strongest delta-channel comparison if we need a separate
   hidden-state communication block.

## Exact Next Ablations

1. `dynalign_module_replace_residrank16 + fixed gauge-fix wrapper`
   Why now: the matched `tokenbasis + rank16` control failed to reproduce the
   lift, so the next live branch should wrap the surviving dynalign row.

2. `dynalign_module_replace_residrank16 + adaptive canonicalization`
   Why now: strongest follow-up if one fixed gauge wrapper is too brittle.

3. Preserve-then-quantize residual split on the better rank16 lane
   Why now: best compression-inspired follow-up if plain low-rank repair ties
   but does not improve beyond the current smoke ceiling.

## Current Read

- We still are not ICLR-ready.
- The first real same-pair lift now comes from explicit residual correction,
  not from more teacher-side variants.
- The matched `tokenbasis + rank16` control failed to reproduce the lift, so
  the next proof step is a gauge/canonicalization wrapper on top of the live
  `dynalign + rank16 residual` lane.
- If that wrapper also fails, the same-pair benchmark story remains too brittle
  for paper mode and the positive burden shifts back toward the low-shot
  shared-basis lane.
