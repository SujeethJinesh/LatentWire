# Live Branch Follow-Up Refs (2026-04-21)

Purpose: capture the latest web-backed follow-up after the matched
`tokenbasis + rank16` control failed, so the real same-pair branch is now
`dynalign + rank16 residual` plus a possible gauge/canonicalization wrapper.

## Gauge / Canonicalization Wrappers

1. Complete Characterization of Gauge Symmetries in Transformer Architectures
   Link: https://openreview.net/forum?id=KrkbYbK0cH
   Why it matters: strongest transformer-specific reference for what symmetry
   group the wrapper should actually remove before alignment.

2. Bispectral Invariants for Transformers
   Link: https://openreview.net/forum?id=QxVvKboznV
   Why it matters: supports gauge-invariant comparison after the continuous
   symmetry is stripped away.

3. RECON
   Link: https://openreview.net/forum?id=bpWzTPDybh
   Why it matters: strongest practical canonical-orientation normalization
   reference for a plug-in wrapper around an existing lane.

4. Adaptive Canonicalization
   Link: https://arxiv.org/abs/2509.24886
   Why it matters: strongest backup if one fixed gauge choice is too brittle on
   prompt-varying latents.

## Residual Repair / Compression Transfer

1. ResQ
   Link: https://arxiv.org/abs/2412.14363
   Why it matters: preserve the dominant subspace and repair only the tail.

2. EoRA
   Link: https://arxiv.org/abs/2410.21271
   Why it matters: eigenspace low-rank correction is the strongest direct
   analogue for the bridge error we still see after alignment.

3. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: supports a preserve-core, repair-tail decomposition rather
   than a monolithic residual head.

4. LQER
   Link: https://arxiv.org/abs/2402.02446
   Why it matters: simplest low-rank error reconstruction baseline.

5. CommVQ
   Link: https://openreview.net/forum?id=sbbyCB39HN
   Why it matters: strongest codebook fallback if low-rank repair saturates.

## Lateral Connector Ideas

1. The Vision Wormhole
   Link: https://arxiv.org/abs/2602.15382
   Why it matters: strongest recent heterogeneous latent-connector idea; a
   wormhole-style side channel is the clearest lateral alternative to more
   pairwise bridge tuning.

2. COMPASS
   Link: https://arxiv.org/abs/2604.02056
   Why it matters: proxy tokens from a shared latent space suggest a compact
   shared-token sidecar.

3. RecursiveVLM
   Link: https://arxiv.org/abs/2602.09080
   Why it matters: recursive refinement is a better repair template than the
   current one-shot gate if we revisit repair later.

## Benchmark Contract / Table Structure

1. C2C
   Link: https://arxiv.org/abs/2510.03215
   Why it matters: live external bar for same-pair cache communication.

2. KVComm
   Link: https://arxiv.org/abs/2510.03346
   Why it matters: strongest selective KV-sharing comparator and budgeted
   baseline.

3. Let Models Speak Ciphers
   Link: https://arxiv.org/abs/2310.06272
   Why it matters: clean embedding-space communication baseline.

4. State Delta Trajectory
   Link: https://aclanthology.org/2025.emnlp-main.518/
   Why it matters: strongest hidden-state delta-channel baseline.

## Exact Next Ablations

1. `dynalign_module_replace_residrank16 + fixed gauge wrapper`
   Why now: the next exact real same-pair branch after the negative
   `tokenbasis + rank16` control.

2. `dynalign_module_replace_residrank16 + adaptive canonicalization`
   Why now: strongest backup if the fixed wrapper is too brittle.

3. `dynalign_module_replace_residrank16 + EoRA-style eigenspace residual`
   Why now: best next residual variant if the gauge wrapper does not move the
   frozen contract.

4. Wormhole-style proxy-token sidecar
   Why now: strongest lateral fallback if the same-pair bridge lane saturates.

## Current Read

- The live same-pair branch is now explicitly `dynalign + rank16 residual`.
- The matched `tokenbasis + rank16` control is negative, so the residual gain is
  not generic across the whole family.
- The first fixed gauge/canonicalization wrappers are also negative, so the
  next exact proof step is **adaptive** canonicalization or an eigenspace-aware
  residual, not another fixed wrapper and not another teacher variant.
- The main table should stay medium-split and budgeted, with same-pair rows
  kept separate from cross-family and multimodal rows.
