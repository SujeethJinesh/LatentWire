# Multimodal / Residual / Contract Follow-Up Refs (2026-04-21)

Purpose: capture the next lateral inspirations after the first real
`rank16` same-pair residual lift, with emphasis on heterogeneous connectors,
quantization-style residual repair, and fair benchmark-table structure.

## Heterogeneous / Multimodal Connector Ideas

1. The Vision Wormhole
   Link: https://arxiv.org/abs/2602.15382
   Why it matters: strongest recent heterogeneous latent-connector pattern;
   supports a hub-and-spoke “wormhole” side channel rather than pairwise
   translators everywhere.

2. COMPASS
   Link: https://arxiv.org/abs/2604.02056
   Why it matters: proxy tokens generated from a shared latent space suggest a
   compact shared-token sidecar for LatentWire.

3. RecursiveVLM
   Link: https://arxiv.org/abs/2602.09080
   Why it matters: recursive refinement is a stronger template for latent
   repair than the current one-shot gate.

4. AlignVLM
   Link: https://arxiv.org/abs/2502.01341
   Why it matters: explicit shared-space alignment plus robustness framing is a
   close analogue of the current interface-stress and bridge-stability story.

5. OmniBridge
   Link: https://arxiv.org/abs/2509.19018
   Why it matters: two-stage latent alignment then guided refinement is a good
   abstraction for “shared basis first, residual correction second.”

## Residual Repair / Quantization Transfer

1. ResQ
   Link: https://arxiv.org/abs/2412.14363
   Why it matters: preserve a dominant subspace and only repair the tail.

2. EoRA
   Link: https://arxiv.org/abs/2410.21271
   Why it matters: eigenspace low-rank repair is the clearest direct bridge
   residual analogue.

3. Preserve-Then-Quantize
   Link: https://arxiv.org/abs/2602.02001
   Why it matters: preserve the core shared directions before allocating
   residual rank.

4. RILQ
   Link: https://arxiv.org/abs/2412.01129
   Why it matters: optimize compensation against the real discrepancy signal,
   not a proxy.

5. LQER
   Link: https://arxiv.org/abs/2402.02446
   Why it matters: simplest low-rank error reconstruction baseline for the
   bridge lane.

6. CommVQ
   Link: https://openreview.net/forum?id=sbbyCB39HN
   Why it matters: best codebook fallback if low-rank repair saturates.

## Benchmark Contract / Comparison Norms

1. C2C
   Link: https://arxiv.org/abs/2510.03215
   Why it matters: current live same-pair external bar for cache-space
   communication.

2. KVComm
   Link: https://arxiv.org/abs/2510.03346
   Why it matters: strongest selective KV-sharing comparator and the cleanest
   budgeted KV baseline.

3. Let Models Speak Ciphers
   Link: https://arxiv.org/abs/2310.06272
   Why it matters: clean embedding-space communication baseline for token-free
   exchange.

4. State Delta Trajectory
   Link: https://aclanthology.org/2025.emnlp-main.518/
   Why it matters: strongest hidden-state delta-channel baseline for a
   separate communication-medium block.

5. Direct Semantic Communication via Vector Translation
   Link: https://arxiv.org/abs/2511.03945
   Why it matters: useful latent-translation comparator for learned bridge
   baselines.

## Exact Next Ablations

1. `dynalign_module_replace_residrank16 + fixed gauge wrapper`
   Why now: the matched `tokenbasis + rank16` control failed, so the next live
   real branch should wrap the surviving dynalign residual row.

2. `dynalign_module_replace_residrank16 + adaptive canonicalization`
   Why now: strongest backup if one fixed gauge choice is too brittle.

3. EoRA-style eigenspace residual repair
   Why now: strongest next residual variant if plain low-rank rank16 only ties
   the current benchmark ceiling.

4. Wormhole-style proxy-token sidecar
   Why now: strongest lateral connector idea if same-pair residual repair
   saturates and we need a more explicit shared interface.

## Current Read

- The best current same-pair real clue is still residual repair on top of
  output-aware alignment, but the lift is now clearly dynalign-specific rather
  than generic across the full family.
- The best next wrapper is gauge/canonicalization, not another teacher variant
  and not another blind residual sweep over the token-grounded lane.
- The best lateral fallback is a proxy-token or wormhole-style shared side
  channel inspired by heterogeneous multimodal systems.
- The best codec-side add-on is now “preserve dominant anchors, redesign the
  tail,” not “drop in a naive codebook tail.”
- The next main table should stay split by communication medium and budget,
  with same-pair rows kept separate from cross-family and multimodal rows.
