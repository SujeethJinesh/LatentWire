# Train-Only Receiver Permuted-Null Gap References

This memo supports
`paper/source_private_train_only_receiver_permuted_null_gap_20260501.md`.

## Primary Comparisons

- Cache-to-Cache (C2C) projects and fuses source KV cache into a receiver KV
  cache. LatentWire's permuted-null gap decoder differs by transmitting a tiny
  source-private packet and decoding with candidate side information rather than
  exposing or fusing source KV state. Sources: https://arxiv.org/abs/2510.03215
  and https://openreview.net/forum?id=LeatkxrBCi.
- LatentMAS and Interlat are close latent-communication baselines because they
  move agent communication into hidden-state space. The current contribution is
  narrower: rate-capped packet decoding with destructive source controls and a
  deterministic null receiver. Sources: https://arxiv.org/abs/2511.20639 and
  https://arxiv.org/abs/2511.09149.
- Relative Representations motivate a shared coordinate system built from
  anchors. The permuted-null gap decoder instead uses the null basis as a
  leakage guard around candidate-local residual scoring. Source:
  https://openreview.net/forum?id=SrC-nwieGJ.

## Mathematical Framing

- Slepian-Wolf / decoder-side-information coding motivates sending only the
  source evidence missing from receiver side information:
  https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources.
- Error-correcting output codes motivate candidate-code and syndrome decoding
  views: https://arxiv.org/abs/cs/9501101.
- Concept erasure / control-null projection is a useful analogy for removing
  control-predictive directions, but the local method uses a paired null
  receiver score rather than representation erasure.

## Quantization And Systems Inspiration

- QJL and TurboQuant motivate public random/quantized bases and null/sketch
  controls as systems-conscious analogies, but they target KV/vector
  compression rather than source-private candidate decoding:
  https://arxiv.org/abs/2406.03482 and https://arxiv.org/abs/2504.19874.
- vLLM/PagedAttention and MLPerf-style metrics remain required for native
  serving claims; the current receiver result is a method result, not a native
  GPU systems result. Sources: https://arxiv.org/abs/2309.06180,
  https://docs.vllm.ai/en/stable/usage/metrics/, and
  https://mlcommons.org/benchmarks/inference-datacenter/.

## Local Evidence

The new receiver mode `candidate_local_permuted_null_gap_residual_norm` passes
`6/6` n512 cross-family seed-repeat rows across seeds `47`, `53`, and `59`.
Cross-family accuracy is `0.500-0.625` versus target `0.250`, with maximum
best destructive control `0.260`.

This is not a duplicate of the above work because the core mechanism is a
source-private packet decoder that subtracts a deterministic permuted receiver
null score under strict source-destroying controls.
