# ARC/OpenBookQA Soft-Prefix Preflight References

Date: 2026-05-02

## Purpose

This memo supports the first target-loss soft-prefix preflight for
ARC/OpenBookQA and records the novelty boundary for a future positive
connector claim.

## Primary Sources Checked

- Prefix-Tuning keeps the LM frozen and learns continuous task prefixes, so it
  is the direct baseline for any soft-token receiver interface. Source:
  https://arxiv.org/abs/2101.00190
- BLIP-2 bridges frozen encoders and frozen LLMs with a lightweight Querying
  Transformer. It supports query bottlenecks as a plausible architecture but
  makes "learned queries bridge modules" non-novel by itself. Source:
  https://arxiv.org/abs/2301.12597
- Flamingo uses a Perceiver-style resampler and gated cross-attention to feed
  non-text features into a language model. This is the strongest architectural
  analogy for a fixed query-token connector. Source:
  https://arxiv.org/abs/2204.14198
- Cache-to-Cache (C2C) directly communicates projected/fused KV-cache state
  between LLMs, making it the closest semantic communication competitor.
  Source: https://openreview.net/forum?id=LeatkxrBCi
- KVComm-style selective KV sharing is a systems/latency competitor because it
  transmits cache-like state rather than fixed source-private packets. Source:
  https://arxiv.org/abs/2510.03346
- QJL gives a 1-bit quantized Johnson-Lindenstrauss transform for KV-cache
  quantization and is a required byte-floor baseline. Source:
  https://arxiv.org/abs/2406.03482
- TurboQuant is a rate/distortion-oriented online vector quantization baseline
  for KV/cache-like vectors. Source: https://arxiv.org/abs/2504.19874
- Sparse autoencoder scaling/evaluation work is relevant for interpretable
  common-basis features, but reconstruction quality is not enough unless
  target loss improves under source-destroying controls. Source:
  https://arxiv.org/abs/2406.04093
- Consistency Models are useful inspiration for one-step/few-step latent
  refinement, but iterative refinement must be evaluated with help/harm and
  stop-rule telemetry before becoming a LatentWire contribution. Source:
  https://arxiv.org/abs/2303.01469

## Novelty Boundary

The publishable claim is not that soft prefixes, query tokens, or KV transfer
exist. The publishable claim must be narrower:

- per-example source-conditioned communication;
- fixed-rate packet/query budget and explicit source-exposure accounting;
- frozen target or train-only receiver calibration;
- target-loss improvement, not representation reconstruction alone;
- source-destroying controls that rule out target-cache-only, static-prefix,
  shuffled-source, random-prefix, and same-byte-text explanations.

## Implication For The Current Gate

The current n8 CPU result is negative but useful. It establishes the
target-loss soft-prefix scaffold and shows that a single selected-choice Qwen
hidden summary is not enough on the tiny ARC smoke surface. The next method
should use a real tokenwise/query bottleneck and larger validation slices
before any ICLR claim.
