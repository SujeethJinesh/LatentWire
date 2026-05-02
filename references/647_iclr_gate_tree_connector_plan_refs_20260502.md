# ICLR Gate Tree Connector Plan References

Date: 2026-05-02

## Purpose

This memo supports the gate-tree consolidation and the next tokenwise
soft-prefix/query connector branch. It records the boundary between what
LatentWire can claim now and what prior work already covers.

## Primary Sources Checked

- BLIP-2 / Q-Former: a lightweight Querying Transformer bridges frozen visual
  encoders and frozen LLMs. This is a close precedent for learned query
  bottlenecks between heterogeneous modules. Source:
  https://arxiv.org/abs/2301.12597
- Flamingo: uses a Perceiver-style resampler and gated cross-attention to
  condition a language model on non-text inputs. This bounds any claim that a
  resampler/query bridge is novel by itself. Source:
  https://arxiv.org/abs/2204.14198
- Perceiver IO: gives the general latent-array/query architecture for mapping
  structured inputs and outputs. Source: https://arxiv.org/abs/2107.14795
- Prefix-Tuning: optimizes continuous prefixes while keeping the LM frozen.
  This is the mandatory baseline for any soft-token receiver interface.
  Source: https://arxiv.org/abs/2101.00190
- Cache-to-Cache (C2C): direct semantic communication between LLMs through
  cache-state channels. LatentWire must differ on byte rate, source exposure,
  and destructive controls. Source: https://openreview.net/forum?id=LeatkxrBCi
- KVComm and KVCOMM: selective/cross-context KV sharing are cache-transfer
  communication baselines, not fixed-byte source-private packet methods.
  Sources: https://arxiv.org/abs/2510.03346 and
  https://arxiv.org/abs/2510.12872
- QJL: 1-bit quantized JL transform for KV-cache quantization. It is a
  required byte-floor baseline for systems claims. Source:
  https://arxiv.org/abs/2406.03482
- TurboQuant: online vector quantization with rate/distortion guarantees. It
  is a required quantization/compression boundary for future native systems
  rows. Source: https://arxiv.org/abs/2504.19874
- Diffusion Transformers and Consistency Models: useful inspiration for
  iterative refinement, but not direct evidence for current LatentWire claims.
  Sources: https://arxiv.org/abs/2212.09748 and
  https://arxiv.org/abs/2303.01469
- SAE feature-space universality: supports the plausibility of shared sparse
  feature spaces, but makes any generic "common latent language" claim
  non-novel unless LatentWire shows source-private fixed-byte behavior gains.
  Source: https://arxiv.org/abs/2410.06981
- Relative Representations: model comparison through relative coordinates is a
  common-basis prior-art boundary for learned alignment claims. Source:
  https://openreview.net/forum?id=SrC-nwieGJ

## Novelty Boundary

The unique claim is not "soft prefixes work" or "queries bridge models." Those
are prior art. The unique LatentWire claim must be:

- per-example source-conditioned messages;
- fixed byte budget and explicit source-state exposure accounting;
- frozen target or train-only receiver calibration;
- destructive controls showing the transmitted packet carries source-specific
  information rather than target-cache leakage or label copying;
- matched comparison against KV/cache transfer and quantization byte floors.

## Gate Implication

The next experiment should be a target-loss connector, not another mean-cache
proxy. A Mac-local ARC n32 soft-prefix preflight is useful only as an
implementation gate. The paper-facing positive result needs OpenBookQA 3B or a
larger frozen ARC/OpenBookQA slice with paired uncertainty and strict controls.
