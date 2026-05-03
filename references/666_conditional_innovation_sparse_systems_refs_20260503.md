# Conditional-Innovation, Sparse-Feature, and Systems References

Date: 2026-05-03

## Purpose

This memo pins the uniqueness and systems boundary after the ARC
conditional-innovation packet gate. The result is not promotable, but it
clarifies the next method branch: source evidence must be transformed into a
receiver-conditioned sparse/common-feature object, not just repackaged as
candidate scores.

## Conditional Innovation

- Wyner-Ziv side information:
  https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf
  - Supports the decoder-side-information framing. LatentWire's receiver has
    its own target-side evidence, so the source packet should encode the
    missing innovation rather than the full message.
- Quest query-aware KV selection:
  https://arxiv.org/abs/2406.10774
  - Motivation for receiver-query-conditioned selection. The useful
    communicated object may depend on what the receiver is trying to resolve.
- SnapKV query-focused cache compression:
  https://arxiv.org/abs/2404.14469
  - Another query-focused compression precedent. Query-conditioned packet
    design is a method inspiration, not a baseline win until measured.

## Sparse and Common-Feature Coordinates

- Sparse autoencoders:
  https://arxiv.org/abs/2309.08600
  - Motivates sparse feature dictionaries as a more interpretable basis than
    raw hidden vectors.
- SAE feature universality:
  https://arxiv.org/abs/2410.06981
  - Supports the hypothesis that some sparse features can align across models.
    LatentWire still needs downstream packet controls to claim a common
    language.
- QJL:
  https://arxiv.org/abs/2406.03482
  - Useful systems/math inspiration for fixed-rate sign sketches of vector
    residuals. It is a KV quantization method, not a direct latent
    communication baseline.
- TurboQuant:
  https://openreview.net/forum?id=tO3ASKZlok
  - Defines a current vector/KV quantization comparison surface. LatentWire's
    packet rows should compare bytes/exposure against it, and native
    throughput only after shared hardware runs.

## Direct Cache and KV Communication Competitors

- C2C / Cache-to-Cache:
  https://openreview.net/forum?id=LeatkxrBCi
  - Closest direct semantic communication competitor. C2C projects and fuses
    source KV cache into target cache. LatentWire must be framed as a tiny
    source-private task packet unless native KV baselines are run.
- KVComm:
  https://arxiv.org/abs/2510.03346
  - Selective KV-sharing communication. It transmits KV pairs, not tiny
    fixed-byte task packets.
- KVCOMM:
  https://arxiv.org/abs/2510.12872
  - Cross-context KV-cache communication for multi-agent systems. Use as a
    systems competitor for native GPU rows, not as a current Mac-local
    throughput claim.

## Serving and Cache-Movement Systems

- vLLM / PagedAttention:
  https://arxiv.org/abs/2309.06180
  - Required serving substrate for native rows; report TTFT, TPOT, ITL,
    goodput, throughput, memory, and concurrency.
- LMCache:
  https://arxiv.org/abs/2510.09665
  - Closest open-source KV cache movement/reuse system surface for native
    comparison.
- CacheGen:
  https://cs.stanford.edu/~keithw/sigcomm2024/sigcomm24-final1571-acmpaginated.pdf
  - KV compression/streaming competitor. LatentWire can claim a different
    communicated object now, but not lower serving latency until a matched
    hardware run exists.

## Decision Boundary

The conditional-innovation packet result weakens score-surface methods:

- matched innovation beats learned source-index and quantized score decoders
  by `+0.033557`, but the CI crosses zero;
- matched innovation beats direct source-label text by only `+0.006711`;
- the flip audit is only `+1` net heldout example over source label.

Promote only if the next sparse/query-conditioned packet beats source-label,
source-index, source-score, same-byte text, target-derived packet,
row-shuffle, candidate-roll, and label-shuffle controls with positive paired
uncertainty and seed repeats.
