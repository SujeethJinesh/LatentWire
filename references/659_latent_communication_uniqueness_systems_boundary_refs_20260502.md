# Latent Communication Uniqueness and Systems Boundary References

Date: 2026-05-02

## Purpose

This memo records the primary-source boundary for the current LatentWire claim.
The paper should not claim generic latent communication, prompt compression, or
KV/cache transfer as its novelty. The defensible claim is narrower:
source-private, fixed-byte evidence packets with destructive controls and
systems-facing byte/exposure accounting.

## Closest Communication Competitors

- Cache-to-Cache (C2C), ICLR 2026:
  https://arxiv.org/abs/2510.03215
  C2C projects and fuses source KV-cache state into a target model. This is a
  direct semantic-communication competitor, but it transfers model state rather
  than a fixed-byte source-private task packet.
- KVComm selective KV sharing, ICLR 2026:
  https://arxiv.org/abs/2510.03346
  KVComm selects informative KV pairs for inter-model communication. It is a
  required systems comparator because it sends KV state, including source-model
  internal objects.
- Interlat, ACL 2026:
  https://arxiv.org/abs/2511.09149
  Interlat sends continuous last hidden states and learns latent compression.
  It is a latent-agent communication comparator, not a source-private packet
  method.
- Relative Representations, ICLR 2023 notable:
  https://arxiv.org/abs/2209.15430
  This is the closest anchor-space precedent: samples are represented by their
  similarities to a fixed anchor set to support latent-space communication. For
  LatentWire, this means anchor/Fourier rows must be framed as public-coordinate
  evidence packets with controls, not as a novel anchor representation.

## Compression and Systems Comparators

- QJL:
  https://arxiv.org/abs/2406.03482
  QJL uses a Johnson-Lindenstrauss transform and sign quantization for KV-cache
  compression. It provides a quantized-state byte floor and inner-product
  estimator boundary.
- TurboQuant:
  https://arxiv.org/abs/2504.19874
  TurboQuant uses random rotations plus scalar quantization and a QJL residual
  stage for near-optimal vector quantization. It is a compression comparator,
  not a task-evidence packet baseline.
- Gist Tokens:
  https://arxiv.org/abs/2304.08467
  Gisting compresses prompts into learned tokens for compute reuse. It is a
  prompt-compression baseline boundary.
- LLMLingua:
  https://arxiv.org/abs/2310.05736
  LLMLingua compresses visible text prompts. Same-byte visible text controls
  are needed so reviewers can separate packet evidence from ordinary prompt
  compression.
- vLLM/PagedAttention:
  https://arxiv.org/abs/2309.06180
  vLLM is the natural target-only serving baseline for native GPU rows.
- SGLang:
  https://arxiv.org/abs/2312.07104
  SGLang is a required serving baseline for native throughput and latency rows.

## Inspiration Boundaries, Not Current Claims

- Diffusion Transformers:
  https://arxiv.org/abs/2212.09748
  DiT motivates transformer-based latent denoising and scalable latent-token
  processing. For LatentWire this is only future-method inspiration for
  iterative packet repair or denoising receivers; it is not current evidence.
- Consistency Models:
  https://arxiv.org/abs/2303.01469
  Consistency models motivate one-step/few-step repair from noisy latent states.
  The current ARC learned repair branches failed, so this should be framed only
  as a future receiver-design idea.
- SAE feature-space universality:
  https://arxiv.org/abs/2410.06981
  SAE universality supports the hypothesis that feature spaces can align across
  models under suitable transformations. It does not by itself validate
  LatentWire; it motivates future common-language probes and feature-space
  controls.

## Claim Boundary

Allowed current claim:

- LatentWire demonstrates fixed-byte source-private evidence packets on
  public-basis ARC/OpenBookQA gates, with destructive controls that distinguish
  matched source packets from zero-source, wrong-row, deranged, and visible-text
  controls.
- The systems contribution is currently byte/exposure accounting plus a native
  ingest gate and claim-boundary matrix.

Forbidden current claim:

- Do not claim that LatentWire generally beats C2C, KVComm, or Interlat.
- Do not claim a native GPU throughput, latency, or HBM win until vLLM/SGLang,
  C2C/KVComm, QJL/TurboQuant, same-byte text, and label-copy rows are measured
  and ingested.
- Do not claim the anchor mechanism itself is novel relative to Relative
  Representations. The novelty is the source-private packet protocol, controls,
  rate accounting, and benchmark gate.

## Next Reviewer-Facing Comparison

For COLM, cite these works to show that LatentWire is not rediscovering prompt
compression, hidden-state transfer, or KV-cache communication. For ICLR, add
native rows and a stronger cross-family positive row before presenting the
method as competitive with C2C/KVComm-style systems.
