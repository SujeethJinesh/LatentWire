# ARC Soft-Prefix Connector Uniqueness and Systems References

Date: 2026-05-03

## Purpose

This memo records the prior-work boundary for the ARC n32 target-loss
soft-prefix/query preflight. The connector is not novel because it uses soft
prefixes. The only defensible novelty is a source-conditioned, per-example,
rate-limited communication object that is tested against packet-only,
source-free, source-destroying, same-byte text, and Qwen-substituted controls.

## Closest Latent-Communication Prior Work

- Cache-to-Cache / C2C: https://arxiv.org/abs/2510.03215
  - Closest high-bandwidth source-KV transfer baseline. LatentWire must not
    claim C2C-style KV fusion unless a native baseline is run.
- InterLat: https://openreview.net/forum?id=rmYbgsehTd
  - Direct latent communication baseline. Any source-hidden-to-target-prefix
    branch must be framed as lower-rate/source-private rather than simply
    "latent communication."
- Communicating Activations Between Language Model Agents:
  https://openreview.net/forum?id=W6RPXUUFic
  - Activation-mixing precedent. LatentWire differs only if the transmitted
    object is compact, source-private, and source-necessary under controls.
- Direct Semantic Communication via Vector Translation:
  https://arxiv.org/abs/2511.03945
  - Recent vector-translation bridge for LLMs. It is a direct novelty threat
    for generic cross-model vector transfer claims.
- Latent Space Communication via K-V Cache Alignment:
  https://arxiv.org/abs/2601.06123
  - Shared KV latent-space adapters and portable soft prompts. It is a direct
    high-bandwidth comparator for any shared-latent-space framing.
- Relative Representations:
  https://openreview.net/forum?id=SrC-nwieGJ
  - Mandatory anchor/common-coordinate precedent. LatentWire should claim
    downstream source-private communication, not relative coordinates alone.

## Prefix, Prompt, and Steering Baselines

- Prefix-Tuning: https://aclanthology.org/2021.acl-long.353/
  - Establishes frozen-LM conditioning with learned continuous prefixes.
- Prompt Tuning: https://arxiv.org/abs/2104.08691
  - Establishes soft prompt adaptation as a baseline family.
- P-Tuning v2: https://arxiv.org/abs/2110.07602
  - Establishes deep prompt tuning; cite when using learned prefix states.
- Gist Tokens: https://arxiv.org/abs/2304.08467
  - Prompt-compression/soft-token baseline and same-byte text control context.
- Activation Addition: https://arxiv.org/abs/2308.10248
  - Steering-vector baseline for compact activation control.
- Representation Engineering: https://arxiv.org/abs/2310.01405
  - Broader activation-steering and representation-control baseline.
- Function Vectors: https://openreview.net/forum?id=AwyxtyMwaG
  - Compact task-vector/function-vector baseline.

## Common Feature and Query Connector Priors

- Sparse Autoencoders: https://openreview.net/forum?id=F76bwRSLeK
  - Supports feature-bus motivation, not a communication claim by itself.
- Anthropic monosemantic features:
  https://transformer-circuits.pub/2023/monosemantic-features/
  - Interpretability motivation for sparse features.
- Crosscoders:
  https://transformer-circuits.pub/2024/crosscoders/
  - Shared/private feature separation baseline for model-diffing.
- BLIP-2 / Q-Former: https://arxiv.org/abs/2301.12597
  - Query bottleneck connector precedent.
- Flamingo: https://arxiv.org/abs/2204.14198
  - Perceiver-style resampler precedent for tokenwise source pooling.
- Wyner-Ziv side information:
  https://www.itsoc.org/publications/papers/the-rate-distortion-function-for-source-coding-with-side-information-at-the-decoder
  - Theoretical support for encoding only source innovation relative to target
    side information.

## Systems and Quantization Boundaries

- KVCOMM: https://arxiv.org/abs/2510.12872
  - Online cross-context KV-cache communication and offset alignment. Cite for
    serving-side KV reuse; LatentWire has only accounting until native rows run.
- CacheGen: https://arxiv.org/abs/2310.07240
  - KV-cache compression/streaming baseline.
- LMCache: https://arxiv.org/abs/2510.09665
  - Serving-layer KV reuse/offload baseline.
- KIVI: https://arxiv.org/abs/2402.02750
  - Tuning-free asymmetric low-bit KV-cache quantization baseline.
- KVQuant: https://arxiv.org/abs/2401.18079
  - Sub-4-bit KV-cache quantization baseline.
- TurboQuant: https://arxiv.org/abs/2504.19874
  - Near-optimal online vector quantization, random rotation, and QJL residual
    correction. Treat as a vector/KV compression baseline and connector
    inspiration, not as semantic communication.
- vLLM/PagedAttention: https://arxiv.org/abs/2309.06180
  - Native serving baseline for TTFT, TPOT, memory, and prefix caching.
- SGLang/RadixAttention: https://arxiv.org/abs/2312.07104
  - Native structured-serving and prefix/KV reuse baseline.

## Decision Boundary

For ICLR, the soft-prefix/query connector is alive only if matched source
features beat target-only, source-free/static prefixes, zero-source,
row-shuffled, same-norm noise, label-shuffled, same-byte visible text,
packet-only source-index, and Qwen-substituted packet controls with paired
uncertainty. If packet-only source-index wins, the branch should be framed as a
negative connector result and the next method should move toward conditional
innovation/sparse-feature packets rather than larger soft prefixes.
