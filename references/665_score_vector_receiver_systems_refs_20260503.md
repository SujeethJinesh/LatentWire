# Score-Vector, Receiver, and Systems Boundary References

Date: 2026-05-03

## Purpose

This memo records the uniqueness and systems boundary after the ARC score-vector
fusion gate and the HellaSwag non-Qwen receiver-family gate.

## Latent Communication and Representation Boundaries

- C2C / Cache-to-Cache: https://arxiv.org/abs/2510.03215
  - Closest source-KV fusion competitor. LatentWire should be framed as tiny
    task packets unless native KV-transfer baselines are run.
- KVComm / KV communication: https://arxiv.org/abs/2510.03346 and
  https://arxiv.org/abs/2510.12872
  - Direct KV-sharing and cross-context reuse competitors. Do not claim
    serving superiority without native rows.
- InterLat: https://openreview.net/forum?id=rmYbgsehTd
  - Direct hidden-state communication prior. LatentWire differs only under a
    fixed-byte, source-private, source-necessary packet contract.
- Communicating Activations Between Language Model Agents:
  https://openreview.net/forum?id=W6RPXUUFic
  - Activation communication precedent. Avoid generic activation-language
    claims.
- Relative Representations: https://openreview.net/forum?id=SrC-nwieGJ
  - Anchor/common-coordinate precedent. Shared coordinates are not novel by
    themselves.
- Prefix-Tuning, Prompt Tuning, and Gist Tokens:
  https://aclanthology.org/2021.acl-long.353/ ,
  https://arxiv.org/abs/2104.08691 ,
  https://arxiv.org/abs/2304.08467
  - Soft/prefix/gist connectors are baselines, not novelty.
- Sparse autoencoders and Crosscoders:
  https://openreview.net/forum?id=F76bwRSLeK ,
  https://transformer-circuits.pub/2024/crosscoders/index.html
  - Motivate a sparse/common-feature bus, but do not validate a packet method
    without downstream controls.
- Wyner-Ziv side information:
  https://www.itsoc.org/publications/papers/the-rate-distortion-function-for-source-coding-with-side-information-at-the-decoder
  - Supports conditional innovation packets: encode what source knows beyond
    target side information.

## Systems and Quantization Boundaries

- vLLM / PagedAttention: https://arxiv.org/abs/2309.06180
- SGLang / RadixAttention: https://arxiv.org/abs/2312.07104 and
  https://sgl-project-sglang-93.mintlify.app/concepts/radix-attention
- LMCache: https://arxiv.org/abs/2510.09665
- CacheGen: https://arxiv.org/abs/2310.07240
- KIVI: https://arxiv.org/abs/2402.02750
- QJL: https://arxiv.org/abs/2406.03482
- TurboQuant: https://arxiv.org/abs/2504.19874

These systems papers define the native-serving baseline surface: TTFT, TPOT,
goodput, peak GPU memory, HBM traffic, cache reuse, and quantized-KV quality.
LatentWire's current systems result is byte/exposure accounting only.

## Decision Boundary

The 2026-05-03 gates say:

- ARC quantized source-score fusion does not beat source-label/source-index.
- HellaSwag TinyLlama packets transfer to a Phi-3 receiver slice, but the
  learned receiver does not beat packet-only.

Therefore the ICLR branch should stop global source-score fusion and simple
receiver tuning. The next high-value method should transmit conditional
innovation or sparse/common-feature evidence and must beat packet-only,
source-index, quantized source-score, same-byte text, candidate-roll, and
source-destroying controls.
