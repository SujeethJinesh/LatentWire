# 2026-05-04 Post-Receiver Systems Reference Memo

## Purpose

This memo records the systems-related primary sources used to bound the
post-receiver-failure Mac packet-ring transport claim. The Mac run strengthens
LatentWire's measured local packet-transport evidence, but it must not be
presented as native serving evidence until the NVIDIA/vLLM/SGLang rows are run.

## Source-Private Packet Claim Boundary

LatentWire communicates compact source-private decision packets. The current
artifact measures local pack-copy-verify movement for 4B and 7B framed packet
records, plus text/log/KV byte-floor buffers. This differs from methods that
communicate source model internals, source KV cache, or quantized source-state
sketches.

## Related Systems Baselines

- C2C / cache-to-cache communication: `https://openreview.net/forum?id=LeatkxrBCi`
  and `https://arxiv.org/abs/2510.03215`. C2C is the closest model-to-model
  state-sharing competitor because it transfers/fuses cache state rather than
  fixed source-private packets.
- KVComm / KVCOMM selective KV communication:
  `https://openreview.net/forum?id=F7rUng23nw`,
  `https://arxiv.org/abs/2510.03346`, and
  `https://arxiv.org/abs/2510.12872`. These are KV/cache communication
  baselines and should be compared with source-KV exposure explicitly marked.
- QJL quantized Johnson-Lindenstrauss sketching:
  `https://arxiv.org/abs/2406.03482`. QJL is a compact source-state sketch
  comparator, not a source-private packet protocol.
- TurboQuant low-bit KV-cache quantization:
  `https://arxiv.org/abs/2504.19874`. TurboQuant is a low-bit KV-cache
  substrate/comparator, not evidence for LatentWire until run on the same
  serving surface.
- KIVI KV-cache quantization: `https://arxiv.org/abs/2402.02750`.
  KIVI supplies a quantized KV byte-floor comparator for source-state exposure.
- vLLM / PagedAttention serving substrate:
  `https://arxiv.org/abs/2309.06180`. Native claims should report TTFT, TPOT,
  goodput, memory, and cache behavior on a vLLM-compatible serving run.
- SGLang / RadixAttention serving substrate:
  `https://arxiv.org/abs/2312.07104`. SGLang is another required native
  endpoint surface for fair systems comparison.

## Review Boundary

Safe paper wording:

> A Mac-local packet-ring microbenchmark shows stable packed-record movement for
> LatentWire's source-private packet format; native serving wins remain future
> work pending NVIDIA endpoint measurements.

Unsafe paper wording:

> LatentWire is faster than C2C, KVComm, QJL, TurboQuant, vLLM, or SGLang.

The latter is not supported by the current artifacts.
