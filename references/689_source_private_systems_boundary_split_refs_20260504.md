# Source-Private Systems Boundary Split References

Date: 2026-05-04

## Purpose

This memo records the primary-source boundaries for the LatentWire systems
split. The safe claim is byte/exposure accounting for a source-private packet
communication object. Native serving-speed claims remain disabled.

## Primary-Source Boundaries

| Work | Source | Boundary for LatentWire |
|---|---|---|
| C2C | https://arxiv.org/abs/2510.03215 and https://openreview.net/pdf?id=LeatkxrBCi | C2C projects and fuses source KV cache into a target cache. LatentWire must not claim a native speed win without running C2C; the safe distinction is packet bytes and no raw KV transfer. |
| KVComm | https://arxiv.org/abs/2510.03346 and https://openreview.net/pdf?id=F7rUng23nw | KVComm selectively shares KV pairs. Treat it as a selected-KV communication baseline, not as a packet protocol. |
| DroidSpeak | https://arxiv.org/abs/2411.02820 | DroidSpeak reuses KV caches across same-architecture LLMs. It is a native systems neighbor; do not compare latency without matched serving rows. |
| Interlat | https://arxiv.org/abs/2511.09149 | Interlat transmits continuous last hidden states. LatentWire is not first latent communication; the distinction is a fixed source-private packet rather than dense hidden-state transfer. |
| QJL | https://arxiv.org/abs/2406.03482 | QJL is a low-bit KV/source-state sketch. Use as a one-bit KV byte floor, not as a defeated native baseline. |
| KIVI | https://arxiv.org/abs/2402.02750 | KIVI is tuning-free asymmetric 2-bit KV-cache quantization. Use as a 2-bit KV size floor. |
| KVQuant | https://arxiv.org/abs/2401.18079 | KVQuant is a sub-4-bit KV-cache quantization baseline. It exposes compressed KV/source state, not a source-private packet. |
| TurboQuant | https://arxiv.org/abs/2504.19874 and https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/ | TurboQuant is a strong modern vector/KV quantization comparator. Do not claim to beat its latency without native serving rows. |
| vLLM/PagedAttention | https://arxiv.org/abs/2309.06180 | vLLM is a serving substrate for native TTFT/TPOT/goodput/HBM measurement. It is not closed by Mac byte accounting. |
| SGLang/RadixAttention | https://arxiv.org/abs/2312.07104 | SGLang is a serving substrate and KV-reuse system. Treat it as a native measurement target. |
| Prefix tuning | https://aclanthology.org/2021.acl-long.353/ | Prefix tuning learns virtual tokens for task adaptation. LatentWire should distinguish packet communication from prompt tuning. |
| Gist tokens | https://arxiv.org/abs/2304.08467 | Gist tokens compress prompts into learned tokens. LatentWire should not present fixed candidate packets as general prompt compression. |

## Safe Claims

- LatentWire's cached-source row measures the communication object, not the full
  source-side computation.
- The end-to-end row separately discloses source scoring and receiver decode
  timing when phase traces exist.
- The packet row exposes no source text, raw hidden vector, raw score/logit
  vector, or KV cache.
- Dense KV/source-state byte floors remain much larger than the packet rows
  under explicit byte accounting.

## Unsafe Claims

- Native speedup over C2C, KVComm, DroidSpeak, QJL, TurboQuant, vLLM, or SGLang.
- TTFT, TPOT, HBM, PCIe/NVLink, goodput, or throughput improvements.
- General cross-model latent reasoning or universal latent-language transfer.
- Equating fp16 score/logit vectors with source-private packets.
