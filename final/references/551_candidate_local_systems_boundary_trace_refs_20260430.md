# Candidate-Local Systems Boundary Trace References

- date: `2026-04-30`
- purpose: primary-source memo for the candidate-local systems boundary trace.

## Claim Boundary

The boundary trace supports a narrow systems claim:

> The live candidate-local residual receiver is a source-private 8B-payload /
> 11B-record packet interface with explicit transfer accounting and Mac resident
> sparse-decode evidence. Cache/KV systems are reported as exposure, byte-floor,
> and native systems baselines, not defeated competitors.

## Primary Systems Anchors

- C2C / Cache-to-Cache directly projects and fuses source KV-cache into the
  target model. It is the closest semantic cache-communication competitor, but
  exposes source KV under the LatentWire threat model.
  Source: https://arxiv.org/abs/2510.03215
- KVComm/KVCOMM and Q-KVComm share or compress KV-cache state across contexts
  or agents. These are communication competitors only under a source-KV-visible
  access model.
  Sources: https://arxiv.org/abs/2510.03346,
  https://arxiv.org/abs/2510.12872, and https://arxiv.org/abs/2512.17914
- TurboQuant is an online vector/KV quantization method with near-optimal
  distortion-rate claims. It is a KV byte-floor and native-kernel baseline, not
  a source-private packet baseline.
  Source: https://arxiv.org/abs/2504.19874
- KIVI and KVQuant are low-bit KV-cache quantization systems. They motivate the
  byte-floor rows and must not be claimed as beaten without native KV runs.
  Sources: https://arxiv.org/abs/2402.02750 and
  https://arxiv.org/abs/2401.18079
- CacheGen compresses and streams KV caches for context loading; it is a
  serving/cache-reuse baseline with KV exposure.
  Source: https://arxiv.org/abs/2310.07240
- vLLM/PagedAttention and DistServe define the serving metrics and runtime
  substrate needed for a future native systems table: TTFT, TPOT, goodput, GPU
  memory, and prefill/decode effects.
  Sources: https://arxiv.org/abs/2309.06180 and
  https://arxiv.org/abs/2401.09670

## Result-Guided Implication

The current artifact should be used to separate access models:

- LatentWire packet: same-slice measured, source-private, no source text/KV,
  8B payload, 11B record, 64B/128B/4KB transfer accounting, resident sparse
  decode p50 `5.231934 us/request`.
- matched-byte text: same-slice measured, private text exposed, target floor.
- query-aware/full-log text: Mac/accounting proxy rows, private text exposed.
- QJL/KIVI/KVQuant/TurboQuant/CacheGen: KV byte-floor or native systems rows,
  not source-private packet equivalents.
- C2C/KVComm/Q-KVComm: native cache-communication rows that require source-KV
  visibility and future NVIDIA/vLLM instrumentation.
- Mac proxy C2C/KVComm/Q-KVComm/TurboQuant/CacheGen byte floors: deterministic
  source-KV accounting rows derived from the local Qwen3 endpoint proxy, not
  native kernel measurements.

The KV proxy rows use the standard decoder-cache byte formula:

```text
KV_bytes(T, L, H_kv, D_head, bits)
  = T * L * 2 * H_kv * D_head * bits / 8
```

For selected-layer systems:

```text
KVComm_bytes = T * ceil(p * L) * 2 * H_kv * D_head * bits / 8
```

For ratio-compression systems:

```text
Compressed_KV_bytes = Full_KV_bytes / compression_ratio
```

Measured proxy floors now recorded in the artifact:

- C2C fp16 source-KV lower-bound proxy: `344064B`;
- KVComm 30%-layer fp16 lower-bound proxy: `103219.2B`;
- Q-KVComm 6x compressed lower-bound proxy: `57344B`;
- TurboQuant 3.5-bit lower-bound proxy: `75264B`;
- TurboQuant 2.5-bit aggressive lower-bound proxy: `53760B`;
- CacheGen 4.3x compressed lower-bound proxy: `80014.9B`.

The minimum proxy/live-record ratio is `4887.3x` (`53760B / 11B`), but this is a
byte-floor contrast only. It is not a claim that LatentWire beats those systems
on native quality or serving latency.

Safe claim:

> The live method occupies a byte-scale packet-boundary point that is distinct
> from private text relay and source-KV transport, and the current Mac proxy
> table quantifies how large optimistic KV floors are under the local endpoint
> configuration.

Unsafe claim:

> The current Mac trace does not prove production serving speedup and does not
> beat C2C/KVComm/TurboQuant/CacheGen/KIVI/KVQuant on native workloads.
