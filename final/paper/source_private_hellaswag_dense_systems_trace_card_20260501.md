# HellaSwag Dense Packet Systems Trace Card

## Status

This is a Mac-local systems boundary card for the dense HellaSwag
hidden-innovation packet. It is not a native NVIDIA/vLLM/SGLang throughput
claim.

Current systems story: the positive HellaSwag method communicates through a
`2B` raw / `5B` framed packet and does not transmit source text, KV cache, raw
hidden vectors, or raw score vectors. The current win is source-state exposure
and byte accounting, plus Mac CPU extraction timing. Native TTFT, TPOT, goodput,
HBM, and GPU-memory rows remain blocking.

## Local Evidence

Method artifact:
`results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation7168_8192/hellaswag_hidden_innovation_eval_slice_stress.json`

Aggregate artifact:
`results/source_private_hellaswag_hidden_innovation_multi_slice_stress_20260501_qwen05_validation0_8192/hellaswag_hidden_innovation_multi_slice_stress.json`

Validation status:

- latest slice: validation `7168:8192`
- latest slice pass: `true`
- aggregate: validation `0:8192`
- aggregate slices passing: `8/8`
- aggregate rows: `8192`
- weighted selected accuracy: `0.522949`
- weighted best label-copy accuracy: `0.482056`
- weighted score-only accuracy: `0.476929`
- min slice delta vs best label-copy: `+0.034180`

## Packet Boundary

| Field | Value |
|---|---:|
| raw payload bytes | `2` |
| framed record bytes | `5` |
| source text transmitted | `false` |
| source KV cache transmitted | `false` |
| raw hidden vector transmitted | `false` |
| raw score vector transmitted | `false` |
| single-request cacheline accounting | `64B` |
| batch-64 cacheline bytes/request | `5B` |
| batch-64 DMA bytes/request | `6B` |

The packet carries a selected candidate id plus a quantized debug/confidence bin
under a predeclared model-bank policy. The model-bank id is experiment metadata,
not a per-request transmitted field.

## Mac CPU Timing

Measured on the local Mac for validation rows `7168:8192`.

| Phase | Total | Per Request |
|---|---:|---:|
| source score cache | `543.655s` | `530.9ms` |
| source hidden cache | `380.325s` | `371.4ms` |
| full slice script | `962.351s` | `939.8ms` |

Cache artifacts for this slice:

- score cache: `139K`
- hidden cache: `13M`
- final gate JSON: `63K`

These timings are CPU/offline artifact-generation timings. They should not be
used as serving throughput claims.

## Source-State Byte Floors

These are byte/exposure floors against the `5B` framed packet, using the local
Qwen2.5-0.5B source configuration recorded in the cross-benchmark systems
comparator. They are not quality-matched native baselines.

| Comparator floor | Bytes/request | Ratio vs 5B packet | Exposure |
|---|---:|---:|---|
| LatentWire framed packet | `5` | `1.0x` | no source text/KV/raw hidden/raw scores |
| FP16 one-token KV | `12288` | `2457.6x` | source KV/state |
| KVComm 30% FP16 KV floor | `3686.4` | `737.3x` | source KV/state |
| QJL 1-bit KV sketch | `768` | `153.6x` | source-state sketch |
| QJL 30% layer floor | `230.4` | `46.1x` | source-state sketch |
| TurboQuant 3.5-bit KV floor | `2688` | `537.6x` | source-state quantization |
| KIVI 2-bit KV floor | `1536` | `307.2x` | source KV quantization |
| KVQuant 3-bit KV floor | `2304` | `460.8x` | source KV quantization |

## Reviewer-Safe Claim

Safe systems claim:

> On Mac, LatentWire demonstrates a source-private byte-scale communication
> boundary for HellaSwag: a `2B` raw / `5B` framed packet preserves a positive
> validation gain over label-copy and score-only controls across `8192`
> contiguous rows while exposing no source text, source KV cache, raw hidden
> vector, or raw score vector.

Forbidden systems claim:

> LatentWire is faster than C2C, KVComm, QJL, TurboQuant, KIVI, KVQuant, vLLM,
> or SGLang in native serving.

That claim remains blocked until native NVIDIA rows measure TTFT, TPOT, ITL,
goodput, peak GPU memory, KV-cache use, HBM traffic, and PCIe/NVLink traffic.

## Next Systems Gate

The next systems artifact should be a native table with:

1. target-only vLLM and SGLang;
2. same-byte visible text control;
3. LatentWire cached packet decode;
4. LatentWire end-to-end source scoring plus packet decode;
5. C2C or faithful cache-fusion proxy;
6. KVComm or faithful selective-KV proxy;
7. QJL/TurboQuant/KIVI/KVQuant source-state floors or native implementations.

Required metrics: accuracy with paired CI, TTFT p50/p95, TPOT p50/p95, ITL,
prefill/decode time, request goodput at SLO, generated tokens/s, peak GPU
memory, HBM read/write bytes/request, PCIe/NVLink bytes/request, payload bytes,
transferred source-state bytes, and source-exposure flags.
