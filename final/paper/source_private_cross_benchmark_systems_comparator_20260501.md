# Cross-Benchmark Systems Comparator

- pass gate: `True`
- headline-eligible benchmarks: `2`
- diagnostic benchmarks: `1`
- framed packet bytes: `5-15B`
- min QJL 1-bit one-token KV floor vs framed packet: `51.2x`
- min QJL 30%-layer one-token KV floor vs framed packet: `15.4x`
- min KVComm30 fp16 one-token KV floor vs framed packet: `245.8x`
- min TurboQuant 3.5-bit one-token KV floor vs framed packet: `179.2x`
- native systems complete: `False`

## Benchmark Rows

| Dataset | Role | Seeds | Packet | Accuracy | Target | Text | QJL 1-bit floor | HellaSwag label-copy threat |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ARC-Challenge | headline_public_benchmark | 5/5 | 12B raw / 15B framed | 0.344 | 0.265 | 0.311 | 51.2x | `False` |
| OpenBookQA | headline_second_public_benchmark | 5/5 | 3B raw / 6B framed | 0.378 | 0.276 | 0.350 | 128.0x | `False` |
| HellaSwag | diagnostic_not_headline_label_copy_threat | 5/5 | 2B raw / 5B framed | 0.461 | 0.233 | 0.386 | 153.6x | `True` |

## Checks

| Check | Pass | Value |
|---|---:|---|
| `two_public_headline_benchmarks_eligible` | `True` | `2` |
| `all_packet_rows_source_private_boundary` | `True` | `no source text or source KV exposed` |
| `one_token_qjl_floor_at_least_50x_framed_packet` | `True` | `51.2` |
| `kvcomm30_qjl_floor_at_least_15x_framed_packet` | `True` | `15.359999999999998` |
| `hellaswag_label_copy_threat_marked_not_headline` | `True` | `[('hellaswag_validation1024_2b', True, False)]` |
| `native_baseline_non_claims_explicit` | `True` | `native C2C/KVComm/QJL/TurboQuant wins forbidden` |

## External Baselines

| Method | Source | Communicated object | Local status | Claim boundary |
|---|---|---|---|---|
| C2C cache-to-cache communication | https://arxiv.org/abs/2510.03215 | projected/fused source KV cache | not_run_native | closest semantic-cache baseline; byte floors are not a native C2C result |
| KVComm selective KV sharing | https://arxiv.org/abs/2510.03346 | selected source KV layers/pairs | not_run_native | 30% layer fraction is used only as an assumption row unless rerun locally |
| KVCOMM cross-context KV-cache communication | https://arxiv.org/abs/2510.12872 | aligned/reused KV caches for multi-agent prefill | not_run_native | systems neighbor, not same threat model as source-private public packets |
| QJL 1-bit KV sketch | https://arxiv.org/abs/2406.03482 | 1-bit JL/sign sketch of KV/cache state | byte_floor_only | mathematical byte-floor comparator, not a defeated native baseline |
| TurboQuant online vector quantization | https://arxiv.org/abs/2504.19874 | low-bit vector/KV state | byte_floor_only | quantization inspiration and byte-floor proxy only |
| vLLM/PagedAttention | https://arxiv.org/abs/2309.06180 | paged KV-cache serving substrate | pending_nvidia | native serving target; Mac rows cannot close this gate |

## Claim Boundary

Cross-benchmark byte/state-exposure comparator for source-private packets. KV/cache rows are one-source-token byte floors from the local Qwen2.5-0.5B config, not native C2C/KVComm/QJL/TurboQuant quality or throughput measurements.

This artifact strengthens the systems story by making the source-state exposure cost explicit. It does not close the NVIDIA/vLLM native systems blocker.
