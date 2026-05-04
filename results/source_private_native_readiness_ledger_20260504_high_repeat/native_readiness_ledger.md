# Source-Private Native Readiness Ledger

- native ready: `False`
- local measured rows: `3`
- pending native rows: `5`
- blocker: NVIDIA/vLLM/SGLang run is still required for systems claims.

## Readiness Rows

| Row | Evidence | Source private | Source KV exposed | Accuracy | Native status | Claim allowed |
|---|---|---:|---:|---:|---|---|
| LatentWire train-donor anti-shuffle packet (12-14B frontier) | measured_mac_same_slice | true | false | 0.652344 | mac_accuracy_only | source-private cross-family packet accuracy and byte-boundary evidence |
| LatentWire packet-ring packed-record microbench | measured_mac_packet_ring | true | false |  | mac_transport_proxy | Mac-local packed-record transport sanity check |
| LatentWire Mac endpoint packet proxy | measured_mac_same_slice | true | false | 0.675 | mac_endpoint_proxy | Mac TTFT proxy and source-exposure accounting |
| C2C cache-to-cache communication | external_reference_only | false | true |  | pending_native_required | related-work native baseline to run |
| KVComm/KVCOMM selective KV communication | external_reference_only | false | true |  | pending_native_required | related-work systems comparator |
| TurboQuant-style low-bit KV cache | external_reference_only | false | true |  | pending_native_required | quantization substrate inspiration and byte-floor comparator |
| QJL-style quantized Johnson-Lindenstrauss sketch | external_reference_only | false | true |  | pending_native_required | mathematical systems inspiration and byte-floor comparator |
| vLLM/PagedAttention serving substrate | external_reference_only | false | true |  | pending_native_required | native serving target for the next experiment phase |

## Interpretation

This ledger makes the systems boundary explicit. The paper may claim Mac-local source-private packet accuracy, byte accounting, and packed-record transport proxy evidence. It may not claim native NVIDIA/vLLM throughput, HBM traffic, or wins over C2C, KVComm, TurboQuant, QJL, or vLLM until those rows are measured in their native serving setting.

## Non-Claims

- No native GPU throughput claim.
- No HBM read/write or peak-memory claim.
- No claim that byte-floor KV/cache accounting beats native C2C/KVComm/TurboQuant.
- No claim that visible text relay is privacy-equivalent to source-private packets.
