# Source-Private Packet Trace Card v2

- pass gate: `True`
- checklist: `7/7`
- packet raw bytes min: `2.00`
- single-request cache-line bytes: `64.00`
- batch-64 line bytes/request: `5.00`
- batch-64 DMA bytes/request: `6.00`
- query-aware text raw ratio: `7.00x`
- query-aware text cache-line ratio: `1.00x`
- full-log raw ratio min: `183.25x`
- KV raw ratio min: `10752.00x`
- Qwen receiver pass rows: `2/2`
- Qwen CPU p50 latency max: `915.66 ms`

## Claim Checklist

| Check | Pass | Value | Reviewer risk reduced |
|---|---:|---|---|
| `strict_source_controls` | `True` | endpoint 2/2; qwen 2/2 | packet lift is not from zero/shuffled/random/text/deranged controls |
| `same_byte_text_negative` | `True` | 0.250 | same-byte text relay does not explain packet accuracy |
| `query_aware_text_raw_gap` | `True` | 7.0x raw bytes; 1.0x cache-line bytes | strongest visible-text relay needs more semantic payload and exposes private text |
| `full_log_transport_gap` | `True` | 183.2x raw; 6.0x line; +164.27 ms TTFT proxy | full private text relay is not the same operating point |
| `kv_byte_floor_gap` | `True` | 10752x raw; 336x line | KV/cache movement is a different and much larger transport object |
| `batch_amortization` | `True` | 5.00 line B/request; 6.00 DMA B/request at batch 64 | the one-line single-request floor can be amortized by packed packet records |
| `production_overclaim_guard` | `True` | This is not measured accelerator throughput.; It assumes ideal contiguous packing of packet records inside a batch.; It does not model receiver compute or dictionary/cache locality beyond packet bytes. | Mac trace card does not pretend to be NVIDIA/HBM throughput |

## Trace Rows

| Group | Method | Raw B | Line B | Batch64 line B/req | Accuracy | Exposure | Scope |
|---|---|---:|---:|---:|---:|---|---|
| `endpoint_packet` | LatentWire endpoint packet | 2.00 | 64.00 | 5.00 | 0.675 | source-private | mac_endpoint_proxy |
| `endpoint_packet` | LatentWire endpoint packet | 2.00 | 64.00 | 5.00 | 0.688 | source-private | mac_endpoint_proxy |
| `endpoint_text_relay` | query-aware diagnostic text | 14.00 | 64.00 |  |  | private text | text_relay_comparator |
| `endpoint_text_relay` | full hidden-log relay | 366.50 | 384.00 |  |  | private text | text_relay_comparator |
| `endpoint_text_relay` | query-aware diagnostic text | 14.00 | 64.00 |  |  | private text | text_relay_comparator |
| `endpoint_text_relay` | full hidden-log relay | 373.50 | 384.00 |  |  | private text | text_relay_comparator |
| `semantic_anchor_medium` | semantic-anchor source-private packet | 4.00 | 64.00 | 7.00 |  | source-private | medium_confirmation_accounting |
| `semantic_anchor_medium` | semantic-anchor source-private packet | 8.00 | 64.00 | 11.00 |  | source-private | medium_confirmation_accounting |
| `kv_byte_floor` | QJL-style 1-bit source KV byte floor | 21504.00 | 21504.00 |  |  | source KV | kv_cache_lower_bound |
| `kv_byte_floor` | KIVI/KVQuant-style 2-bit source KV byte floor | 43008.00 | 43008.00 |  |  | source KV | kv_cache_lower_bound |

## Allowed Claim

LatentWire packets are a source-private, byte-scale side-information interface with strict controls. They are smaller than visible text relays in raw payload, avoid private text and source KV exposure, and amortize below one cache line per request when packed across batches.

## Non-Claims

- This does not prove production GPU serving throughput.
- This does not beat KV compression on native KV-cache tasks.
- This does not make the exact-table receiver a protocol-free latent-transfer method.
- Qwen CPU verifier latency is model-consumption evidence, not a systems win.
