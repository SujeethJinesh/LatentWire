# Mac Packet-Ring Transport Microbench

- pass gate: `True`
- repeats: `5`
- target bytes/repeat: `134217728`
- packet batch64 p50 ns/request: `0.65`
- packet batch64 p95 ns/request: `0.69`
- packet batch64 line bytes/request: `5.00`
- packet batch64 DMA bytes/request: `6.00`
- PQ packet batch64 p50 ns/request: `0.64`
- PQ packet batch64 p95 ns/request: `0.66`
- PQ packet batch64 line bytes/request: `7.00`
- PQ packet batch64 DMA bytes/request: `8.00`
- query-text p95 ratio vs packet at batch64: `1.01x`
- query-text p95 ratio vs PQ packet at batch64: `1.05x`
- full-log p50 ratio vs packet at batch64: `8.28x`
- KV-floor p50 ratio vs packet at batch64: `617.46x`

## Checks

| Check | Pass | Value |
|---|---:|---|
| `packet_batch64_p95_under_1us` | `True` | 0.69 ns/request |
| `packet_batch64_line_bytes` | `True` | 5.00 line B/request; 6.00 DMA B/request |
| `pq_packet_batch64_p95_under_1us` | `True` | 0.66 ns/request |
| `pq_packet_batch64_line_bytes` | `True` | 7.00 line B/request; 8.00 DMA B/request |
| `query_text_not_much_faster` | `True` | 1.01x packet p95 at batch64 |
| `full_log_measured_slower` | `True` | 8.28x packet p50 at batch64 |
| `kv_floor_measured_slower` | `True` | 617.46x packet p50 at batch64 |
| `packet_repeat_stability` | `True` | batch64 packet CV 0.0252; batch64 PQ CV 0.0121; diagnostic max packet CV 0.0252 |

## Batch-64 Rows

| Profile | Record B | Line B/req | DMA B/req | p50 ns/req | p95 ns/req | p95 ratio vs packet | Exposure |
|---|---:|---:|---:|---:|---:|---:|---|
| `packet_2b_payload_5b_record` | 5 | 5.00 | 6.00 | 0.65 | 0.69 | 1.00 | source-private |
| `pq_packet_4b_payload_7b_record` | 7 | 7.00 | 8.00 | 0.64 | 0.66 | 0.96 | source-private |
| `query_aware_text_14b` | 14 | 14.00 | 14.00 | 0.67 | 0.69 | 1.01 | private text |
| `full_hidden_log_370b` | 370 | 370.00 | 370.00 | 5.36 | 5.65 | 8.22 | private text |
| `qjl_1bit_kv_floor_21504b` | 21504 | 21504.00 | 21504.00 | 400.15 | 404.17 | 587.66 | source KV |
| `kivi_2bit_kv_floor_43008b` | 43008 | 43008.00 | 43008.00 | 1164.67 | 1169.56 | 1700.52 | source KV |

Local Mac packet-ring microbenchmark for contiguous pack-copy-verify transport. It measures boundary movement for tiny packet records, query-aware private text, full private logs, and KV byte-floor buffers across batch sizes. It is not GPU serving throughput.
