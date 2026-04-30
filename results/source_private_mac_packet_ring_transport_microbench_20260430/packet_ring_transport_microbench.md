# Mac Packet-Ring Transport Microbench

- pass gate: `True`
- repeats: `5`
- target bytes/repeat: `134217728`
- packet batch64 p50 ns/request: `0.64`
- packet batch64 p95 ns/request: `0.65`
- packet batch64 line bytes/request: `5.00`
- packet batch64 DMA bytes/request: `6.00`
- query-text p95 ratio vs packet at batch64: `1.02x`
- full-log p50 ratio vs packet at batch64: `8.80x`
- KV-floor p50 ratio vs packet at batch64: `671.23x`

## Checks

| Check | Pass | Value |
|---|---:|---|
| `packet_batch64_p95_under_1us` | `True` | 0.65 ns/request |
| `packet_batch64_line_bytes` | `True` | 5.00 line B/request; 6.00 DMA B/request |
| `query_text_not_much_faster` | `True` | 1.02x packet p95 at batch64 |
| `full_log_measured_slower` | `True` | 8.80x packet p50 at batch64 |
| `kv_floor_measured_slower` | `True` | 671.23x packet p50 at batch64 |
| `packet_repeat_stability` | `True` | max packet CV 0.1354 |

## Batch-64 Rows

| Profile | Record B | Line B/req | DMA B/req | p50 ns/req | p95 ns/req | p95 ratio vs packet | Exposure |
|---|---:|---:|---:|---:|---:|---:|---|
| `packet_2b_payload_5b_record` | 5 | 5.00 | 6.00 | 0.64 | 0.65 | 1.00 | source-private |
| `query_aware_text_14b` | 14 | 14.00 | 14.00 | 0.65 | 0.66 | 1.02 | private text |
| `full_hidden_log_370b` | 370 | 370.00 | 370.00 | 5.66 | 5.78 | 8.91 | private text |
| `qjl_1bit_kv_floor_21504b` | 21504 | 21504.00 | 21504.00 | 431.64 | 435.18 | 670.94 | source KV |
| `kivi_2bit_kv_floor_43008b` | 43008 | 43008.00 | 43008.00 | 1140.14 | 1149.17 | 1771.74 | source KV |

Local Mac packet-ring microbenchmark for contiguous pack-copy-verify transport. It measures boundary movement for tiny packet records, query-aware private text, full private logs, and KV byte-floor buffers across batch sizes. It is not GPU serving throughput.
