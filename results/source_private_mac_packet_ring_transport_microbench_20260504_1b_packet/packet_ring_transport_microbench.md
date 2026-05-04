# Mac Packet-Ring Transport Microbench

- pass gate: `True`
- repeats: `5`
- target bytes/repeat: `67108864`
- packet batch64 p50 ns/request: `0.65`
- packet batch64 p95 ns/request: `0.66`
- packet batch64 line bytes/request: `4.00`
- packet batch64 DMA bytes/request: `4.00`
- legacy 2B packet batch64 p95 ns/request: `0.64`
- PQ packet batch64 p50 ns/request: `0.66`
- PQ packet batch64 p95 ns/request: `0.66`
- PQ packet batch64 line bytes/request: `7.00`
- PQ packet batch64 DMA bytes/request: `8.00`
- query-text p95 ratio vs packet at batch64: `1.02x`
- query-text p95 ratio vs PQ packet at batch64: `1.02x`
- full-log p50 ratio vs packet at batch64: `8.81x`
- KV-floor p50 ratio vs packet at batch64: `608.67x`

## Checks

| Check | Pass | Value |
|---|---:|---|
| `packet_batch64_p95_under_1us` | `True` | 0.66 ns/request |
| `packet_batch64_line_bytes` | `True` | 4.00 line B/request; 4.00 DMA B/request |
| `pq_packet_batch64_p95_under_1us` | `True` | 0.66 ns/request |
| `pq_packet_batch64_line_bytes` | `True` | 7.00 line B/request; 8.00 DMA B/request |
| `query_text_not_much_faster` | `True` | 1.02x packet p95 at batch64 |
| `full_log_measured_slower` | `True` | 8.81x packet p50 at batch64 |
| `kv_floor_measured_slower` | `True` | 608.67x packet p50 at batch64 |
| `packet_repeat_stability` | `True` | batch64 packet CV 0.0015; batch64 PQ CV 0.0029; diagnostic max packet CV 0.1392 |

## Batch-64 Rows

| Profile | Record B | Line B/req | DMA B/req | p50 ns/req | p95 ns/req | p95 ratio vs packet | Exposure |
|---|---:|---:|---:|---:|---:|---:|---|
| `packet_1b_payload_4b_record` | 4 | 4.00 | 4.00 | 0.65 | 0.66 | 1.00 | source-private |
| `packet_2b_payload_5b_record` | 5 | 5.00 | 6.00 | 0.64 | 0.64 | 0.98 | source-private |
| `pq_packet_4b_payload_7b_record` | 7 | 7.00 | 8.00 | 0.66 | 0.66 | 1.00 | source-private |
| `query_aware_text_14b` | 14 | 14.00 | 14.00 | 0.66 | 0.67 | 1.02 | private text |
| `full_hidden_log_370b` | 370 | 370.00 | 370.00 | 5.76 | 5.97 | 9.11 | private text |
| `qjl_1bit_kv_floor_21504b` | 21504 | 21504.00 | 21504.00 | 398.32 | 1000.24 | 1525.71 | source KV |
| `kivi_2bit_kv_floor_43008b` | 43008 | 43008.00 | 43008.00 | 1173.22 | 1197.88 | 1827.17 | source KV |

Local Mac packet-ring microbenchmark for contiguous pack-copy-verify transport. It measures boundary movement for tiny packet records, query-aware private text, full private logs, and KV byte-floor buffers across batch sizes. It is not GPU serving throughput.
