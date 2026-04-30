# PQ Transport + Receiver Waterfall

- pass gate: `True`
- PQ record bytes: `7`
- PQ transport batch64 p95: `0.66 ns/request`
- PQ receiver batch64 p50: `0.01628 ms/request`
- transport share of receiver p50: `0.000041`
- query-aware text record ratio vs PQ: `2.00x`
- full-log transport p50 ratio vs PQ: `8.34x`
- KV-floor transport p50 ratio vs PQ: `622.05x`

## Checks

| Check | Pass | Value |
|---|---:|---|
| `transport_gate_passes` | `True` | True |
| `receiver_gate_passes` | `True` | True |
| `pq_transport_under_1us` | `True` | 0.66 ns/request |
| `pq_receiver_under_0p25ms` | `True` | batch p95 0.01787 ms; resident p95 0.01883 ms |
| `pq_receiver_exact` | `True` | 0 |
| `query_text_larger_and_exposes_text` | `True` | 2.00x bytes; exposes text=True |
| `full_log_transport_slower` | `True` | 8.34x PQ p50 transport |
| `kv_floor_transport_slower` | `True` | 622.05x PQ p50 transport |

## Rows

| Component | Profile | Batch | Record B | Line B/req | DMA B/req | p50 ns/req | p95 ns/req | p50 ms/req | p95 ms/req | Exposure | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `packet` | `packet_2b_payload_5b_record` | 64 | 5 | 5 | 6 | 0.648052 | 0.687764 | 6.48052e-07 | 6.87764e-07 | source-private | legacy 2B diagnostic packet record |
| `transport` | `pq_packet_4b_payload_7b_record` | 64 | 7 | 7 | 8 | 0.643269 | 0.660897 | 6.43269e-07 | 6.60897e-07 | source-private | 4B PQ payload plus 3B record overhead |
| `transport` | `query_aware_text_14b` | 64 | 14 | 14 | 14 | 0.672894 | 0.691878 | 6.72894e-07 | 6.91878e-07 | private text | query-aware private text comparator |
| `transport` | `full_hidden_log_370b` | 64 | 370 | 370 | 370 | 5.36274 | 5.65224 | 5.36274e-06 | 5.65224e-06 | private text | full private-log relay comparator |
| `transport` | `qjl_1bit_kv_floor_21504b` | 64 | 21504 | 21504 | 21504 | 400.146 | 404.175 | 0.000400146 | 0.000404175 | source KV | 1-bit KV byte-floor comparator |
| `transport` | `kivi_2bit_kv_floor_43008b` | 64 | 43008 | 43008 | 43008 | 1164.67 | 1169.56 | 0.00116467 | 0.00116956 | source KV | 2-bit KV byte-floor comparator |
| `receiver` | `pq_resident_table_decode_max` | 1 | 7 |  |  |  |  | 0.016667 | 0.018833 | source-private | max resident lookup over all remap/variant rows |
| `receiver` | `pq_batch64_decode_max` | 64 | 7 |  |  |  |  | 0.0162751 | 0.0178717 | source-private | max batched decode over all remap/variant rows |

## Interpretation

This artifact joins measured packet-ring transport with the PQ resident receiver microbench. It supports a Mac-local boundary-traffic plus receiver-kernel claim for 7-byte PQ packet records, not a production GPU serving or protocol-free latent-reasoning claim.
