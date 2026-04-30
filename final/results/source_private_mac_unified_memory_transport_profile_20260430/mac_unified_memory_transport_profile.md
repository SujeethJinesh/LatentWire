# Mac Unified-Memory Transport Profile

This artifact profiles the existing `n=160` core and held-out Mac endpoint rows as
source-target transport objects: raw bytes, cache-line rounding, DMA-burst rounding,
batch-packed packet records, prompt/KV byte floors, and private-state exposure.

## Headline

- pass gate: `True`
- exact ID parity: `True`
- packet payload bytes: `[2.0]`
- matched packet min delta vs target: `0.425`
- max source-destroying control delta vs target: `0.000`
- query-aware text raw ratio: `7.00x`
- query-aware text 64B-line ratio: `1.00x`
- full-log raw ratio: `183.25x`
- full-log 64B-line ratio: `6.00x`
- full-log QJL 1-bit prompt-KV delta / packet byte: `313353.6x`
- packet batch-64 packed line bytes/request: `5.00`
- packet batch-64 packed DMA bytes/request: `6.00`

## Host Profile

- mac_model: `MacBookPro18,2`
- cpu_brand: `Apple M1 Max`
- machine: `arm64`
- memory_gib: `64.0`
- platform: `macOS-26.4.1-arm64-arm-64bit`
- execution note: Profile generation is CPU-only deterministic artifact accounting over existing endpoint rows; it does not run an MPS generation job.

## Rows

| Surface | Condition | Acc | Payload B | 64B line B | 128B DMA B | Prompt delta | QJL KV delta B | TTFT p50 | Exposure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| core n160 label_strict | `target_only` | 0.250 | 0.0 | 0.0 | 0.0 | 0.00 | 0.0 | 482.7 | none |
| core n160 label_strict | `matched_packet` | 0.675 | 2.0 | 64.0 | 128.0 | 0.00 | 0.0 | 471.6 | packet |
| core n160 label_strict | `matched_byte_text_2` | 0.250 | 2.0 | 64.0 | 128.0 | 0.00 | 0.0 | 473.0 | text |
| core n160 label_strict | `random_same_byte_packet` | 0.000 | 2.0 | 64.0 | 128.0 | 0.00 | 0.0 | 459.3 | none |
| core n160 label_strict | `deranged_candidate_diag_table` | 0.244 | 2.0 | 64.0 | 128.0 | 0.00 | 0.0 | 465.9 | none |
| core n160 label_strict | `query_aware_diag_span` | 0.694 | 14.0 | 64.0 | 128.0 | 3.31 | 23699.2 | 452.0 | text |
| core n160 label_strict | `structured_json_diag` | 0.575 | 21.0 | 64.0 | 128.0 | 5.00 | 35840.0 | 478.6 | text |
| core n160 label_strict | `structured_free_text_diag` | 0.713 | 17.0 | 64.0 | 128.0 | 3.00 | 21504.0 | 474.2 | text |
| core n160 label_strict | `full_hidden_log` | 0.463 | 366.5 | 384.0 | 384.0 | 87.43 | 626707.2 | 635.9 | text |
| holdout n160 label_strict | `target_only` | 0.250 | 0.0 | 0.0 | 0.0 | 0.00 | 0.0 | 556.3 | none |
| holdout n160 label_strict | `matched_packet` | 0.688 | 2.0 | 64.0 | 128.0 | 0.00 | 0.0 | 547.2 | packet |
| holdout n160 label_strict | `matched_byte_text_2` | 0.250 | 2.0 | 64.0 | 128.0 | 0.00 | 0.0 | 545.2 | text |
| holdout n160 label_strict | `random_same_byte_packet` | 0.000 | 2.0 | 64.0 | 128.0 | 0.00 | 0.0 | 547.1 | none |
| holdout n160 label_strict | `deranged_candidate_diag_table` | 0.244 | 2.0 | 64.0 | 128.0 | 0.00 | 0.0 | 549.4 | none |
| holdout n160 label_strict | `query_aware_diag_span` | 0.688 | 14.0 | 64.0 | 128.0 | 3.31 | 23699.2 | 558.9 | text |
| holdout n160 label_strict | `structured_json_diag` | 0.594 | 21.0 | 64.0 | 128.0 | 5.00 | 35840.0 | 552.5 | text |
| holdout n160 label_strict | `structured_free_text_diag` | 0.719 | 17.0 | 64.0 | 128.0 | 3.00 | 21504.0 | 557.7 | text |
| holdout n160 label_strict | `full_hidden_log` | 0.531 | 373.5 | 384.0 | 384.0 | 90.18 | 646419.2 | 730.8 | text |

## Interpretation

Mac unified-memory transport profile over existing CPU endpoint rows. It separates raw packet bytes, 64B line rounding, 128B DMA-burst rounding, batch-64 packet packing, prompt/KV byte floors, TTFT proxy telemetry, and private source-state exposure.

The important boundary is not just raw byte count. A 2-byte packet still rounds
to one transfer quantum for a single request. The win is that the packet avoids
private source text and source KV/cache movement, and it becomes materially tiny
when packed across a batch. Short query-aware text can tie the packet at one 64B
line, but it is a private-text relay and uses 7x raw semantic payload.

## Non-Claims

- This is not a native MPS/GPU kernel benchmark and not a production serving benchmark.
- Unified-memory line/DMA values are deterministic accounting proxies, not measured hardware counters.
- KV rows are prompt-token byte floors under published quantization-style bit widths, not a KVComm implementation.
- Short query-aware text can tie a single packet at one 64B line while still exposing private text.
