# Source-Private Serving SLO Envelope

- pass gate: `True`
- rows: `10`
- TTFT proxy rows: `4`
- goodput claim rows: `0`
- packet min raw bytes: `2.00`
- packet min batch-64 line bytes/request: `5.00`
- packet min batch-64 DMA bytes/request: `6.00`
- packet min 500 ms TTFT margin: `-47.21 ms`

## Envelope Rows

| method | surface | private | text exposed | KV exposed | raw bytes | line B1 | line B64 | TTFT p50 | 500 ms margin | claim |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LatentWire endpoint packet | core n160 label_strict | True | False | False | 2.00 | 64.00 | 5 | 471.636 | 28.3643 | Mac endpoint TTFT proxy plus packet efficacy; production TPOT/goodput not claimed |
| LatentWire endpoint packet | holdout n160 label_strict | True | False | False | 2.00 | 64.00 | 5 | 547.21 | -47.2105 | Mac endpoint TTFT proxy plus packet efficacy; production TPOT/goodput not claimed |
| query-aware diagnostic text | core n160 label_strict | False | True | False | 14.00 | 64.00 |  |  |  | visible private-text byte/privacy comparator; no TTFT measurement |
| full hidden-log relay | core n160 label_strict | False | True | False | 366.50 | 384.00 |  | 635.909 | -135.909 | Mac endpoint TTFT proxy for full private text relay; production goodput not claimed |
| query-aware diagnostic text | holdout n160 label_strict | False | True | False | 14.00 | 64.00 |  |  |  | visible private-text byte/privacy comparator; no TTFT measurement |
| full hidden-log relay | holdout n160 label_strict | False | True | False | 373.50 | 384.00 |  | 730.755 | -230.755 | Mac endpoint TTFT proxy for full private text relay; production goodput not claimed |
| semantic-anchor source-private packet | heldout paraphrase n512 x 3 seeds | True | False | False | 4.00 | 64.00 | 7 |  |  | medium source-private rate/control evidence; no serving TTFT claim |
| semantic-anchor source-private packet | heldout paraphrase n512 x 3 seeds | True | False | False | 8.00 | 64.00 | 11 |  |  | medium source-private rate/control evidence; no serving TTFT claim |
| QJL-style 1-bit source KV byte floor | endpoint source context byte accounting | False | False | True | 21504.00 | 21504.00 |  |  |  | KV/cache byte-floor comparator only; native KV transport not run |
| KIVI/KVQuant-style 2-bit source KV byte floor | endpoint source context byte accounting | False | False | True | 43008.00 | 43008.00 |  |  |  | KV/cache byte-floor comparator only; native KV transport not run |

## Interpretation

This envelope translates the source-private packet systems evidence into serving vocabulary: what crosses the source-target boundary, which private state is exposed, how transfers round under single-request and batch-64 accounting, which TTFT proxy rows exist, and why TPOT/goodput remain explicit non-claims until native GPU serving is available.

## Non-Claims

- No row claims production GPU throughput, TPOT, or goodput.
- Mac endpoint TTFT is a proxy measurement, not an accelerator serving benchmark.
- KV/cache rows are lower-bound comparators and not a native KV transport implementation.
- Batch-64 packet rows assume contiguous packet-record packing.
