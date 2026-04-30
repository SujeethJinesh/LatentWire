# Source-Private PQ Receiver Batch Microbench

- pass gate: `True`
- rows: `18`
- pass rows: `18`
- max resident table p50 ms: `0.0167`
- max batch-64 per-request p50 ms: `0.0163`
- min batch-64 speedup vs scalar table p50: `0.9838x`
- raw payload bytes/request: `4`
- packet record bytes/request: `7`
- batch-256 amortized 128B raw payload bytes/request: `4.00`
- batch-256 amortized 128B packet record bytes/request: `7.00`
- max table mismatches: `0`
- max batch mismatches: `0`

## Rows

| Remap | Variant | Acc | Table p50 ms | Batch64 p50 ms | Batch64 speedup | Batch256 record 128B bytes/req | Table bytes | Rotation bytes | Mismatch | Invariant | Pass |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | canonical | 0.482 | 0.01662 | 0.01623 | 1.02x | 7.00 | 16384 | 0 | 0/0 | `True` | `True` |
| 101 | utility_balanced | 0.494 | 0.01650 | 0.01623 | 1.02x | 7.00 | 16384 | 0 | 0/0 | `True` | `True` |
| 101 | opq_procrustes | 0.498 | 0.01650 | 0.01608 | 1.03x | 7.00 | 16384 | 1048576 | 0/0 | `True` | `True` |
| 101 | utility_opq_procrustes | 0.480 | 0.01667 | 0.01622 | 1.03x | 7.00 | 16384 | 1048576 | 0/0 | `True` | `True` |
| 101 | protected_hadamard | 0.498 | 0.01667 | 0.01612 | 1.03x | 7.00 | 16384 | 1048576 | 0/0 | `True` | `True` |
| 101 | utility_protected_hadamard | 0.504 | 0.01642 | 0.01612 | 1.02x | 7.00 | 16384 | 1048576 | 0/0 | `True` | `True` |
| 103 | canonical | 0.508 | 0.01608 | 0.01602 | 1.00x | 7.00 | 16384 | 0 | 0/0 | `True` | `True` |
| 103 | utility_balanced | 0.502 | 0.01563 | 0.01582 | 0.99x | 7.00 | 16384 | 0 | 0/0 | `True` | `True` |
| 103 | opq_procrustes | 0.498 | 0.01583 | 0.01595 | 0.99x | 7.00 | 16384 | 1048576 | 0/0 | `True` | `True` |
| 103 | utility_opq_procrustes | 0.504 | 0.01600 | 0.01614 | 0.99x | 7.00 | 16384 | 1048576 | 0/0 | `True` | `True` |
| 103 | protected_hadamard | 0.514 | 0.01646 | 0.01613 | 1.02x | 7.00 | 16384 | 1048576 | 0/0 | `True` | `True` |
| 103 | utility_protected_hadamard | 0.504 | 0.01625 | 0.01611 | 1.01x | 7.00 | 16384 | 1048576 | 0/0 | `True` | `True` |
| 107 | canonical | 0.520 | 0.01650 | 0.01620 | 1.02x | 7.00 | 16384 | 0 | 0/0 | `True` | `True` |
| 107 | utility_balanced | 0.516 | 0.01596 | 0.01622 | 0.98x | 7.00 | 16384 | 0 | 0/0 | `True` | `True` |
| 107 | opq_procrustes | 0.502 | 0.01650 | 0.01618 | 1.02x | 7.00 | 16384 | 1048576 | 0/0 | `True` | `True` |
| 107 | utility_opq_procrustes | 0.514 | 0.01621 | 0.01616 | 1.00x | 7.00 | 16384 | 1048576 | 0/0 | `True` | `True` |
| 107 | protected_hadamard | 0.512 | 0.01654 | 0.01604 | 1.03x | 7.00 | 16384 | 1048576 | 0/0 | `True` | `True` |
| 107 | utility_protected_hadamard | 0.516 | 0.01642 | 0.01628 | 1.01x | 7.00 | 16384 | 1048576 | 0/0 | `True` | `True` |

## Interpretation

This is a receiver-kernel systems gate for geometry-mitigated source-private PQ packets. It measures the target-side operation after public candidate state has been cached: summing PQ distance-table entries for a few source-private byte indices. It supports a boundary-traffic and batching claim, not an end-to-end vLLM/GPU serving claim.

## Pass Rule

Every remap/variant row must exactly match the canonical geometry decoder under resident table lookup and all batch kernels; predictions must be invariant across batch sizes; resident table p50 and every batch p95 must stay below 0.25 ms/request; and the largest batch must amortize 128B packet-record traffic down to the packet record byte count per request. Codebook fit and public table build costs are reported separately and are not claimed as per-token model speedups.
