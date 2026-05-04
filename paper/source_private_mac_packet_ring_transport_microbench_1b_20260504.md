# Mac Packet-Ring Transport Microbench 1B Packet

Date: 2026-05-04

## Status

- Current paper readiness: COLM systems positioning is stronger; ICLR systems
  claims still require native serving counters on NVIDIA.
- Current story: the current HellaSwag fixed-hybrid packet has a `1B` raw /
  `4B` framed source-private systems boundary, and local Mac transport shows
  the tiny record is below the measurement scale where text/KV state movement
  becomes expensive.
- Exact gap: this is a local transport microbenchmark, not model-serving TTFT,
  TPOT, HBM traffic, or energy.

## Gate

Artifact:
`results/source_private_mac_packet_ring_transport_microbench_20260504_1b_packet/`

Script:
`scripts/build_source_private_mac_packet_ring_transport_microbench.py`

Change:

- Added `packet_1b_payload_4b_record` to the C transport profiles.
- Kept the older `packet_2b_payload_5b_record` row for continuity.
- Uses the `1B/4B` packet as the reference packet for same-batch ratios.

## Result

The microbench passes.

| Metric | Value |
| --- | ---: |
| packet record bytes | `4` |
| batch64 line bytes/request | `4.00` |
| batch64 DMA bytes/request | `4.00` |
| packet batch64 p50 | `0.654399 ns/request` |
| packet batch64 p95 | `0.655591 ns/request` |
| legacy 2B/5B packet p95 | `0.644327 ns/request` |
| query-text p95 ratio vs packet | `1.02x` |
| full-log p50 ratio vs packet | `8.81x` |
| QJL 1-bit KV floor p50 ratio vs packet | `608.67x` |
| pass gate | `true` |

## Interpretation

This is a systems-boundary accounting result. It supports the claim that the
current packet has a sharply smaller communicated record than text logs or KV
state floors, and that batched transport for a `4B` record is below `1us` on
the local Mac microbenchmark.

It does not prove end-to-end model speed. Native systems claims still require
matched-quality vLLM/SGLang or equivalent runs with TTFT, TPOT, goodput,
memory, HBM/PCIe/NVLink bytes, and energy.

## Lay Explanation

This benchmark copies tiny packet records and larger text/KV-like records
through memory. The tiny packet is so small that it is essentially free at this
microbenchmark level, while KV-like state movement is much larger. That helps
the systems story, but it does not replace GPU serving measurements.
