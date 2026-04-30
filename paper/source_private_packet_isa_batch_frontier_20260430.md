# Source-Private Packet ISA Batch Frontier

- date: `2026-04-30`
- gate: `source_private_packet_isa_batch_frontier`
- artifact: `results/source_private_packet_isa_batch_frontier_20260430/`
- status: pass as a Mac-local hardware-accounting artifact

## Question

Does the packet systems claim survive a more hardware-realistic packet format:
headers, parity/check bytes, cache-line rounding, DMA-burst rounding, and batch
packing?

## Method

I added:

```bash
./venv_arm64/bin/python scripts/build_source_private_packet_isa_batch_frontier.py \
  --output-dir results/source_private_packet_isa_batch_frontier_20260430
```

The artifact uses the packet contract from
`results/source_private_hardware_packet_frontier_20260430/` and sweeps:

- payload bytes: `2, 4, 8, 16, 32`
- batch sizes: `1, 4, 16, 64`
- packet overhead: `2` header bytes + `1` parity/check byte
- transfer assumptions: 64B cache lines and 128B DMA bursts

## Results

Headline values:

- minimum packet bytes with overhead: `5`
- single request, 2B payload: `64` line bytes/request and `128` DMA bytes/request
- batch 64, 2B payload: `5.0` line bytes/request and `6.0` DMA bytes/request
- max 64B-line packing efficiency: `12.8x`
- max 128B-burst packing efficiency: `21.33x`
- 2B payload packet can pack `12` requests per 64B line under the chosen header
  and parity format

## Interpretation

This strengthens the systems contribution while making a real caveat explicit.
For a single request, the packet is line/burst limited. The value of an extreme
rate packet becomes clearer under batching: contiguous packet records amortize
the transfer quantum, while full logs and KV/cache tensors remain large.

This is a hardware-accounting artifact, not a measured accelerator benchmark.
It supports the paper's systems claim by turning the packet into a concrete ISA
record rather than an abstract byte count.

## Next Systems Gate

Add Mac-local TTFT decomposition or, once NVIDIA access is available, run a
serving benchmark with TTFT/TPOT/throughput/HBM counters for target-only, packet
relay, query-aware text, full-log relay, and one KV/cache comparator.
