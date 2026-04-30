# Source-Private Memory Traffic Ledger

- date: `2026-04-30`
- gate: `source_private_memory_traffic_ledger`
- artifact: `results/source_private_memory_traffic_ledger_20260430/`
- status: pass as a deterministic systems trace-card, not as measured
  accelerator throughput

## Question

Can the systems contribution be made reviewer-readable as boundary traffic:
what crosses the source-target interface, what private state is exposed, how
small packets round to hardware transfer quanta, and where TTFT evidence is only
a Mac-local proxy?

## Method

I added:

```bash
./venv_arm64/bin/python scripts/build_source_private_memory_traffic_ledger.py \
  --output-dir results/source_private_memory_traffic_ledger_20260430
```

The script joins:

- `results/source_private_systems_rate_assumption_frontier_20260430/`
- `results/source_private_hardware_packet_frontier_20260430/`
- `results/source_private_packet_isa_batch_frontier_20260430/`

It emits JSON/CSV/Markdown rows with raw bytes, single-request 64B cache-line
traffic, single-request 128B DMA-burst traffic, batch-64 packet amortization,
source-text/KV exposure, destructive-control status, and TTFT proxy deltas.

## Results

Headline values:

- packet raw payload minimum: `2` bytes
- packet single-request transfer: `64` line bytes and `128` DMA bytes
- packet batch-64 amortized traffic: `5.0` line bytes/request and `6.0` DMA
  bytes/request for the 2-byte payload plus header/parity
- query-aware diagnostic text: `7.0x` raw bytes but `1.0x` cache-line bytes
  versus a single packet request
- full hidden-log relay: at least `183.25x` raw bytes and `6.0x` cache-line
  bytes, with p50 TTFT delta at least `+164.27 ms`
- KV byte-floor rows: at least `10752.0x` raw bytes and `336.0x` cache-line
  bytes

## Interpretation

This strengthens the systems contribution by separating four claims:

1. Semantic payload: packets are far smaller than structured text, full private
   logs, or KV/cache-state movement.
2. Hardware transfer quantum: an isolated 2-byte packet still costs at least one
   line/burst, so query-aware text can tie at one 64B line.
3. Privacy/interface: query-aware text exposes private source text; packets do
   not expose private text or source KV/cache.
4. Batching: contiguous packet records can amortize the one-line floor to
   `5.0` line bytes/request at batch 64.

This is the right systems framing for the paper today. It is stronger and more
honest than saying "2 bytes beats everything" because it explicitly marks the
cache-line tie and keeps production throughput as a future NVIDIA/HBM gate.

## Decision

Promote this as a third systems contribution:

- source-private evidence-packet benchmark/method
- strict destructive-control protocol
- hardware-facing memory traffic ledger and packet-ISA trace card

Do not use it as a substitute for the learned receiver blocker. The method
frontier still needs a non-hand-coded target-preserving receiver, with the next
candidate being a JEPA/Q-Former-style query resampler trained with
source-destroying negatives.

## Next Gate

Implement `jepa_query_resampler` as a materially different learned receiver:

- smoke: `n=64`, budgets `4/8`, `K={4,8}`, one seed
- strict gate: `n=256`, seeds `47/53/59`, bidirectional held-out synonym plus
  same-family
- pass: matched accuracy at least target `+0.25`, all destructive controls
  within target `+0.03`, and query effective rank not collapsed
