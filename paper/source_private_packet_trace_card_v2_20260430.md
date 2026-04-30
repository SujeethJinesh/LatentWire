# Source-Private Packet Trace Card v2 and Mac Transport Microbench

- date: `2026-04-30`
- gates:
  - `source_private_packet_trace_card_v2`
  - `source_private_mac_packet_ring_transport_microbench`
- artifacts:
  - `results/source_private_packet_trace_card_v2_20260430/`
  - `results/source_private_mac_packet_ring_transport_microbench_20260430/`
- status: pass as a systems/trace-card contribution; not a production GPU
  serving benchmark

## Cycle Start

1. Current ICLR readiness and distance: strong scoped positive-method paper,
   still not comfortable broad latent-transfer ICLR. The near-term ICLR path is
   a rigorous systems/benchmark paper plus a clearly bounded side-information
   method.
2. Current story: source-private packets communicate hidden evidence to a
   target with public side information; strict controls and deranged-table
   tests separate real packet use from target priors or public-only shortcuts.
3. Exact blocker: reviewers can still dismiss the method as a public-table
   protocol unless the systems contribution is precise and the claim boundaries
   are explicit.
4. Current live branch: packet systems trace card.
5. Highest-priority gate: packet trace-card v2 plus local measured packet-ring
   transport microbench.
6. Scale-up rung: systems/accounting confirmation on Mac, not GPU serving.

## Layman Version

A two-byte clue is not literally a two-byte hardware transaction: computers move
data in chunks such as cache lines and DMA bursts. This gate asks what really
crosses the source-target boundary: the tiny clue, private text, or huge KV
state. Then it times a local Mac copy-and-check loop for packed packet records
versus private text and KV-sized buffers.

## What Changed

I added:

- `scripts/build_source_private_packet_trace_card_v2.py`
- `scripts/source_private_packet_ring_transport_microbench.c`
- `scripts/build_source_private_mac_packet_ring_transport_microbench.py`
- tests for both builders.

The trace card joins the existing rate frontier, memory traffic ledger, packet
ISA batch frontier, and Qwen binary-verifier uncertainty artifact. The
microbench compiles a local C program and measures contiguous pack-copy-verify
for five record types at batch sizes `1, 4, 16, 64, 256`.

No SSH or remote accelerator was used.

## Trace Card Results

Artifact: `results/source_private_packet_trace_card_v2_20260430/`

Headline:

- pass gate: `True`
- checklist: `7/7`
- packet raw bytes min: `2.00`
- single-request cache-line bytes: `64.00`
- batch-64 line bytes/request: `5.00`
- batch-64 DMA bytes/request: `6.00`
- query-aware text raw ratio: `7.00x`
- query-aware text cache-line ratio: `1.00x`
- full-log raw ratio min: `183.25x`
- KV raw ratio min: `10752.00x`
- Qwen receiver pass rows: `2/2`
- Qwen CPU p50 latency max: `915.66 ms`

Claim checklist:

| Check | Pass | Value |
|---|---:|---|
| strict source controls | `True` | endpoint `2/2`; Qwen `2/2` |
| same-byte text negative | `True` | max same-byte text accuracy `0.250` |
| query-aware text raw gap | `True` | `7.0x` raw bytes, but `1.0x` cache-line bytes |
| full-log transport gap | `True` | `183.2x` raw, `6.0x` line, `+164.27 ms` TTFT proxy |
| KV byte-floor gap | `True` | `10752x` raw, `336x` line |
| batch amortization | `True` | `5.00` line B/request, `6.00` DMA B/request at batch 64 |
| production overclaim guard | `True` | explicitly not accelerator throughput |

## Mac Packet-Ring Microbench Results

Artifact: `results/source_private_mac_packet_ring_transport_microbench_20260430/`

Headline:

- pass gate: `True`
- repeats: `5`
- target bytes/repeat: `134217728`
- packet batch64 p50: `0.64 ns/request`
- packet batch64 p95: `0.65 ns/request`
- packet batch64 line bytes/request: `5.00`
- packet batch64 DMA bytes/request: `6.00`
- query-aware text p95 ratio vs packet: `1.02x`
- full hidden-log p50 ratio vs packet: `8.80x`
- QJL/KV floor p50 ratio vs packet: `671.23x`
- max packet repeat CV: `0.1354`

Batch-64 measured rows:

| Profile | Record B | Line B/req | DMA B/req | p50 ns/req | p95 ns/req | Exposure |
|---|---:|---:|---:|---:|---:|---|
| packet 2B payload / 5B record | 5 | 5.00 | 6.00 | 0.64 | 0.65 | source-private |
| query-aware text | 14 | 14.00 | 14.00 | 0.65 | 0.66 | private text |
| full hidden log | 370 | 370.00 | 370.00 | 5.66 | 5.78 | private text |
| QJL 1-bit KV floor | 21504 | 21504.00 | 21504.00 | 431.64 | 435.18 | source KV |
| KIVI 2-bit KV floor | 43008 | 43008.00 | 43008.00 | 1140.14 | 1149.17 | source KV |

## Interpretation

This gives the systems contribution a defensible shape:

- a single request is cache-line limited, so a 2-byte packet is not honestly a
  2-byte hardware transfer;
- packed packet records amortize to `5.00` line bytes/request and `6.00` DMA
  bytes/request at batch 64;
- query-aware text is close in local copy latency and ties the single cache-line
  quantum, but it exposes private source text and uses `7x` raw payload;
- full private logs and KV/cache floors are materially different transport
  objects and are much larger in both accounting and measured local copy cost.

This should be promoted as a systems-facing contribution:

> Source-private packets are a byte-scale boundary interface with explicit
> packet layout, batch amortization, privacy exposure accounting, strict
> source-destroying controls, and a local measured transport trace.

## Non-Claims

- This is not production NVIDIA/HBM/vLLM throughput.
- This does not beat KV compression on native KV-cache tasks.
- Qwen CPU verifier latency is model-consumption evidence, not a serving win.
- The exact-table receiver remains a public side-information method, not
  protocol-free latent transfer.

## Next Gate

For the systems side:

```text
source_private_nvidia_serving_trace_card_202605xx
```

when hardware is available: vLLM or equivalent, TTFT/TPOT/goodput/HBM/KV bytes,
packet/text/full-log rows, batch sweeps, and the same non-claim boundaries.

For the method side, do not tune masked consistency again. The only live learned
branches worth trying are:

- learned synonym-invariant sparse crosscoder packet;
- source-control-trained product-codebook / QJL residual packet.

## Tests

```bash
./venv_arm64/bin/python -m py_compile \
  scripts/build_source_private_packet_trace_card_v2.py \
  scripts/build_source_private_mac_packet_ring_transport_microbench.py

./venv_arm64/bin/python -m pytest \
  tests/test_build_source_private_packet_trace_card_v2.py \
  tests/test_build_source_private_mac_packet_ring_transport_microbench.py
```

Outcome: `2 passed in 0.03s`.
