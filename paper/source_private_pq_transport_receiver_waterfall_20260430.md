# Source-Private PQ Transport + Receiver Waterfall

Date: 2026-04-30

## Status

Current readiness: COLM workshop ready with a clear systems/evaluation story;
not comfortably ICLR-full until a stronger non-hand-decoded receiver or native
GPU serving profile lands.

Current paper story: a source can send a tiny source-private residual packet
instead of exposing source text or source KV/cache. The target combines that
packet with public candidate side information, decodes it exactly with a small
resident receiver, and the artifact accounts for boundary bytes, private-state
exposure, and strict source-destroying controls.

Exact blocker addressed here: the previous PQ receiver batch microbench proved
that the receiver lookup is exact and cheap, but it did not join receiver cost
with measured packet transport. This gate extends the Mac packet-ring transport
profile to 7-byte PQ records and 14-byte query-aware text records, then joins
transport and receiver timing in one reviewer-facing waterfall.

Layman explanation: imagine the source model is not allowed to send its notes.
It can only send a tiny receipt-like code. This experiment measures two things:
how cheaply that tiny code can be moved, and how cheaply the target can use the
code to choose the right answer. We compare it to sending private text or KV
cache state, which are larger and expose more of the source model.

## Gate

- transport script:
  `scripts/build_source_private_mac_packet_ring_transport_microbench.py`
- transport C kernel:
  `scripts/source_private_packet_ring_transport_microbench.c`
- join script:
  `scripts/build_source_private_pq_transport_receiver_waterfall.py`
- tests:
  `tests/test_build_source_private_mac_packet_ring_transport_microbench.py`
  and `tests/test_build_source_private_pq_transport_receiver_waterfall.py`
- transport artifact:
  `results/source_private_mac_packet_ring_transport_microbench_pq7_20260430/`
- waterfall artifact:
  `results/source_private_pq_transport_receiver_waterfall_20260430/`
- receiver input artifact:
  `results/source_private_pq_receiver_batch_microbench_20260430/`
- references:
  `references/541_pq_transport_receiver_waterfall_refs_20260430.md`

## Pass Rule

The joined waterfall passes only if:

- the transport microbench passes,
- the PQ receiver batch microbench passes,
- PQ packet batch64 p95 transport is below `1 us/request`,
- receiver batch/resident p95 decode is below `0.25 ms/request`,
- receiver mismatches remain `0`,
- query-aware text uses at least `2x` PQ record bytes and exposes private text,
- full-log transport is at least `5x` slower than PQ transport,
- and the KV byte-floor transport row is at least `100x` slower than PQ
  transport.

## Headline Results

The joined waterfall passes.

- PQ packet record: `7` bytes/request line traffic and `8` bytes/request DMA
  traffic at batch64
- PQ transport p95: `0.66 ns/request`
- PQ transport batch64 CV: `0.0121`
- PQ receiver batch64 p50: `0.01628 ms/request`
- PQ receiver resident p50: `0.01667 ms/request`
- receiver mismatch count: `0`
- query-aware private text: `2.00x` PQ record bytes and exposes private text
- full-log transport p50: `8.34x` PQ transport p50
- QJL-style 1-bit KV byte floor p50: `622.05x` PQ transport p50
- transport share of receiver p50: `0.0041%`

Representative waterfall rows:

| Component | Profile | Batch | Record B | Line B/req | DMA B/req | p50 ns/req | p95 ns/req | Exposure |
|---|---|---:|---:|---:|---:|---:|---:|---|
| packet | legacy 2B diagnostic packet | 64 | 5 | 5 | 6 | 0.648 | 0.688 | source-private |
| transport | 4B PQ packet + 3B record | 64 | 7 | 7 | 8 | 0.643 | 0.661 | source-private |
| transport | query-aware text | 64 | 14 | 14 | 14 | 0.673 | 0.692 | private text |
| transport | full hidden log | 64 | 370 | 370 | 370 | 5.363 | 5.652 | private text |
| transport | QJL 1-bit KV byte floor | 64 | 21504 | 21504 | 21504 | 400.146 | 404.175 | source KV |
| receiver | max PQ batch64 decode | 64 | 7 | n/a | n/a | n/a | n/a | source-private |

The query-aware text row can copy at a similar nanosecond scale in this
microbench, so the claim is not "text is always slower." The claim is that
the text comparator exposes private text and uses `2x` the PQ record bytes,
while the packet row stays source-private and remains exact under the receiver
gate.

## Interpretation

Promote:

- This is the strongest Mac-local systems artifact so far: packet transport,
  receiver decode, private-state exposure, and byte-floor comparators are in
  one reproducible waterfall.
- The measured 7-byte PQ record path is exact at the receiver and tiny at the
  boundary. It now has a concrete systems story rather than just static byte
  accounting.
- The result supports a source-private boundary-traffic contribution that is
  meaningfully distinct from KV/cache communication methods, which communicate
  source internal state at much higher byte counts.

Do not overclaim:

- This is not a native vLLM or NVIDIA GPU serving result.
- This is not TTFT, TPOT, goodput, HBM, PCIe, or NVLink telemetry.
- This is not protocol-free cross-model latent reasoning.
- The receiver remains a deterministic public-table decoder, not a learned
  model-mediated receiver.

## Readiness Impact

The paper now has at least three defensible technical contributions:

1. a strict source-private packet/control protocol that separates matched
   source evidence from target-only and source-destroying controls,
2. a frozen target-verifier path showing packet consumption can survive a model
   verifier under strict controls,
3. geometry-mitigated PQ residual packets with n500 source-causal lift and
   lookup-risk diagnostics,
4. a systems contribution: 7-byte packet transport plus exact batched receiver
   waterfall with private-text/KV exposure accounting.

COLM workshop: strong enough as a scoped source-private packet communication
paper if we keep the claims precise.

ICLR full: still needs one major gate:

- a learned or frozen model-mediated receiver that uses the packet beyond
  hand-decoded public-table lookup,
- or native NVIDIA/vLLM/KV serving telemetry showing real TTFT/TPOT/goodput
  wins,
- or a larger and less synthetic cross-family benchmark where the packet still
  beats text/KV baselines under paired uncertainty.

## Next Exact Gate

Mac-local method gate: implement a learned score adapter / control-regularized
innovation receiver with deranged public-table controls. It should start at
cheap n256/n500 probes and only scale if matched packets beat target-only,
random-packet, shuffled-source, and deranged-public-handle controls.

Systems gate when NVIDIA is available: reproduce the PQ packet receiver path in
a serving loop and report TTFT, TPOT, goodput, HBM bytes, and interconnect
traffic against text relay, C2C/KVComm-style cache sharing, and KV compression
byte floors.

## Tests

```bash
./venv_arm64/bin/python -m py_compile \
  scripts/build_source_private_mac_packet_ring_transport_microbench.py \
  scripts/build_source_private_pq_transport_receiver_waterfall.py

./venv_arm64/bin/python -m pytest \
  tests/test_build_source_private_mac_packet_ring_transport_microbench.py \
  tests/test_build_source_private_pq_transport_receiver_waterfall.py -q
```

Outcome: `2 passed in 0.03s`.
