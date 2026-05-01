# Source-Private ARC-Challenge Systems Trace, 2026-05-01

## Status

- artifact:
  `results/source_private_arc_challenge_systems_trace_20260501/`
- code:
  `scripts/build_source_private_arc_challenge_systems_trace.py`
  and updated `scripts/run_source_private_arc_challenge_fixed_packet_gate.py`
- tests:
  `tests/test_build_source_private_arc_challenge_systems_trace.py`
  and `tests/test_run_source_private_arc_challenge_fixed_packet_gate.py`
- references:
  `references/571_arc_challenge_systems_trace_refs_20260501.md`

## What Changed

The ARC shared-basis validation/test artifacts were regenerated with a
Mac-local systems trace embedded in `systems_trace`. The trace records phase
timings, peak process RSS, candidate feature-cache footprint, payload bytes,
framed record bytes, and cacheline/DMA accounting.

The new systems card is also included in the ICLR evidence bundle.

## Main Evidence

Official ARC-Challenge test remains positive:

- matched/target/same-byte text: `0.344/0.265/0.311`;
- CI95 lower bound versus target: `0.044`;
- projection-seed stability: `5/5` test seeds.

Systems readout on the regenerated test artifact:

- source scoring: `251.0 ms/question` on local CPU Qwen2.5-0.5B;
- receiver sparse decode: `31.7/104.3 us` p50/p95;
- raw payload/framed record: `12B/15B`;
- single-request cacheline/DMA: `64B/128B`;
- batch-64 line/DMA per request: `15.0B/16.0B`;
- test candidate feature cache: `13.73 MiB` float64, `6.87 MiB` fp32 floor;
- peak process RSS: `7261.3 MiB`.

## Interpretation

This strengthens the systems side of the ARC contribution. The paper can now
show that the positive public endpoint is not only byte-bounded in principle:
the full official ARC test artifact records source scoring time, receiver-side
decode time, payload framing, cacheline/DMA accounting, and process RSS.

The right claim remains narrow. This is a Mac-local Python/NumPy/PyTorch trace,
not native vLLM throughput. It supports the source-private boundary-record
story and gives reviewers a concrete systems table, while still marking
NVIDIA/vLLM TTFT, TPOT, goodput, GPU memory, HBM traffic, C2C, KVComm,
TurboQuant, and QJL as pending native rows.

## Lay Explanation

We measured not just whether the tiny hint helps, but what it costs. The hint
is 12 bytes of useful payload. If we add a small header/check byte budget, it is
a 15-byte record. Even when rounded up to computer memory movement units, a
single request is one cache line or DMA burst, and a packed batch stays at about
15-16 bytes per request. The slow part on the Mac is asking Qwen to score each
question; using the hint once it exists is tens of microseconds in the current
Python code path.

## Remaining ICLR Gap

Comfortable ICLR still needs native serving measurements on NVIDIA/vLLM:
TTFT, TPOT, goodput, GPU memory, HBM traffic, and faithful C2C/KVComm/
TurboQuant/QJL baselines with source exposure annotated. The next method gate is
still a second public benchmark or a stronger hidden-state endpoint.
