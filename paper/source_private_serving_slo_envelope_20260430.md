# Source-Private Serving SLO Envelope

- date: `2026-04-30`
- gate: `source_private_serving_slo_envelope`
- artifact: `results/source_private_serving_slo_envelope_20260430/`
- status: pass as a reviewer-facing systems/SLO trace-card; not a production
  throughput claim

## Question

Can the systems contribution be stated in the language a serving/hardware
reviewer expects: what crosses the boundary, what private state is exposed, how
traffic rounds under batching, which TTFT rows are measured, and where TPOT /
goodput remain unmeasured?

## Method

I added:

```bash
./venv_arm64/bin/python scripts/build_source_private_serving_slo_envelope.py \
  --output-dir results/source_private_serving_slo_envelope_20260430
```

The script joins:

- `results/source_private_memory_traffic_ledger_20260430/`
- `results/source_private_packet_isa_batch_frontier_20260430/`

It emits JSON/CSV/Markdown rows with source privacy flags, source text/KV
exposure, raw payload bytes, single-request 64B-line and 128B-DMA accounting,
batch-64 packet accounting, TTFT proxy margins at `500/750/1000 ms`, and
explicit flags for TPOT/goodput non-claims.

## Results

Headline values:

- rows: `10`
- TTFT proxy rows: `4`
- production goodput claim rows: `0`
- GPU counter required rows: `10`
- packet raw payload minimum: `2` bytes
- packet batch-64 traffic: `5.0` line bytes/request and `6.0` DMA
  bytes/request
- packet minimum TTFT margin: `-47.21 ms` at a `500 ms` SLO, `+202.79 ms` at a
  `750 ms` SLO, and `+452.79 ms` at a `1000 ms` SLO

The negative `500 ms` margin is intentional and useful: the held-out Mac proxy
packet row is not always inside a strict `500 ms` TTFT SLO, so the paper should
not imply production-serving readiness from this artifact.

## Interpretation

This strengthens the systems story by replacing a vague "small packets are
faster" argument with an audit table:

1. **Boundary payload:** packets move tiny source-private records rather than
   source text or KV/cache tensors.
2. **Transfer granularity:** a single request is line/burst limited, but
   packet records amortize under batch packing.
3. **Serving vocabulary:** TTFT is measured only as a Mac proxy; TPOT and
   goodput are explicitly not measured.
4. **Reviewer non-claims:** every row requires native GPU counters before any
   production serving claim.

This is a stronger systems contribution for a Tambe-lab-style review because it
states both the hardware-relevant upside and the exact missing counters.

## Decision

Promote the serving SLO envelope as a systems artifact layered on top of the
memory traffic ledger and packet ISA. It should appear in the paper as
"systems-facing accounting and SLO envelope," not as a throughput benchmark.

## Next Gate

Once the held-out target-decoder run completes, update the receiver table with
paired uncertainty. The next systems gate remains native GPU/server telemetry:
TTFT, TPOT, goodput, and memory movement under a real serving stack.

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_build_source_private_serving_slo_envelope.py
```

Outcome: `1 passed in 0.04s`.
