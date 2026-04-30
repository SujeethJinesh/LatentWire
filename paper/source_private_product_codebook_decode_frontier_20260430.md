# Source-Private Product-Codebook Decode Frontier

Date: 2026-04-30

## Cycle Start

1. Current ICLR readiness and distance: improved on the systems axis, but still
   not comfortable ICLR-full. The paper now has a learned compression-native
   packet that is functionally positive and target-decodes quickly; it still
   needs paired uncertainty, one stronger generalization/model-mediated receiver
   gate, and eventually native GPU/server measurements.
2. Current paper story: source-private packets exploit decoder side information
   to transmit private evidence at extreme rate. Product-codebook packets are a
   learned discrete codec: each byte selects a source-trained centroid, and the
   receiver scores the packet against public candidates.
3. Exact blocker before this gate: the product-codebook packet gate passed
   functionally but failed the strict systems row because the previous timing
   path reported `8.3-10.3 ms` p50.
4. Current live branch: product-codebook source-private packets, backed by
   semantic-anchor/scalar WZ positives and rotation-sign/QJL/protected-residual
   ablations.
5. Highest-priority gate: isolate true receiver-side product-codebook decode
   from source packet construction and harness feature hashing.
6. Scale-up rung: strict small/medium Mac CPU systems frontier (`n=256`, three
   remapped codebooks, `2/4/6` byte budgets).

## Commands

Focused tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_build_source_private_product_codebook_decode_frontier.py \
  tests/test_build_source_private_product_codebook_packet_gate.py
```

Compile check:

```bash
./venv_arm64/bin/python -m py_compile \
  scripts/build_source_private_product_codebook_decode_frontier.py
```

Gate:

```bash
./venv_arm64/bin/python scripts/build_source_private_product_codebook_decode_frontier.py \
  --timing-repeats 3 \
  --batch-repeats 50
```

MPS guard:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

No PID was present. This gate is CPU-only.

## Method

The new script builds a systems readout around the already frozen
product-codebook packet rows:

- source packet construction: current harness path, including source projection
  from private evidence features.
- source packet kernel: product-codebook index selection once the
  source-projected vector already exists.
- cold receiver decode: canonical decoder with candidate feature construction.
- cached vector decode: canonical vector reconstruction and nearest-candidate
  scoring with public candidate matrices cached.
- request-public table decode: build public product-codebook distance tables for
  the request, then score packet codes through table lookups.
- resident table decode: packet parse and PQ distance-table lookup when public
  target-side tables are already resident.
- batch amortized decode: vectorized cached batch-256 kernel, reported as
  throughput/headroom rather than a batch-1 latency substitute.

The table decoder is required to exactly match the canonical decoder on every
example before latency can count.

## Result

Artifact:
`results/source_private_product_codebook_decode_frontier_20260430/`.

Headline:

- pass gate: `True`
- rows: `9`
- pass rows: `8`
- remaps with pass: `[101, 103, 107]`
- max request-public table p50: `0.4942 ms`
- max resident table p50: `0.02000 ms`
- max cached-vector p50: `0.0257 ms`
- min speedup vs prior recorded timing: `371.893x`
- prediction mismatches: `0`
- table prediction mismatches: `0`

Rows:

| Remap | Budget | Functional pass | Prior p50 ms | Source packet kernel p50 | Request table p50 | Resident table p50 | Cached vector p50 | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 2 | `True` | 10.285 | 0.0669 | 0.3599 | 0.01687 | 0.0228 | `True` |
| 101 | 4 | `True` | 9.659 | 0.1711 | 0.4204 | 0.01846 | 0.0242 | `True` |
| 101 | 6 | `True` | 9.944 | 0.1290 | 0.4942 | 0.01988 | 0.0257 | `True` |
| 103 | 2 | `True` | 9.039 | 0.0734 | 0.3547 | 0.01675 | 0.0226 | `True` |
| 103 | 4 | `True` | 9.415 | 0.1429 | 0.4117 | 0.01856 | 0.0242 | `True` |
| 103 | 6 | `True` | 9.499 | 0.1316 | 0.4915 | 0.01975 | 0.0255 | `True` |
| 107 | 2 | `False` | 8.313 | 0.0709 | 0.3519 | 0.01675 | 0.0221 | `False` |
| 107 | 4 | `True` | 9.005 | 0.1758 | 0.4203 | 0.01838 | 0.0238 | `True` |
| 107 | 6 | `True` | 9.759 | 0.1479 | 0.4911 | 0.02000 | 0.0255 | `True` |

## Interpretation

The prior `8.3-10.3 ms` timing was not a true receiver-decode bottleneck. It
mostly measured source packet construction from private evidence features. Once
the source-side projected vector and target-side public candidate state are
available, product-codebook packet transmission has a fast receiver:
request-level public table decode stays below `0.5 ms` p50, and resident
distance-table lookup stays around `0.017-0.020 ms` p50.

This is now a systems contribution, but it must be stated carefully. The result
does not claim end-to-end model inference speedup, GPU serving speedup, or full
source-agent latency reduction. It shows that the learned packet's receiver is
not the bottleneck and can be implemented as a small public-table lookup.

## Decision

Promote product-codebook packets from “functional but systems-blocked” to the
live learned compression-native method candidate with a receiver-side systems
win. Keep the failed 2-byte remap-107 row visible because its source-control
failure is useful reviewer evidence against cherry-picking.

Next exact gate: run paired bootstrap/stability for product-codebook packets on
the same `n=256` rows and then run the model-mediated target decoder `n=256`
gate. If only one more gate fits on Mac, prioritize the target-decoder gate
because it addresses the hand-coded receiver objection.

## Literature / Uniqueness Update

New reference memo:
`references/515_product_codebook_decode_frontier_refs_20260430.md`.

Product quantization and asymmetric-distance computation are not novel by
themselves. The paper's novelty is applying a learned PQ/VQ packet as a
source-private communication channel with decoder side information, exact
source-destroying controls, and explicit byte/latency accounting.

## Tests

- `2 passed in 31.64s`
- `1 passed in 2.00s`
- `py_compile` passed
