# Source-Private Product-Codebook Packet Gate

Date: 2026-04-30

## Cycle Start

1. Current ICLR readiness and distance: stronger than the previous cycle, but
   still not comfortable ICLR-full. The paper now has a real compression-native
   packet candidate; it still needs optimized systems latency, paired
   uncertainty, and/or a non-hand-coded target receiver.
2. Current paper story: source-private packets exploit decoder side information
   to transmit private evidence at extreme rate. Scalar WZ and semantic anchors
   are positive; pure signs fail; product-codebook packets now show that learned
   discrete vector quantization can preserve useful source margins.
3. Exact blocker: product-codebook packets functionally pass but fail strict
   Python p50 decode latency. The method needs either an optimized cached/table
   decoder or a scoped claim that separates functional communication from
   serving-speed claims.
4. Current live branch: product-codebook source-private packets as the top
   compression-native candidate; semantic-anchor/scalar WZ remain promoted
   evidence rows.
5. Highest-priority gate: remapped slot-codebook source-control gate across
   `2/4/6` byte budgets, compared against scalar WZ, QJL, protected residual,
   and rotation-sign.
6. Scale-up rung: strict small/medium Mac CPU confirmation (`n=256`, three
   remapped codebooks, three byte budgets).

## Commands

Focused tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_tool_trace_compression_baselines.py \
  tests/test_build_source_private_product_codebook_packet_gate.py -q
```

Additional aggregate-builder test after adding functional/systems split:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_build_source_private_product_codebook_packet_gate.py -q
```

Gate:

```bash
./venv_arm64/bin/python scripts/build_source_private_product_codebook_packet_gate.py \
  --output-dir results/source_private_product_codebook_packet_gate_20260430 \
  --remap-seeds 101 103 107 \
  --budgets 2 4 6 \
  --train-examples 512 \
  --eval-examples 256 \
  --feature-dim 512 \
  --train-seed 29 \
  --eval-seed 30
```

MPS guard:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

No PID was present. This gate is CPU-only.

## Method

`product_codebook` fits a learned product quantizer over the source-projected
training vectors. Each packet byte is one centroid index for one vector
subspace. The target reconstructs the product-codebook vector and decodes by
nearest public candidate vector.

Source-destroying controls:

- label-shuffled ridge encoder
- constrained shuffled source
- answer-masked source
- permuted product-code indices
- random same-byte code indices

## Result

Artifact:
`results/source_private_product_codebook_packet_gate_20260430/`.

Headline:

- strict pass gate: `False`
- functional pass gate: `True`
- systems latency pass gate: `False`
- rows: `9`
- functional pass rows: `8`
- functional remaps with pass: `[101, 103, 107]`
- systems pass rows: `0`
- max product-codebook accuracy: `0.598`

Rows:

| Remap | Budget | Product codebook | Scalar WZ | QJL | Protected | Rotation-sign | Target | Best PQ control | PQ-control | PQ-scalar | p50 ms | PQ pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 2 | 0.574 | 0.449 | 0.434 | 0.406 | 0.297 | 0.250 | 0.254 | 0.320 | +0.125 | 10.285 | True |
| 101 | 4 | 0.582 | 0.531 | 0.523 | 0.469 | 0.336 | 0.250 | 0.281 | 0.301 | +0.051 | 9.659 | True |
| 101 | 6 | 0.598 | 0.570 | 0.547 | 0.504 | 0.340 | 0.250 | 0.293 | 0.305 | +0.027 | 9.944 | True |
| 103 | 2 | 0.539 | 0.469 | 0.430 | 0.461 | 0.312 | 0.250 | 0.285 | 0.254 | +0.070 | 9.039 | True |
| 103 | 4 | 0.531 | 0.500 | 0.535 | 0.473 | 0.348 | 0.250 | 0.293 | 0.238 | +0.031 | 9.415 | True |
| 103 | 6 | 0.551 | 0.539 | 0.531 | 0.535 | 0.332 | 0.250 | 0.273 | 0.277 | +0.012 | 9.499 | True |
| 107 | 2 | 0.512 | 0.449 | 0.449 | 0.484 | 0.309 | 0.250 | 0.312 | 0.199 | +0.062 | 8.313 | False |
| 107 | 4 | 0.527 | 0.488 | 0.500 | 0.496 | 0.316 | 0.250 | 0.293 | 0.234 | +0.039 | 9.005 | True |
| 107 | 6 | 0.523 | 0.520 | 0.504 | 0.492 | 0.332 | 0.250 | 0.281 | 0.242 | +0.004 | 9.759 | True |

## Interpretation

This is the first compression-native packet variant in this line that should
remain live. Unlike rotation-sign, product-codebook packets preserve enough
source-projected magnitude/geometry to beat scalar WZ on every row. They also
beat the strongest PQ destructive control by `+0.199` to `+0.320`, except the
2-byte remap-107 row still fails the target+0.05 control criterion.

The strict aggregate stays false because the current simple Python decoder has
`8.3-10.3 ms` p50 latency. This is not acceptable for a systems-latency claim,
but it is not a scientific failure of the communication method. The next
systems gate should precompute candidate matrices / table distances or add a
dedicated microbench that separates codec kernel latency from feature-hashing
overhead.

## Decision

Promote product-codebook packets to the live compression-native candidate
branch, not as a final headline claim yet. This adds a stronger third technical
contribution candidate:

1. Source-private packet/control protocol.
2. Semantic-anchor/scalar WZ positive method evidence.
3. Product-codebook source-private packets as a learned discrete codec that
   preserves source evidence better than scalar, QJL/protected residual, and
   sign-only sketches on this remapped surface.

Next exact gate: optimized/cached product-codebook decode frontier plus paired
bootstrap on the same rows. A second useful branch is target-decoder `n=256`
because reviewers will still object to deterministic decoding.

## Literature / Uniqueness Update

New reference memo:
`references/514_product_codebook_packet_refs_20260430.md`.

The novelty story is now more defensible than after the rotation-sign prune:
PQ/TurboQuant/QJL motivate the codec, but LatentWire evaluates task-level
source-private communication under destructive controls rather than vector/KV
reconstruction alone.

## Tests

- `12 passed in 209.78s`
- `1 passed in 23.40s`
