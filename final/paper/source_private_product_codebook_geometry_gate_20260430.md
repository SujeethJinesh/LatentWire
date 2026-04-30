# Source-Private Product-Codebook Geometry Gate

Date: 2026-04-30

## Cycle Header

1. Current ICLR readiness and distance: stronger scoped candidate, not
   comfortable ICLR-full. Distance remains one stronger non-protocol method or
   a narrower airtight claim boundary, plus native systems telemetry.
2. Current paper story: source-private packets transmit private task evidence
   at very low rate; product-codebook packets are a learned discrete codec with
   paired source-control lift and fast target-side decode.
3. Exact blocker: canonical product-codebook packets are still decoded by a
   deterministic PQ L2 receiver and do not clearly dominate scalar WZ under
   paired uncertainty.
4. Current live branch: source-private packet/control protocol with
   semantic-anchor/scalar WZ, product-codebook packets, and systems/SLO
   accounting as the strongest contribution stack.
5. Highest-priority gate: change the product-codebook geometry itself before
   spending more cycles on failed prompt receivers.
6. Scale-up rung: Mac-local n256 exact-ID geometry smoke on remap/budget rows.

## Layman Summary

Each byte in the PQ packet points to one learned bucket. The question was:
should each byte summarize a different part of the internal vector than the
current default split? We tried utility-balanced splits and OPQ-style rotations.
The answer so far is mostly no: the default split is already strong. OPQ does
repair one weak row's controls, but not by enough to become a new claim.

## Implemented

`scripts/build_source_private_product_codebook_geometry_gate.py` adds a
geometry gate over the existing source-private product-codebook surface.

Variants:

- `canonical`: contiguous product-codebook subspaces, matching the existing PQ
  branch.
- `utility_round_robin`: dimensions ranked by source-to-gold candidate
  separation and distributed round-robin across subspaces.
- `utility_balanced`: high-utility dimensions greedily balanced across
  subspaces.
- `random_balanced`: random balanced dimension regrouping.
- `opq_procrustes`: OPQ-style orthogonal Procrustes loop over canonical
  subspaces.
- `utility_opq_procrustes`: utility permutation initialization followed by the
  OPQ loop.

Controls stay unchanged: label-shuffled ridge, constrained shuffled source,
answer-masked source, permuted codes, and random same-byte.

Promotion rule: a non-canonical geometry must pass source controls and beat
canonical PQ by at least `+0.03` accuracy at the same remap, budget, and exact
eval IDs.

## Results

### Utility Regrouping

Artifact:
`results/source_private_product_codebook_geometry_gate_20260430/remap101_budget4_n256/`

| Variant | Source | Target | Best control | Source-canonical | Pass controls |
|---|---:|---:|---:|---:|---:|
| canonical | 0.582 | 0.250 | 0.254 | 0.000 | true |
| utility_round_robin | 0.574 | 0.250 | 0.297 | -0.008 | true |
| utility_balanced | 0.582 | 0.250 | 0.289 | 0.000 | true |
| random_balanced | 0.586 | 0.250 | 0.289 | +0.004 | true |

Gate result: `False`. Best noncanonical gain is only `+0.004`.

### OPQ On Strong Row

Artifact:
`results/source_private_product_codebook_geometry_gate_20260430/remap101_budget4_opq_n256/`

| Variant | Source | Target | Best control | Source-canonical | Pass controls |
|---|---:|---:|---:|---:|---:|
| canonical | 0.582 | 0.250 | 0.254 | 0.000 | true |
| opq_procrustes | 0.578 | 0.250 | 0.270 | -0.004 | true |
| utility_opq_procrustes | 0.578 | 0.250 | 0.316 | -0.004 | false |

Gate result: `False`. OPQ does not improve the strong remap-101/budget-4 row.

### OPQ On Known Weak Row

Artifacts:

- `results/source_private_product_codebook_geometry_gate_20260430/remap107_budget2_opq_n256/`
- `results/source_private_product_codebook_geometry_gate_20260430/remap107_budget2_opq8_n256/`

| Variant | Source | Target | Best control | Source-canonical | Pass controls |
|---|---:|---:|---:|---:|---:|
| canonical | 0.512 | 0.250 | 0.312 | 0.000 | false |
| opq_procrustes | 0.527 | 0.250 | 0.289 | +0.016 | true |
| opq_procrustes, 8 iters | 0.527 | 0.250 | 0.289 | +0.016 | true |

Gate result: `False` under the promotion rule, but diagnostically useful. OPQ
repairs the known control failure and improves source accuracy by `+0.016`,
then saturates with more iterations.

## Decision

Do not promote utility grouping or OPQ-Procrustes as a new technical
contribution yet. The canonical PQ split is already near the local optimum for
this surface, and the OPQ repair effect is too small for a full-paper claim.

Keep this gate as an ablation showing that:

- product-codebook packets are not fragile to simple geometry perturbations;
- OPQ can reduce one destructive-control failure, but does not clearly beat
  canonical PQ;
- the next PQ method branch must train a source-control-aware rotation or
  protected basis directly, not just run generic OPQ.

## Next Exact Gate

Reviewer priority should now shift away from PQ geometry unless we implement a
source-control-trained protected rotation. The higher-value next Mac-local gate
is the reviewer-suggested label-blind anti-lookup scaling (`n=160` or `n=500`)
or the no-NVIDIA systems trace-card v2, depending on whether we want to
strengthen the method claim or the systems contribution next.

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_build_source_private_product_codebook_geometry_gate.py
```

`3 passed in 1.35s`; `py_compile` passed.
