# Source-Private Product-Codebook Paired Uncertainty

Date: 2026-04-30

## Cycle Start

1. Current ICLR readiness and distance: improved but still not comfortable
   ICLR-full. Product-codebook packets now have functional, systems-decode, and
   paired-uncertainty evidence on the current `n=256` remapped surface; broader
   receiver/model generalization remains the blocker.
2. Current paper story: source-private product-codebook packets transmit a few
   learned discrete indices that let the target recover private evidence using
   public decoder side information. The method beats target-only and
   source-destroying controls with a fast receiver.
3. Exact blocker before this gate: the product-codebook result needed paired
   uncertainty so reviewers could distinguish stable source-causal lift from
   aggregate chance.
4. Current live branch: product-codebook source-private packets, backed by
   semantic-anchor/scalar WZ positives and PQ decode-frontier systems evidence.
5. Highest-priority gate: paired bootstrap over existing exact-ID prediction
   files, comparing product-codebook packets against target-only, scalar WZ, and
   the strongest product-codebook destructive control.
6. Scale-up rung: strict small/medium Mac CPU confirmation (`n=256`, three
   remapped codebooks, `2/4/6` byte budgets).

## Commands

Focused test:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_summarize_source_private_product_codebook_uncertainty.py
```

Compile check:

```bash
./venv_arm64/bin/python -m py_compile \
  scripts/summarize_source_private_product_codebook_uncertainty.py
```

Gate:

```bash
./venv_arm64/bin/python scripts/summarize_source_private_product_codebook_uncertainty.py \
  --bootstrap-samples 5000
```

## Result

Artifact:
`results/source_private_product_codebook_uncertainty_20260430/`.

Headline:

- pass gate: `True`
- rows: `9`
- pass rows: `8`
- remaps with pass: `[101, 103, 107]`
- min passing paired CI95 low vs target: `0.191`
- min passing paired CI95 low vs best control: `0.152`
- min paired CI95 low vs scalar WZ: `-0.035`
- max product-codebook accuracy: `0.598`

Rows:

| Remap | Budget | PQ | Target | Best control | Scalar WZ | PQ-control CI low | PQ-target CI low | PQ-scalar CI low | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 2 | 0.574 | 0.250 | 0.254 | 0.449 | 0.234 | 0.238 | 0.066 | `True` |
| 101 | 4 | 0.582 | 0.250 | 0.281 | 0.531 | 0.219 | 0.250 | -0.004 | `True` |
| 101 | 6 | 0.598 | 0.250 | 0.293 | 0.570 | 0.230 | 0.266 | -0.016 | `True` |
| 103 | 2 | 0.539 | 0.250 | 0.285 | 0.469 | 0.172 | 0.203 | 0.004 | `True` |
| 103 | 4 | 0.531 | 0.250 | 0.293 | 0.500 | 0.152 | 0.195 | -0.023 | `True` |
| 103 | 6 | 0.551 | 0.250 | 0.273 | 0.539 | 0.191 | 0.211 | -0.035 | `True` |
| 107 | 2 | 0.512 | 0.250 | 0.312 | 0.449 | 0.117 | 0.180 | 0.004 | `False` |
| 107 | 4 | 0.527 | 0.250 | 0.293 | 0.488 | 0.152 | 0.195 | -0.016 | `True` |
| 107 | 6 | 0.523 | 0.250 | 0.281 | 0.520 | 0.160 | 0.191 | -0.035 | `True` |

## Interpretation

The product-codebook packet claim is stable against target-only and
source-destroying controls on this frozen surface. The known 2-byte remap-107
row remains failed because the original functional gate marked its control row
too high; the uncertainty summary now respects that prior gate instead of
reviving it.

The scalar WZ comparison remains nuanced. Product-codebook beats scalar WZ in
raw accuracy on every row, but the paired CI lower bound versus scalar is
negative on several rows because the margin is small. The honest claim is:
product-codebook is a learned discrete codec that is source-causal and
systems-fast, while scalar WZ remains a strong adjacent comparator rather than a
fully dominated baseline.

## Decision

Promote product-codebook packets as a live third technical contribution:

1. source-private packet/control protocol;
2. semantic-anchor/scalar WZ positive source-private packet evidence;
3. learned product-codebook packets with paired source-control lift and
   low-latency target-side decode.

Next exact gate: a model-mediated target decoder at `n=256`, or a seed-repeat
product-codebook gate if compute is tighter. The model-mediated gate is the
higher reviewer-value next step because it addresses the hand-coded receiver
objection directly.

## Tests

- `1 passed in 0.04s`
- `py_compile` passed
