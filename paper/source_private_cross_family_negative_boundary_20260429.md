# Source-Private Cross-Family Negative Boundary

- date: `2026-04-29`
- status: passed boundary/limitation packaging gate
- result root: `results/source_private_cross_family_negative_boundary_20260429/`

## Current Readiness

This artifact improves reviewer trust by making the paper's failure boundary
explicit. It does not add a positive cross-family method. The current defensible
paper claim remains source-private evidence-packet communication under decoder
side information, not broad cross-family latent transfer.

## What This Gate Adds

I added `scripts/build_source_private_cross_family_negative_boundary.py`, which
aggregates cross-family failures and asymmetric positives from the pass/fail
ledger plus masked-innovation receiver summaries.

The appendix covers:

- learned Wyner-Ziv / scalar syndrome packets;
- canonical RASP;
- consistent posterior packets;
- anchor-relative sparse packets;
- learned target-preserving receivers;
- masked innovation receivers.

## Result

- pass gate: `true`
- total rows: `27`
- method families: `6`
- claim-ready cross-family methods: `0`
- oracle-headroom rows: `6`

Family summary:

| Family | Rows | Negative boundary | Asymmetric/incomplete | Best accuracy | Best delta vs control | Claim status |
|---|---:|---:|---:|---:|---:|---|
| anchor-relative sparse packet | 8 | 6 | 2 | 0.496 | 0.246 | not claimed |
| canonical RASP | 2 | 1 | 1 | 0.492 | 0.242 | not claimed |
| consistency posterior packet | 2 | 1 | 1 | 0.495 | 0.245 | not claimed |
| learned Wyner-Ziv / scalar syndrome | 6 | 5 | 1 | 0.623 | 0.373 | not claimed |
| learned target-preserving receiver | 4 | 4 | 0 | 0.453 | 0.143 | not claimed |
| masked innovation receiver | 5 | 5 | 0 | 0.266 | 0.016 | not claimed |

## Interpretation

The negative boundary is important because it prevents the paper from making a
claim the evidence does not support. Several failed rows preserve strong oracle
headroom, especially the masked innovation receiver where full diagnostic
oracle remains `1.000` while the learned packet stays at the target/control
floor. This means the benchmark can represent source information, but current
learned/static interfaces do not transfer it bidirectionally under controls.

The right paper language is:

> Cross-family learned latent communication remains an open boundary in this
> benchmark. We report it explicitly rather than promoting asymmetric or
> control-explained rows.

## Next Method Gate

The only learned-method branch worth testing next is materially different from
the failed ones: a shared sparse crosscoder / shared-dictionary packet with
feature knockout. Pass only if both directions clear target and controls and
top shared-atom knockout removes the lift.

## Artifacts

- `cross_family_negative_boundary.json`
- `cross_family_negative_boundary.csv`
- `cross_family_negative_boundary.md`
- `manifest.json`
- `manifest.md`
