# Source-Private Wyner-Ziv Cross-Family Gate

- date: `2026-04-29`
- artifact: `results/source_private_wyner_ziv_cross_family_gate_20260429/`
- script: `scripts/build_source_private_wyner_ziv_cross_family_gate.py`
- test: `tests/test_build_source_private_wyner_ziv_cross_family_gate.py`
- scale rung: cross-family falsification

## Purpose

This gate tests whether the learned Wyner-Ziv packet can support a broad
cross-family claim. The same scalar source-private packet setup that passed
remapped same-family/all-family rows is trained and evaluated bidirectionally:

- `core -> holdout`
- `holdout -> core`

Budgets are `2`, `4`, and `6` bytes. The strict pass rule requires every
direction-budget row to beat target-only and the best source-destroying scalar
control by at least `+0.15`.

## Result

The gate fails. Cross-family remains asymmetric.

| Direction | Budget | Scalar WZ | Target | Best scalar control | Raw sign | QJL | Canonical RASP | Scalar pass | Canonical pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_to_holdout | 2 | 0.127 | 0.250 | 0.623 | 0.283 | 0.252 | 0.125 | false | false |
| core_to_holdout | 4 | 0.174 | 0.250 | 0.529 | 0.365 | 0.131 | 0.207 | false | false |
| core_to_holdout | 6 | 0.146 | 0.250 | 0.584 | 0.326 | 0.131 | 0.207 | false | false |
| holdout_to_core | 2 | 0.328 | 0.250 | 0.275 | 0.246 | 0.381 | 0.375 | false | false |
| holdout_to_core | 4 | 0.338 | 0.250 | 0.250 | 0.129 | 0.414 | 0.498 | false | true |
| holdout_to_core | 6 | 0.623 | 0.250 | 0.250 | 0.109 | 0.564 | 0.498 | true | true |

## Interpretation

The learned WZ packet should not be promoted as a bidirectional cross-family
method. It is a useful learned same-family/remap packet contribution, but
`core_to_holdout` fails below target and is explained by source-destroying
controls. `holdout_to_core` has a real 6-byte positive row, matching the
historical asymmetry seen in canonical RASP.

This is valuable negative evidence: it prevents overclaiming and narrows the
next method search toward architectures that explicitly handle family shift,
such as anchor-relative dictionary packets, learned query bottlenecks, or
family-normalized source innovations.

## Next Gate

Do not spend another cycle tuning this exact scalar WZ cross-family setup. The
next cross-family branch needs a new mechanism, not another seed:

1. anchor-relative dictionary packet with held-out family normalization, or
2. small query-bottleneck decoder trained with source-destroying controls, or
3. endpoint/system hardening for the already positive same-family/remap story.
