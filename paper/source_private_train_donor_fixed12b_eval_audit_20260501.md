# Train-Donor Anti-Shuffle Fixed-12B Eval Audit, 2026-05-01

## Status

- paper readiness: stronger but not ICLR-ready; this closes an eval-surface
  concern but not the validation-selection concern.
- current story: a single global `12B` train-donor anti-shuffle packet rate
  passes all available n512 cross-family seed rows.
- exact blocker: the paper still needs a clean pre-eval validation rule or
  predeclared rate-band justification for choosing `12B`.

## Artifact

- code: `scripts/build_source_private_train_donor_fixed_budget_eval_audit.py`
- output:
  `results/source_private_train_donor_antishuffle_fixed12b_eval_audit_20260501/`
- input runs:
  `results/source_private_train_donor_antishuffle_seed47_n512_budget12_cross/`,
  `results/source_private_train_donor_antishuffle_seed53_n512_budget12_14/`,
  `results/source_private_train_donor_antishuffle_seed59_n512_budget12_14/`

Lay explanation: we asked whether the same packet size can work for every
seed, instead of tuning a different size per seed. A `12B` packet works on all
six cross-family rows we currently have.

## Evidence

| Seed | Direction | Candidate | Base | Target | Best control | CI95 low vs base | Pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 47 | core_to_holdout | 0.750 | 0.625 | 0.250 | 0.275 | 0.098 | yes |
| 47 | holdout_to_core | 0.652 | 0.500 | 0.250 | 0.256 | 0.123 | yes |
| 53 | core_to_holdout | 0.750 | 0.625 | 0.250 | 0.260 | 0.098 | yes |
| 53 | holdout_to_core | 0.652 | 0.500 | 0.250 | 0.256 | 0.119 | yes |
| 59 | core_to_holdout | 0.750 | 0.625 | 0.250 | 0.266 | 0.098 | yes |
| 59 | holdout_to_core | 0.652 | 0.500 | 0.250 | 0.252 | 0.119 | yes |

Headline: `6/6` n512 cross-family rows pass at a fixed `12B`; minimum
candidate accuracy is `0.652`, maximum best-control accuracy is `0.275`, and
minimum paired CI95 lower bound versus base is `0.098`.

## Interpretation

Promoted: a fixed-rate eval story. This is cleaner than the prior per-seed
`12-14B` selected frontier because it removes the appearance that seed `47`
needed a hand-picked `14B` row.

Still not solved: validation. The separate disjoint-validation memo shows that
the smallest-passing source-private selector chooses `10B`, which fails one
n512 row. The next selector must justify choosing `12B` before final eval, or
the paper should present `12B` as a predeclared conservative operating point
inside the measured rate frontier.

## Next Gate

Run a margin-aware train-family disjoint selector for seeds `47/53/59` and
check whether it selects `12B` without final-eval labels. If it does not, write
the method as a fixed `12B` operating point and report adjacent rates as
sensitivity rather than claiming automatic rate selection.
