# Train-Donor Stable-Gap Selector, 2026-05-01

## Status

- paper readiness: materially stronger, still not fully ICLR-ready.
- current story: a global `12B` train-donor anti-shuffle packet can now be
  selected by a predeclared validation rule and then passes all current n512
  cross-family eval rows.
- exact blocker: this selector uses source-private validation gaps and treats
  matched-byte structured text as a separate visible-text access model; the
  final ICLR paper still needs public benchmark transfer and native systems
  evidence.

## Artifact

- code: `scripts/build_source_private_train_donor_locked_rate_frontier.py`
- output:
  `results/source_private_train_donor_antishuffle_stable_gap_seed47_53_59_20260501/`
- validation inputs:
  `results/source_private_train_donor_antishuffle_train_family_disjoint_seed47_n128/`,
  `results/source_private_train_donor_antishuffle_train_family_disjoint_seed53_n128_train256/`,
  `results/source_private_train_donor_antishuffle_train_family_disjoint_seed59_n128_train256/`

Lay explanation: instead of picking the first tiny message size that happened
to look clean, we required the chosen size to sit inside a stable band. `12B`
is chosen because `10B`, `12B`, and `14B` are all useful under the validation
gap rule, so `12B` is not a one-off lucky minimum.

## Selector

Selector: `stable_interior` with `source_private_gap` validation scope.

Rule:

- predeclare budgets `{10,12,14,16}`;
- ignore visible-text controls during source-private budget selection, but keep
  them reported separately;
- require candidate accuracy to beat target-only, source-private destructive
  controls, and the learned-synonym base by margins on validation;
- choose the smallest budget that is an interior point of a clean validation
  band, meaning the selected budget and its neighboring budgets pass;
- evaluate only the selected budget on the n512 cross-family rows, using the
  original strict all-controls eval gate.

## Evidence

The selector chooses `12B` globally for seeds `47/53/59`.

| Seed | Direction | Budget | Candidate | Base | Target | Best control | CI95 low vs base | Pass |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 47 | core_to_holdout | 12 | 0.750 | 0.625 | 0.250 | 0.275 | 0.098 | yes |
| 47 | holdout_to_core | 12 | 0.652 | 0.500 | 0.250 | 0.256 | 0.123 | yes |
| 53 | core_to_holdout | 12 | 0.750 | 0.625 | 0.250 | 0.260 | 0.098 | yes |
| 53 | holdout_to_core | 12 | 0.652 | 0.500 | 0.250 | 0.256 | 0.119 | yes |
| 59 | core_to_holdout | 12 | 0.750 | 0.625 | 0.250 | 0.266 | 0.098 | yes |
| 59 | holdout_to_core | 12 | 0.652 | 0.500 | 0.250 | 0.252 | 0.119 | yes |

Headline: `6/6` selected rows pass; minimum candidate accuracy `0.652`;
maximum best-control accuracy `0.275`; minimum paired CI95 lower bound versus
base `0.098`.

## Interpretation

Promoted: validation-selected global fixed-rate story. This is stronger than
the prior per-seed `12-14B` frontier and stronger than the eval-only fixed-12B
audit because the budget is selected before reading the final n512 rows.

Still limited: same-family all-controls validation remains blocked by
matched-byte structured text. The paper must say that visible text is a
different access model, not hide it. Also, seeds `53/59` used the cheaper
`256`-train disjoint validation surface because the `1024`-train disjoint runs
were too slow on the Mac.

## Next Gate

Use this selector as the ICLR default operating-point rule, then rerun it on a
public benchmark slice and, when available, native NVIDIA/vLLM systems rows.
