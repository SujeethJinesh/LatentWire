# Source-Private Train-Family Disjoint Validation Selector, 2026-05-01

## Status

- paper readiness: not ICLR-ready; this stricter selector branch is now a
  failed gate.
- current story: train-donor anti-shuffle still passes n512 cross-family under
  the frozen 12-14B frontier, but train-family disjoint validation is not yet a
  reliable way to select the byte budget.
- exact blocker: the all-controls validation rule is blocked by matched-byte
  structured text, while the source-private-controls-only rule selects a 10B
  budget that fails one n512 cross-family direction.

## Artifact

- code:
  `scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py`
- output:
  `results/source_private_train_donor_antishuffle_train_family_disjoint_seed47_n128/`
- source-private selector audit:
  `results/source_private_train_donor_antishuffle_source_private_validation_seed47_budget10_20260501/`
- n512 budget-10 eval:
  `results/source_private_train_donor_antishuffle_seed47_n512_budget10_cross/`

Lay explanation: instead of choosing the byte budget on the final cross-family
test, we tried to choose it on examples from the same family that trained the
sender and receiver, but with disjoint example IDs. That is a cleaner practice
test. It failed because a short text hint baseline was too strong on the
same-family validation split.

## Evidence

Seed `47`, `1024` train examples, `128` disjoint validation examples, budgets
`{10,12,14,16}`.

| Direction | Train family | Validation family | Train start | Validation start | ID overlap | Budget | Candidate | Base | Target | Best control | Pass |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| core_to_holdout | core | core | 0 | 1024 | 0 | 10 | 1.000 | 0.750 | 0.250 | 0.258 | yes |
| core_to_holdout | core | core | 0 | 1024 | 0 | 12 | 1.000 | 0.750 | 0.250 | 0.250 | yes |
| core_to_holdout | core | core | 0 | 1024 | 0 | 14 | 1.000 | 0.750 | 0.250 | 0.266 | yes |
| core_to_holdout | core | core | 0 | 1024 | 0 | 16 | 1.000 | 0.750 | 0.250 | 0.289 | no |
| holdout_to_core | holdout | holdout | 0 | 1024 | 0 | 10 | 1.000 | 0.875 | 0.250 | 0.375 | no |
| holdout_to_core | holdout | holdout | 0 | 1024 | 0 | 12 | 1.000 | 0.875 | 0.250 | 0.375 | no |
| holdout_to_core | holdout | holdout | 0 | 1024 | 0 | 14 | 1.000 | 0.875 | 0.250 | 0.375 | no |
| holdout_to_core | holdout | holdout | 0 | 1024 | 0 | 16 | 1.000 | 0.875 | 0.250 | 0.375 | no |

The failing best control in `holdout_to_core` is `structured_text_matched`.
Source-private destructive controls remain near the target-only band, but the
strict current gate treats matched-byte text as a blocker.

## Source-Private-Only Selector Check

To test the obvious escape hatch, I reran the locked frontier builder with
`--validation-control-scope source_private_controls`. This excludes visible
text/target sidecars from the budget-selection rule, but still reports them in
the table.

On seed `47`, that rule selects `10B`. It is too optimistic:

| Eval direction | Budget | Candidate | Base | Target | Best control | CI95 low vs base | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| core_to_holdout | 10 | 0.750 | 0.625 | 0.250 | 0.260 | 0.094 | yes |
| holdout_to_core | 10 | 0.652 | 0.625 | 0.250 | 0.252 | 0.014 | no |

The failed row is not a control leak: `random_same_byte` is only `0.252` and
the source-private controls are clean. The failure is statistical margin
against the learned-synonym base. The candidate improves base by only `0.027`,
below the locked gate's `0.050` paired lower-bound requirement.

## Interpretation

Ruled out:

- a strict same-family, all-controls validation selector for the current
  train-donor packet method. It cannot select a budget because the text baseline
  is too strong on holdout-family validation.
- a smallest-passing source-private-controls-only selector. It selects `10B`,
  but that row fails the final n512 holdout-to-core CI margin.

Still alive: a source-private selector with a stronger validation margin or a
predeclared rate band. The current positive evidence is better described as a
`12-14B` rate-frontier method than as a single automatically selected minimum
budget.

Promoted next branch: use validation to choose a conservative rate band or a
margin-aware budget rule, then test it across seeds `47/53/59`. The selector
must not choose the smallest clean byte budget unless that budget also has
validation margin against the base model.

## Next Gate

Implement a margin-aware locked selector. Candidate rule: among
source-private-clean budgets, choose the smallest budget whose validation
candidate-minus-base margin clears a stricter threshold in both train-family
directions; otherwise report the full `12-14B` rate band rather than a single
budget. Then run the same disjoint validation/eval audit for seeds `53/59`.
