# Source-Private Train-Donor Locked Rate Frontier, 2026-05-01

## Status

- paper readiness: stronger, but still not a comfortable ICLR full paper.
- current story: the train-donor anti-shuffle method survives n512 seed
  repeats when the byte budget is selected from a frozen validation frontier.
- exact blocker: the current locked rule is per-seed (`47 -> 14B`,
  `53/59 -> 12B`), while a global fixed-budget rule still does not clear.

## Artifact

- code:
  `scripts/build_source_private_train_donor_locked_rate_frontier.py`
- test:
  `tests/test_build_source_private_train_donor_locked_rate_frontier.py`
- output:
  `results/source_private_train_donor_antishuffle_locked_rate_frontier_20260501/`
- bundle:
  `results/source_private_iclr_evidence_bundle_20260501/`

Lay explanation: before checking the larger test set, we use a smaller
validation set to decide how many bytes the sender is allowed to send. This is
like choosing a volume knob on practice data, then locking it before the final
exam. The goal is to avoid making the result look good by picking the best byte
count after seeing test performance.

## Evidence

The primary readout uses the per-seed policy because the global fixed-budget
policy does not yet pass. The selected n512 rows all pass:

| Seed | Selected budget | Direction | Candidate | Base | Target | Best control | CI95 low vs base | Pass |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 47 | 14 | core_to_holdout | 0.750 | 0.625 | 0.250 | 0.273 | 0.100 | yes |
| 47 | 14 | holdout_to_core | 0.652 | 0.500 | 0.250 | 0.254 | 0.123 | yes |
| 53 | 12 | core_to_holdout | 0.750 | 0.625 | 0.250 | 0.260 | 0.098 | yes |
| 53 | 12 | holdout_to_core | 0.652 | 0.500 | 0.250 | 0.256 | 0.119 | yes |
| 59 | 12 | core_to_holdout | 0.750 | 0.625 | 0.250 | 0.266 | 0.098 | yes |
| 59 | 12 | holdout_to_core | 0.652 | 0.500 | 0.250 | 0.252 | 0.119 | yes |

The validation frontier explains the frontier behavior:

- seed 47 at 10B clears only `core_to_holdout`; `holdout_to_core` ties the
  base packet, and 12B leaks a destructive control in `core_to_holdout`;
- seed 47 recovers at 14B;
- seeds 53 and 59 fail the hard `holdout_to_core` direction at 10B and clear
  at 12B.

## Interpretation

Promoted: train-donor anti-shuffle is no longer just a post-hoc 12-14B row.
It has a locked validation readout that selects the byte budget before n512
scoring and preserves all selected cross-family positives.

Still blocked: this is not yet the cleanest ICLR version because the global
fixed-budget policy is false. Reviewers may accept a rate-frontier method, but
the paper is stronger if one global budget clears all seeds or if the adaptive
budget rule is defined by source/packet diagnostics rather than seed identity.

Next exact gate: run a larger train-only validation split with budgets
`{10,12,14,16}` and either promote a global budget or replace the per-seed rule
with an example-level diagnostic rule that chooses 12B versus 14B without using
test labels.

Recommended final selector: for `core_to_holdout`, train and validate only on
disjoint `core` IDs; for `holdout_to_core`, train and validate only on disjoint
`holdout` IDs. The validation seed should be separate from both train and final
eval seeds, and the artifact should log exact-ID hashes plus overlap audits.
This is stricter than the current small frozen frontier and is the right next
ICLR gate before public benchmark transfer.
