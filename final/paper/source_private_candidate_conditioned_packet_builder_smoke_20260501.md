# Source-Private Candidate-Conditioned Packet Builder Smoke, 2026-05-01

## Status

- paper readiness: promoted as the strongest current positive-method result for
  a scoped COLM paper; still not sufficient for a comfortable ICLR full paper.
- artifact:
  `results/source_private_candidate_conditioned_packet_builder_smoke_20260501/`
- seed repeats:
  `results/source_private_candidate_conditioned_packet_builder_smoke_20260501_seed53/`
  and
  `results/source_private_candidate_conditioned_packet_builder_smoke_20260501_seed59/`
- code:
  `scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py`
- test:
  `tests/test_run_source_private_candidate_conditioned_packet_builder_smoke.py`
- references:
  `references/557_candidate_conditioned_packet_builder_refs_20260501.md`

## Question

Can we improve the live source-private candidate-local residual packet by
learning sender-side packet construction, instead of only calibrating the
receiver over a fixed packet?

The runner keeps the candidate-local residual receiver fixed. It learns a ridge
map from source-private atom evidence to the receiver's candidate-atom basis on
public eval-disjoint calibration examples. At evaluation time, the sender sees
only the source-private hidden-test log, emits the same low-byte atom packet,
and the target receiver scores the public candidate set. The gate compares this
learned packet against the live hand-built source-atom packet on the same rows
and applies the strict source-destroying controls.

## Result

The public-disjoint learned packet passes on three n512 seeds.

| Seed | Direction | Learned packet | Live base | Target | Best control | Pass |
|---:|---|---:|---:|---:|---:|---|
| 47 | core-to-holdout | `0.875` | `0.500` | `0.250` | `0.250` | yes |
| 47 | holdout-to-core | `0.875` | `0.625` | `0.250` | `0.250` | yes |
| 47 | same-family-all | `0.875` | `0.562` | `0.250` | `0.250` | yes |
| 53 | core-to-holdout | `0.875` | `0.500` | `0.250` | `0.250` | yes |
| 53 | holdout-to-core | `0.875` | `0.625` | `0.250` | `0.258` | yes |
| 53 | same-family-all | `0.875` | `0.562` | `0.250` | `0.250` | yes |
| 59 | core-to-holdout | `0.875` | `0.500` | `0.250` | `0.252` | yes |
| 59 | holdout-to-core | `0.875` | `0.625` | `0.250` | `0.256` | yes |
| 59 | same-family-all | `0.875` | `0.562` | `0.250` | `0.250` | yes |

Top-level evidence bundle impact:

- `9/9` learned-packet n512 rows pass.
- `3/3` seed repeats pass the bidirectional cross-family rule.
- minimum learned-packet lift over live base is `+0.250`.
- maximum strict destructive-control accuracy is `0.258`, within the
  target-plus-control band.

## Interpretation

Promoted:

- learned source-to-candidate packet construction is now the strongest
  positive-method branch in the repo.
- the live hand-built packet remains an important baseline and ablation, but is
  no longer the best row.

Weakened / still blocked:

- true unseen-family packet-builder generalization. A `.debug/` train-only
  packet-builder check failed cross-family: core-to-holdout learned packet
  `0.375` versus base `0.500`; holdout-to-core learned packet `0.500` versus
  base `0.625`, with one control leak. This means the current promoted claim is
  public eval-disjoint calibration, not zero-shot unseen-family transfer.

## Lay Explanation

The old method sent the raw private clue words it found in the source log. The
new method first learns how those private clue words should be translated into
the receiver's hint vocabulary, then sends the translated tiny hint. Fake hints
from shuffled, random, or zeroed sources do not help, which is what we need for
a source-private communication claim.

## Next Gate

Keep this method in the evidence bundle, but do not overclaim it. The next ICLR
gate is a stricter packet-builder generalization run: train the builder on a
narrower family split or leave-one-family-out calibration while preserving the
`0.875` public-disjoint row and clean controls. Native NVIDIA/vLLM systems rows
remain the systems blocker.
