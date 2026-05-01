# Train-Only Sender Source-Prioritized Packet Builder

Date: 2026-05-01

## Status

This is now the strongest current generalization-facing LatentWire method.
Compared with the previous leave-one-family-out packet-builder result, the
sender packet builder is trained only on the train family split. The receiver
dictionary still uses public eval-disjoint calibration, so this does not yet
solve full train-only cross-family generalization.

## Method

The packet uses the same source-prioritized innovation form:

```text
packet_vector = source_to_candidate_ridge_train_only(source_atoms) + 0.75 * source_atoms
```

The sender then transmits the top atoms in a 12-byte packet. The builder
calibration is `train_only`; the candidate receiver dictionary calibration is
`all_public_eval_disjoint`.

Lay explanation: the sender learns how to translate clues only from training
families. At test time, it sends the learned translation plus enough of the
original clue to prevent drift. The receiver still has a public dictionary for
reading candidate descriptions, which is the part we have not made train-only
yet.

## Evidence

Artifacts:

- `results/source_private_candidate_conditioned_packet_builder_train_builder_hybrid_w075_20260501/`
- `results/source_private_candidate_conditioned_packet_builder_train_builder_hybrid_w075_20260501_seed53/`
- `results/source_private_candidate_conditioned_packet_builder_train_builder_hybrid_w075_20260501_seed59/`
- `results/source_private_candidate_conditioned_packet_builder_train_builder_hybrid_w075_rate_20260501/`

Three-seed n512 result:

| Seed | Direction | N | Candidate packet | Live source packet | Target | Best control | Pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 47 | core_to_holdout | 512 | 0.625 | 0.500 | 0.250 | 0.250 | True |
| 47 | holdout_to_core | 512 | 0.625 | 0.500 | 0.250 | 0.258 | True |
| 47 | same_family_all | 512 | 0.750 | 0.500 | 0.250 | 0.250 | True |
| 53 | core_to_holdout | 512 | 0.625 | 0.500 | 0.250 | 0.250 | True |
| 53 | holdout_to_core | 512 | 0.625 | 0.500 | 0.250 | 0.256 | True |
| 53 | same_family_all | 512 | 0.750 | 0.500 | 0.250 | 0.250 | True |
| 59 | core_to_holdout | 512 | 0.625 | 0.500 | 0.250 | 0.250 | True |
| 59 | holdout_to_core | 512 | 0.625 | 0.500 | 0.250 | 0.254 | True |
| 59 | same_family_all | 512 | 0.750 | 0.500 | 0.250 | 0.250 | True |

Headline: `9/9` rows pass; min lift over the live source packet is `+0.125`;
max best destructive control is `0.258`; min paired CI95 lower bound versus the
live source packet is `0.092`.

Single-seed rate curve:

| Direction | Budget | Candidate packet | Live source packet | Target | Best control | Pass |
|---|---:|---:|---:|---:|---:|---|
| core_to_holdout | 8 | 0.500 | 0.500 | 0.250 | 0.250 | False |
| core_to_holdout | 10 | 0.625 | 0.500 | 0.250 | 0.250 | True |
| core_to_holdout | 12 | 0.625 | 0.500 | 0.250 | 0.250 | True |
| core_to_holdout | 16 | 0.500 | 0.500 | 0.250 | 0.250 | False |
| holdout_to_core | 8 | 0.625 | 0.625 | 0.250 | 0.252 | False |
| holdout_to_core | 10 | 0.625 | 0.625 | 0.250 | 0.260 | False |
| holdout_to_core | 12 | 0.625 | 0.500 | 0.250 | 0.258 | True |
| holdout_to_core | 16 | 0.625 | 0.500 | 0.250 | 0.258 | True |
| same_family_all | 8 | 0.812 | 0.562 | 0.250 | 0.250 | True |
| same_family_all | 10 | 0.750 | 0.562 | 0.250 | 0.250 | True |
| same_family_all | 12 | 0.750 | 0.500 | 0.250 | 0.250 | True |
| same_family_all | 16 | 0.688 | 0.500 | 0.250 | 0.250 | True |

Interpretation: 12B is the cleanest bidirectional cross-family operating point
for the train-only sender hybrid. 8B is not stable enough, and larger packets
can admit extra atoms that dilute the hard core-to-holdout direction.

## Calibration Split

Cheap n128 probes isolate the remaining blocker:

- full train-only receiver+sender: cross-family fails; same-family passes.
- public receiver dictionary + train-only sender builder: cross-family nearly
  passes at n128 and passes at n512.
- train-only receiver dictionary + LOO public builder: cross-family fails.

This says the train-only sender is alive, but the receiver/candidate dictionary
still needs a stronger train-only common basis.

## Reviewer-Facing Claim

Safe claim: sender-side packet construction can be trained without public
eval-family packet-builder calibration and still beat the live source packet
under strict controls.

Unsafe claim: full train-only cross-family latent reasoning. That still fails
with the current receiver dictionary.

## Next Gate

The next ICLR gate is a train-only receiver basis repair: target-side innovation
syndrome, gauge-fixed common basis, or train-only candidate dictionary with
hard candidate-only/codeword controls.
