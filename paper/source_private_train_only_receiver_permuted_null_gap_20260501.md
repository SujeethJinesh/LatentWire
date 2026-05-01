# Train-Only Receiver Permuted-Null Gap Decoder

Status: promoted as a live receiver-basis method candidate for the cross-family
gate; not yet a complete ICLR solution.

## Claim

A train-only receiver can decode source-private source-atom packets across
family splits when candidate-local residual scoring is explicitly blocked by a
deterministic permuted-receiver null model.

Lay description: the receiver has a real train-only dictionary and a scrambled
copy of that dictionary. It only trusts the sender's tiny hint when the real
dictionary explains the hint better than the scrambled dictionary. This directly
targets the earlier failure where scrambled receiver bases still looked useful.

## Method

The decoder score mode is
`candidate_local_permuted_null_gap_residual_norm`.

For each candidate, it computes:

```text
score(candidate) =
  candidate_local_residual_score(real_train_only_receiver)
  - 0.75 * candidate_local_residual_score(permuted_null_receiver)
```

The receiver preserves the target prior unless the best score is at least
`0.30`. The null receiver is deterministic and fitted from the same train-only
calibration rows, but with the semantic-anchor target coordinates permuted.

## Commands

```bash
for seed in 47 53 59; do
  PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
    --output-dir results/source_private_train_only_receiver_permuted_null_gap_20260501_seed${seed}_n512 \
    --budgets 12 --train-examples 768 --eval-examples 512 --seed ${seed} \
    --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress \
    --candidate-calibration train_only --calibration-examples 768 \
    --feature-dim 384 --ridge 0.05 --top-k 8 --min-score 0.0 \
    --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher \
    --decoder-score-mode candidate_local_permuted_null_gap_residual_norm \
    --permuted-null-weight 0.75 --min-decision-score 0.30
done
```

## Results

| Seed | Direction | Pass | Matched | Target | Best control | Gap | CI95 low vs target |
|---:|---|---|---:|---:|---:|---:|---:|
| 47 | core_to_holdout | true | 0.625 | 0.250 | 0.250 | 0.375 | 0.334 |
| 47 | holdout_to_core | true | 0.500 | 0.250 | 0.250 | 0.250 | 0.213 |
| 47 | same_family_all | false | 0.812 | 0.250 | 0.312 | 0.500 | 0.510 |
| 53 | core_to_holdout | true | 0.625 | 0.250 | 0.250 | 0.375 | 0.334 |
| 53 | holdout_to_core | true | 0.500 | 0.250 | 0.260 | 0.240 | 0.213 |
| 53 | same_family_all | false | 0.812 | 0.250 | 0.312 | 0.500 | 0.510 |
| 59 | core_to_holdout | true | 0.625 | 0.250 | 0.252 | 0.373 | 0.334 |
| 59 | holdout_to_core | true | 0.500 | 0.250 | 0.250 | 0.250 | 0.213 |
| 59 | same_family_all | false | 0.812 | 0.250 | 0.312 | 0.500 | 0.510 |

Aggregate evidence now folded into
`results/source_private_iclr_evidence_bundle_20260501/`:

- `6/6` n512 cross-family seed-repeat rows pass
- `3/3` seeds cross-family pass
- cross-family accuracy range: `0.500-0.625`
- target accuracy: `0.250`
- max cross-family best destructive control: `0.260`
- min passing cross-family CI95 lower bound vs target: `0.213`

## Interpretation

Promoted:

- control-blocked train-only receiver basis for source-atom packets
- permuted null receiver as a practical leakage guard for candidate-local
  residual decoding

Still not promoted:

- full train-only sender+receiver stack
- same-family claim under the current strict control set, because
  `structured_text_matched` reaches `0.3125`
- broader real-task latent communication
- native NVIDIA/vLLM systems win

## Reviewer Framing

Safe claim: a train-only receiver basis can generalize cross-family when the
decoder subtracts a deterministic permuted-receiver null score, showing that the
receiver-basis blocker is not fundamental on this benchmark.

Unsafe claim: full train-only cross-model latent communication. The current
row decodes source-atom packets, not the train-only learned sender packet, and
same-family remains control-limited.

## Next Gate

Run the full train-only sender+receiver stack with the permuted-null gap
decoder. Promote only if the candidate packet beats the base source packet in
both cross-family directions with controls at target. If it ties the base
packet, keep this as a receiver-basis contribution rather than the headline
method.

## Follow-Up Stack Smoke

The first n128 full train-only sender+receiver stack was run in `.debug/` with
the train-only packet builder, the same permuted-null gap decoder, and thresholds
`0.30/0.35/0.40/0.45`.

Outcome: not promoted. Core-to-holdout passes, but holdout-to-core ties the
base packet and can leak shuffled-source control.

| Threshold | core_to_holdout | holdout_to_core | same_family_all | Decision |
|---:|---|---|---|---|
| 0.30 | candidate 0.750 vs base 0.625, pass | candidate 0.500 vs base 0.500, shuffled control 0.500 | candidate 1.000 vs base 0.812, structured control 0.312 | fail |
| 0.35 | candidate 0.750 vs base 0.500, pass | candidate 0.375 vs base 0.375, shuffled control 0.375 | candidate 1.000 vs base 0.750, structured control 0.312 | fail |
| 0.40 | candidate 0.750 vs base 0.500, pass | candidate 0.375 vs base 0.375 | candidate 1.000 vs base 0.812, structured control 0.312 | fail |
| 0.45 | candidate 0.750 vs base 0.500, pass | candidate 0.375 vs base 0.375 | candidate 1.000 vs base 0.750, structured control 0.312 | fail |

Interpretation: the receiver-side null gap is useful, but the current
train-only learned sender packet does not add cross-family information in the
hard holdout-to-core direction under a train-only receiver. The next sender-side
idea should explicitly suppress shuffled-source equivalence before any n512
stack run.
