# Full Train-Only Sender+Receiver Gates, 2026-05-01

## Status

- paper readiness: ICLR is still blocked; COLM remains plausible with the
  public-disjoint and leave-one-family-out sender evidence.
- current story: train-only sender and train-only receiver are independently
  positive, but the unified train-only sender+receiver stack still fails the
  hardest cross-family control.
- exact blocker: `holdout_to_core` must beat the live source-atom packet while
  `shuffled_source` stays inside the target-only band.

## Question

Can a train-only sender packet builder and train-only permuted-null receiver be
combined into a single positive method that survives strict source-destroying
controls?

Lay explanation: we already have one translator that can write good tiny hints
and one reader that can read private hints without public examples from the
test families. This asks whether the writer and reader work together when both
are trained only on the training side.

## New Branches Tested

All runs used the train-only receiver dictionary, semantic-anchor features,
`candidate_local_permuted_null_gap_residual_norm`, null weight `0.75`,
threshold `0.30`, seed `47`, `n=128`, train-only packet-builder calibration,
and the same strict destructive controls as the live gate.

| Branch | Artifact | Best positive row | Hard failure |
|---|---|---|---|
| projected candidate-basis packet | `.debug/iclr_20260501_trainonly_sender_receiver_project_add_source_w075_n128/` | `core_to_holdout` 12B passes: candidate `0.750`, base `0.625`, best control `0.250` | `holdout_to_core` shuffled-source reaches `0.625`, matching candidate |
| answer-minus-candidate-mean innovation | `.debug/iclr_20260501_trainonly_sender_receiver_innovation_mean_add_source_w075_n128/` | `core_to_holdout` 4B/8B pass; same-family 4B passes with candidate `1.000` | `holdout_to_core` candidate/base tie at `0.625/0.625`; shuffled-source `0.375` |
| answer-minus-prior innovation | `.debug/iclr_20260501_trainonly_sender_receiver_innovation_prior_add_source_w075_n128/` | same-family 4B passes | cross-family does not improve over base; several controls leak |
| innovation source-weight sweep | `.debug/iclr_20260501_trainonly_sender_receiver_innovation_mean_add_source_w0p25_n128/`, `w0p5`, `w075`, `w1p0` | weights `0.5-1.0` preserve core-to-holdout positives | every weight fails `holdout_to_core`; either candidate ties base or shuffled-source leaks |
| train-mean centered packets | `.debug/iclr_20260501_trainonly_sender_receiver_answer_candidate_centered_add_source_w075_n128/`, `.debug/iclr_20260501_trainonly_sender_receiver_answer_minus_candidate_mean_centered_add_source_w075_n128/` | centered mean-innovation keeps core-to-holdout and same-family positives | `holdout_to_core` still fails; best shuffled-source remains `0.375` |

## Interpretation

Weakened:

- candidate projection. It made the sender speak in the receiver's public
  candidate basis, but the hard shuffled-source control projected to the answer
  at the same rate as the real packet.
- simple answer-vs-decoy innovation. It reduces shuffled leakage from `0.625`
  to `0.375`, but not to the clean target band.
- train-mean centering. It suppresses some common packet mass, but the remaining
  source-independent candidate shortcut still survives in `holdout_to_core`.

Promoted for the next method gate:

- anti-shuffle innovation sender. The next packet objective should score atoms
  by incremental matched-source gain minus shuffled-source/null gain under the
  actual train-only permuted-null receiver. This directly attacks the observed
  failure instead of adding another generic basis transform.

Still alive:

- the train-only permuted-null receiver contribution, which remains positive on
  live source-atom packets.
- the source-prioritized packet-builder contribution, which remains positive
  when the receiver has public eval-disjoint calibration.

## Next Exact Gate

Implement `antishuffle_innovation` as a packet-builder composition and run a
seed-47 n128 gate first:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py \
  --output-dir .debug/iclr_20260501_trainonly_sender_receiver_antishuffle_innovation_seed47_n128 \
  --budgets 10 12 14 \
  --train-examples 256 --eval-examples 128 --seed 47 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration train_only \
  --packet-builder-calibration train_only \
  --calibration-examples 256 \
  --packet-builder-examples 256 \
  --feature-dim 384 \
  --ridge 0.05 \
  --packet-builder-ridge 0.1 \
  --top-k 8 \
  --min-score 0.0 \
  --packet-min-score 0.0 \
  --text-feature-mode semantic_anchor \
  --adapter-target-mode semantic_anchor_teacher \
  --decoder-score-mode candidate_local_permuted_null_gap_residual_norm \
  --permuted-null-weight 0.75 \
  --packet-builder-target-mode answer_minus_candidate_mean \
  --packet-builder-composition antishuffle_innovation \
  --source-identity-weight 0.75 \
  --min-decision-score 0.30 \
  --bootstrap-samples 200
```

Promotion rule: do not widen unless `holdout_to_core` beats the base source
packet by at least `0.03`, shuffled-source is at most `target+0.03`, the matched
packet beats the best destructive control by at least `0.10`, and both
cross-family directions pass.

## Follow-Up Result

See `paper/source_private_antishuffle_innovation_sender_20260501.md`.
Eval-donor anti-shuffle passes the seed-47/53/59 n128 cross-family gate at 12B
and passes a seed-47 n512 cross-family gate. The stricter train-mean contrast
variant does not pass, so the live next branch is sampled train-donor
anti-shuffle rather than claiming the eval-donor diagnostic as the final ICLR
method.

Second follow-up: see
`paper/source_private_train_donor_antishuffle_sender_20260501.md`.
Sampled train-donor anti-shuffle now passes the unified train-only
sender+receiver cross-family gate. The promoted setting uses 12 train donors,
donor/null/generic weights `0.50/0.75/0.10`, source identity weight `0.75`,
and a 12-14B frontier. It passes n128 seeds `47/53/59` and seed-47 n512
cross-family, while same-family remains unpromoted due the structured-text
control. This promotes the full train-only stack from "blocked" to "live ICLR
method branch"; next gates are n512 seed repeats and public benchmark transfer.
