# HellaSwag Train-Only Public Receiver Repair Probe

Date: 2026-05-01

## Status

This branch is pruned as a public-feature-only repair method. It is useful as
a reviewer-facing falsification because it shows that the obvious train-only
receiver-side lexical scorer does not recover the HellaSwag source top-2
headroom.

## Plain-Language Goal

The sender model often has the right answer in its top two HellaSwag guesses,
but not always as its first guess. This probe asked whether the receiver could
learn a cheap public "second opinion" from HellaSwag train labels and use the
sender's top two guesses as a tiny hint.

If this worked, it would mean a one-byte top-2 hint plus a train-learned
receiver rule could beat simply sending the sender's favorite answer label.
It did not work.

## Protocol

Artifact:
`results/source_private_hellaswag_public_receiver_repair_probe_20260501_qwen05_validation1024/`

Script:
`scripts/build_source_private_hellaswag_public_receiver_repair_probe.py`

Frozen evaluation slice:
`results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation_first1024.jsonl`

Train split:
`results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_train.jsonl`

Source score cache:
`results/source_private_hellaswag_score_packet_headroom_20260501_qwen05_validation1024/source_score_cache.json`

The public receiver model is a hashed lexical perceptron trained only on the
official HellaSwag train rows. Hyperparameters are fixed, and the selected
epoch is chosen on an internal train/dev split. Validation labels are used only
for final scoring.

## Readout

| Condition | Accuracy | Correct |
|---|---:|---:|
| source-label copy | `0.462` | `473/1024` |
| public target-only receiver | `0.260` | `266/1024` |
| top-2 public rerank | `0.364` | `373/1024` |
| public-if-in-source-top-2 gate | `0.413` | `423/1024` |

Additional diagnostics:

- selected internal dev accuracy: `0.309`
- best repair condition: `public_if_in_source_top2`
- best repair minus source-label copy: `-0.049`
- source top-2 oracle accuracy: `0.716`
- public prediction in source top-2 rate: `0.485`
- pass gate: `false`

## Interpretation

The public receiver model is too weak to decide when the source's runner-up
should replace the source's top answer. The source top-2 oracle remains high,
so the problem is not lack of headroom. The missing ingredient is a better
source-error signal.

For ICLR framing, this should be cited as a negative gate:

- do not claim HellaSwag positive-method success from public top-2 reranking
- do not keep tuning eval-only margin or public lexical features
- next HellaSwag branch must use train-split source scores or source hidden
  summaries, with a frozen validation evaluation and label-copy controls

## Next Gate

Build a train-only source-score calibration packet:

1. score a bounded official HellaSwag train slice with the same source model
2. train a repair policy using only train labels and train source scores
3. evaluate once on the frozen `1024` validation rows
4. require at least `494/1024` correct, i.e. `+0.02` over source-label copy
5. compare against trained source-label-copy, same-byte visible text, shuffled
   source packet, candidate derangement, label permutation, and metadata priors

If that fails, HellaSwag should stay a diagnostic surface and the ICLR headline
should remain ARC/OpenBookQA plus the stronger train-donor method branch.
