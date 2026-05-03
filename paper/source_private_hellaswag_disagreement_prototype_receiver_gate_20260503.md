# HellaSwag Disagreement-Prototype Receiver Gate

Date: 2026-05-03

## Status

This gate fails. It is a useful branch-kill result for ICLR: shallow
official-train disagreement prototypes do not close the HellaSwag
target-or-packet oracle headroom.

Current paper readiness remains unchanged: COLM-workshop plausible as a
fixed-byte source-private packet artifact; ICLR full still needs a stronger
positive connector/receiver method plus native NVIDIA systems rows.

## Experiment

Artifact:

`results/source_private_hellaswag_disagreement_prototype_receiver_gate_20260503/`

Script:

`scripts/build_source_private_hellaswag_disagreement_prototype_receiver_gate.py`

Test:

`tests/test_build_source_private_hellaswag_disagreement_prototype_receiver_gate.py`

Protocol:

- source packet: TinyLlama hidden-innovation packet, `2B` raw / `5B` framed;
- receiver side information: Qwen target scores and Qwen hidden-confidence
  features;
- calibration: official HellaSwag train, out-of-bag packet predictions,
  `1487` usable rows;
- official-train fit/dev split: `1115/372`;
- eval: full HellaSwag validation, `10042` rows;
- receiver: fit helpful and harmful disagreement prototypes on official-train
  rows, then override the TinyLlama packet only when validation features are
  closer to helpful than harmful prototypes;
- controls: label-permutation, candidate-roll alternative, and same-rate
  random override.

Lay explanation: the receiver looks for recurring shapes of disagreement. It
learns examples where Qwen fixes TinyLlama and examples where Qwen would hurt
TinyLlama. At test time, it only switches away from the TinyLlama packet when
the row looks like the helpful group.

## Results

Headline:

| row | accuracy | delta vs packet | CI95 low |
|---|---:|---:|---:|
| TinyLlama packet-only | `0.619199` | `0.000000` | `0.000000` |
| Qwen target score | `0.480880` | n/a | n/a |
| Qwen hybrid packet | `0.532464` | n/a | n/a |
| target-or-packet oracle | `0.686815` | `+0.067616` | n/a |
| predeclared prototype receiver | `0.619299` | `+0.000100` | `-0.000896` |
| best diagnostic prototype receiver | `0.620394` | `+0.001195` | `-0.000199` |
| best control | `0.618701` | `-0.000498` | `-0.001095` |

Best diagnostic row:

- alternative: `hybrid_vote_on_score_agreement_prediction`;
- feature view: `score_hidden_confidence`;
- positive prototypes: `8`;
- negative prototypes: `16`;
- aggregation: `top2`.

Flip audit:

| row | same as packet | fixed packet errors | broke packet-correct | net |
|---|---:|---:|---:|---:|
| predeclared prototype | `10008/10042` | `13` | `12` | `+1` |
| best diagnostic prototype | `9963/10042` | `31` | `19` | `+12` |

Block stability for the predeclared row fails: two of five validation blocks
are negative, two are exactly tied, and only one is positive.

## Decision

Kill shallow disagreement-prototype receivers on this HellaSwag surface:

- the target-or-packet oracle is real at `0.686815`, but prototypes recover
  only `0.001195` best-scout accuracy over packet-only;
- the predeclared row nets only one extra correct validation example;
- CI lower bounds remain negative;
- the result is below the `+0.005` receiver-improvement bar and fails block
  stability.

This further weakens score/hidden-confidence geometry as the missing
cross-model common language. The next gate should be a target-loss
query/soft-prefix repair or another true nonlinear query bottleneck, not
another scalar, kNN, or prototype selector.

## What To Cut

Cut from ICLR-positive claims:

- official-train scalar acceptance;
- official-train relative-kNN acceptance;
- official-train disagreement-prototype receiver;
- linear CCA/crosscoder hidden-code packets as a headline method.

Keep as paper support:

- HellaSwag receiver-headroom decomposition;
- fixed-byte TinyLlama packet utility;
- failure evidence showing that the receiver problem is real and not solved by
  shallow selectors.

## Next Gate

Run the ARC TinyLlama-to-Qwen n64 target-loss soft-prefix preflight recommended
by the literature scout, with target-only, static-prefix, zero-source,
shuffled-source, same-norm noise, candidate-roll source, same-byte text, and
Qwen-substituted controls.

Pass only if matched soft-prefix beats every non-oracle control by at least
`+0.02` accuracy on heldout half of the n64 slice. If it passes, widen to n160
or the full TinyLlama/Qwen disagreement slice with paired uncertainty.
