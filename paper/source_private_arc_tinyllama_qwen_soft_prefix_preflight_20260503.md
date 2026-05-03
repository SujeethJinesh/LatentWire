# ARC TinyLlama-to-Qwen Soft-Prefix Preflight

Date: 2026-05-03

## Paper Status

Current paper readiness: COLM-workshop plausible, ICLR full paper blocked.

Current paper story: LatentWire studies source-private fixed-byte
model-to-model evidence packets with destructive controls and systems
byte/exposure accounting.

Exact submission gap: the positive learned receiver/connector is still
missing. The current TinyLlama-to-Qwen target-loss soft-prefix branch does not
beat packet-only, target-only, static-prefix, Qwen-substituted, or same-byte
text controls.

## Artifact

- script: `scripts/run_source_private_arc_openbookqa_soft_prefix_preflight.py`
- test: `tests/test_run_source_private_arc_openbookqa_soft_prefix_preflight.py`
- result:
  `results/source_private_arc_openbookqa_soft_prefix_preflight_20260503_arc_tinyllama_to_qwen_n64_mps_batched_gradaccum/`
- references:
  `references/668_arc_tinyllama_qwen_soft_prefix_failure_refs_20260503.md`

## Gate

The run uses ARC-Challenge validation rows with cached TinyLlama source
predictions and Qwen-substituted packet controls. It trains a frozen-target
soft-prefix connector from answer-key-forbidden TinyLlama candidate hidden
innovation features plus a source selection score channel into Qwen3-0.6B
input-embedding prefixes.

Configuration:

- source: TinyLlama-1.1B-Chat hidden candidate pool on MPS, eager attention
- receiver: Qwen3-0.6B on MPS, eager attention
- row limit: `64`
- fit/eval rows: `32/32`
- feature mode:
  `hf_choice_hidden_score_public_innovation_candidate_pool_residual`
- prefix length: `8`
- hidden dim: `64`
- epochs: `8`
- contrastive controls:
  `zero_source,shuffled_source,same_norm_noise,candidate_roll_source`

## Outcome

The branch fails decisively.

| Condition | Accuracy | Correct / N | Mean Margin |
|---|---:|---:|---:|
| matched soft-prefix | 0.218750 | 7 / 32 | -0.718780 |
| target only | 0.406250 | 13 / 32 | -0.097709 |
| slots-only prefix | 0.468750 | 15 / 32 | -0.031533 |
| packet-only source index | 0.468750 | 15 / 32 | -0.062500 |
| Qwen-substituted packet | 0.437500 | 14 / 32 | -0.125000 |
| same-byte visible text | 0.500000 | 16 / 32 | 0.021598 |
| shuffled source | 0.375000 | 12 / 32 | -0.540985 |
| zero source | 0.250000 | 8 / 32 | -0.393371 |

Headline:

- pass gate: `False`
- matched minus best-control accuracy: `-0.281250`
- matched minus best-control margin: `-0.740378`
- best control by accuracy and margin: `same_byte_visible_text`
- paired accuracy versus target-only: mean `-0.187500`, CI95 low `-0.406250`
- paired accuracy versus packet-only source index: mean `-0.250000`, CI95
  low `-0.468750`

Flip audit:

- versus target-only: matched fixes `3` target errors, breaks `9`
  target-correct rows;
- versus packet-only source index: matched fixes `4` packet errors, breaks
  `12` packet-correct rows;
- versus Qwen-substituted packet: matched fixes `4`, breaks `11`;
- versus same-byte visible text: matched fixes `2`, breaks `11`.

## Implementation Notes

This turn fixed two Mac-local systems blockers in the preflight harness:

- source hidden-state extraction now forces eager attention and releases
  source-model memory before loading the target model;
- multiple-choice target scoring is batched across choices, and the connector
  now uses row-wise gradient accumulation instead of retaining every fit-row
  target graph until a single backward pass.

The batched scorer is objective-preserving: the unit test compares it against
the previous per-choice implementation and checks prefix gradient flow.

The run also exposed a control-design bug: `candidate_roll_source` is a no-op
for the rank-3 query-pooling connector because the connector is permutation
invariant over the candidate set. The harness now adds
`candidate_score_roll_source`, which preserves candidate hidden geometry but
rolls the source selection/score channel across candidates. Future
candidate-pool gates should include this control.

## Decision

Rule out this answer-CE target-loss soft-prefix/query branch as a positive
ICLR method on the current Mac-local ARC surface. It is not merely
statistically weak; it is worse than target-only, worse than packet-only
source index, worse than Qwen-substituted packet, worse than slots-only
prefixes, and worse than same-byte text.

The next branch should not be another shallow answer-CE soft-prefix run unless
there is a crisp new mechanism. The highest expected-value options are:

1. a query-conditioned sparse/common-feature innovation packet with a
   meaningful candidate-score-roll control;
2. a receiver-side resonance objective that matches target-model logit or
   selective hidden-state behavior before answer CE;
3. a stronger native-GPU connector run only after the Mac-local objective has
   a positive small-slice sign.

## Lay Explanation

This experiment asked whether TinyLlama could send Qwen a tiny learned hidden
hint so Qwen would answer science multiple-choice questions better. It did
not work. The learned hint made Qwen worse than just using Qwen alone, worse
than simply trusting TinyLlama's selected answer, and worse than showing Qwen
a tiny visible text hint with the same byte budget.
