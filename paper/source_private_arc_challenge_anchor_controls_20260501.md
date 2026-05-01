# Source-Private ARC-Challenge Anchor Controls, 2026-05-01

## Status

- anchor-relative positive artifacts:
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_relative_validation/`
  and
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_relative_test/`
- coordinate-mismatch artifacts:
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_id_shuffle_validation/`
  and
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_id_shuffle_test/`
- value-mismatch artifacts:
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_value_shuffle_validation/`
  and
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_value_shuffle_test/`
- random-anchor artifacts:
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_random_anchors_validation/`
  and
  `results/source_private_arc_challenge_seed_stability_20260501_qwen05_random_anchors_test/`
- references:
  `references/573_arc_challenge_anchor_controls_refs_20260501.md`

## Control Design

The positive anchor-relative endpoint uses the same deterministic public
coordinate chart on the source and receiver. These controls separate three
questions:

- `anchor_id_shuffle`: source and receiver use the same anchor values, but the
  receiver shuffles the coordinate identities.
- `anchor_value_shuffle`: source and receiver keep coordinate slots, but the
  receiver deranges the anchor values assigned to those slots.
- `random_anchors_same_count`: source and receiver share synthetic random
  anchors with the same count as the train anchors.

All runs reuse the same answer-key-forbidden Qwen source-choice cache and vary
only the receiver packet basis/control.

## Results

| Split | Basis/control | Seeds | Pass | Matched mean/min/max | Target | Same-byte text | Min lift vs target | Min CI95 low |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| validation | anchor-relative | 5 | 5/5 | 0.386 / 0.381 / 0.388 | 0.244 | 0.348 | 0.137 | 0.065 |
| test | anchor-relative | 5 | 5/5 | 0.344 / 0.344 / 0.345 | 0.265 | 0.311 | 0.078 | 0.039 |
| validation | anchor-id shuffle | 5 | 0/5 | 0.249 / 0.237 / 0.264 | 0.244 | 0.348 | -0.007 | -0.075 |
| test | anchor-id shuffle | 5 | 0/5 | 0.248 / 0.240 / 0.262 | 0.265 | 0.311 | -0.026 | -0.061 |
| validation | anchor-value shuffle | 5 | 0/5 | 0.241 / 0.217 / 0.271 | 0.244 | 0.348 | -0.027 | -0.090 |
| test | anchor-value shuffle | 5 | 0/5 | 0.246 / 0.235 / 0.257 | 0.265 | 0.311 | -0.031 | -0.064 |
| validation | random shared anchors | 5 | 5/5 | 0.388 / 0.388 / 0.388 | 0.244 | 0.348 | 0.144 | 0.070 |
| test | random shared anchors | 5 | 5/5 | 0.344 / 0.344 / 0.344 | 0.265 | 0.311 | 0.078 | 0.038 |

## Interpretation

The controls rule out a trivial zero-byte target-cache explanation for the
anchor-relative result: when the source and receiver do not agree on coordinate
identities or anchor values, the matched packet collapses to target-level
accuracy on validation and test.

The random-anchor result also rules out a stronger semantic-train-anchor claim.
The train anchor semantics are not necessary on ARC; a shared random public
coordinate chart is enough. The safe contribution is therefore public
common-basis packet communication under fixed-byte source-private controls, not
semantic anchor superiority.

## Next Gate

The next highest-value ICLR gate is a second public benchmark or a learned
hidden-state endpoint that preserves this common-basis/control structure. The
systems blocker remains native NVIDIA/vLLM or SGLang TTFT, TPOT, goodput, HBM,
and KV/cache baseline measurements.
