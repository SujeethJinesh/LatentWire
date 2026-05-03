# HellaSwag Score-Channel True Candidate-Text Permutation Gate

Date: 2026-05-03

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full remains
  blocked.
- Current story: the HellaSwag fixed-byte packet row is now hardened by full
  cached validation, cached option-position controls, and a fresh physical
  candidate-text permutation smoke for the source score/candidate-id channel.
- Exact gap: this is not a full fixed-hybrid hidden-innovation permutation
  rerun and not a learned receiver/common-basis method. ICLR still needs the
  full hidden fixed-hybrid text permutation gate, broader benchmark evidence,
  and native systems rows.

## Contribution Status

The current contribution set remains:

1. Source-private fixed-byte candidate/evidence packets with destructive
   controls.
2. HellaSwag fixed hybrid vote-on-score-agreement packet policy over full
   cached validation.
3. Candidate-position and candidate-text hardening for the packet stack.
4. Byte/exposure accounting that separates packet protocols from KV/cache
   transfer systems.

The weak points are unchanged: learned receiver novelty, common-basis latent
reasoning, cross-family generality, and native NVIDIA systems measurements.

## Gate

Artifact:
`results/source_private_hellaswag_score_channel_true_candidate_text_permutation_gate_20260503_validation0_128/`

Script:
`scripts/build_source_private_hellaswag_score_channel_true_candidate_text_permutation_gate.py`

Test:
`tests/test_build_source_private_hellaswag_score_channel_true_candidate_text_permutation_gate.py`

Design:

- Load HellaSwag validation rows `0:128`.
- Use eight fixed candidate permutations per example.
- Physically reorder candidate endings before Qwen2.5-0.5B-Instruct
  continuation-likelihood scoring.
- Update the display answer index for the permuted row.
- Map the displayed source prediction back to the canonical candidate ID.
- Compare canonical remapped predictions against the identity candidate-id
  packet and against intentionally wrong remaps.

This is a true candidate-text permutation rerun for the score/candidate-id
channel because the local model sees reordered candidate strings and the score
cache is freshly generated for the reordered rows:

- `source_model.cache_hit`: `false`
- `source_model.device`: `cpu`
- `source_model.dtype`: `float32`
- `source_model.prompt_mode`: `continuation`
- `source_model.latency_s`: `288.956527`

## Result

The component smoke passes.

| Metric | Value |
| --- | ---: |
| eval rows | `128` |
| permuted evaluations | `1024` |
| permutations per example | `8` |
| identity accuracy | `0.468750` |
| remapped accuracy | `0.468750` |
| canonical packet consistency | `1.000000` |
| max accuracy delta from identity | `0.000000` |
| accuracy std across permutations | `0.000000` |
| unremapped accuracy | `0.255859` |
| wrong-remap accuracy | `0.176758` |
| wrong-remap CI95 high vs remapped | `-0.248022` |
| packet raw bytes | `1` |
| framed record bytes | `4` |
| source text/KV/hidden/score-vector/logits transmitted | `false` |

Per answer-position remapped accuracies exactly match identity accuracies:

| Answer Slot | Examples | Identity | Remapped | Consistency |
| --- | ---: | ---: | ---: | ---: |
| 0 | `31` | `0.387097` | `0.387097` | `1.000000` |
| 1 | `30` | `0.366667` | `0.366667` | `1.000000` |
| 2 | `38` | `0.500000` | `0.500000` | `1.000000` |
| 3 | `29` | `0.620690` | `0.620690` | `1.000000` |

## Interpretation

This weakens the simplest candidate-slot explanation for the score-channel
packet: when candidate strings move to different displayed option slots, the
source score packet follows the candidate text after canonical remapping. If we
skip remapping or intentionally apply the wrong remap, accuracy collapses.

This does not claim that the full fixed hybrid hidden-innovation packet is
candidate-order invariant. The fixed hybrid uses cached hidden/score bagged
prediction rows, so the stronger reviewer-facing gate is to rerun the hidden
innovation pipeline under physically permuted candidate endings and compare
canonical remapped fixed-hybrid predictions with the original same-row
predictions.

## Lay Explanation

We shuffled the answer endings before asking the local source model to score
them. Then we translated the displayed option number back to the original
option number. The same original candidate was selected after shuffling, so the
score-channel hint is following the text of the answer rather than the display
slot.

## Decision

Promote this as score-channel true candidate-text permutation hardening for the
HellaSwag packet stack.

Do not promote it as:

- a full fixed-hybrid hidden permutation audit;
- a learned receiver/common-basis contribution;
- an ICLR-complete positive method;
- a native systems throughput result.

The next exact gate is a `512`-row or larger true candidate-text permutation
rerun of the hidden fixed-hybrid pipeline, with canonical remapping and paired
uncertainty.
