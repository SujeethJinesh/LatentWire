# Target-Aligned Sparse Resonance Packet ARC Gate

Date: 2026-05-04

## Readiness Status

- COLM_v1: frozen except for cleanup/reproducibility/style/citation/build
  fixes.
- COLM_v2: active Sparse Resonance Packet pivot, but still negative on the
  current strict ARC scouting gate.
- ICLR: blocked until a positive source-private packet beats target-only,
  source-index/rank/score, same-byte text, target-derived, destructive atom
  controls, and source-family substitution with paired positive uncertainty.

## Method Tested

This gate tests a target-aligned sparse packet instead of the previous
source-only PCA packet.

1. Compute answer-key-forbidden TinyLlama candidate hidden representations.
2. Remove public question/choice-hash predictable components using train-only
   ridge innovation.
3. Compute answer-key-forbidden Qwen3 candidate hidden representations on the
   same candidate texts and remove the same public component.
4. Fit a target PCA basis on train Qwen3 innovations.
5. Learn a train-only ridge map from TinyLlama innovations into Qwen3 PCA
   coordinates.
6. Transmit only sparse top-k target-basis atom IDs plus quantized
   coefficients to the Qwen3 soft-prefix receiver.

Added destructive controls:

- `coefficient_shuffle`: reverses coefficients across atom IDs;
- `top_atom_knockout`: zeros the strongest atom per candidate summary.

These complement the existing `atom_shuffle`, `source_row_shuffle`,
`candidate_roll`, `same_byte_visible_text`, source-index/rank/score, and
Qwen-substitution controls.

## Artifacts

- Code:
  `scripts/run_source_private_arc_openbookqa_soft_prefix_preflight.py`
  and
  `scripts/build_source_private_arc_challenge_soft_prefix_resonance_gate.py`.
- Tests:
  `tests/test_run_source_private_arc_openbookqa_soft_prefix_preflight.py`
  and
  `tests/test_build_source_private_arc_challenge_soft_prefix_resonance_gate.py`.
- Main scouting run:
  `results/source_private_arc_challenge_sparse_resonance_packet_gate_20260504_tinyllama_to_qwen3_disagreement_n8_target_aligned_top2q3/`.
- High-rate ablation:
  `results/source_private_arc_challenge_sparse_resonance_packet_gate_20260504_tinyllama_to_qwen3_disagreement_n8_target_aligned_top8q8_noresid/`.

## Results

| run | packet | matched | best required control | worst CI95 low | pass |
|---|---:|---:|---:|---:|---|
| target-aligned top2 q3 residual | 5 B/row | 0.250 | target-derived 0.625 | -0.750 | no |
| target-aligned top8 q8 no residual | 44 B/row | 0.125 | Qwen-substitution 0.625 | -0.875 | no |

The low-rate run preserved high train-side packet diagnostics:

- packet rank: `4`;
- top-k atoms: `2`;
- coefficient bits: `3`;
- fit sparse energy ratio: `0.921`;
- target-coordinate train R2: `0.999986`;
- target PCA explained variance ratio: `0.491`.

The high-rate ablation also preserved the target-aligned coordinates:

- packet rank: `8`;
- top-k atoms: `8`;
- coefficient bits: `8`;
- fit sparse energy ratio: `1.002`;
- target-coordinate train R2: `0.999979`;
- target PCA explained variance ratio: `0.719`.

Despite this, matched accuracy did not improve. Atom shuffle, coefficient
shuffle, candidate roll, target-only, same-byte text, and Qwen-substitution
controls remain tied or better. This points away from packet quantization as
the immediate failure and toward receiver decode / task relevance: the
target-aligned coordinates are reconstructable on the train candidates, but the
one-step soft-prefix receiver is not using them as causal task information on
held-out disagreement rows.

## Decision

Demote shallow target-PCA alignment plus one-step soft-prefix decoding for the
ARC n8 scouting slice. Do not widen this exact branch.

Promote the next branch only if it changes the receiver objective or packet
semantics, not just the byte budget:

1. target-side logit-margin/transcoder packet trained to predict target answer
   margin deltas rather than hidden coordinates;
2. anchor-relative packet with target candidate-local atoms and explicit
   atom/coeff destructive controls;
3. confidence/error-coded Wyner-Ziv style packet that uses target uncertainty
   as side information before decoding.

## Lay Explanation

We tried translating TinyLlama's clue into Qwen's own coordinate system before
sending it. The translation itself looked numerically clean, and even a larger
packet did not help. That means the current receiver is probably not turning
those coordinates into better answer choices; the next method needs a receiver
or packet that is trained around target behavior, not just hidden-state
alignment.
