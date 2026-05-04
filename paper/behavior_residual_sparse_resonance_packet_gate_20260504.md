# Behavior-Residual Sparse Resonance Packet ARC Gate

Date: 2026-05-04

## Readiness Status

- COLM_v1: frozen except for cleanup/reproducibility/style/citation/build
  fixes.
- COLM_v2: active Sparse Resonance Packet pivot, still without a positive
  strict SRP gate.
- ICLR: blocked until a source-private packet beats target-only,
  source-index/rank/score, same-byte text, target-derived, destructive packet
  controls, and source-family substitution with positive paired uncertainty.

## Method Tested

After source-only PCA and target-aligned PCA packets failed, this gate changed
the receiver objective from hidden-coordinate reconstruction to target behavior
correction.

The packet builder:

1. extracts answer-key-forbidden TinyLlama candidate hidden states;
2. removes train-only public question/choice-hash predictable components;
3. scores the same rows with Qwen3 using the full multiple-choice prompt;
4. fits a train-only ridge map from TinyLlama candidate innovations to
   `gold_one_hot - target_probability` behavior residuals;
5. transmits a candidate-local sparse packet: top-k candidate atom IDs plus
   quantized residual coefficients;
6. decodes by adding the packet as a small correction to Qwen3 target scores.

This is a small target-side residual decoder, not a soft-prefix receiver.

## Artifacts

- Code:
  `scripts/build_source_private_arc_challenge_behavior_residual_packet_gate.py`.
- Tests:
  `tests/test_build_source_private_arc_challenge_behavior_residual_packet_gate.py`.
- Main corrected-prompt run:
  `results/source_private_arc_challenge_behavior_residual_packet_gate_20260504_tinyllama_to_qwen3_disagreement_n8_top2q4_corrected_prompt/`.
- Conservative fixed-weight run:
  `results/source_private_arc_challenge_behavior_residual_packet_gate_20260504_tinyllama_to_qwen3_disagreement_n8_top2q4_corrected_prompt_w1/`.

## Results

| run | packet | matched | target-only | best required control | worst CI95 low | pass |
|---|---:|---:|---:|---:|---:|---|
| top2 q4, train-selected weight | 1.5 B/row | 0.375 | 0.375 | Qwen-substitution 0.625 | -0.750 | no |
| top2 q4, fixed weight 1.0 | 1.5 B/row | 0.375 | 0.375 | Qwen-substitution 0.625 | -0.750 | no |

Important diagnostics from the corrected-prompt train-selected run:

- source behavior-map fit R2: `0.911`;
- selected residual weight on train: `8.0`;
- packet payload: `12` bits / `1.5` B per row, framed to `2` B;
- cache-line accounting: `64` B/row;
- DMA-burst accounting: `128` B/row;
- target-derived packet: `0.000`;
- atom shuffle: `0.125`;
- coefficient shuffle: `0.125`;
- candidate roll: `0.125`;
- top-atom knockout: `0.500`;
- Qwen-substitution: `0.625`.

The initial run used a target scoring prompt without an explicit choices list;
that was treated as a scout-only invalid surface and corrected before the
reported runs above. The corrected gate uses the same full multiple-choice
prompt style as the strict soft-prefix wrapper.

## Interpretation

Behavior residual packets are not dead, but this simple ridge/transcoder
variant is not a positive method. It fits train behavior well but does not
generalize beyond target-only on the held-out n8 disagreement rows. More
importantly, top-atom knockout is better than the matched packet, which means
the strongest transmitted residual atom is often harmful rather than causally
useful.

This diagnoses receiver/objective failure more sharply:

- hidden-coordinate packet failure was not only a basis problem;
- naive behavior-supervised residual decoding overfits the tiny train slice;
- simply lowering residual weight does not recover a positive row;
- the next branch needs selective harm control or a more discrete
  side-information code, not another always-on residual correction.

## Decision

Demote the always-on ridge behavior-residual packet. Do not widen this exact
variant.

Promote a selective confidence/error-coded packet branch:

1. emit a tiny top1/top2/ECOC-style candidate code plus confidence header;
2. decode only when target uncertainty is high and source reliability is high;
3. include parity/header shuffles, candidate-roll, source-index/rank/score,
   same-byte text, target-derived, and Qwen-substitution controls;
4. measure fired-row helps/harms separately from total accuracy.

This aligns with the new confidence/error-coded side-information memo and the
behavior-transcoder scout recommendations.

## Lay Explanation

We stopped trying to send hidden-state coordinates and instead sent a tiny
"how should Qwen change its answer scores?" packet. It learned the training
questions too well and then did not improve new questions. When we removed the
strongest packet clue, results got better, so the packet is often pushing Qwen
in the wrong direction. The next version should fire only when the target is
uncertain and the source clue looks trustworthy.
