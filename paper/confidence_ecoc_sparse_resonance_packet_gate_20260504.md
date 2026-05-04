# Confidence/ECOC Sparse Resonance Packet ARC Gate

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

This gate tested a selective 2-byte side-information packet after hidden
coordinate packets and always-on behavior residual packets failed.

The packet contains:

1. an 8-bit ECOC-style candidate codeword;
2. an 8-bit reliability/header field: source top-2 identity, source margin bin,
   source entropy bin, and a parity/check bit.

The receiver decodes the packet against Qwen3 target scores and fires only when
the target is uncertain and the packet passes train-calibrated reliability and
parity checks. If the gate does not fire, the target-only decision is preserved.

## Artifacts

- Code:
  `scripts/build_source_private_arc_challenge_confidence_ecoc_packet_gate.py`.
- Tests:
  `tests/test_build_source_private_arc_challenge_confidence_ecoc_packet_gate.py`.
- Initial source-confidence scout:
  `results/source_private_arc_challenge_confidence_ecoc_packet_gate_20260504_tinyllama_to_qwen3_disagreement_n16_2b/`.
- Main uncertainty-gated scout:
  `results/source_private_arc_challenge_confidence_ecoc_packet_gate_20260504_tinyllama_to_qwen3_disagreement_n16_2b_uncertainty/`.

## Results

| run | packet | matched | target-only | best required control | fired | helps / harms | worst CI95 low | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| source confidence + disagreement | 2 B/row | 0.1875 | 0.2500 | Qwen-substitution 0.3750 | 5/16 | 1 / 2 | -0.4375 | no |
| target-uncertainty gated | 2 B/row | 0.2500 | 0.2500 | source-row shuffle 0.3125 | 0/16 | 0 / 0 | -0.3750 | no |

Main uncertainty-gated diagnostics:

- target-only correct: `4/16`;
- TinyLlama source top1 correct: `5/16`;
- source-or-target oracle correct: `8/16`;
- source-helpable target-wrong rows: `4/16`;
- source-harm-risk target-right rows: `3/16`;
- Qwen-substitution correct: `4/16`;
- packet bytes: `16` bits / `2` B per row, `64` B cache-line accounting,
  `128` B DMA-burst accounting;
- native serving throughput was not measured.

## Interpretation

The simple confidence/ECOC packet is not a positive method. The looser version
fires and causes net harm on held-out rows; the stricter uncertainty-gated
version avoids harm by never firing. This rules out the current hand-thresholded
source-margin/source-entropy header as a reliable receiver gate.

The result is still useful: the source-or-target oracle is `0.50` while
target-only is `0.25`, so the slice has real repair headroom. The failure is
not packet byte budget; it is helpability prediction. The next branch should
learn an event-triggered innovation/defer gate over target uncertainty, source
packet reliability, source-target disagreement, and source/target score-shape
features, with calibration split and fired-row helps/harms reported.

## Decision

Demote simple hand-binned confidence/ECOC packets. Promote an
event-triggered innovation controller:

1. keep the 2-4 B packet schema;
2. train a tiny helpability/defer model on calibration rows;
3. fire only when predicted override value has positive lower bound;
4. compare against target-only, source-index/rank/score, source-score
   quantization, same-byte text, wrong-row, bit/header shuffles,
   candidate-roll, Qwen-substitution, and learned target-derived gates.

## Lay Explanation

We tried sending Qwen a tiny coded hint plus a confidence stamp. When we let
Qwen listen too often, the hint fixed one question but broke two. When we made
Qwen listen only when it was uncertain, it never listened on the test rows.
There are still rows where either TinyLlama or Qwen knows the answer, but the
simple confidence stamp cannot tell which hints are safe.
