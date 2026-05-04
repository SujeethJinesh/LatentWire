# Innovation-Defer Sparse Resonance Packet ARC Gate

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

This gate keeps the 2-byte confidence/ECOC packet transport but replaces the
hand-binned receiver rule with a learned event-triggered defer controller.

The target-local receiver:

1. builds a 2-byte source-private packet from TinyLlama source scores:
   candidate codeword plus reliability/header bits;
2. computes target-local score-shape features from Qwen3;
3. fits a ridge value model on the first half of validation disagreement rows
   to predict whether overriding to the source packet will help or harm;
4. selects a firing threshold on the second half of validation disagreement
   rows;
5. evaluates once on held-out test disagreement rows.

This is a learned receiver gate, not a new source encoder.

## Artifacts

- Code:
  `scripts/build_source_private_arc_challenge_innovation_defer_packet_gate.py`.
- Tests:
  `tests/test_build_source_private_arc_challenge_innovation_defer_packet_gate.py`.
- Main scout:
  `results/source_private_arc_challenge_innovation_defer_packet_gate_20260504_tinyllama_to_qwen3_disagreement_n64_2b/`.
- Ridge sensitivity:
  `results/source_private_arc_challenge_innovation_defer_packet_gate_20260504_tinyllama_to_qwen3_disagreement_n64_2b_ridge01/`.

## Results

| run | packet | matched | target-only | best required control | fired | helps / harms | worst CI95 low | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ridge 3.0 | 2 B/row | 0.2344 | 0.2344 | same-byte text 0.3594 | 3/64 | 0 / 0 | -0.2500 | no |
| ridge 0.1 | 2 B/row | 0.2188 | 0.2344 | same-byte text 0.3594 | 9/64 | 1 / 2 | -0.2500 | no |

Main scout diagnostics:

- value-model fit R2: `0.105`;
- calibration fired rows: `1/32`;
- test fired rows: `3/64`;
- target-only correct: `15/64`;
- TinyLlama source top1 correct: `16/64`;
- source-or-target oracle correct: `30/64`;
- source-helpable target-wrong rows: `15/64`;
- source-harm-risk target-right rows: `14/64`;
- same-byte visible text: `23/64`;
- packet bytes: `16` bits / `2` B per row, `64` B cache-line accounting,
  `128` B DMA-burst accounting;
- native serving throughput was not measured.

## Interpretation

The learned event-triggered receiver is safer than always-on source override,
but it is not useful enough. With ridge `3.0`, it mostly abstains and ties
target-only. With ridge `0.1`, it fires more often but causes net harm. The
source-or-target oracle remains high, so the task still has repair headroom;
the failure is that observable candidate-packet and target-score features do
not identify safe source overrides well enough.

This weakens the whole candidate-override branch. Source-index, source-rank,
source-score, and same-byte text remain too strong. The next branch should stop
making the source-selected candidate the main transmitted object.

## Decision

Demote simple learned candidate-override defer gates. Promote a residual /
syndrome packet with target side information:

1. encode source-target residual diagnostics rather than source top1 identity;
2. decode against target logits as side information, in a Wyner-Ziv /
   Slepian-Wolf style framing;
3. include source-index/rank/score, same-byte text, wrong-row, label/candidate
   shuffle, target-derived syndrome, and cross-family substitution controls;
4. keep oracle headroom and fired-row helps/harms in the main table.

## Lay Explanation

We taught Qwen a small rule for when to trust TinyLlama's tiny hint. The rule
was cautious and usually refused to listen. When made less cautious, it listened
more but broke more answers than it fixed. That means the source still has
useful information, but sending "which answer the source picked" is not the
right shape of message.
