# Hidden-Atom Decoder Sparse Resonance Packet Gate

Date: 2026-05-04

## Question

Can a compact source-hidden atom packet become useful if the receiver decodes it
with a target-conditioned residual model, rather than reconstructing a
soft-prefix or sending source-choice/score signals?

## Method

The source extracts answer-key-forbidden TinyLlama candidate hidden states,
removes public question/choice hashed features with a train-only ridge
residualizer, fits train-only PCA atoms, and sends only sparse top-k atom IDs
plus quantized coefficients. The receiver sees Qwen3 target candidate scores
and the sparse packet, then predicts a target score residual with a small
target-conditioned ridge decoder. A train-only rule selects when to apply the
residual.

This is a source-private packet in the receiver threat model: no source text,
source KV cache, raw source hidden vector, or raw source score vector is exposed
to the receiver. The packet is still evaluated against explicit source-index,
source-rank, source-score, quantized score, same-byte text, target-derived, and
atom-destruction controls.

## Artifacts

- implementation:
  `scripts/build_source_private_arc_challenge_hidden_atom_decoder_gate.py`
- tests:
  `tests/test_build_source_private_arc_challenge_hidden_atom_decoder_gate.py`
- primary scout:
  `results/source_private_arc_challenge_hidden_atom_decoder_gate_20260504_tinyllama_to_qwen3_disagreement_n16_rank8_top2q4/`
- regularization diagnostic:
  `results/source_private_arc_challenge_hidden_atom_decoder_gate_20260504_tinyllama_to_qwen3_disagreement_n16_rank8_top2q4_ridge100/`
- reference synthesis:
  `references/730_hidden_atom_decoder_packet_refs_20260504.md`

## Results

| Variant | Packet | Framed bytes | Matched | Target | Best required control | Worst CI95 low | Fired | Helps | Harms |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| ridge 10 | rank8 top2 q4 | 7 | 0.3125 | 0.2500 | top_atom_knockout, 0.4375 | -0.3750 | 12/16 | 4 | 3 |
| ridge 100 | rank8 top2 q4 | 7 | 0.2500 | 0.2500 | qwen_substituted_packet, 0.4375 | -0.4375 | 0/16 | 0 | 0 |

The ridge-10 scout is the first May 4 ARC SRP variant with nonzero positive
target lift: matched beats target-only by one row and has net help +1. It still
fails the strict gate because target-derived/zero-source/source-score/same-byte
controls tie matched, Qwen-substitution reaches 0.4375, and top-atom knockout
also reaches 0.4375. The top-atom knockout result means the largest transmitted
atom is often harmful rather than causally useful.

The ridge-100 diagnostic collapses to a no-fire receiver and ties target-only.
This weakens the "just regularize harder" explanation. The current failure is
better explained as atom/basis semantics and helpability calibration, not only
decoder overfit magnitude.

## Diagnostics

For ridge 10, the target-conditioned decoder fit R2 is 0.508 versus 0.259 for
the target-only decoder, so the source packet does add train-predictive signal.
Held out, however, zero-source and target-derived packet controls tie matched,
and destructive controls do not collapse cleanly. Atom shuffle drops to 0.1875,
which suggests some atom ordering matters, but coefficient shuffle and
candidate roll tie matched, and top-atom knockout improves.

The source-or-target oracle is 0.50, target-only is 0.25, source top-1 is
0.3125, and Qwen substitution is 0.4375. There is headroom, but the PCA atom
packet is not yet a reliable way to select safe repairs.

## Decision

Demote plain PCA hidden atoms with a ridge target-conditioned residual decoder.
Keep the code path because it is the smallest strict ARC hidden-atom packet
harness, but do not widen this exact configuration.

Promote a behavior-trained atom basis rather than a reconstruction PCA basis:

1. Kalman-gated behavior-innovation SRP: source sends hidden/behavior atoms,
   target predicts a bounded gain from uncertainty and source residual
   reliability.
2. BatchTopK behavior-crosscoder/transcoder atoms: train atoms directly for
   target-margin residual behavior, then packetize atom IDs and coefficients.
3. Same-source-choice wrong-row controls must be first-class before claiming
   source-hidden communication.

Lay explanation: we compressed TinyLlama's hidden evidence into a few learned
feature clues and taught Qwen a small correction rule. Qwen improved slightly
over its own guesses, but removing the strongest clue improved even more. That
means the current atoms are not reliable evidence atoms yet.
