# Behavior-Atom Decoder Sparse Resonance Packet Gate

Date: 2026-05-04

## Question

Can source-private hidden atom packets become more causally useful if the atom
basis is trained toward the target model's behavioral residual needs, rather
than toward unsupervised source-hidden variance?

## Method

The source extracts answer-key-forbidden TinyLlama candidate hidden states and
removes public question/choice hashed features with a train-only residualizer.
Instead of fitting PCA atoms, this branch fits behavior-supervised atom
directions from the train split by taking the cross-covariance between source
hidden innovations and target residual behavior features. At evaluation time,
the source transmits only sparse top-k atom IDs plus quantized coefficients.

The receiver sees Qwen3 target candidate scores and the sparse packet, then
predicts a target score residual with a small target-conditioned ridge decoder.
The gate includes same-source-choice wrong-row packets from the start, because a
packet that merely encodes the source's selected answer should not pass.

The script also supports zero-packet-baseline subtraction. In that mode, the
receiver subtracts the residual it would have produced from an empty packet, so
only source-specific packet information can move the target scores.

## Artifacts

- implementation:
  `scripts/build_source_private_arc_challenge_behavior_atom_decoder_gate.py`
- tests:
  `tests/test_build_source_private_arc_challenge_behavior_atom_decoder_gate.py`
- primary scout:
  `results/source_private_arc_challenge_behavior_atom_decoder_gate_20260504_tinyllama_to_qwen3_disagreement_n16_rank8_top2q4/`
- top-1 diagnostic:
  `results/source_private_arc_challenge_behavior_atom_decoder_gate_20260504_tinyllama_to_qwen3_disagreement_n16_rank8_top1q4/`
- zero-baseline-subtracted diagnostic:
  `results/source_private_arc_challenge_behavior_atom_decoder_gate_20260504_tinyllama_to_qwen3_disagreement_n16_rank8_top2q4_zsub/`
- packet-only innovation diagnostic:
  `results/source_private_arc_challenge_behavior_atom_decoder_gate_20260504_tinyllama_to_qwen3_disagreement_n16_rank8_top2q4_packet_innovation/`
- reference synthesis:
  `references/731_behavior_atom_decoder_packet_refs_20260504.md`

## Results

| Variant | Packet | Framed bytes | Matched | Target | Best required control | Worst CI95 low | Fired | Helps | Harms |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| behavior atoms | rank8 top2 q4 | 7 | 0.3750 | 0.2500 | top_atom_knockout, 0.4375 | -0.3750 | 8/16 | 2 | 0 |
| top-1 behavior atoms | rank8 top1 q4 | 4 | 0.2500 | 0.2500 | source_row_shuffle, 0.4375 | -0.4375 | 10/16 | 2 | 2 |
| zero-subtracted behavior atoms | rank8 top2 q4 | 7 | 0.3125 | 0.2500 | qwen_substituted_packet, 0.4375 | -0.3750 | 6/16 | 1 | 0 |
| packet-only innovation decoder | rank8 top2 q4 | 7 | 0.2500 | 0.2500 | qwen_substituted_packet, 0.4375 | -0.4375 | 7/16 | 1 | 1 |

The primary behavior-atom scout is a partial positive signal: matched accuracy
beats target-only by two rows and has helped `2`, harmed `0`. It also beats
atom shuffle, coefficient shuffle, target-derived packets, same-byte visible
text, and generic source-row shuffle on this slice.

It still fails the strict gate. Zero-source and same-source-choice wrong-row
tie matched at `0.3750`, candidate roll ties matched, Qwen substitution reaches
`0.4375`, and top-atom knockout also reaches `0.4375`. That means the current
decoder still contains target-side/no-source correction and does not establish
that the transmitted behavior atoms are source-necessary.

The top-1 diagnostic collapses to target-only and harms as often as it helps.
The zero-baseline-subtracted diagnostic is cleaner but weaker: it beats
zero-source and target-only by one row, with no harms, but loses to
same-source-choice wrong-row, top-atom knockout, source-score controls, and
Qwen substitution.

The packet-only innovation diagnostic removes the receiver intercept entirely:
its no-intercept decoder can only produce a residual from packet-dependent
features, so a zero packet is guaranteed to decode to zero residual. This
rules out the specific zero-source shortcut, but the held-out row loses the
positive signal. Matched ties target-only, zero-source, same-source-choice
wrong-row, same-byte visible text, top-atom knockout, and candidate derangement
at `0.2500`; it loses to source-index/rank/score controls at `0.3125`,
candidate roll at `0.3125`, coefficient shuffle at `0.3750`, and
Qwen-substitution at `0.4375`. The train decoder still fits strongly
(`fit_r2 = 0.653`) and the train gate reaches `0.75` accuracy, so the failure is
held-out receiver generalization and harm calibration rather than inability to
fit train residuals.

## Diagnostics

The behavior basis fit is train-predictive but not held-out causal enough. For
the primary scout, packet-to-behavior fit R2 is `0.325`, the
target-conditioned decoder fit R2 is `0.546`, and the train-selected gate fires
on half of held-out rows. The no-harm held-out behavior is encouraging, but the
same-source-choice and zero-source controls prevent a positive claim.

The source-or-target oracle remains `0.50` on the n16 slice while target-only
is `0.25` and source top-1 is `0.3125`. There is still repair headroom. The
failure is now narrowed to source-specific helpability and atom causality, not
absence of source signal.

The packet-only innovation result further narrows the diagnosis: subtracting or
forbidding target-only receiver bias is necessary but not sufficient. Linear
packet-dependent interactions do not identify when the packet should safely
override Qwen on this n16 slice. The next live receiver therefore needs an
explicit accept/abstain or corruption-to-no-op objective, or a behavior-trained
DFC/crosscoder atom bank that separates source-private innovation from shared
and target-private features.

## Decision

Demote the current linear behavior-atom residual decoder as an ICLR-positive
method. Keep the implementation because it is now the cleanest strict ARC
behavior-atom packet harness and includes same-source-choice wrong-row and
zero-packet-baseline subtraction.

Promote the next branch only if it changes the receiver enough to remove the
zero-source/same-source-choice failure:

1. event-triggered accept/abstain decoder: train matched packets as possible
   act events and corrupted/wrong-row packets as no-op events, so the receiver
   returns target-only unless packet gain is predicted to exceed harm risk;
2. consistency/corruption-trained decoder: train the receiver so matched
   packets are stable under coefficient noise but wrong-row and atom-ID
   permutations collapse;
3. small BatchTopK/DFC/crosscoder/transcoder packet atoms that explicitly
   separate shared, source-private, and target-private behavior features.

Lay explanation: we trained the feature clues to point at the kinds of mistakes
Qwen makes, rather than at generic TinyLlama hidden-state variation. That helped
a bit and did not break answers, but fake/blank/damaged clues were still too
competitive. We need the next receiver to listen only to the part of the packet
that cannot be produced without the source row.
