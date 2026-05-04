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
- event-triggered zero-subtracted diagnostic:
  `results/source_private_arc_challenge_behavior_atom_decoder_gate_20260504_tinyllama_to_qwen3_disagreement_n16_rank8_top2q4_event_triggered/`
- event-triggered no-subtraction diagnostic:
  `results/source_private_arc_challenge_behavior_atom_decoder_gate_20260504_tinyllama_to_qwen3_disagreement_n16_rank8_top2q4_event_triggered_no_zsub/`
- corruption-noop unweighted diagnostic:
  `results/source_private_arc_challenge_behavior_atom_decoder_gate_20260504_tinyllama_to_qwen3_disagreement_n16_rank8_top2q4_corruption_noop_decoder/`
- corruption-noop weighted diagnostic:
  `results/source_private_arc_challenge_behavior_atom_decoder_gate_20260504_tinyllama_to_qwen3_disagreement_n16_rank8_top2q4_corruption_noop_w01_decoder/`
- corruption-noop higher-weight diagnostic:
  `results/source_private_arc_challenge_behavior_atom_decoder_gate_20260504_tinyllama_to_qwen3_disagreement_n16_rank8_top2q4_corruption_noop_w025_decoder/`
- corruption-noop weighted zero-subtracted diagnostic:
  `results/source_private_arc_challenge_behavior_atom_decoder_gate_20260504_tinyllama_to_qwen3_disagreement_n16_rank8_top2q4_corruption_noop_w01_zsub_decoder/`
- candidate-aligned no-op weighting diagnostic:
  `results/source_private_arc_challenge_behavior_atom_decoder_gate_20260504_tinyllama_to_qwen3_disagreement_n16_rank8_top2q4_candidate_aligned_noop_w005_emphasis/`
- candidate/atom packet-integrity diagnostic:
  `results/source_private_arc_challenge_behavior_atom_decoder_gate_20260504_tinyllama_to_qwen3_disagreement_n16_rank8_top2q4_integrity_candidate_atom_w01/`
- reference synthesis:
  `references/731_behavior_atom_decoder_packet_refs_20260504.md`
  and `references/733_event_triggered_packet_gate_refs_20260504.md`
  and `references/734_corruption_noop_receiver_refs_20260504.md`

## Results

| Variant | Packet | Framed bytes | Matched | Target | Best required control | Worst CI95 low | Fired | Helps | Harms |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| behavior atoms | rank8 top2 q4 | 7 | 0.3750 | 0.2500 | top_atom_knockout, 0.4375 | -0.3750 | 8/16 | 2 | 0 |
| top-1 behavior atoms | rank8 top1 q4 | 4 | 0.2500 | 0.2500 | source_row_shuffle, 0.4375 | -0.4375 | 10/16 | 2 | 2 |
| zero-subtracted behavior atoms | rank8 top2 q4 | 7 | 0.3125 | 0.2500 | qwen_substituted_packet, 0.4375 | -0.3750 | 6/16 | 1 | 0 |
| packet-only innovation decoder | rank8 top2 q4 | 7 | 0.2500 | 0.2500 | qwen_substituted_packet, 0.4375 | -0.4375 | 7/16 | 1 | 1 |
| event-triggered zero-subtracted decoder | rank8 top2 q4 | 7 | 0.3125 | 0.2500 | qwen_substituted_packet, 0.4375 | -0.3750 | 5/16 | 1 | 0 |
| event-triggered decoder, no subtraction | rank8 top2 q4 | 7 | 0.3750 | 0.2500 | top_atom_knockout, 0.4375 | -0.3750 | 8/16 | 2 | 0 |
| corruption-noop decoder, weight 1.0 | rank8 top2 q4 | 7 | 0.2500 | 0.2500 | qwen_substituted_packet, 0.4375 | -0.4375 | 3/16 | 0 | 0 |
| corruption-noop decoder, weight 0.1 | rank8 top2 q4 | 7 | 0.4375 | 0.2500 | candidate_roll, 0.5000 | -0.3125 | 7/16 | 3 | 0 |
| corruption-noop decoder, weight 0.25 | rank8 top2 q4 | 7 | 0.3750 | 0.2500 | top_atom_knockout, 0.5000 | -0.3750 | 7/16 | 2 | 0 |
| corruption-noop decoder, weight 0.1, zero-subtracted | rank8 top2 q4 | 7 | 0.3750 | 0.2500 | top_atom_knockout, 0.4375 | -0.3750 | 7/16 | 2 | 0 |
| candidate-aligned no-op emphasis | rank8 top2 q4 | 7 | 0.4375 | 0.2500 | top_atom_knockout, 0.4375 | -0.3125 | 8/16 | 3 | 0 |
| candidate/atom integrity gate | rank8 top2 q4 | 7 | 0.2500 | 0.2500 | qwen_substituted_packet, 0.4375 | -0.4375 | 0/16 | 0 | 0 |

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

The event-triggered diagnostics add a learned accept/abstain gate trained with
matched packets as possible act events and zero, wrong-row, target-derived,
atom-shuffled, coefficient-shuffled, top-atom-knockout, and candidate-rolled
packets as no-op negatives. The zero-subtracted event gate is cleaner than the
plain zero-subtracted row: it reaches `0.3125` matched accuracy, fires on
`5/16`, helps `1`, harms `0`, and beats target-derived, zero-source,
source-row shuffle, atom shuffle, coefficient shuffle, and candidate
derangement. It still ties or loses to same-source-choice wrong-row,
candidate roll, source-index/rank/score controls, same-byte text, top-atom
knockout, and Qwen substitution.

The no-subtraction event gate preserves the larger matched lift: `0.3750`
matched accuracy versus `0.2500` target-only, firing on `8/16`, helping `2`,
and harming `0`. However, same-source-choice wrong-row and candidate roll tie
matched at `0.3750`, source-index/rank/score controls remain close at
`0.3125`, and top-atom knockout plus Qwen substitution reach `0.4375`. The
event gate therefore improves harm control but not source necessity or atom
causality.

The corruption-to-no-op decoder moves the destructive controls into receiver
training. The unweighted objective is too conservative: with nine corruption
families per matched example, it collapses matched lift to target-only
(`0.2500`). A balanced corruption weight of `0.1` restores and improves the
held-out lift: matched reaches `0.4375` versus target-only `0.2500`, fires on
`7/16`, helps `3`, and harms `0`. However, it still fails strict causality:
candidate roll reaches `0.5000`, top-atom knockout and Qwen substitution tie
matched at `0.4375`, and same-source-choice wrong-row remains high at
`0.3750`.

The no-op residual diagnostics show why this is not yet a positive method.
With corruption weight `0.1`, zero-source and generic source-row shuffle are
partly suppressed, but candidate roll, candidate derangement, atom shuffle,
coefficient shuffle, same-source-choice wrong-row, and top-atom knockout all
decode to residual norms close to matched. Increasing the corruption weight to
`0.25` weakens matched lift before it solves top-atom-knockout; zero-baseline
subtraction also weakens matched lift without clearing the strongest controls.

Condition-specific no-op weighting, with higher weights on candidate roll,
candidate derangement, same-source-choice wrong-row, and top-atom knockout,
preserves the best matched point estimate (`0.4375`) and keeps harms at zero.
It improves same-source-choice wrong-row to `0.2500`, but candidate roll still
ties matched at `0.4375`, top-atom knockout still ties matched at `0.4375`,
and Qwen substitution also ties. Residual norms for candidate roll and
top-atom knockout remain close to matched, so targeted linear no-op pressure is
not sufficient.

The candidate/atom packet-integrity gate is a separate label-free accept/reject
layer trained only from matched packets versus synthetic packet corruptions.
It uses packet-to-target-feature alignment and atom-profile diagnostics rather
than answer labels. The idea is sound as a protocol direction, but this first
linear integrity classifier fails held-out generalization: it accepts `13/16`
matched train packets and `53/144` train corruptions, but accepts `0/16`
held-out matched packets. Matched falls to target-only (`0.2500`) because every
real held-out packet is rejected.

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

The packet-only innovation and event-triggered results further narrow the
diagnosis: subtracting or forbidding target-only receiver bias is necessary but
not sufficient, and a row-level accept/abstain gate alone does not make the
current behavior atoms source-necessary. The corruption-noop and integrity
results narrow it again: no-op receiver training can recover a stronger
no-harm matched lift, but linear behavior atoms still do not carry enough
candidate-aligned causal information for a held-out integrity gate to trust
them. The next live method must change the atom basis so candidate-rolled and
top-atom-knockout packets naturally collapse.

## Decision

Demote the current linear behavior-atom residual decoder and weighted
corruption-noop variant as ICLR-positive methods. Keep the implementation
because it is now the cleanest strict ARC behavior-atom packet harness and
includes same-source-choice wrong-row, zero-packet-baseline subtraction,
corruption-to-no-op receiver training, and no-op residual diagnostics.

Promote the next branch only if it changes the atom basis enough to remove the
candidate-roll/top-atom-knockout/same-source-choice failure:

1. small BatchTopK/DFC/crosscoder/transcoder packet atoms that explicitly
   separate shared, source-private, and target-private behavior features.
2. Reintroduce packet-integrity only after the atom bank has stable held-out
   candidate alignment and top-atom knockout sensitivity.

Lay explanation: we trained the feature clues to point at the kinds of mistakes
Qwen makes, rather than at generic TinyLlama hidden-state variation. That helped
a bit and did not break answers, but fake/blank/damaged clues were still too
competitive. We need the next receiver to listen only to the part of the packet
that cannot be produced without the source row.
