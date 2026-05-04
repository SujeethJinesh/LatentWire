# Residual/Syndrome Sparse Resonance Packet Gate

Date: 2026-05-04

## Question

Can a fixed-byte source-private syndrome over source pairwise answer
preferences transmit useful source innovation when decoded with target-side
scores, without exposing source text, source KV cache, raw source hidden states,
or raw source score vectors?

## Method

The source converts its answer-key-forbidden candidate scores into pairwise
comparison bits, sends a short parity syndrome plus a confidence/check header,
and the receiver decodes the coset member closest to its own target pairwise
score bits. The receiver then applies only the pairwise residual between the
decoded bits and target bits as a target score correction.

This is a Mac-local ARC-Challenge disagreement scout using TinyLlama source
score caches and Qwen3-0.6B target scoring. It tests the Slepian-Wolf/Wyner-Ziv
side-information framing without claiming native GPU throughput or C2C parity.

## Artifacts

- implementation:
  `scripts/build_source_private_arc_challenge_residual_syndrome_packet_gate.py`
- tests:
  `tests/test_build_source_private_arc_challenge_residual_syndrome_packet_gate.py`
- primary n32 / 4-syndrome-bit result:
  `results/source_private_arc_challenge_residual_syndrome_packet_gate_20260504_tinyllama_to_qwen3_disagreement_n32_2b/`
- bit-budget diagnostic n32 / 6-syndrome-bit result:
  `results/source_private_arc_challenge_residual_syndrome_packet_gate_20260504_tinyllama_to_qwen3_disagreement_n32_2b_syndrome6/`
- reference synthesis:
  `references/729_residual_syndrome_packet_refs_20260504.md`

## Results

| Variant | Packet bits | Framed bytes | Matched acc. | Target acc. | Best required control | Worst CI95 low | Fired | Helps | Harms |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 4-bit syndrome | 9 | 2 | 0.21875 | 0.25000 | candidate_derangement, 0.31250 | -0.37500 | 14/32 | 2 | 3 |
| 6-bit syndrome | 11 | 2 | 0.21875 | 0.25000 | qwen_substituted_packet, 0.37500 | -0.32891 | 14/32 | 3 | 4 |

Both variants fail the strict gate. The 6-bit diagnostic spends more syndrome
bits while staying in the same 2-byte framed packet, but does not improve
matched accuracy. The failure is therefore not explained by the smallest
syndrome budget alone.

## Diagnostics

For the 4-bit packet, source-or-target oracle accuracy is 0.53125, target-only
accuracy is 0.25, and source top-1 accuracy is 0.3125. There are source-helpable
rows, but the syndrome receiver does not identify them reliably enough.

For the 6-bit packet, source-or-target oracle accuracy remains 0.53125, while
Qwen-substitution rises to 0.375 and beats the matched TinyLlama packet. The
packet fires on 14/32 rows but has net help -1. Target-side information removal
does not hurt the 6-bit matched row, and wrong parity matrix decoding can match
or beat it. That weakens the specific claim that this implementation is using
target side information in a causally useful way.

## Decision

Demote pairwise score-syndrome packets as currently implemented. They are a
cleaner object than candidate-ECOC/source-index packets, but still collapse to
a weak source score-shape correction and do not beat source-score, same-byte
visible text, or destructive controls.

Promote only branches that change the observable message shape or receiver,
not merely the syndrome length:

1. target-conditioned residual features beyond pairwise score bits, such as
   source hidden behavior atoms or target-side learned residual decoders;
2. same-source-choice wrong-row controls before any new positive claim;
3. if score-surface work continues, use a learned posterior/risk model that
   predicts row-level source innovation directly and report risk/coverage, not
   always-on residual correction.

Lay explanation: we sent Qwen a tiny checksum of TinyLlama's pairwise answer
preferences and let Qwen use its own guesses to decode the checksum. The
checksum did not help enough; it fixed a few questions but broke as many or
more, and simple source-score/text controls still did better.
