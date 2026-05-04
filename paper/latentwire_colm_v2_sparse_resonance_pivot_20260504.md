# LatentWire COLM_v2 / ICLR Sparse Resonance Pivot

Date: 2026-05-04

## Readiness Status

- COLM_v1: freeze except for cleanup, reproducibility, style, citation, and
  packaging fixes. The defensible story remains a scoped source-private packet
  evaluation paper with explicit source-index/rank limitations.
- COLM_v2: newly opened. It should inherit the strict controls from COLM_v1
  but pivot the method to Sparse Resonance Packets.
- ICLR: not ready. The blocker is still a positive learned receiver that beats
  target-only, explicit source-index/rank/score, same-byte text, wrong-row,
  atom-shuffle, candidate-roll, and source-family substitution controls with
  paired uncertainty.

## Current Story

LatentWire_v2 should not copy C2C's dense KV-cache fusion. C2C is the strongest
high-bandwidth semantic communication baseline. LatentWire's meaningful win
condition is lower-rate, source-private, interpretable, quantized packets that
are more hardware-friendly and auditable while remaining competitive on task
accuracy.

The desired ICLR claim is:

> Sparse Resonance Packets provide a low-rate, source-private, interpretable
> alternative to dense cache fusion for model-to-model communication, achieving
> favorable utility per byte under strict leakage and destructive controls.

## First Implemented Gate

Implemented a train-fit sparse PCA packet mode in the strict ARC soft-prefix
wrapper:

- source feature mode:
  `hf_choice_hidden_public_innovation_sparse_pca_packet_candidate_pool`;
- packet: top-k PCA atom coordinates from answer-key-forbidden TinyLlama
  candidate hidden-state public innovations;
- quantization: signed uniform coefficient quantization;
- receiver: Qwen3 target-native soft prefix;
- new control: `atom_shuffle`, which preserves coefficient magnitudes but
  rolls atom identities;
- systems sideband: packet bytes per candidate / estimated packet bytes per row
  are reported, explicitly not as native serving throughput.

## Runs

| Run | Packet | Estimated packet bytes / row | Matched | Best required control | Pass |
|---|---:|---:|---:|---:|---|
| `n8_pca_top2q3` | rank 4, top-2, 3-bit coeffs | 5.0 | 0.250 | 0.625 (`target_derived_prefix`) | False |
| `n8_pca_top4q4_noresid` | rank 8, top-4, 4-bit coeffs | 14.0 | 0.250 | 0.625 (`slots_only_prefix`) | False |

For the rerun artifact with final systems sideband, the strict controls on the
`n8_pca_top2q3` gate were:

| Control | Accuracy | Matched - Control | CI95 low |
|---|---:|---:|---:|
| `target_only` | 0.500 | -0.250 | -0.625 |
| `slots_only_prefix` | 0.500 | -0.250 | -0.625 |
| `target_derived_prefix` | 0.625 | -0.375 | -0.750 |
| `zero_source` | 0.500 | -0.250 | -0.625 |
| `source_row_shuffle` | 0.375 | -0.125 | -0.375 |
| `atom_shuffle` | 0.375 | -0.125 | -0.441 |
| `candidate_roll` | 0.250 | 0.000 | 0.000 |
| `source_rank_control` | 0.125 | 0.125 | -0.250 |
| `source_score_control` | 0.125 | 0.125 | -0.250 |
| `same_byte_visible_text` | 0.500 | -0.250 | -0.625 |
| `qwen_substituted_packet` | 0.625 | -0.375 | -0.750 |

## Interpretation

This first PCA packet is not a positive method. The source-private sparse
packet is tiny and cleanly instrumented, but the target-native soft-prefix
receiver does not use it better than target-derived, target-only, same-byte, or
Qwen-substitution controls.

The failure is more likely a basis/selection problem than a channel-format
problem:

- the top-2/rank-4 packet retains only about 0.509 train-fit PCA variance;
- atom-shuffle and source-row-shuffle are close to or better than matched,
  which means atom identity is not yet causally meaningful to the receiver;
- candidate-roll ties matched, showing the receiver is not reliably using
  candidate-specific packet fields;
- target-derived controls dominate, so target cache effects remain the main
  failure mode.

## Decision

Promote Sparse Resonance Packets as the COLM_v2/ICLR story, but demote plain
PCA-on-public-innovation packets as currently implemented. The next branch
should improve the basis before widening:

1. train a target-aligned or shared basis rather than source-only PCA;
2. test CCA/Procrustes/anchor-relative coordinates before SAE scale-up;
3. add coefficient shuffle and top-atom knockout controls;
4. only then try small SAE/crosscoder atoms on Mac or NVIDIA.

Do not widen to larger slices until the n8/n16 gate beats target-only,
target-derived, atom-shuffle, candidate-roll, source-index/rank/score, and
same-byte controls.

Lay explanation: we tried sending Qwen only a few compressed "feature clue"
numbers from TinyLlama instead of TinyLlama's text or cache. The packet was
very small, but Qwen did not use it reliably; simpler no-source and
target-derived hints were just as good or better. The interface is now ready,
but the feature basis is not good enough yet.
