# LatentWire COLM_v2 / ICLR Sparse Resonance Pivot

Date: 2026-05-04

## Readiness Status

- COLM_v2: viable as a narrow workshop paper if the claim stays scoped to
  source-private byte-scale packets, strict destructive controls, and honest
  saturation analysis. It is not yet a broad positive-method paper.
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

## Latest Live Triage Update

The current generated triage is
`paper/iclr_colm_v2_live_branch_triage_20260504.md` and should be treated as
the active dashboard.

Promoted for COLM_v2:

- conditional-PQ shared-schema packet: `16/16` disjoint n500 rows pass and
  `4/4` budget-2 rows pass, but cross-family remains `0/28`;
- HellaSwag fixed hybrid candidate packet: full validation accuracy `0.532464`
  vs candidate-only `0.526688`, CI95 low `+0.002888`, useful mainly as a
  systems/privacy packet row.

Alive but not yet a method:

- target self-resonance oracle soft-prefix capacity: `3/3` tiny oracle rows
  pass and best optimized agreement reaches `0.937500`; this is only headroom
  evidence because the prefix is optimized on evaluation rows.

Ruled out or weakened for the current implementation family:

- public-zscore and public-SVD conditional-PQ held-out-family rescues;
- corruption-to-noop conditional-PQ receiver;
- ARC PCA/target-aligned soft-prefix packets;
- HellaSwag protected-rival, receiver-calibrated, harm-bucket, top2-bucket,
  denoising-syndrome, and sparse/common-basis score switchers;
- target self-resonance chunk/distill/query-resampler encoders;
- source-conditioned source-hidden/codebook/refinement target-native receivers.

The next exact gate is no longer "try another target-prefix decoder." First
backport the live triage into COLM_v2 tables/figures, then run a small
complementarity-frontier diagnostic: isolate rows where the target is wrong and
source top1/top2 could help, and test whether any source-private packet field
has source-causal signal beyond source-choice, wrong-row, candidate-roll,
same-byte, and target-derived controls.

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
PCA-on-public-innovation packets as currently implemented. Also demote the
current target-native resonance receiver family: oracle soft-prefixes show
capacity, but held-out learned/source-conditioned receivers do not yet pass
controls. The next branch should diagnose complementarity before widening:

1. identify whether there are stable rows where source information can help
   beyond target-only and source-choice shortcuts;
2. only train another receiver if that diagnostic shows separable
   source-causal signal;
3. if it does, prefer a new basis or packet field, not another shallow
   chunk/query/source-to-prefix decoder;
4. if it does not, pivot to a benchmark/method where the source has measurable
   complementarity.

Do not widen to larger slices until the n8/n16 gate beats target-only,
target-derived, atom-shuffle, candidate-roll, source-index/rank/score, and
same-byte controls.

Lay explanation: we tried sending Qwen only a few compressed "feature clue"
numbers from TinyLlama instead of TinyLlama's text or cache. The packet was
very small, but Qwen did not use it reliably; simpler no-source and
target-derived hints were just as good or better. The interface is now ready,
but the feature basis is not good enough yet.
