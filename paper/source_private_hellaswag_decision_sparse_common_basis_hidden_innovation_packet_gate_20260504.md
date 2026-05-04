# HellaSwag Decision-Sparse Common-Basis Hidden Packet Gate

Date: 2026-05-04

## Readiness

- Current paper readiness: COLM remains plausible; ICLR is still blocked by a
  learned positive method that beats the current source-private packet.
- Current story: the fixed HellaSwag packet is robust under physical
  candidate-text permutation, but shallow learned hidden/common-basis packets
  are not yet adding stable accuracy beyond the packet candidate.
- Exact remaining blocker: a learned latent/common-basis method must beat
  packet-only, compact candidate-only, and target-only decoders with paired
  uncertainty and destructive source/atom controls.

## Lay Explanation

This gate asks whether TinyLlama can send Qwen more than just "I pick answer
2" while still using only one byte. We first train a shared coordinate system
between TinyLlama and Qwen hidden states on train examples. Then we train a
sparse autoencoder-style basis with a decision loss, so some hidden features
should light up when a candidate looks correct. At evaluation time, TinyLlama
sends one tiny packet: the candidate id plus one sparse atom id. Qwen then
uses its own matching atom features to decide whether that extra atom helps.

The receiver never sees TinyLlama text, KV cache, raw hidden vectors, raw score
vectors, logits, or raw scores.

## Artifact

`results/source_private_hellaswag_decision_sparse_common_basis_hidden_innovation_packet_gate_20260504_validation1024_2048/hellaswag_decision_sparse_common_basis_hidden_innovation_packet_gate.json`

Supporting files:

- `frontier_rows.csv`
- `control_rows.csv`
- `default_blocks.csv`
- `predictions.jsonl`
- `manifest.json`

## Method

- Calibration surface: official HellaSwag train only.
- Retained calibration rows: `1487`; split `1115/372`.
- Evaluation surface: frozen HellaSwag validation slice `1024:2048`, `1024`
  rows.
- Source model: TinyLlama-1.1B-Chat final-layer choice-mean hidden states.
- Target side: Qwen scores, Qwen fixed alternatives, and Qwen common-basis atom
  features.
- Shared basis: train-only PCA plus linear CCA/SVD-style TinyLlama/Qwen shared
  coordinates.
- Sparse encoder: train-only SAE-style encoder on source shared candidate
  vectors with reconstruction loss, L1 activation penalty, and binary decision
  loss for correct candidate detection.
- Packet contract: one atom slot plus candidate low bits, max `256` symbols,
  `1B` raw / `4B` framed.

Promotion required the train-dev-selected row to beat packet-only by at least
`0.010` with positive paired CI95 low, beat compact-candidate and Qwen-side-only
common-basis decoders by at least `0.010`, degrade under top-atom knockout, be
positive on at least `4/5` contiguous blocks, and separate from destructive
atom/source controls.

## Result

The gate fails.

| Row | Accuracy | Delta vs Packet | CI95 Low |
|---|---:|---:|---:|
| Packet-only | `0.501953` | `0.000000` | `0.000000` |
| Qwen-side-only common-basis decoder | `0.464844` | `-0.037109` | `-0.064453` |
| Compact-candidate common-basis decoder | `0.503906` | `+0.001953` | `0.000000` |
| Train-dev-selected sparse common-basis packet | `0.503906` | `+0.001953` | `-0.001953` |
| Best diagnostic sparse common-basis packet | `0.503906` | `+0.001953` | `-0.001953` |

The selected row is `cca_pca64_d8_sae64_top2_dw0p2_l10p001` with decoder ridge
`100.0`. It uses `64` trained SAE atoms, transmits `63` possible atoms plus a
reserved zero slot, and produces `88` unique eval codes. The SAE itself is not
collapsed during training: reconstruction loss is `0.006239`, train decision
accuracy is `0.711211`, and active rate is `0.548332`. However, every eval row
has a nonzero transmitted atom, so the discrete atom channel is not sparse at
the per-example packet level.

Controls for the selected row:

| Control | Accuracy | Delta vs Packet |
|---|---:|---:|
| source-shared shuffle before encoding | `0.503906` | `+0.001953` |
| atom-index permutation mismatch | `0.504883` | `+0.002930` |
| SAE label-permutation encoder | `0.505859` | `+0.003906` |
| top-atom knockout | `0.499023` | `-0.002930` |
| row-shuffled sparse atom code | `0.283203` | `-0.218750` |
| candidate-roll code | `0.235352` | `-0.266602` |
| random same-byte code | `0.277344` | `-0.224609` |
| zero-source code | `0.295898` | `-0.206055` |
| label-permutation decoder | `0.330078` | `-0.171875` |

Block deltas versus packet-only are `+0.004878`, `0.000000`, `+0.009756`,
`0.000000`, and `-0.004902`, so block stability fails.

## Interpretation

This weakens shallow decision-supervised SAE/common-basis packets on the
current HellaSwag surface. There is a small atom signal: top-atom knockout
drops the selected row below packet-only. But the signal is too small, is not
stable by paired CI or blocks, and is not source-specific enough because
source-row shuffle, atom-index permutation, and label-permuted SAE controls
match or exceed the selected row.

The result strengthens the reviewer-facing branch discipline: do not claim SAE
common-basis transfer, universal sparse features, or latent reasoning from this
gate. The current learned method is saturated by the candidate packet and
target-side decoder surface.

## Decision

Mark this branch as weakened, not promoted.

Next exact gate: build the systems boundary split requested by the systems
review, with separate cached-source communication-object and end-to-end
source-scoring rows for the current strongest packet. The method next branch
should not be another shallow atom code on this same surface unless it changes
the information bottleneck substantially, for example a nonlinear resampler
with an explicit no-source-cache target-only control or a benchmark with more
source/target complementarity.

## Contribution Status

Defensible:

1. A reviewer-clean negative ablation for decision-supervised sparse
   common-basis hidden packets.
2. Atom-level controls showing why the branch does not earn an interpretability
   or source-specific transfer claim.
3. A sharper novelty boundary against SAE universality, sparse crosscoders,
   prefix/gist tokens, latent reasoning, and KV-cache communication.

Not promoted:

1. SAE/common-basis hidden packet as a positive ICLR method.
2. Universal latent language or cross-model sparse feature transfer.
3. C2C/KVComm/DroidSpeak/TurboQuant superiority.

