# HellaSwag Linear Crosscoder Hidden-Code Scout

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains plausible; ICLR is still blocked by a
  positive learned latent/common-basis method.
- Current story: compact one-byte HellaSwag packets remain the strongest
  method and systems row. Shallow score, hidden, anchor-relative, and now
  linear crosscoder/CCA hidden-code branches do not robustly beat packet-only.
- Exact remaining blocker: a nonlinear joint connector, cross-attention
  resampler, or less packet-saturated benchmark must produce a train-only
  positive method with paired uncertainty and destructive controls.

## Lay Explanation

This scout tries to teach TinyLlama and Qwen a shared hidden-coordinate system
from train examples only. It first compresses each model's candidate hidden
states with PCA, then finds correlated directions between the two models, like
a small linear translation table. TinyLlama then sends only a one-byte code
derived from its side of that shared coordinate system, and Qwen tries to use
that code plus its own hidden/score side information.

The receiver never sees TinyLlama's text, KV cache, raw hidden vectors, or raw
score vectors.

## Artifacts

Primary reduced sweep:

`results/source_private_hellaswag_crosscoder_hidden_code_scout_20260502_tinyllama_validation1024_2048/hellaswag_crosscoder_hidden_code_scout.json`

Focused repeat:

`results/source_private_hellaswag_crosscoder_hidden_code_scout_20260502_seed97_tinyllama_validation1024_2048/hellaswag_crosscoder_hidden_code_scout.json`

## Method

- Calibration surface: official HellaSwag train only.
- Retained calibration rows: `1487`; split `1115/372`.
- Evaluation surface: frozen HellaSwag validation slice `1024:2048`, `1024`
  rows.
- Source model: TinyLlama-1.1B-Chat final-layer choice-mean hidden states.
- Target side: Qwen score side information plus Qwen linear-crosscoder
  candidate coordinates.
- Packet contract: max `256` symbols, `1B` raw / `4B` framed.
- Shared-basis fit: train-only PCA followed by CCA/SVD-style cross-covariance
  directions over paired TinyLlama/Qwen candidate hidden features.
- Encoder families:
  - source packet-candidate shared-coordinate k-means plus candidate id;
  - train-only source shared-coordinate reliability quantiles plus candidate
    id.
- Decoder controls include packet-only, Qwen-side-only crosscoder decoder,
  compact-candidate crosscoder decoder, row-shuffled source code,
  source-shared shuffle before encoding, codebook permutation mismatch, random
  same-byte code, zero source code, and label permutation.

Promotion required the train-dev-selected row to beat packet-only,
Qwen-side-only crosscoder decoding, and compact-candidate crosscoder decoding
by at least `0.010`, with positive paired CI95 lower bound, positive `4/5`
block stability, and separation from destructive controls.

## Result

The scout fails.

Primary reduced sweep:

| Row | Accuracy | Delta vs Packet | CI95 Low |
|---|---:|---:|---:|
| Packet-only | `0.501953` | `0.000000` | `0.000000` |
| Qwen-side crosscoder decoder | `0.464844` | n/a | n/a |
| Compact-candidate crosscoder decoder | `0.501953` | `0.000000` | `0.000000` |
| Train-dev-selected crosscoder code | `0.503906` | `+0.001953` | `-0.004883` |
| Best diagnostic crosscoder code | `0.507812` | `+0.005859` | `-0.002930` |

The train-dev-selected row is
`cca_pca64_d8_relconf_q32_ridge10` with decoder ridge `10.0`. It improves only
two net examples on the 1024-row slice and fails CI/block stability. The best
diagnostic row, `cca_pca32_d16_relconf_q32_ridge10`, is also far below the
`+0.010` scout bar.

Focused repeat on the best family gives the same weak result:

| Row | Accuracy | Delta vs Packet | CI95 Low |
|---|---:|---:|---:|
| Best focused diagnostic | `0.507812` | `+0.005859` | `-0.002930` |
| Focused train-dev-selected k-means | `0.502930` | `+0.000977` | `-0.005396` |

Controls for the primary selected row:

| Control | Accuracy | Delta vs Packet |
|---|---:|---:|
| compact-candidate crosscoder decoder | `0.501953` | `0.000000` |
| packet-only | `0.501953` | `0.000000` |
| candidate-only code | `0.500000` | `-0.001953` |
| qwen-side-only crosscoder decoder | `0.464844` | `-0.037109` |
| row-shuffled crosscoder code | `0.276367` | `-0.225586` |
| source-shared shuffle before encoding | `0.255859` | `-0.246094` |
| codebook permutation mismatch | `0.290039` | `-0.211914` |
| random same-byte code | `0.277344` | `-0.224609` |
| zero source code | `0.258789` | `-0.243164` |
| label permutation decoder | `0.291016` | `-0.210938` |

The destructive controls mostly collapse, which is good, but the matched
crosscoder code itself does not beat compact packet-only by enough to matter.

## Interpretation

This weakens linear shared-projection/common-basis packets on the current
HellaSwag surface. CCA-style shared coordinates are a stronger basis than raw
source-hidden PCA, but the fixed-byte source code still does not expose stable
additional TinyLlama information beyond the candidate id.

The result supports the reviewer diagnosis: analytic or linear basis alignment
is not the missing ICLR method. The next high-value branch should either train
a nonlinear cross-attention/Q-former-style resampler with an explicit
information bottleneck, or move to a less packet-saturated benchmark where
source/target complementarity leaves more headroom.

## Contribution Status

Defensible:

1. A reviewer-clean negative ablation for train-only linear crosscoder/CCA
   hidden-code packets.
2. Evidence that Qwen-side crosscoder coordinates alone do not explain a
   source-code win.
3. A sharper novelty boundary: shared projections are prior alignment tools;
   LatentWire's possible novelty must come from fixed-byte source-private
   packets and controls.

Not promoted:

1. Linear CCA/crosscoder hidden-code communication.
2. A shared latent-basis claim.
3. Full-validation materialization for this linear crosscoder family.

## Decision

Do not widen linear CCA/crosscoder codebooks on this HellaSwag slice. The next
exact gate should be a nonlinear learned connector/resampler, or a benchmark
headroom gate that finds a surface where packet-only does not already absorb
most of the source signal.
