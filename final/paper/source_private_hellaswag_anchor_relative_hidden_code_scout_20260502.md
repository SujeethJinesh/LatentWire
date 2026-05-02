# HellaSwag Anchor-Relative Hidden-Code Scout

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains plausible; ICLR is still blocked by a
  positive learned latent/common-basis method.
- Current story: compact one-byte HellaSwag packets are still the strongest
  method and systems row. Shallow score codes, hidden PCA/reliability codes,
  and now shallow anchor-relative hidden codes do not robustly beat packet-only.
- Exact remaining blocker: a joint common-basis/crosscoder/resampler objective
  must beat compact packet-only under train-only selection and destructive
  controls, or the positive-method search should move to a less
  packet-saturated benchmark.

## Lay Explanation

The previous hidden-code scout clustered TinyLlama hidden states in
TinyLlama's own private coordinate system. This scout tries a more principled
common-basis idea: describe TinyLlama and Qwen hidden states by similarity to
the same official-train anchor examples. That is like agreeing on a small set
of landmarks, then describing a new point by how close it is to each landmark.

The source still sends only a one-byte packet: candidate id plus a small
anchor-relative code. Qwen gets its own anchor-relative hidden features and
tries to decode the answer. No source text, source KV cache, raw hidden vector,
or raw score vector is transmitted.

## Artifacts

Primary sweep:

`results/source_private_hellaswag_anchor_relative_hidden_code_scout_20260502_tinyllama_validation1024_2048/hellaswag_anchor_relative_hidden_code_scout.json`

Anchor-seed repeats:

- `results/source_private_hellaswag_anchor_relative_hidden_code_scout_20260502_seed41_tinyllama_validation1024_2048/hellaswag_anchor_relative_hidden_code_scout.json`
- `results/source_private_hellaswag_anchor_relative_hidden_code_scout_20260502_seed59_tinyllama_validation1024_2048/hellaswag_anchor_relative_hidden_code_scout.json`
- `results/source_private_hellaswag_anchor_relative_hidden_code_scout_20260502_seed83_tinyllama_validation1024_2048/hellaswag_anchor_relative_hidden_code_scout.json`

## Method

- Calibration surface: official HellaSwag train only.
- Retained calibration rows: `1487`; split `1115/372`.
- Evaluation surface: frozen HellaSwag validation slice `1024:2048`, `1024`
  rows.
- Source model: TinyLlama-1.1B-Chat final-layer choice-mean hidden states.
- Target side: Qwen score side information plus Qwen hidden
  anchor-relative coordinates.
- Packet contract: max `256` symbols, `1B` raw / `4B` framed.
- Anchor counts in primary sweep: `16`, `32`, `64`.
- Encoder families:
  - nearest anchor id plus candidate id;
  - source relative-coordinate PCA/k-means plus candidate id;
  - train-only source relative-coordinate reliability quantiles plus candidate
    id.
- Decoder controls include packet-only, Qwen-side-only relative decoder,
  compact-candidate relative decoder, row-shuffled source code,
  source-relative shuffle before encoding, codebook permutation mismatch,
  random same-byte code, zero source code, and label permutation.

Promotion required the train-dev-selected row to beat packet-only,
Qwen-side-only relative decoding, and compact-candidate relative decoding by at
least `0.010`, with positive paired CI95 lower bound, positive `4/5` block
stability, and separation from destructive controls.

## Result

The scout fails.

Primary sweep:

| Row | Accuracy | Delta vs Packet | CI95 Low |
|---|---:|---:|---:|
| Packet-only | `0.501953` | `0.000000` | `0.000000` |
| Qwen-side relative decoder | `0.464844` | n/a | n/a |
| Compact-candidate relative decoder | `0.501953` | `0.000000` | `0.000000` |
| Train-dev-selected anchor code | `0.503906` | `+0.001953` | `-0.006836` |
| Best diagnostic anchor code | `0.511719` | `+0.009766` | `-0.000977` |

The best diagnostic row, `anchor64_relconf_q16_ridge10`, is a near miss but
does not pass: it is below the predeclared `+0.010` scout threshold and its
paired CI still crosses zero. The train-dev-selected row,
`anchor64_relpca16_kmeans32`, is only `+0.001953` over packet-only.

Anchor-seed repeats on the near-miss family show the near miss is not stable:

| Anchor seed | Best Row | Best Accuracy | Delta vs Packet | CI95 Low |
|---:|---|---:|---:|---:|
| `23` | `anchor64_relconf_q16_ridge10` | `0.511719` | `+0.009766` | `-0.000977` |
| `41` | `anchor64_relconf_q16_ridge10` | `0.503906` | `+0.001953` | `-0.001953` |
| `59` | `anchor64_relpca4_kmeans4` | `0.502930` | `+0.000977` | `0.000000` |
| `83` | `anchor64_relconf_q16_ridge10` | `0.503906` | `+0.001953` | `0.000000` |

Primary controls for the selected row:

| Control | Accuracy | Delta vs Packet |
|---|---:|---:|
| compact-candidate relative decoder | `0.501953` | `0.000000` |
| packet-only | `0.501953` | `0.000000` |
| qwen-side-only relative decoder | `0.464844` | `-0.037109` |
| row-shuffled anchor-relative code | `0.506836` | `+0.004883` |
| source-relative shuffle before encoding | `0.498047` | `-0.003906` |
| codebook permutation mismatch | `0.265625` | `-0.236328` |
| random same-byte code | `0.262695` | `-0.239258` |
| zero source code | `0.249023` | `-0.252930` |
| label permutation decoder | `0.295898` | `-0.206055` |

The row-shuffled control being positive, while the matched row is small,
further argues against a robust example-specific source signal.

## Interpretation

This weakens shallow anchor-relative common-basis packets on this HellaSwag
surface. Anchor-relative coordinates are a better-motivated basis than raw
TinyLlama PCA, but the current one-byte code does not produce stable lift
beyond the candidate id. The near-miss seed is useful signal for future method
design, but it is not evidence for a paper claim.

The next live method branch should not be another shallow codebook on the same
hidden features. The highest-value branches are:

1. a joint crosscoder/SAE-style shared dictionary trained with a receiver
   decoding objective;
2. a Q-former/Perceiver-style resampler bottleneck with an explicit byte or
   vector budget;
3. a less packet-saturated benchmark where the source candidate id leaves more
   headroom for extra latent evidence.

## Contribution Status

Defensible:

1. A reviewer-clean negative ablation for anchor-relative hidden-code packets.
2. Evidence that Qwen-side hidden relative coordinates alone do not explain a
   source-code win.
3. A sharper boundary between prior anchor/common-basis representation work and
   our fixed-byte source-private packet contract.

Not promoted:

1. Anchor-relative hidden-code communication.
2. A shared latent basis claim.
3. Full-validation materialization for this shallow anchor family.

## Decision

Do not widen this shallow anchor-relative branch. Keep the primary ICLR method
search on a true joint crosscoder/resampler objective or switch to a benchmark
with more headroom above packet-only.
