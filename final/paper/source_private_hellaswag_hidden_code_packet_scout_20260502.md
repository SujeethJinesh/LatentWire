# HellaSwag Hidden-Code Packet Scout

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains plausible; ICLR is still blocked by a
  positive learned latent/common-basis method.
- Current story: the strongest current HellaSwag contribution is still the
  one-byte fixed candidate packet. Source-score codes and this source-hidden
  code scout do not beat packet-only under train-only calibration.
- Exact remaining blocker: find a learned source-hidden/code/connector method
  that beats compact packet-only with paired uncertainty, destructive controls,
  and eventually full-validation plus second-benchmark confirmation.

## Lay Explanation

This experiment asks whether TinyLlama can send Qwen a tiny learned label about
what TinyLlama's internal hidden state looked like. The label is at most one raw
byte, so Qwen does not see TinyLlama's text, KV cache, score vector, or hidden
vector. If the label captured useful hidden geometry, Qwen should make better
HellaSwag choices than it can with the one-byte candidate-id packet alone.

## Artifact

`results/source_private_hellaswag_hidden_code_packet_scout_20260502_tinyllama_validation1024_2048/hellaswag_hidden_code_packet_scout.json`

Supporting files:

- `results/source_private_hellaswag_hidden_code_packet_scout_20260502_tinyllama_validation1024_2048/hellaswag_hidden_code_packet_scout.md`
- `results/source_private_hellaswag_hidden_code_packet_scout_20260502_tinyllama_validation1024_2048/manifest.json`
- `results/source_private_hellaswag_hidden_code_packet_scout_20260502_tinyllama_validation1024_2048/source_eval_hidden_cache.npz`

## Method

- Calibration surface: official HellaSwag train only.
- Retained calibration rows: `1487` after duplicate and out-of-bag overlap
  filtering.
- Fit/dev split: `1115/372`.
- Evaluation surface: frozen HellaSwag validation slice `1024:2048`, `1024`
  rows.
- Source model: TinyLlama-1.1B-Chat, local MPS, `float16`, continuation
  prompts, final-layer choice-mean hidden states.
- Packet contract: max `256` symbols, `1B` raw / `4B` framed.
- Encoder families:
  - source-hidden PCA followed by k-means, with the compact candidate id in the
    low two bits;
  - supervised train-only hidden reliability quantiles that estimate when the
    TinyLlama candidate should be trusted, again with the candidate id in the
    low two bits.
- Decoder: Qwen side information plus the learned source-hidden code.
- No source text, source KV cache, raw hidden vectors, or raw source score
  vectors are transmitted.

Promotion required the train-dev-selected hidden-code packet to beat compact
packet-only by at least `0.010`, have positive paired CI95 lower bound, beat
the best prior receiver scout by at least `0.005`, stay positive on at least
`4/5` contiguous blocks, and separate from same-byte destructive controls.

## Result

The scout fails.

| Row | Accuracy | Delta vs Packet | CI95 Low |
|---|---:|---:|---:|
| Compact packet-only | `0.501953` | `0.000000` | `0.000000` |
| Train-dev-selected hidden code | `0.500000` | `-0.001953` | `-0.007812` |
| Best diagnostic hidden code | `0.507812` | `+0.005859` | `-0.001953` |
| Best supervised reliability code | `0.502930` | `+0.000977` | `-0.001953` |
| Qwen target-score baseline | `0.409180` | n/a | n/a |

The selected hidden encoder is `hidden_pca4_kmeans64` with `256` symbols and
decoder ridge `10.0`. Its default row is negative versus packet-only and below
the best prior receiver scout by `-0.120594`. The best diagnostic row,
`hidden_pca16_kmeans8`, gains only six net examples on the 1024-row slice and
has a CI crossing zero.

Default contiguous blocks are not stable:

| Block | Rows | Delta vs Packet |
|---:|---:|---:|
| 0 | `205` | `+0.009756` |
| 1 | `205` | `+0.009756` |
| 2 | `205` | `-0.009756` |
| 3 | `205` | `-0.004878` |
| 4 | `204` | `-0.014706` |

Controls for the selected hidden encoder:

| Control | Accuracy | Delta vs Packet |
|---|---:|---:|
| compact candidate-only decoder | `0.501953` | `0.000000` |
| packet-only | `0.501953` | `0.000000` |
| row-shuffled hidden code | `0.291992` | `-0.209961` |
| hidden-feature shuffle before encoding | `0.279297` | `-0.222656` |
| codebook permutation mismatch | `0.250977` | `-0.250977` |
| random same-byte code | `0.274414` | `-0.227539` |
| zero source code | `0.258789` | `-0.243164` |

The controls show that the decoder is sensitive to code identity, but the
matched hidden code still does not beat the compact candidate packet.

## Interpretation

This weakens the simple hidden-state discrete-code branch. TinyLlama hidden
geometry is not useless, but unsupervised PCA/k-means and supervised reliability
quantization do not expose a stable extra byte-scale signal beyond the
candidate-id packet on this HellaSwag slice. Because the 1024-row scout fails,
full validation hidden materialization is not currently justified.

The next live branch should not be another shallow scalar or unsupervised
codebook on the same surface. The higher-value options are:

1. a real crosscoder/SAE-style shared basis trained with a receiver decoding
   objective;
2. an anchor-relative or random-feature common basis that explicitly addresses
   representation gauge mismatch;
3. a less packet-saturated benchmark where source/target complementarity leaves
   headroom beyond the candidate id.

## Contribution Status

Defensible:

1. The compact one-byte HellaSwag packet remains the strongest systems/rate
   row.
2. This artifact adds a reviewer-clean negative ablation for source-hidden
   byte-scale codebooks.
3. The branch clarifies that simple hidden confidence/codebook summaries do not
   solve cross-model latent communication.

Not promoted:

1. Learned hidden-state discrete packet communication.
2. Supervised hidden reliability sideband.
3. Full-validation hidden materialization for this simple codebook family.

## Decision

Kill simple source-hidden PCA/k-means and hidden-reliability byte codes on this
HellaSwag surface. Promote the next gate to a stronger common-basis objective
or to a benchmark with more room above source candidate-id copying.
