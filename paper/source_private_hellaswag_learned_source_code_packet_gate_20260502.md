# HellaSwag Learned Source-Code Packet Gate

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains plausible; ICLR is still blocked by a
  positive learned source-code or connector method.
- Current story: the compact HellaSwag candidate-id packet is the strongest
  current method and systems row. Receiver-only, residual-logit, and now
  source-score-derived learned discrete codes do not improve over packet-only.
- Exact remaining blocker: a learned source code must beat compact packet-only
  under official-train calibration, paired uncertainty, destructive controls,
  and preferably a second benchmark.

## Lay Explanation

The previous experiment tried adding a hand-designed score sketch. This one
instead lets the training data choose a small discrete code. The source side
looks only at TinyLlama's private score pattern and packet choice, maps that to
a one-byte code, and Qwen tries to decode the final answer using that code plus
its own side information.

## Artifact

`results/source_private_hellaswag_learned_source_code_packet_gate_20260502/hellaswag_learned_source_code_packet_gate.json`

Supporting files:

- `results/source_private_hellaswag_learned_source_code_packet_gate_20260502/hellaswag_learned_source_code_packet_gate.md`
- `results/source_private_hellaswag_learned_source_code_packet_gate_20260502/manifest.json`

## Method

- Calibration surface: official HellaSwag train only.
- Retained calibration rows: `1487` after dropping `5` duplicates and `44`
  out-of-bag overlap rows.
- Fit/dev split: `1115/372`.
- Validation surface: full HellaSwag validation, `10042` rows.
- Packet contract: one learned discrete source code, max `256` symbols,
  `1B` raw / `4B` framed.
- Source encoder inputs: TinyLlama source-only score features and the existing
  compact packet candidate id.
- Decoder inputs: learned source code plus Qwen target-side score/prediction
  side information.
- Encoder families: candidate-only baseline, quantile subcodes over
  packet-score, top-2 margin, and packet-rank features, plus k-means source
  feature subcodes.
- No source text, source KV cache, raw hidden vectors, or raw score vectors are
  transmitted.

Promotion required a train-dev-selected learned code to beat compact
packet-only by at least `0.010`, have positive CI95 lower bound, beat the best
prior receiver scout by at least `0.005`, stay positive on at least `4/5`
blocks, and separate from destructive same-byte controls.

## Result

The gate fails.

| Row | Accuracy | Delta vs Packet | CI95 Low |
|---|---:|---:|---:|
| Compact packet-only | `0.619199` | `0.000000` | `0.000000` |
| Train-dev-selected learned code | `0.615316` | `-0.003884` | `-0.006326` |
| Best scout code | `0.619896` | `+0.000697` | `-0.000498` |
| Best prior receiver scout | `0.620594` | n/a | n/a |

The selected learned encoder was `packet_z_quantile_32` with `128` symbols and
ridge `10.0`. It was negative in `4/5` contiguous blocks and below the best
prior receiver scout by `-0.005278`.

Controls:

| Control | Accuracy | Delta vs Packet |
|---|---:|---:|
| compact candidate-only decoder | `0.619199` | `0.000000` |
| Qwen-side-only decoder | `0.532364` | `-0.086835` |
| candidate-only code under selected decoder | `0.616311` | `-0.002888` |
| source-feature shuffle before encoding | `0.275344` | `-0.343856` |
| codebook permutation mismatch | `0.256423` | `-0.362776` |
| row-shuffled source code | `0.272057` | `-0.347142` |
| qwen-derived code | `0.488050` | `-0.131149` |
| label-permutation decoder | `0.328520` | `-0.290679` |

## Interpretation

This weakens the simple learned source-code branch. The source-score features
do carry the compact packet signal, but the learned subcodes do not expose a
stable extra bit of Tiny-to-Qwen information on full validation. The tiny
positive best scout is too small and its CI crosses zero.

Together with the failed residual-logit gate, this suggests that the current
HellaSwag calibration surface is not missing a scalar or low-dimensional
source-score code. The next live branch should change the source feature
family, for example by using source hidden states with a real learned
quantizer/crosscoder objective, or move to a benchmark where source/target
complementarity is less saturated by the compact candidate id.

## Contribution Status

Defensible:

1. The compact one-byte packet remains the promoted systems row.
2. This gate adds a reviewer-clean negative ablation for learned discrete
   source-score codes.
3. The artifact clarifies that Qwen-side-only and compact-candidate-only
   controls do not explain a hidden learned-code win.

Not promoted:

1. Learned source-score discrete packet communication.
2. Cross-model common-language transfer.
3. Any claim that source-score VQ/k-means/quantile codes beat packet-only.

## Decision

Demote source-score-derived learned discrete codes to a negative ablation. Do
not tune this family further on HellaSwag without a new source feature surface,
such as source hidden-state codes or a true learned quantizer trained with a
target-side decoding objective.
