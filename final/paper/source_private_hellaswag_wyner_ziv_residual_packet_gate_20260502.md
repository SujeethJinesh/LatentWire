# HellaSwag Wyner-Ziv Residual-Logit Packet Gate

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains plausible; ICLR is still blocked by a
  positive method that improves over packet-only under train-only calibration.
- Current story: fixed-byte source-private HellaSwag packets are strong, and
  the one-byte compaction is a useful systems rate point. Receiver/common-basis
  variants, including this residual-logit packet, do not yet improve on the
  packet-only baseline.
- Exact remaining blocker: a learned source code or connector must beat the
  compact packet-only row with paired uncertainty, destructive controls, and a
  second benchmark or native systems evidence.

## Lay Explanation

The earlier packet only says which answer TinyLlama would choose. This
experiment tried giving Qwen a little more source-private information: the
TinyLlama answer choice plus a tiny quantized sketch of TinyLlama's four answer
scores. In plain terms, it asks whether Qwen can use its own scores plus this
small hint from TinyLlama to choose better than TinyLlama's answer alone.

## Artifact

`results/source_private_hellaswag_wyner_ziv_residual_packet_gate_20260502/hellaswag_wyner_ziv_residual_packet_gate.json`

Supporting files:

- `results/source_private_hellaswag_wyner_ziv_residual_packet_gate_20260502/hellaswag_wyner_ziv_residual_packet_gate.md`
- `results/source_private_hellaswag_wyner_ziv_residual_packet_gate_20260502/manifest.json`

## Method

- Calibration surface: official HellaSwag train rows only, with `1487`
  retained rows after dropping `5` duplicate rows and `44` out-of-bag overlap
  rows.
- Fit/dev split: `1115/372`.
- Validation surface: full HellaSwag validation, `10042` rows.
- Packet contract: `2B` raw / `5B` framed.
- Default packet fields:
  - TinyLlama selected candidate id.
  - Four quantized TinyLlama centered score-residual bins.
- Default selected packet: `4` quantizer bins, `10` packet bits per request,
  ridge `1.0`.
- Decoder: candidate-wise ridge model combining the packet with Qwen
  side-information features.
- No source text, source KV cache, raw hidden vectors, or raw score vectors are
  transmitted.

The promotion rule was intentionally strict: the train-dev-selected full
decoder must beat TinyLlama packet-only by at least `0.010`, have positive
paired CI95 lower bound, beat Qwen target-score by at least `0.020`, beat the
best prior official-train receiver scout by at least `0.005`, remain positive
on at least `4/5` contiguous validation blocks, and separate from destructive
same-byte controls.

## Result

The gate fails.

| Row | Accuracy | Delta vs Packet | CI95 Low |
|---|---:|---:|---:|
| TinyLlama packet-only | `0.619199` | `0.000000` | `0.000000` |
| Default Wyner-Ziv residual packet | `0.616013` | `-0.003187` | `-0.004979` |
| Best scout row | `0.619199` | `0.000000` | `0.000000` |
| Best prior receiver scout | `0.620594` | n/a | n/a |

The default is below the best prior receiver scout by `-0.004581`. All five
contiguous blocks are negative versus packet-only, with deltas from `-0.000498`
to `-0.005478`.

Destructive controls did not explain a hidden gain because there was no gain.
The strongest control was Qwen-derived score sketch at `0.604860`, still
`-0.014340` versus packet-only.

## Interpretation

This branch does not promote. It was the right falsification test because it
changed the transmitted source code instead of tuning another receiver
selector. Under official-train-only calibration, however, a tiny residual-logit
source code does not beat the compact packet-only baseline.

This weakens or rules out the current residual-logit Wyner-Ziv packet branch on
HellaSwag. Together with scalar acceptance, prototype receivers, and sparse
query receivers, it says the live receiver/common-language path is saturated on
this calibration surface.

## Contribution Status

Promoted or still defensible:

1. Source-private fixed-byte HellaSwag packet evidence.
2. One-byte candidate-id compaction as the current systems rate point.
3. A rigorous negative receiver/code ablation suite showing what does not close
   the Tiny/Qwen oracle headroom.

Not promoted:

1. Wyner-Ziv residual-logit packets.
2. Learned syndrome/source-code communication on HellaSwag.
3. Receiver/common-language transfer from TinyLlama to Qwen.

## Decision

Demote residual-logit Wyner-Ziv packets to a negative ablation. Do not tune this
branch further unless a new preregistered train-only calibration surface or a
true learned source-code objective is introduced. The next exact ICLR gate
should either learn the source code directly or shift to a stronger benchmark
surface where the packet is not already saturated.
