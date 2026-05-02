# HellaSwag Official-Train Receiver Calibration

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains defensible around the full-validation
  fixed-byte packet result; ICLR is still blocked by a positive
  receiver/common-language method, a second benchmark, and native GPU systems
  rows.
- Current story: Qwen and TinyLlama source-private hidden-innovation packets
  both work on full HellaSwag, and TinyLlama/Qwen complementary errors create
  a receiver oracle above packet-only.
- Exact remaining blocker: the receiver must learn when to use target-side
  Qwen evidence without consuming validation labels and must beat TinyLlama
  packet-only with positive paired uncertainty.

## Lay Explanation

The previous receiver learned from early validation rows. This experiment asks
for a cleaner test: can we train the receiver on official HellaSwag train
rows, then freeze it and evaluate on all validation rows? To avoid letting a
packet model score a row it trained on, every official-train calibration row
uses out-of-bag packet models trained on other train samples, and rows that
overlap those packet-training samples are dropped.

## Artifact

`results/source_private_hellaswag_official_train_receiver_calibration_20260502/hellaswag_official_train_receiver_calibration.json`

Supporting files:

- `results/source_private_hellaswag_official_train_receiver_calibration_20260502/hellaswag_official_train_receiver_calibration.md`
- `results/source_private_hellaswag_official_train_receiver_calibration_20260502/manifest.json`

## Method

- Official train calibration rows come from aligned TinyLlama and Qwen train
  caches for seeds `2027`, `2039`, and `2053`.
- The calibration builder drops duplicate rows and any row that appears in an
  included out-of-bag packet model's training sample.
- TinyLlama packet predictions on calibration rows are produced by packet
  component models from the other train-sample seeds.
- Qwen receiver alternatives are Qwen target-score top-1, Qwen mean-zscore
  packet, and Qwen hybrid vote-on-score-agreement packet.
- Receiver families are benefit ridge and relative-kNN benefit over score-only
  and Qwen hidden-confidence feature views.
- Fit/dev split is official-train-only; validation labels are used only for
  final evaluation.

Strict promotion requires the predeclared official-train calibrated receiver
to beat TinyLlama packet-only on full validation by at least `+0.005` with
positive paired CI95 lower bound, beat Qwen target-only by at least `+0.02`,
and remain positive across contiguous validation blocks.

## Result

Gate outcomes:

- pass gate: `false`;
- best-scout pass gate: `false`;
- predeclared default pass gate: `false`;
- target-transfer gate: `true`;
- block-stability gate: `false`.

Calibration audit:

| Quantity | Value |
|---|---:|
| official-train calibration rows | `1487` |
| duplicate rows dropped | `5` |
| out-of-bag overlap rows dropped | `44` |
| fit rows | `1115` |
| dev rows | `372` |
| validation rows | `10042` |

Headline rows:

| Row | Accuracy | Packet-only | Delta | CI95 low |
|---|---:|---:|---:|---:|
| predeclared default | `0.618701` | `0.619199` | `-0.000498` | `-0.001394` |
| best scout | `0.620594` | `0.619199` | `+0.001394` | `-0.000597` |

Oracle/headroom:

- official-train Tiny packet accuracy: `0.591123`;
- official-train Qwen hybrid accuracy: `0.523874`;
- official-train Tiny-or-Qwen-hybrid oracle: `0.662408`;
- full-validation Tiny packet accuracy: `0.619199`;
- full-validation Qwen hybrid accuracy: `0.532464`;
- full-validation Tiny-or-Qwen-hybrid oracle: `0.686815`.

Systems/accounting row:

- packet remains `2B` raw / `5B` framed;
- full-validation logical packet payload is `20084B` raw or `50210B`
  framed;
- no source text, source KV, raw hidden vectors, or raw score vectors are
  transmitted;
- feature construction and receiver selection are Mac-local cached operations;
- native GPU serving claims remain disabled.

## Interpretation

This is a stronger branch-kill result than the validation-prefix acceptance
gate. The receiver headroom is still real: the Tiny/Qwen oracle is
`0.686815`, about `+0.067616` over packet-only. But simple scalar acceptance,
even with official-train-only calibration, cannot exploit it. The best scout
row is only `+0.001394` over packet-only and has a negative CI lower bound,
while the predeclared default is slightly worse than packet-only.

The next branch should change the information structure. The highest-value
Mac-feasible next gates are:

1. a train-only disagreement-prototype receiver that clusters reusable
   Tiny/Qwen error types instead of thresholding global confidence;
2. a sparse/crosscoder or relative-representation common basis over
   disagreement rows;
3. a tiny learned query-bottleneck receiver, Q-Former/Perceiver style, with
   source-destroying controls and heldout-only model selection.

## Contribution Status

Promoted:

1. A validation-label-free receiver-calibration artifact.
2. A sharper negative result ruling out official-train scalar acceptance as
   the missing common-language mechanism.
3. A reusable out-of-bag calibration harness that preserves the fixed-byte
   packet and source-state exposure boundary.

Still blocked:

1. Positive receiver improvement over packet-only.
2. Second benchmark under the same packet discipline.
3. Native NVIDIA/vLLM/SGLang systems rows and direct C2C/KVComm/KV-quant
   comparisons.

## Decision

Do not spend more turns tuning ridge/kNN confidence selectors on this surface.
The next exact gate should be a train-only disagreement-prototype/common-basis
receiver with random-prototype, label-permutation, packet-shuffle, and
score-row-shuffle controls.
