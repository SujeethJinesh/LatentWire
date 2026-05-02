# HellaSwag Disagreement-Prototype Receiver

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains defensible around fixed-byte
  source-private packet evidence; ICLR is still blocked by a positive
  receiver/common-language method, broader benchmark coverage, and native GPU
  systems rows.
- Current story: TinyLlama and Qwen packets each show full-validation
  HellaSwag utility, and their complementary errors create large oracle
  headroom, but simple receiver acceptance has not captured it.
- Exact remaining blocker: a train-only receiver must beat TinyLlama
  packet-only on full validation with positive paired uncertainty and
  destructive controls.

## Lay Explanation

The scalar receiver asked whether the packet or Qwen looked globally more
confident. This experiment instead builds train-only prototypes of rows where
TinyLlama and Qwen disagree. At validation time, it asks whether a row is near
past disagreement types where Qwen helped more than it harmed. Random,
label-permuted, and score-row-shuffled controls test whether any lift comes
from real local disagreement structure.

## Artifact

`results/source_private_hellaswag_disagreement_prototype_receiver_20260502/hellaswag_disagreement_prototype_receiver.json`

Supporting files:

- `results/source_private_hellaswag_disagreement_prototype_receiver_20260502/hellaswag_disagreement_prototype_receiver.md`
- `results/source_private_hellaswag_disagreement_prototype_receiver_20260502/manifest.json`

## Method

- Calibration source: official HellaSwag train caches for TinyLlama and Qwen
  seeds `2027`, `2039`, and `2053`.
- Leakage guard: out-of-bag TinyLlama packet predictions, duplicate train rows
  dropped, and rows overlapping included packet-training samples removed.
- Receiver alternatives:
  - Qwen target-score top-1;
  - Qwen mean-zscore packet;
  - Qwen hybrid vote-on-score-agreement packet.
- Feature views:
  - score-only packet/target confidence features;
  - score plus Qwen hidden-confidence features.
- Prototype pool: official-train fit rows where TinyLlama packet and Qwen
  alternative disagree.
- Prototype selection: farthest-first disagreement prototypes, with local
  train-only benefit estimates from nearest fit neighbors.
- Model selection: threshold is selected on official-train dev rows only;
  validation labels are used only for final evaluation.
- Controls:
  - random prototypes with the same count;
  - label-permuted local benefit values;
  - score-row-shuffled features.

Strict promotion requires the predeclared receiver to beat TinyLlama
packet-only on full validation by at least `+0.005` with positive paired CI95
lower bound, beat Qwen target-only by at least `+0.02`, remain positive across
contiguous validation blocks, and separate from controls by at least `+0.003`
delta.

## Result

Gate outcomes:

- pass gate: `false`;
- best-scout pass gate: `false`;
- predeclared default pass gate: `false`;
- target-transfer gate: `true`;
- block-stability gate: `false`;
- control-separation gate: `false`.

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
| predeclared default | `0.618602` | `0.619199` | `-0.000597` | `-0.001295` |
| best scout | `0.620693` | `0.619199` | `+0.001494` | `+0.000299` |

Predeclared default:

- alternative: Qwen hybrid vote-on-score-agreement packet;
- feature view: score plus Qwen hidden-confidence features;
- prototypes: `64`;
- neighbor `k`: `25`;
- top-k prototypes at validation: `5`;
- override rate: `0.004880`.

Best scout:

- alternative: Qwen mean-zscore packet;
- feature view: score-only;
- prototypes: `128`;
- neighbor `k`: `25`;
- top-k prototypes at validation: `3`;
- improvement is statistically positive but only `+0.001494`, below the
  preregistered `+0.005` promotion bar.

Controls:

| Control | Accuracy | Delta vs packet-only | CI95 low |
|---|---:|---:|---:|
| random prototypes | `0.619299` | `+0.000100` | `-0.000996` |
| label permutation | `0.614519` | `-0.004680` | `-0.006874` |
| score row shuffle | `0.604163` | `-0.015037` | `-0.018921` |

Default block deltas versus packet-only:

| Block | Rows | Delta |
|---:|---:|---:|
| `0` | `2009` | `0.000000` |
| `1` | `2009` | `-0.000996` |
| `2` | `2008` | `0.000000` |
| `3` | `2008` | `-0.000498` |
| `4` | `2008` | `-0.001494` |

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
- official-train calibration took `14.53s`, feature construction `0.54s`,
  selector sweep `35.43s`, and total Mac-local wall time `51.43s`;
- native GPU serving claims remain disabled.

## Interpretation

This branch does not promote. Local disagreement prototypes preserve the
source-private packet boundary and do find a tiny best-scout lift, but the
lift is too small for an ICLR positive method and the predeclared row is
slightly worse than packet-only. The default block table also fails stability,
so the result cannot be framed as a robust receiver improvement.

The important scientific update is that the remaining Tiny/Qwen oracle
headroom is not captured by scalar confidence, relative-kNN, or simple local
prototype benefit estimates. The next method branch should change the
representation class rather than tune this family harder.

## Contribution Status

Promoted:

1. A train-only disagreement-prototype diagnostic artifact with destructive
   controls.
2. A sharper branch-kill result for local prototype/common-basis selectors.
3. A systems-side packet/receiver accounting row that preserves the fixed
   `2B` raw / `5B` framed sideband boundary.

Still blocked:

1. Positive receiver improvement over packet-only.
2. Second benchmark under the same packet discipline.
3. Native NVIDIA/vLLM/SGLang systems rows and direct C2C/KVComm/KV-quant
   comparisons.

## Decision

Do not keep tuning local prototype thresholds on this surface. The next exact
gate should be a richer receiver that can represent reusable disagreement
atoms: a sparse/crosscoder dictionary, a learned query-bottleneck receiver, or
a relative-anchor basis with train-only public anchors and stronger
source-destroying controls.
