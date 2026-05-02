# HellaSwag Sparse-Query Receiver

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains defensible around fixed-byte
  source-private packet evidence; ICLR is still blocked by a positive
  receiver/common-language method, broader benchmark coverage, and native GPU
  systems rows.
- Current story: TinyLlama and Qwen packets each show full-validation
  HellaSwag utility, and their complementary errors create large oracle
  headroom, but receiver-side arbitration has not captured it.
- Exact remaining blocker: a train-only receiver must beat TinyLlama
  packet-only on full validation with positive paired uncertainty and controls
  that collapse.

## Lay Explanation

Previous receivers only looked at score summaries or small local prototypes.
This experiment looks at the full hidden-state difference between Qwen's
candidate and the TinyLlama packet candidate, compresses that large residual
into a tiny set of train-only query features, and learns when those features
say Qwen should override the packet.

## Artifact

`results/source_private_hellaswag_sparse_query_receiver_20260502/hellaswag_sparse_query_receiver.json`

Supporting files:

- `results/source_private_hellaswag_sparse_query_receiver_20260502/hellaswag_sparse_query_receiver.md`
- `results/source_private_hellaswag_sparse_query_receiver_20260502/manifest.json`

## Method

- Calibration source: official HellaSwag train caches for TinyLlama and Qwen
  seeds `2027`, `2039`, and `2053`.
- Leakage guard: out-of-bag TinyLlama packet predictions, duplicate train rows
  dropped, and rows overlapping included packet-training samples removed.
- Receiver alternatives:
  - Qwen target-score top-1;
  - Qwen mean-zscore packet;
  - Qwen hybrid vote-on-score-agreement packet.
- Hidden design: for each row, build a `5376`-dimension candidate-residual
  design from Qwen hidden contrasts: alternative-minus-packet, top1-minus-
  packet, alternative-minus-top1, top1-minus-top2, alternative-minus-mean, and
  packet-minus-mean.
- Query basis: train-only supervised benefit directions plus randomized PCA
  directions from official-train fit rows.
- Sparse query features: project hidden residuals into `4` to `64` query
  directions, optionally keep only the largest absolute query activations, and
  combine them with score-only or score-plus-hidden-confidence scalar features.
- Selector: benefit ridge over the sparse-query features, with threshold and
  hyperparameters selected on official-train dev rows only.
- Controls:
  - random query basis;
  - label-permuted benefit targets;
  - hidden-row shuffle.

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
| predeclared default | `0.617407` | `0.619199` | `-0.001792` | `-0.003784` |
| best scout | `0.619797` | `0.619199` | `+0.000597` | `-0.000996` |

Predeclared default:

- alternative: Qwen mean-zscore packet;
- scalar view: score plus Qwen hidden-confidence features;
- query basis: supervised plus randomized PCA;
- query count: `16`;
- active queries: `4`.

Best scout:

- alternative: Qwen mean-zscore packet;
- scalar view: score-only;
- query count: `32`;
- active queries: `4`;
- improvement is only `+0.000597`, has negative CI95 lower bound, and is far
  below the preregistered `+0.005` promotion bar.

Controls:

| Control | Accuracy | Delta vs packet-only | CI95 low |
|---|---:|---:|---:|
| random query basis | `0.619797` | `+0.000597` | `-0.000498` |
| label permutation | `0.602968` | `-0.016232` | `-0.020116` |
| hidden-row shuffle | `0.619797` | `+0.000597` | `-0.000498` |

Default block deltas versus packet-only:

| Block | Rows | Delta |
|---:|---:|---:|
| `0` | `2009` | `-0.002489` |
| `1` | `2009` | `0.000000` |
| `2` | `2008` | `-0.000996` |
| `3` | `2008` | `-0.002988` |
| `4` | `2008` | `-0.002490` |

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
- no source text, source KV, source raw hidden vectors, or source raw score
  vectors are transmitted;
- official-train calibration took `14.03s`, feature construction `1.79s`,
  selector sweep `15.55s`, and total Mac-local wall time `33.35s`;
- native GPU serving claims remain disabled.

## Interpretation

This branch does not promote. It is a stronger negative than the prototype
gate because it gave the receiver access to a low-rank view of the full Qwen
candidate-residual hidden space, not just score summaries or local prototypes.
The best scout is tiny, uncertain, and matched by random-query and
hidden-row-shuffle controls. That means the observed lift is not evidence that
the sparse query basis learned a reusable common language.

The current HellaSwag receiver-selector family should be considered saturated:
scalar confidence, ridge/kNN, local prototypes, and full-hidden sparse queries
all fail to close the Tiny/Qwen oracle headroom under official-train
calibration. The next positive-method branch should change the packet/code
itself or collect substantially more train-only calibration, rather than
tuning receiver selectors harder.

## Contribution Status

Promoted:

1. A train-only full-hidden sparse-query receiver diagnostic artifact.
2. A sharper negative result showing that low-rank hidden residual queries do
   not explain the receiver oracle headroom.
3. A Mac-local systems-side accounting row for sparse-query receiver cost under
   the fixed `2B` raw / `5B` framed packet boundary.

Still blocked:

1. Positive receiver improvement over packet-only.
2. Second benchmark under the same packet discipline.
3. Native NVIDIA/vLLM/SGLang systems rows and direct C2C/KVComm/KV-quant
   comparisons.

## Decision

Do not continue tuning HellaSwag receiver selectors on this calibration
surface. The next exact method gate should change the information structure:
learn a new source packet/code under a Wyner-Ziv side-information objective,
expand train-only calibration rows, or move the query-bottleneck to a true
joint source/receiver connector when GPU access is available.
