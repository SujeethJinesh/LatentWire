# HellaSwag Receiver Headroom Decomposition

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains strong for a bounded fixed-byte
  source-private packet paper; ICLR remains gated by receiver improvement,
  benchmark breadth, and native NVIDIA systems rows.
- Current story: Qwen and TinyLlama each support a `2B` raw / `5B` framed
  hidden-innovation packet on full HellaSwag validation. The receiver-family
  scout showed TinyLlama packets are useful to a Qwen receiver, but the
  receiver did not beat packet-only.
- Exact remaining blocker: we need a train-only receiver/common-basis method
  that captures TinyLlama/Qwen complementary wins without eval-label tuning.

## Lay Explanation

This experiment asks whether two models make different enough mistakes that a
receiver could combine them. TinyLlama sends its tiny packet; Qwen has its own
score-based and hidden-packet guesses. The oracle is an unfair upper bound
that peeks at the answer and picks whichever model was right. The simple
selector is the fair cheap version: choose a rule using only the first
validation prefix, then freeze it on heldout rows.

## Artifact

`results/source_private_hellaswag_receiver_headroom_decomposition_20260502/hellaswag_receiver_headroom_decomposition.json`

Supporting files:

- `results/source_private_hellaswag_receiver_headroom_decomposition_20260502/hellaswag_receiver_headroom_decomposition.md`
- `results/source_private_hellaswag_receiver_headroom_decomposition_20260502/manifest.json`

## Method

- TinyLlama source packet: full-validation selected hidden-innovation packet.
- Qwen alternatives: target score top-1, mean-zscore hidden packet, and hybrid
  vote-on-score-agreement hidden packet.
- Train split: HellaSwag validation rows `0:1024`.
- Heldout eval split: HellaSwag validation rows `1024:10042`.
- Diagnostic selectors: train-prefix-selected threshold overrides on packet
  margin or target-score margin, plus an explicit eval-only best selector that
  is marked non-promotable.

## Result

Heldout rows `1024:10042` (`9018` examples):

| Row | Accuracy |
|---|---:|
| TinyLlama packet-only | `0.629741` |
| Qwen target-score | `0.483034` |
| Qwen mean-zscore packet | `0.528277` |
| Qwen hybrid packet | `0.534043` |
| best Tiny+Qwen oracle | `0.692947` |
| train-selected simple selector | `0.608228` |
| eval-only best simple selector | `0.629851` |

Paired deltas versus TinyLlama packet-only:

| Comparison | Delta | CI95 low | Status |
|---|---:|---:|---|
| best oracle | `+0.063207` | `+0.058658` | headroom pass |
| train-selected selector | `-0.021513` | `-0.028055` | receiver fail |
| eval-only best selector | `+0.000111` | `0.000000` | non-promotable |

Overlap with TinyLlama packet-only on heldout rows:

| Qwen row | Qwen-only correct count | Qwen-only correct rate |
|---|---:|---:|
| target-score | `487` | `0.054003` |
| mean-zscore packet | `538` | `0.059658` |
| hybrid packet | `570` | `0.063207` |

Systems/accounting row:

- packet remains `2B` raw / `5B` framed;
- logical full-validation packet sideband is `20,084` raw bytes or `50,210`
  framed bytes for `10,042` requests;
- source text, source KV, raw hidden vectors, and raw score vectors are not
  exposed;
- native GPU systems claims remain disabled.

## Interpretation

This is a useful failure. The oracle shows genuine complementary signal:
Qwen's hybrid packet is correct on `570` heldout rows where the TinyLlama
packet is wrong, yielding `+0.063207` oracle headroom. But simple confidence
thresholds cannot identify those rows under train-only selection; the chosen
selector actually hurts.

Therefore the receiver branch is alive, but the next branch should not be
another packet-margin threshold. The highest-value next method is a
train-only common-basis receiver: relative coordinates, sparse/crosscoder
atoms, or a tiny Q-Former/Perceiver-style selector that learns when target
evidence should override the source packet.

## Contribution Status

Promoted:

1. Receiver-headroom decomposition: Qwen and TinyLlama contain complementary
   candidate evidence on `9018` heldout HellaSwag rows.
2. Reviewer boundary: simple confidence selectors fail, preventing an
   overclaim that the current receiver learned cross-model latent reasoning.
3. Systems boundary: the packet-sideband remains fixed-byte and source-private
   even in the receiver analysis.

Still blocked:

1. A train-only receiver that beats packet-only by at least `+0.005` with
   positive paired CI95 low.
2. A second benchmark under the same packet discipline.
3. Native NVIDIA/vLLM/SGLang rows against C2C/KVComm/KV compression baselines.

## Decision

Promote this artifact as the receiver/common-basis decision surface. The next
exact gate is a train-only selective residual/common-basis receiver that
targets the `0.692947` oracle while preserving the `2B` raw / `5B` framed
source-private packet contract and destructive controls.
