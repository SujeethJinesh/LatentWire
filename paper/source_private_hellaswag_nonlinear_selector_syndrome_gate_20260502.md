# HellaSwag Nonlinear Selector/Syndrome Gate

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains plausible around fixed-byte
  source-private packet systems/evaluation; ICLR remains blocked by a positive
  learned transfer method.
- Current story: HellaSwag has stable TinyLlama/Qwen complementarity, but
  linear and now nonlinear train-only selector/syndrome receivers fail to
  recover enough headroom from the current packet surface.
- Exact remaining blocker: a method must beat compact packet-only by a
  meaningful margin with positive paired uncertainty, block stability, and
  source-destroying controls.

## Lay Explanation

This experiment asks whether the previous selector failed only because it was
too simple. TinyLlama still sends a tiny byte packet. A small nonlinear referee
looks at that packet plus Qwen's own confidence signals and decides whether to
trust TinyLlama or switch to Qwen. The referee is trained only on official
HellaSwag train examples and then frozen before full validation.

## Artifact

`results/source_private_hellaswag_nonlinear_selector_syndrome_gate_20260502/`

Supporting files:

- `results/source_private_hellaswag_nonlinear_selector_syndrome_gate_20260502/hellaswag_nonlinear_selector_syndrome_gate.json`
- `results/source_private_hellaswag_nonlinear_selector_syndrome_gate_20260502/hellaswag_nonlinear_selector_syndrome_gate.md`
- `results/source_private_hellaswag_nonlinear_selector_syndrome_gate_20260502/manifest.json`

## Method

- Calibration surface: official HellaSwag train rows with duplicate and
  out-of-bag packet-training overlaps removed.
- Fit/dev split: `1115/372` rows.
- Evaluation: full HellaSwag validation, `10042` rows.
- Source packet: TinyLlama compact candidate packet plus optional source-side
  quantile syndrome bins, still `1B` raw / `4B` framed.
- Receiver alternatives: Qwen target-score top-1, Qwen mean-zscore packet, and
  Qwen hybrid vote-on-score-agreement packet.
- Nonlinearity: random Fourier feature approximations to RBF kernels, followed
  by ridge benefit prediction.
- Objective: predict whether overriding the TinyLlama packet with Qwen's
  candidate will help or harm.
- Selection: hyperparameters and thresholds are selected on official-train dev
  only; the full validation set is used once for reporting.
- Promotion rule: the train-dev-selected row must beat packet-only by at least
  `+0.020`, have positive paired CI95 low, capture at least `20%` of measured
  oracle headroom, improve at least `4/5` contiguous blocks, and separate from
  destructive controls.

## Result

The gate fails.

| Quantity | Value |
|---|---:|
| validation rows | `10042` |
| packet-only accuracy | `0.619199` |
| Qwen hybrid accuracy | `0.532464` |
| default nonlinear accuracy | `0.616610` |
| default delta vs packet-only | `-0.002589` |
| default CI95 low vs packet-only | `-0.003884` |
| default help / harm | `11 / 37` |
| default oracle-headroom capture | `-0.031669` |
| best eval-only scout accuracy | `0.621390` |
| best eval-only scout delta | `+0.002191` |
| best eval-only oracle-headroom capture | `0.026797` |

The official-train-dev-selected row uses `top2_margin_q16`, Qwen target-score,
`64` RFF components, RBF gamma `1.0`, seed `19`, harm weight `1.0`, ridge
`1.0`, and threshold `0.075`. It improved official-train dev by `+0.008065`
but hurt every contiguous validation block, indicating that the selector has
not learned a stable transfer rule.

The best eval-only diagnostic row uses `candidate_only` and Qwen hybrid. It
improves validation by only `+0.002191`, captures `2.68%` of the packet-or-Qwen
oracle headroom, and is below both the `+0.020` promotion bar and the `20%`
headroom-capture target.

## Controls

Destructive controls do not create a hidden positive result. All selected-row
controls remain below packet-only:

| Control | Delta vs packet-only |
|---|---:|
| row-shuffle source code | `-0.002689` |
| Qwen-derived source code | `-0.008763` |
| random same-byte code | `-0.003286` |
| random subcode preserve packet | `-0.002589` |
| candidate-derangement packet code | `-0.002987` |
| wrong alternative roll | `-0.127265` |
| packet-only candidate code | `-0.003585` |
| zero source code | `-0.013543` |
| RFF projection seed control | `-0.001394` |
| label-permutation benefit decoder | `-0.025095` |

## Systems Boundary

The method preserves the compact packet contract:

- `1B` raw / `4B` framed per request;
- no source text, source KV, raw hidden vector, raw score vector, or continuous
  syndrome vector transmitted;
- selected public receiver state is `49272` bytes;
- native GPU claims remain disabled.

Mac-local cached-packet microbenchmarks for the selected row:

| Batch | End-to-end cached selector p50 us/request | p95 us/request |
|---:|---:|---:|
| `1` | `54.67` | `62.38` |
| `4` | `14.15` | `15.73` |
| `16` | `4.29` | `4.80` |
| `64` | `4.33` | `4.85` |
| `256` | `1.82` | `2.09` |

These timings measure cached packet decode, RFF transform, and selector
decision on the Mac. They do not include model forward extraction, TTFT, TPOT,
HBM traffic, GPU memory, or native vLLM/SGLang serving throughput.

## Interpretation

This result demotes the HellaSwag selector/syndrome family for the current
packet surface. The oracle headroom is real, but neither scalar/linear nor
bounded nonlinear train-only benefit prediction recovers it in a stable,
reviewer-defensible way.

For ICLR, the next live method should not be another threshold/selector tuning
run on the same features. The high-value branches are:

1. change the source information itself, for example a learned source-private
   code trained directly for conditional transfer rather than reliability bins;
2. move to a true joint connector/resampler on NVIDIA hardware with native C2C
   and KVComm comparisons;
3. cut the HellaSwag receiver-improvement claim and keep HellaSwag as a
   complementarity/headroom plus systems-rate diagnostic.

## Decision

Mark nonlinear selector/syndrome packets as a negative method gate. Keep the
artifact as an ablation that reviewers can inspect, but stop spending Mac-local
time on selector variants over the current HellaSwag packet/Qwen-score surface.
