# HellaSwag Conditional Selector/Syndrome Gate

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains plausible around fixed-byte
  source-private packet systems/evaluation; ICLR remains blocked by a positive
  learned transfer method.
- Current story: HellaSwag has stable TinyLlama/Qwen complementarity, but the
  current learned selector family cannot recover it under train-only
  calibration.
- Exact remaining blocker: a method must beat compact packet-only by a
  meaningful margin with positive paired uncertainty, block stability, and
  source-destroying controls.

## Lay Explanation

This experiment trains a tiny referee. TinyLlama sends a compact byte packet.
Qwen also has its own answer candidate. When they disagree, the referee tries
to predict whether switching from TinyLlama's packet to Qwen's candidate will
help or hurt. It is trained only on official HellaSwag train rows and then
frozen for full validation.

## Artifact

`results/source_private_hellaswag_conditional_selector_syndrome_gate_20260502/`

Supporting files:

- `results/source_private_hellaswag_conditional_selector_syndrome_gate_20260502/hellaswag_conditional_selector_syndrome_gate.json`
- `results/source_private_hellaswag_conditional_selector_syndrome_gate_20260502/hellaswag_conditional_selector_syndrome_gate.md`
- `results/source_private_hellaswag_conditional_selector_syndrome_gate_20260502/manifest.json`

## Method

- Calibration surface: official HellaSwag train rows with duplicate and
  out-of-bag packet-training overlaps removed.
- Fit/dev split: `1115/372` rows.
- Evaluation: full HellaSwag validation, `10042` rows.
- Source packet: TinyLlama compact candidate packet, `1B` raw / `4B` framed.
- Optional syndrome subcodes: source-side quantile bins over packet z-score,
  top-2 margin, and packet rank, preserving candidate id in the low two bits.
- Receiver alternatives: Qwen target-score top-1, Qwen mean-zscore packet, and
  Qwen hybrid vote-on-score-agreement packet.
- Objective: train-only ridge benefit predictor for
  `alternative_correct - packet_correct`, then override packet only when the
  predicted benefit clears a dev-selected threshold.
- Promotion rule: the train-dev-selected row must beat packet-only by at least
  `+0.020`, have positive paired CI95 low, improve at least `4/5` contiguous
  blocks, and separate from destructive controls.

## Result

The gate fails.

| Quantity | Value |
|---|---:|
| validation rows | `10042` |
| packet-only accuracy | `0.619199` |
| Qwen hybrid accuracy | `0.532464` |
| default selector accuracy | `0.618901` |
| default delta vs packet-only | `-0.000299` |
| default CI95 low vs packet-only | `-0.000896` |
| default help / harm | `4 / 7` |
| best eval-only scout accuracy | `0.621291` |
| best eval-only scout delta | `+0.002091` |
| best scout oracle-headroom capture | `0.025579` |

The default row selected on official-train dev uses `packet_z_q32` and Qwen
target-score. It slightly hurts validation and fails block stability. The best
diagnostic row uses `top2_margin_q4` and Qwen hybrid, but its `+0.002091`
delta captures only about `2.6%` of the oracle headroom and remains far below
the promotion bar.

## Controls

Destructive controls do not rescue the method. Several controls collapse well
below packet-only, while packet-preserving controls tie or nearly tie
packet-only:

| Control | Delta vs packet-only |
|---|---:|
| row-shuffle source code | `-0.016531` |
| Qwen-derived source code | `-0.044712` |
| random same-byte code | `-0.015535` |
| random subcode preserve packet | `0.000000` |
| candidate-derangement packet code | `-0.029177` |
| packet-only candidate code | `-0.000199` |
| label-permutation benefit decoder | `-0.086537` |

## Systems Boundary

The selected method stays within the compact packet contract:

- `1B` raw / `4B` framed per request;
- no source text, source KV, raw hidden vector, or raw score vector exposed;
- native GPU claims disabled.

The artifact records source-state byte floors only. For the `4B` framed packet,
conservative one-token floors are `3072x` for fp16 KV, `921.6x` for 30% fp16
KVComm-style KV, `192x` for QJL 1-bit, and `672x` for TurboQuant 3.5-bit. These
are byte/exposure comparisons, not native TTFT, TPOT, HBM, goodput, or quality
claims.

## Interpretation

This result weakens the simplest conditional-syndrome story. The HellaSwag
oracle headroom is real, but it is not exposed to a linear train-only benefit
predictor over packet id, source confidence bins, and Qwen score features.

Do not keep tuning scalar/linear selectors on this surface. If we stay on
HellaSwag, the next method must change representation class: a nonlinear
query-bottleneck/resampler that still emits a discrete byte packet, or a richer
source-private dictionary whose transmitted object remains a task-level
syndrome rather than hidden/KV/vector state.

## Decision

Demote linear conditional selector/syndrome packets. Keep the artifact as a
negative method gate and reviewer-facing ablation. The next exact ICLR method
gate is a nonlinear bottleneck or sparse dictionary packet that must beat
packet-only by at least `+0.020` and capture a nontrivial fraction of the
measured oracle headroom under the same destructive controls.
