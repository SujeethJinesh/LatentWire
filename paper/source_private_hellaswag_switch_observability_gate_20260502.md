# HellaSwag Switch Observability Gate

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains plausible around fixed-byte
  source-private packet systems/evaluation; ICLR remains blocked by a positive
  learned transfer method.
- Current story: HellaSwag has real TinyLlama/Qwen complementarity, but the
  current packet/Qwen-score decision surface does not expose a stable learned
  switch rule.
- Exact remaining blocker: a positive method must change the source
  information, train a true joint connector, or move to a benchmark where
  byte-level source information is not saturated by candidate id.

## Lay Explanation

This experiment does not try to make a new model answer better. It asks a
simpler question: can the available numbers tell us when Qwen should overrule
TinyLlama's byte packet? If the answer is mostly no, then more tiny referees
and threshold tuning will not become an ICLR-strength method.

## Artifact

`results/source_private_hellaswag_switch_observability_gate_20260502/`

Supporting files:

- `results/source_private_hellaswag_switch_observability_gate_20260502/hellaswag_switch_observability_gate.json`
- `results/source_private_hellaswag_switch_observability_gate_20260502/hellaswag_switch_observability_gate.md`
- `results/source_private_hellaswag_switch_observability_gate_20260502/manifest.json`

## Method

- Calibration surface: official HellaSwag train rows with duplicate and
  out-of-bag packet-training overlaps removed.
- Fit/dev split: `1115/372` rows.
- Evaluation: full HellaSwag validation, `10042` rows.
- Target labels: whether switching from the source packet to a Qwen alternative
  helps (`Qwen correct, packet wrong`) or harms (`packet correct, Qwen wrong`).
- Feature views:
  - packet id only;
  - TinyLlama source scores plus packet id;
  - Qwen scores plus packet id and Qwen alternative id;
  - combined TinyLlama and Qwen score features.
- Probes: ridge-linear and random Fourier feature RBF probes.
- Metrics:
  - validation AUC for help-vs-harm among decisive rows;
  - dev-selected threshold delta versus packet-only;
  - validation-oracle threshold delta as a diagnostic upper bound, not a
    promotable method row.

## Result

The observability gate fails and triggers the branch-kill rule.

| Quantity | Value |
|---|---:|
| validation rows | `10042` |
| packet-only accuracy | `0.619199` |
| Qwen hybrid accuracy | `0.532464` |
| broader packet-or-any-Qwen oracle delta | `+0.081757` |
| default row | `source_plus_qwen_rff` / `qwen_hybrid` |
| default delta vs packet-only | `-0.000398` |
| default CI95 low vs packet-only | `-0.001095` |
| default validation AUC help-vs-harm | `0.553985` |
| best validation AUC | `0.561172` |
| best validation-oracle threshold delta | `+0.000199` |
| best validation-oracle threshold headroom capture | `0.002436` |

The default diagnostic row uses the strongest intended feature view
(`source_plus_qwen`) with the nonlinear RFF probe and Qwen hybrid alternative.
It is negative versus packet-only. The best AUC row switches to a linear
combined probe over Qwen mean-zscore, but still reaches only `0.561172` AUC
and produces no validation delta. The best validation-oracle threshold
diagnostic gives only `+0.000199`, which is not remotely close to a usable
method effect.

## Decisive-Row Readout

For the default `source_plus_qwen_rff` / `qwen_hybrid` row:

- decisive rows: `2229`;
- help rows: `679`;
- harm rows: `1550`;
- AUC: `0.553985`;
- average precision for help: `0.345528`.

For the best AUC row (`source_plus_qwen_linear` / `qwen_mean_zscore`):

- decisive rows: `2221`;
- help rows: `646`;
- harm rows: `1575`;
- AUC: `0.561172`;
- average precision for help: `0.342943`.

## Interpretation

This explains the prior failures. The HellaSwag complementarity headroom is
real, but the current score/packet surface weakly ranks the rare helpful
switches and cannot threshold them into a meaningful accuracy gain. That is why
linear selectors, nonlinear selectors, learned source-score codes, and shallow
hidden-code packets keep landing near zero.

This is not evidence that cross-model latent communication is impossible. It
is evidence that this particular Mac-local HellaSwag selector/source-score
surface is exhausted.

## Decision

Kill Mac-local selector/source-score tuning on the current HellaSwag surface.
Keep HellaSwag as:

1. a fixed-byte packet systems row;
2. a complementarity/headroom diagnostic;
3. a negative ablation suite showing that shallow selectors do not recover the
   oracle.

For ICLR, the next positive-method branch must be materially different:

1. a new source representation with a target-side decoding objective;
2. a true fixed-query cross-attention/Q-Former-style connector;
3. a decoder-conditioned innovation codec measured against C2C/KVComm and
   quantized KV baselines on NVIDIA hardware.
