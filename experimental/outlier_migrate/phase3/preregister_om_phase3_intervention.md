# OutlierMigrate Phase 3 Intervention Preregistration

**Frozen on**: 2026-05-09
**Frozen by**: human-authorized Phase 3 sprint prompt, before any Phase 3
quantization run
**Pivot depth**: 1 from OutlierMigrate Phase 0
**Gated by**:
- `outlier_migrate_phase0` PASS: `PASS_OM_PHASE0_DECODE_TIME_MIGRATION`
- `outlier_migrate_phase1` PASS: `PASS_OM_PHASE1_REPLICATED_AT_SCALE`
- partial Phase 2 Nemotron-3 PASS:
  `PARTIAL_PASS_OM_PHASE2_NEMOTRON3_ONLY_QWEN36_KIMI_DEFERRED`

## Status Note

This preregistration converts the OutlierMigrate characterization branch into
a bounded positive-method intervention gate. It is authored before any Phase 3
quantization, perplexity, control, or ablation results are inspected.

Existing Phase 0/1/2 result packets may be used only to define the fixed
trace set, fixed decode-position grids, and motivation. They must not be used
to tune Phase 3 thresholds, traces, layers, quantization method, or scoring
windows after this file is committed.

## Hypothesis

Static W4A16 weight quantization with the union of top-1% channels across
decode positions `{100, 1000, 5000, 10000}` as the protected set on
Granite-4.0-H-Tiny recovers a measurable fraction of the
BF16-vs-position100-only-protection perplexity gap at decode position 10000.

## Primary Model

- `ibm-granite/granite-4.0-h-tiny`

## Trace Set

- Source: AIME-2025
- Count: 24 traces
- Selection: deterministic prompts `0-23`, matching the Phase 1 prompt set
- Prompt SHA: use the exact prompt SHA recorded in the Phase 1 result packet
  and repeat it in the Phase 3 result packet before analysis
- Trace reselection is forbidden

## Quantization Regimes

The primary intervention compares exactly three regimes:

1. **BF16 baseline**: unquantized model.
2. **Static-1% protection**: simple symmetric per-channel INT4 weights with
   FP16 activations; protect the top-1% channels measured at decode position
   `100` only.
3. **Migration-aware union protection**: same quantization scheme, but protect
   the union of top-1% channels measured at decode positions
   `{100, 1000, 5000, 10000}`.

The mandatory controls add:

4. **Static-2% matched-budget control**: protect top-2% channels measured at
   decode position `100` only.
5. **Magnitude-averaging control**: protect top-1% channels by mean magnitude
   averaged across decode positions `{100, 1000, 5000, 10000}`.

## Quantization Scheme

- Weights: simple symmetric signed INT4, per output channel.
- Activations: FP16.
- Protected channels: BF16.
- AWQ-style activation-aware scaling is forbidden.
- SmoothQuant-style activation folding is forbidden.
- Calibration losses, continuation losses, AIME answers, and Phase 3
  perplexity values must not be used to choose protected channels.

Protected channels are hidden-state channel indices defined per layer. For a
2D weight tensor:

- if the output dimension equals the model hidden size, rows whose output
  channel is protected remain BF16;
- if the input dimension equals the model hidden size, columns whose input
  channel is protected remain BF16;
- if both dimensions match, protect the union of the corresponding rows and
  columns;
- all unprotected entries use symmetric per-output-channel INT4 dequantized to
  FP16/BF16 compute as implemented by the runner.

Weights without a hidden-size input or output dimension are quantized with the
same symmetric per-output-channel INT4 rule and no protected-channel exception,
unless the runner documents that the tensor is not a linear projection and is
excluded for an architecture-invariant reason. Exclusions must be listed before
metric analysis.

## Protected-Set Construction

For each trace and layer, compute absolute activation magnitudes at the fixed
decode positions. For each layer and protected-set regime:

- compute the top channel set for each required position using mean absolute
  magnitude over the fixed trace set;
- use top `ceil(0.01 * hidden_size)` channels for 1% sets;
- use top `ceil(0.02 * hidden_size)` channels for the static-2% control;
- break ties by lower channel index;
- for union protection, take the set union across the fixed grid;
- for magnitude averaging, rank channels by the mean of their per-position
  mean absolute magnitudes across the fixed grid.

Protected-set JSON artifacts must be written before any perplexity scoring
for the corresponding quantized regime.

## Perplexity Measurement

The scoring target is the BF16 deterministic trace for each fixed prompt. The
runner must use a deterministic decoding configuration and record all decoding
parameters in the packet. Per-trace perplexity at decode position `10000` is
computed on a fixed 64-token scoring window ending at decode position `10000`
on that BF16 trace. If a trace cannot produce the full scoring window, it is
an infrastructure failure unless the packet proves the failure is unrelated
to model behavior and reruns the exact same prompt to the required length.

For each trace, compute:

```text
static_gap = perplexity_static_1pct - perplexity_BF16
recovery = 1 - (perplexity_union - perplexity_BF16) / static_gap
```

If `static_gap <= 0`, set that trace's recovery to `0.0` for the primary
aggregation and report the trace as `no_recoverable_static_gap`. This is a
conservative rule that prevents post-hoc exclusion of traces where the static
regime does not create a recoverable perplexity gap.

## Primary Statistical Readout

- Primary statistic: median per-trace recovery across the 24 fixed traces.
- Confidence interval: bootstrap over traces, `n=1000`, seed `20260509`.
- Report the full per-trace table with BF16, static-1%, union, static gap,
  recovery, and no-gap flag.

## Decision Rule

### PASS

Decision string: `PASS_OM_PHASE3_MIGRATION_AWARE_RECOVERS`

All of the following must hold:

1. Median recovery is `>= 0.50`.
2. Bootstrap 95% CI lower bound is `> 0.30`.
3. The result packet is artifact-complete.
4. The checker exits 0 with the PASS decision string.

### KILL

Decision string: `KILL_OM_PHASE3_INTERVENTION_FAILS`

Any of the following triggers this kill:

1. Median recovery is `< 0.20`.
2. Bootstrap 95% CI upper bound is `< 0.30`.
3. More than 25% of traces are `no_recoverable_static_gap`.
4. The packet is artifact-complete and shows that the migration-aware union
   does not reduce perplexity relative to static-1% on at least 18 of 24
   traces.

### AMBIGUOUS

Decision string: `KILL_OM_PHASE3_AMBIGUOUS`

If the artifact-complete packet is neither PASS nor KILL, the intervention is
treated as ambiguous and does not support a positive-method claim.

### INFRA

Decision string: `FAIL_INFRA_OM_PHASE3`

Use only for non-scientific failures: model load failure, trace generation
failure, malformed artifacts, missing environment/provenance records, checker
crash, out-of-disk, or inability to produce the fixed scoring window for
reasons not attributable to the model/regime comparison.

## Mandatory Controls And Ablations

These run regardless of the primary Phase 3 outcome unless a stop condition or
infrastructure failure prevents execution.

### Position Grid Sensitivity

Evaluate the same metric under two additional union grids:

- sparse: `{100, 5000, 10000}`
- dense: `{100, 500, 1000, 2000, 5000, 7500, 10000}`

This is descriptive only and has no separate pass/kill decision. Report
recovery versus grid density. Saturation supports the selected grid; monotonic
improvement suggests the selected grid is directionally valid but suboptimal.

### Matched-Budget And Averaging Controls

Evaluate:

- static-2% matched-budget control;
- magnitude-averaging control.

For a positive-method paper claim, the migration-aware union must outperform
both controls in median recovery. If either control outperforms the union by
more than `0.10` median recovery, stop and surface to the human because the
union effect may be artifactual. If the primary gate passes but the union does
not outperform both controls, the primary decision remains recorded, but the
paper must not claim a union-specific positive method without human judgment.

## Conditional Follow-Ups

These are authorized only if the primary Phase 3 gate passes.

1. **Nemotron-3 intervention check**:
   repeat BF16 vs static-1% vs union on
   `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, 12 fixed traces, same union
   grid `{100, 1000, 5000, 10000}` and same recovery metric.
2. **Within-set rank-shuffling pilot**:
   on Granite-4.0-H-Tiny only, 12 traces, use union protection and recompute
   quantization scales every 2000 decode positions from current activation
   magnitudes. This is a pilot, not a primary gate.
3. **Decode-length scaling curve**:
   only if primary median recovery is `> 0.50`; evaluate union recovery at
   decode positions `{2000, 5000, 10000, 20000}` on 12 traces.

If the primary Phase 3 gate kills or is ambiguous, skip these follow-ups and
write a diagnostic explaining why simple union-set static protection was
insufficient.

## Mandatory No-GPU Analysis

Layer-stratified migration analysis must be run on existing Phase 0/1/2
packets:

- attention layers;
- SSM/Mamba layers;
- MoE expert layers where present.

Output:
`experimental/outlier_migrate/phase3/results/layer_stratified_migration.md`.

This analysis is post-hoc explanation only. It must not alter the Phase 3
decision.

## Required Result Packet

The Phase 3 result packet must include:

- `environment.json` and `environment.txt` with Python, CUDA, GPU, driver,
  `pip freeze`, and relevant environment variables;
- model provenance with exact HuggingFace snapshot SHA;
- prompt manifest and prompt SHA;
- decoding configuration and all random seeds;
- command metadata and full stdout/stderr logs;
- protected-set artifacts for every regime;
- quantization configuration and excluded tensor list;
- per-trace perplexity/recovery table;
- bootstrap CI JSON;
- control and ablation tables;
- artifact hashes;
- checker output;
- `artifact_check.json`.

## Forbidden Actions

- Modifying this preregistration after the first Phase 3 quantization run.
- Adjusting thresholds after observing Phase 3 data.
- Skipping the static-2% matched-budget control.
- Skipping the magnitude-averaging control.
- Using AWQ-style activation-aware scaling or SmoothQuant-style activation
  folding.
- Cherry-picking decode positions, traces, layers, scoring windows, or tensor
  exclusions post-hoc.
- Downloading Qwen3.6 or Kimi Linear weights.
- Upgrading vLLM.
- Marking any paper camera-ready final.
- Authorizing additional pivots beyond Phase 3 in this sprint.

## On PASS

Update the OutlierMigrate paper as a positive-method paper only if the
mandatory controls also support a union-specific effect. The paper must report
the primary gate, position-grid sensitivity, matched-budget and averaging
controls, layer-stratified migration, and any conditional follow-ups that were
run.

## On KILL Or AMBIGUOUS

Write a Phase 3 diagnostic and update the paper as a characterization plus
negative intervention finding. The paper may explain why simple union-set
static protection is insufficient, but it must not claim a successful positive
method.
