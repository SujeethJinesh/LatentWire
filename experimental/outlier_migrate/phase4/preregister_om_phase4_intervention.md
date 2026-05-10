# OutlierMigrate Phase 4 Granite-Small Intervention Preregistration

**Frozen on**: 2026-05-10
**Frozen by**: human-authorized Phase 4 sprint prompt, before any Phase 4
quantization run
**Pivot depth**: 1 from OutlierMigrate Phase 0
**Pivot relationship**: parallel pivot to Phase 3, not a child of Phase 3
**Gated by**:
- `outlier_migrate_phase0` PASS: `PASS_OM_PHASE0_DECODE_TIME_MIGRATION`
- `outlier_migrate_phase1` PASS: `PASS_OM_PHASE1_REPLICATED_AT_SCALE`
- Phase 3 diagnostic: `KILL_OM_PHASE3_INTERVENTION_FAILS` on
  Granite-4.0-H-Tiny under W4A16, with evidence that the kill was dominated by
  insufficient measurable static-protection gap rather than by a threshold
  change or post-hoc trace selection.

## Status Note

This preregistration defines a fresh positive-method intervention gate for
OutlierMigrate on Granite-4.0-H-Small. It is not an amendment of Phase 3 and
does not alter any Phase 0, Phase 1, Phase 2, or Phase 3 preregistration.

Phase 3 was a depth-1 pivot from OutlierMigrate Phase 0. Phase 4 is also a
depth-1 pivot from the same Phase 0 root hypothesis. The motivation for Phase
4 is the Phase 3 diagnostic: Granite-4.0-H-Tiny under W4A16 was nearly
lossless, with many traces having no recoverable BF16-vs-static-1% gap. Phase
4 keeps the Phase 3 decision thresholds unchanged for direct comparability,
but changes the model to the Phase 1 same-family scale-up model and expands
the scoring window from 64 tokens to 512 tokens before any Phase 4 data is
observed.

Existing Phase 0/1/2/3 packets may be used only for fixed trace identity,
fixed prompt SHA, fixed model snapshot, and motivation. They must not be used
to retune Phase 4 thresholds, protected fractions, trace selection, layer
selection, decode grids, or control definitions after this file is committed.

## Hypothesis

Static W4A16 weight quantization with the union of top-1% channels across
decode positions `{100, 1000, 5000, 10000}` as the protected set on
Granite-4.0-H-Small recovers a measurable fraction of the
BF16-vs-position100-only-protection perplexity gap at decode position 10000.

## Primary Model

- Model: `ibm-granite/granite-4.0-h-small`
- HuggingFace snapshot commit:
  `b8c0982bab7fde4eb48110f5a069527c008fab39`
- This must match the Phase 1 model provenance.

## Trace Set

- Source: AIME-2025
- Count: 24 traces
- Selection: deterministic prompts `0-23`, matching the Phase 1 prompt set
- Prompt SHA:
  `sha256:aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e`
- Trace reselection is forbidden.
- Excluding traces post-hoc for being difficult, short, long, low-gap,
  high-gap, numerically unstable, or inconvenient is forbidden.

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
6. **Sparse-grid union**: protect the union of top-1% channels across
   `{100, 5000, 10000}`.
7. **Dense-grid union**: protect the union of top-1% channels across
   `{100, 500, 1000, 2000, 5000, 7500, 10000}`.

All mandatory controls and grid-sensitivity regimes must run regardless of the
primary outcome unless an infrastructure failure or stop condition prevents
execution.

## Quantization Scheme

- Weights: simple symmetric signed INT4, per output channel.
- Activations: FP16.
- Protected channels: BF16.
- AWQ-style activation-aware scaling is forbidden.
- SmoothQuant-style activation folding is forbidden.
- Calibration losses, continuation losses, AIME answers, and Phase 4
  perplexity values must not be used to choose protected channels.
- The quantization scheme must match Phase 3 except for the model and scoring
  window changes explicitly preregistered here.

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
excluded for an architecture-invariant reason. Exclusions must be listed
before metric analysis.

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

Protected-set JSON artifacts must be written before any perplexity scoring for
the corresponding quantized regime.

## Perplexity Measurement

The scoring target is the BF16 deterministic trace for each fixed prompt. The
runner must use a deterministic decoding configuration and record all decoding
parameters in the packet.

Per-trace perplexity at decode position `10000` is computed on a fixed
512-token scoring window ending at decode position `10000` on that BF16 trace.
This differs from Phase 3's 64-token window and is a preregistered
methodological improvement intended to reduce per-trace variance before any
Phase 4 data is observed.

If a trace cannot produce the full 512-token scoring window, it is an
infrastructure failure unless the packet proves the failure is unrelated to
model behavior and reruns the exact same prompt to the required length.

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
- Confidence interval: bootstrap over traces, `n=1000`, seed `20260510`.
- Report the full per-trace table with BF16, static-1%, union, static gap,
  recovery, and no-gap flag.
- Report the no-gap count and fraction.
- Report the number of traces with union recovery `>0.50`.
- Report mean recovery only as a diagnostic; it is not the gate statistic.

## Decision Rule

### PASS

Decision string: `PASS_OM_PHASE4_MIGRATION_AWARE_RECOVERS`

All of the following must hold:

1. Median recovery is `>= 0.50`.
2. Bootstrap 95% CI lower bound is `> 0.30`.
3. No kill condition below holds.
4. The result packet is artifact-complete.
5. The checker exits 0 with the PASS decision string.

### KILL

Decision string: `KILL_OM_PHASE4_INTERVENTION_FAILS`

Any of the following triggers this kill:

1. Median recovery is `< 0.20`.
2. Bootstrap 95% CI upper bound is `< 0.30`.
3. More than 25% of traces are `no_recoverable_static_gap`.

### AMBIGUOUS

Decision string: `KILL_OM_PHASE4_AMBIGUOUS`

If the artifact-complete packet is neither PASS nor KILL, the intervention is
treated as ambiguous and does not support a positive-method claim. This
includes median recovery in the `0.20-0.50` band with overlapping CI, and any
case where the median is `>=0.50` but the CI lower bound is not greater than
`0.30`.

### INFRA

Decision string: `FAIL_INFRA_OM_PHASE4`

Use only for non-scientific failures: model load failure, trace generation
failure, malformed artifacts, missing environment/provenance records, checker
crash, out-of-disk, out-of-memory, or inability to produce the fixed scoring
window for reasons not attributable to the model/regime comparison.

## Mandatory Controls And Grid Sensitivity

These run regardless of the primary Phase 4 outcome unless a stop condition or
infrastructure failure prevents execution.

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

### Position Grid Sensitivity

Evaluate the same metric under two additional union grids:

- sparse: `{100, 5000, 10000}`;
- primary: `{100, 1000, 5000, 10000}`;
- dense: `{100, 500, 1000, 2000, 5000, 7500, 10000}`.

This is descriptive only and has no separate pass/kill decision. Report
recovery versus grid density. Saturation supports the selected grid; monotonic
improvement suggests the selected grid is directionally valid but suboptimal.

## Conditional Follow-Ups

These are authorized only if the primary Phase 4 gate passes and no stop
condition fires. Run in this priority order:

1. **Nemotron-3 cross-model intervention check**:
   repeat BF16 vs static-1% vs union on
   `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, 12 fixed traces, same union
   grid `{100, 1000, 5000, 10000}` and same recovery metric.
2. **Within-set rank-shuffling pilot**:
   on Granite-4.0-H-Tiny only, 12 traces, use union protection and recompute
   quantization scales every 2000 decode positions from current activation
   magnitudes. This is a pilot, not a primary gate.
3. **Decode-length scaling curve**:
   evaluate union recovery at decode positions `{2000, 5000, 10000, 20000}`
   on 12 traces using Granite-4.0-H-Small.

If the primary Phase 4 gate kills or is ambiguous, skip these follow-ups and
write `experimental/outlier_migrate/phase4/diagnostic.md` explaining whether
the kill appears measurement-driven, such as Granite-Small W4A16 still being
too lossless, or whether it genuinely argues against the intervention.

## Paper Integration Rule

If Phase 4 passes:

- reframe the OutlierMigrate paper as a measurement plus positive-method
  contribution;
- present Phase 3 as motivation that Granite-Tiny W4A16 was too robust to
  measure the intervention;
- present Phase 4 Granite-Small results as the primary intervention;
- include mandatory controls and grid sensitivity;
- include conditional follow-ups if they ran;
- run committee review with explicit instruction that the paper is now a
  positive-method candidate.

If Phase 4 kills:

- retain the characterization-plus-negative-intervention framing;
- add Phase 4 as a stronger intervention attempt;
- report whether the failure was measurement-driven or method-driven;
- run committee review for score adjustments.

If Phase 4 fails infrastructure:

- write `swarm/blocked_phase4_<date>.md`;
- keep the paper in its current Phase 3 state.

## Forbidden Actions

- Modifying any Phase 0, Phase 1, Phase 2, or Phase 3 preregistration.
- Modifying this Phase 4 preregistration after Phase 4 data are observed.
- Adjusting Phase 4 thresholds after observing data.
- Using AWQ-style activation-aware scaling.
- Using SmoothQuant-style activation folding.
- Skipping static-2% matched-budget control.
- Skipping magnitude-averaging control.
- Skipping sparse/primary/dense grid sensitivity.
- Authoring additional pivots beyond the conditional follow-ups listed above.
- Marking any paper camera-ready final.
- Upgrading vLLM.
- Downloading Qwen3.6 or Kimi Linear weights.
- Cherry-picking traces, layers, or positions post-hoc.
- Re-running Phase 3 with different parameters.

## Stop Conditions

- Cumulative `swarm/state.json` `gpu_hours_used` would exceed `60`.
- Git push fails twice in a row.
- Audit detects preregistration drift or other configured integrity failure.
- Three consecutive infrastructure failures during runner execution.
- Granite-4.0-H-Small OOMs on the 96GB GPU under any expected configuration.
- Disk free on `/workspace` drops below `50GB`.
- Either mandatory control outperforms union by more than `0.10` median
  recovery.
- Any subagent proposes additional pivots beyond Phase 4 conditionals.

## Required Result Packet

The Phase 4 result packet must include:

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
- control and grid-sensitivity tables;
- artifact hashes;
- checker output;
- artifact completeness record.

## Required Reporting

The Phase 4 result summary must compare Phase 3 and Phase 4 explicitly:

- model and snapshot;
- scoring-window length;
- BF16/static-1% gap distribution;
- no-gap trace fraction;
- primary union median recovery and CI;
- static-2% and magnitude-average controls;
- sparse/primary/dense grid sensitivity;
- GPU hours used;
- branch decision and paper-status consequence.
