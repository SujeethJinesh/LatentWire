# Phase 9 M26 Stable-Core Protection Preregistration

**Frozen on**: 2026-05-18
**Frozen by**: Codex GPU swarm, under post-M18 mechanism-first queue
authorization
**Status**: Frozen after M11b landed and after the required stable-core size
check, before any M26 scoring run.

## Terminology

This preregistration uses **decode-position channel drift** and
**long-decode channel drift** for the measured phenomenon. The paper may use
"outlier migration" only in a disambiguated terminology note distinguishing
this channel-across-decode-position usage from SmoothQuant's
activation-to-weight scale migration and MoBiQuant's bit-width-dependent
token-sensitivity shift.

## Purpose

M2, M10, M11, M18, and DecDEC all attempted to track or react to changing
channel salience. M11b produced a mechanical budget-scaling pass, but the
bootstrap interval was wide. Post-M18 Analysis 5 found that every measured
model retains a nontrivial stable high-magnitude core: channels that remain
in the top-1% set across all measured decode positions.

M26 tests a structurally different hypothesis: instead of chasing drifting
channels, protect only the empirically stable core `C`. If prior methods fail
because drifting-channel signals are noisy, then stable-core protection could
recover quality by focusing budget on persistent channels.

## Stable-Core Size Precondition

Before this preregistration, the required size check was written to:

`experimental/outlier_migrate/phase9/m26_stable_core_size.md`

Granite-4.0-H-Small stable core:

- total channels across measured layers: `163840`
- stable-core channels: `847`
- stable-core fraction: `0.516967773438%`

The human-authorized suitability band was roughly `0.2%` to `0.8%` of
channels, with skip conditions below `0.05%` or above `2%`. Granite-Small is
inside the suitability band, so M26 proceeds.

## Prior-Art Differentiation

HCP/CHON (arXiv 2602.02047) uses hot-channel compensation for NVFP4
pretraining. It is a pretraining-time intervention, assumes persistent
channels are stable by design, and targets NVFP4.

M26 differs on five axes:

1. inference-time post-training quantization rather than pretraining;
2. W4A16 rather than NVFP4;
3. stable channels identified empirically from long-decode packet data rather
   than assumed by training design;
4. no pretraining or model-weight modification beyond the same simple
   symmetric INT4 path used by prior OutlierMigrate interventions;
5. decision unit is the intersection of top-1% activation channels across
   measured decode positions, not a training-time persistent hot-channel
   compensation table.

DecDEC (OSDI 2025; arXiv 2412.20185) identifies salient channels reactively
at each decoding step. M26 is the opposite: a static stable-core set computed
once from calibration packets and used throughout decode.

PMPD (arXiv 2410.13461) changes precision across token positions. M26 keeps
the format fixed and changes which activation channels are protected.

## Model

Primary model:

- `ibm-granite/granite-4.0-h-small`
- HuggingFace snapshot commit:
  `b8c0982bab7fde4eb48110f5a069527c008fab39`

No cross-model replication is part of this primary M26 gate. If M26 passes on
Granite-Small, replication on Nemotron-3-Nano and DeepSeek-R1-Distill-Qwen-1.5B
is authorized by the prior queue.

## Trace Set

- Source: AIME-2025
- Count: 12 traces under vacation-mode V4 standing decision
- Selection: deterministic prompt indices `0-11`
- Prompt file:
  `experimental/shared/prompts/aime_2025_indices_0_23.jsonl`
- Prompt payload SHA-256 over indices `0-11`: computed by the runner and
  written to the result packet before scoring.

The 12-trace slice matches M2/M10/M11/M18/DecDEC/M11b on Granite-Small.

## Quantization and Scoring

- BF16 baseline target traces are deterministic greedy decode traces.
- Weight quantization: simple symmetric per-channel INT4 represented as
  dequantized tensors for framework compatibility.
- Activations: FP16.
- Protected channels remain in the unquantized/dequantized high-precision
  path.
- Scoring position: decode position `10000`.
- Scoring window: 512 tokens ending at position `10000`.
- Bootstrap samples: `1000`.
- Bootstrap seed: `20260530`.

AWQ-style activation-aware scaling, SmoothQuant scale folding, post-hoc
threshold tuning, and additional unpreregistered core-size variants are
forbidden.

## Stable-Core Construction

For each layer independently:

1. Use the existing Granite-Small calibration packet:
   `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z`.
2. Average activation magnitudes over prompts at each measured decode
   position: `{100, 500, 1000, 5000, 10000, 20000}`.
3. At each position, select the top-1% channels by prompt-averaged magnitude.
4. Define `C_l` as the intersection of those top-1% sets across all measured
   positions.
5. M26 protects exactly `C_l` for layer `l` throughout decode.

No M26 scoring data may be used to alter `C_l`.

## Regimes

Each packet must evaluate:

1. BF16 baseline.
2. Static-1% at calibration position `100` (Phase 4/Phase 9 baseline).
3. M26 stable-core `C`-only protection.
4. `C` plus current top-1% union, using the current top-1% set at the scoring
   position from the calibration trajectory.
5. Random matched-size control: per-layer random channels with exactly
   `|C_l|` channels, generated from bootstrap seed `20260530`.

## Metric

For each trace and non-BF16 regime:

`recovery = 1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_top1 - perplexity_BF16)`

Only traces with a positive recoverable static top-1% gap are included in the
primary recovery median. The packet must still report no-recoverable-gap count
and fraction.

Report:

- per-trace perplexity for all regimes;
- per-trace recovery for static-1%, M26 core-only, core+current-top1, and
  random matched-size control;
- median recovery and bootstrap 95% CI for every non-BF16 regime;
- `|C_l|` per layer and total `|C|`;
- M26 separation from static-1%;
- M26 separation from random matched-size control.

## Decision Rule

### PASS_M26_STABLE_CORE

Return pass if M26 core-only protection satisfies all of:

1. median recovery is at least `0.30`;
2. bootstrap 95% CI lower bound is greater than `0.10`;
3. median recovery beats static-1% by at least `0.15`;
4. median recovery beats random matched-size control by at least `0.20`.

### KILL_M26_NO_IMPROVEMENT

Return this kill if M26 core-only median recovery is within `0.05` of
static-1%.

### KILL_M26_RANDOM_CONTROL_BEATS

Return this kill if the random matched-size control beats M26 core-only by
more than `0.10` median recovery.

### AMBIGUOUS_M26

Return ambiguous for intermediate outcomes, including cases where M26 improves
over static-1% but misses CI or random-control separation thresholds.

### FAIL_INFRA_M26

Return infrastructure failure for model load failure, incomplete packet, OOM
that cannot be fixed by batch-size reduction, missing required artifacts, or
checker failure that prevents applying the mechanical decision rule.

## Required Artifacts

Each M26 packet must contain:

- environment snapshot (`pip freeze`, `nvidia-smi`, CUDA/driver, git SHA);
- model provenance with HuggingFace snapshot commit;
- prompt manifest and prompt SHA;
- exact command line and stdout/stderr logs;
- BF16 target traces or cited cache source with SHA-256;
- stable-core construction file with per-layer `C_l`;
- quantization configuration;
- per-trace perplexity table;
- per-trace recovery table;
- bootstrap CI table;
- checker result and artifact check;
- artifact hashes.

## Forbidden Actions

- Modifying prior preregistration files.
- Adjusting M26 thresholds after observing M26 data.
- Changing the stable-core definition after observing M26 data.
- Adding a larger core, union core, or current-top-K variant beyond the
  preregistered regimes.
- Dropping the random matched-size control.
- Using AWQ-style scaling, SmoothQuant scale folding, or source-code model
  modifications.
