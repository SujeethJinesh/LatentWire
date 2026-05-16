# Phase 9 M11b Budget Scaling Preregistration

**Frozen on**: 2026-05-16
**Frozen by**: Codex GPU swarm, under information-value queue
restructuring authorization
**Status**: Frozen after M11 landed with `KILL_M11_AMBIGUOUS` and before any
M11b scoring run.

## Terminology

This preregistration uses **decode-position channel drift** and
**long-decode channel drift** for the measured phenomenon. The paper may use
"outlier migration" only in a defined terminology note distinguishing this
channel-across-decode-position usage from SmoothQuant's activation-to-weight
difficulty transfer and MoBiQuant's precision-dependent token-sensitivity
shift.

## Purpose

M11 EMA-smoothed drift protection did not pass on Granite-4-H-Small, but it
also was not beaten by its random-walk control. This makes it structurally
different from M2 and M10, where boundary-discontinuous policies were beaten
by random-bin controls.

M11b tests mechanism hypothesis 3b from the information-value queue:
**budget insufficiency**. The question is whether M11 failed because the
smoothed top-1% channel budget was too small, rather than because the
smoothed signal was stale or because error accumulates over long decode.

## Precondition

M11b is authorized after:

- M2 landed with `KILL_M2_RANDOM_CONTROL_BEATS`;
- M10 landed with `KILL_M10_RANDOM_CONTROL_BEATS`;
- M11 landed with `KILL_M11_AMBIGUOUS` and random-walk did not beat M11;
- the human promoted Phase 1 mechanism-distinguishing experiments in this
  order: M18, DecDEC, M11b.

M11b must run on Granite-Small only in Phase 1. Cross-model replication is
reserved for whichever Phase 2 method, if any, passes.

## Prior-Art Differentiation

DecDEC (Park, Hyun, Kim, and Lee, OSDI 2025; arXiv 2412.20185) identifies
salient channels dynamically at each decoding step and fetches residual
compensation. M11b is not reactive per-step DecDEC. It keeps M11's
EMA-smoothed decision surface and asks whether increasing the protected
budget changes the outcome.

PMPD (Progressive Mixed-Precision Decoding, arXiv 2410.13461) states:
"PMPD builds upon the observation that the prefill phase and the initial
tokens of the decoding phase are more sensitive to approximations than later
tokens." M11b does not lower precision monotonically with decode position; it
keeps the format fixed and increases the channel budget for a smoothed
protected set.

KL Lens and layer-level mixed-precision methods operate at a layer or block
decision unit. M11b is channel-level and tests whether channel-budget scaling
is enough to handle decode-position channel drift.

## Model

Primary model:

- `ibm-granite/granite-4.0-h-small`
- HuggingFace snapshot commit:
  `b8c0982bab7fde4eb48110f5a069527c008fab39`

No additional models are part of this Phase 1 M11b gate.

## Trace Set

- Source: AIME-2025
- Count: 12 traces under vacation-mode V4 standing decision
- Selection: deterministic prompt indices `0-11`
- Prompt file:
  `experimental/shared/prompts/aime_2025_indices_0_23.jsonl`
- Prompt payload SHA-256 over indices `0-11`: computed by the runner and
  written to the result packet before analysis.

The 12-trace slice matches M2/M10/M11/M18 on Granite-Small.

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
- Bootstrap seed: `20260527`.

AWQ-style activation-aware scaling, SmoothQuant scale folding, and any
post-hoc threshold tuning are forbidden for M11b.

## M11b Protected-Set Update Rule

For each layer independently, maintain a protected-set score vector `p_l(t)`
with one score per channel.

Initialization:

- `p_l(0)` is the calibration top-1% indicator at decode position `100`.

At each decode position that is a multiple of `100`:

`p_l(t+1) = 0.3 * current_top_budget_indicator_l(t) + 0.7 * p_l(t)`

Budget levels:

- top-1%
- top-5%
- top-10%

For a given budget level, `current_top_budget_indicator_l(t)` marks the top
budget fraction of channels by current mean absolute activation magnitude.
Protected channels at position `t` are the highest-scoring channels in
`p_l(t)` up to that budget. There is no top-3% cap in the 5% and 10% budget
arms because the cap would invalidate the budget-scaling test.

The alpha is fixed at `0.3` by human authorization. Although the completed M11
packet reported alpha `0.5` as the numerically highest median-recovery arm,
M11b uses the explicitly authorized alpha to avoid post-hoc alpha selection.
See `swarm/vacation_decisions/20260516T201300_m11b_alpha_choice.md`.

## Regimes

Each packet must evaluate:

1. BF16 baseline.
2. Static top-1% protected set from position `100`.
3. M11b EMA-smoothed top-1%.
4. M11b EMA-smoothed top-5%.
5. M11b EMA-smoothed top-10%.
6. Static top-10% matched-budget control from position `100`.

## Metric

For each trace and M11b/control regime:

`recovery = 1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_top1 - perplexity_BF16)`

Only traces with a positive recoverable static top-1% gap are included in the
primary recovery median. The packet must still report the count and fraction
of no-recoverable-static-gap traces.

Report:

- per-trace perplexity for all regimes
- per-trace recovery for M11b top-1%, top-5%, top-10%, and static top-10%
- median recovery and bootstrap 95% CI for every non-BF16 regime
- M11b top-5% and top-10% separation from M11b top-1%
- M11b top-10% separation from static top-10%
- effective protected-channel count statistics for every budget level

## Decision Rule

### PASS_M11B_BUDGET_MATTERS

Return pass if either M11b top-5% or M11b top-10% satisfies both:

1. median recovery is at least `0.30`; and
2. median recovery beats static top-10% matched-budget control by at least
   `0.15`.

This supports budget insufficiency (hypothesis 3b).

### KILL_M11B_BUDGET_INSUFFICIENT

Return this kill if both M11b top-5% and M11b top-10% are within `0.05`
median recovery of M11b top-1%.

This supports stale signal (3a) or compound error (3c) over simple budget
insufficiency.

### AMBIGUOUS_M11B

Return ambiguous for intermediate outcomes, including cases where high-budget
M11b improves over top-1% but misses the `0.30` recovery shelf, fails to beat
static top-10% by `0.15`, or has highly overlapping CIs.

### FAIL_INFRA_M11B

Return infrastructure failure for model load failure, incomplete packet, OOM
that cannot be fixed by batch-size reduction, missing required artifacts, or
checker failure that prevents applying the mechanical decision rule.

## Required Artifacts

Each M11b packet must contain:

- environment snapshot (`pip freeze`, `nvidia-smi`, CUDA/driver, git SHA)
- model provenance with HuggingFace snapshot commit
- prompt manifest and prompt SHA
- exact command line and stdout/stderr logs
- BF16 target traces or cited cache source with SHA-256
- activation/top-channel evidence used for EMA updates
- protected-set trajectory for each budget level
- quantization configuration
- per-trace perplexity table
- per-trace recovery table
- bootstrap CI table
- checker result and artifact check
- artifact hashes

## Forbidden Actions

- Modifying prior preregistration files.
- Adjusting M11b alpha, budgets, or decision thresholds after observing M11b
  data.
- Running additional unpreregistered budget levels.
- Dropping the static top-10% matched-budget control.
- Using AWQ-style scaling or SmoothQuant scale folding.
- Selectively reporting only the best budget arm.
- Modifying model source code.

## Paper Integration Rule

If M11b passes, the paper should identify budget insufficiency as a major
mechanism and queue Phase 2 budget-threshold experiments. If M11b kills, the
paper should report that increasing the smoothed protected-channel budget to
top-10% still does not solve the gap, shifting interpretation toward stale
signal or long-horizon compound error. If M11b is ambiguous, report it as a
descriptive budget analysis rather than a positive method.
