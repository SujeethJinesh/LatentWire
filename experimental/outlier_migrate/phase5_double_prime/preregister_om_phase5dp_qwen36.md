# OutlierMigrate Phase 5'' Qwen3.6 Cross-Lineage Preregistration

**Frozen on**: 2026-05-12
**Frozen by**: Codex GPU swarm, under `swarm/goal.md` Phase 5'' authorization
**Pivot depth**: 1 from OutlierMigrate Phase 0
**Status**: Frozen before any Phase 5'' Qwen3.6 inference, activation rows,
or migration statistics were collected.

## Purpose

Phase 5'' tests whether the OutlierMigrate decode-time rank-migration
measurement transfers beyond the Mamba-2 hybrid lineage to a Gated DeltaNet
hybrid model. It is a cross-lineage validation experiment, not an
intervention.

## Model

Primary model:

- `Qwen/Qwen3.6-35B-A3B`
- HuggingFace snapshot commit observed before inference:
  `995ad96eacd98c81ed38be0c5b274b04031597b0`
- Architectural lineage: Gated DeltaNet hybrid + MoE

The model is accessed through the human-created SGLang environment at:

- `/workspace/.sglang`
- SGLang version observed at gate check: `0.5.9`
- Torch version observed at gate check: `2.9.1+cu128`

The original vLLM/torch/triton environment used for Phase 4/5' must remain
untouched.

## Hypothesis

The same migration metric used in OutlierMigrate Phase 0/1/2/5' measured on
Qwen3.6-35B-A3B will fall into one of two predefined regimes:

- **Dynamic regime**: migration > 0.05 and CI lower > 0.05, indicating the
  effect transfers to the Gated DeltaNet lineage.
- **Static regime**: migration < 0.05 and CI upper < 0.05, indicating the
  effect is not present at the measured threshold for this lineage.

The decision is non-directional. Either result is admissible.

## Trace set

- Source: AIME-2025
- Count: 24 traces
- Selection: deterministic prompt indices 0-23
- Prompt SHA-256:
  `aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e`

The trace set must match OutlierMigrate Phase 1 and Phase 5'.

## Decode position grid

Measure at decode positions:

- `{100, 500, 1000, 5000, 10000, 20000}`

## Measurement procedure

1. Run deterministic greedy decoding in the `/workspace/.sglang`
   environment.
2. Capture residual-stream output magnitudes at each block output at every
   preregistered decode position.
3. Where the runtime exposes separable internals without source
   modification, also capture layer-stratified groups:
   - Gated DeltaNet output
   - Gated Attention output
   - MoE output
4. If SGLang exposes only post-block residual outputs without unsupported
   source modification, the gate may proceed with post-block residual output
   only, and the packet must record the missing internal stratification as a
   limitation.
5. If neither post-block residual output nor equivalent activation capture is
   accessible without modifying SGLang/model source, return
   `FAIL_INFRA_QWEN36`.
6. For each layer and trace, identify the top-1% channels by magnitude at
   decode position 100 and compute migration at position 20000 using the
   same rank movement threshold as Phase 0/1/2/5'.
7. Bootstrap over the 24 traces with seed `20260513`.

## Decision rule

### DYNAMIC_REGIME_QWEN36

Return this decision if:

- migration fraction > 0.05, and
- bootstrap 95% CI lower bound > 0.05.

### STATIC_REGIME_QWEN36

Return this decision if:

- migration fraction < 0.05, and
- bootstrap 95% CI upper bound < 0.05.

### AMBIGUOUS_QWEN36

Return this decision for any in-between result or wide confidence interval.

### FAIL_INFRA_QWEN36

Return this decision for infrastructure failure, inaccessible activation
hooks, incomplete trace set, model load failure, missing artifacts, or
packet incompleteness.

## Required artifacts

The result packet must include:

- environment snapshot from `/workspace/.sglang`
- model provenance with HuggingFace snapshot commit
- prompt manifest and prompt SHA
- command line and shell wrapper showing `/workspace/.sglang/bin/activate`
- stdout/stderr logs
- activation evidence or reduced-rank evidence sufficient to recompute the
  metric
- metrics JSON with bootstrap CI
- migration decomposition JSON/Markdown
- checker result and artifact check
- artifact hashes

## Forbidden actions

- Modifying the vLLM environment to run this phase.
- Creating or modifying `/workspace/.sglang`.
- Modifying SGLang or Qwen3.6 source code to force hooks.
- Switching models after observing any Qwen3.6 migration row.
- Changing trace set, decode grid, threshold, rank-delta criterion, or
  bootstrap seed after observing data.
- Claiming Kimi Linear or any other Lineage 3 model replicates this result
  without measured data.

## Integration rule

If Phase 5'' returns `DYNAMIC_REGIME_QWEN36`, the paper may broaden from
"hybrid Mamba-2 reasoning LLMs" to "hybrid Mamba-2 and Gated DeltaNet
reasoning LLMs" for the migration measurement.

If Phase 5'' returns `STATIC_REGIME_QWEN36`, the paper keeps the Mamba-2 /
pure-Transformer evidence separate and reports Qwen3.6 as a scoped
counterpoint.

If Phase 5'' returns `AMBIGUOUS_QWEN36` or `FAIL_INFRA_QWEN36`, the paper
must not use Qwen3.6 to broaden the claim.
