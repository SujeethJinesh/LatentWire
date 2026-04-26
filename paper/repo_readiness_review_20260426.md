# Repo Readiness Review

- date: `2026-04-26`
- status: not ICLR-ready
- estimated distance: one stable positive method plus larger-slice, seed-repeat,
  source-control, and cross-family replication gates

## Current Paper Story

The honest current story is conditional innovation rather than proven latent
transfer. Target-side candidate/self-repair gives a strong decoder floor, C2C
shows real cache-level headroom, and LatentWire has a mature evaluation harness.
What is missing is a deployable LatentWire method whose improvement is both
source-derived and stable under source-destroying controls.

The strongest bound is the SVAMP32 C2C-derived syndrome sidecar:

- strict target-side pool: `14/32`
- clean source-necessary IDs: `2/6`
- controls clean union: `0/6`
- syndrome size: `1` byte
- blocker: the syndrome uses C2C numeric answers, so this is a bound, not a
  deployable positive method

The first C2C-mechanism distillation attempt is now negative:

- scalar prefill trace: matched `11/32`, clean source-necessary `0/6`
- residual prefill trace: matched `12/32`, clean source-necessary `0/6`
- target-only decoder floor: `14/32`
- decision: do not scale summary-feature C2C syndrome distillation without a
  new token/layer-level mechanism reason

The first Perceiver answer-teacher contrastive connector is also negative:

- fixed gates `0.125`, `0.15`, `0.20`: matched-only clean residual IDs `0/6`
- matched-positive clean IDs: `2/6`, but both are explained by shuffled-source,
  target-only, or slots-only controls
- decision: do not run generation for this checkpoint

The same branch also fails on the stronger SVAMP70 C2C-vs-process-repair
surface:

- clean C2C source-only IDs after excluding process-repair: `10`
- teacher-forced gate `0.15`: matched-positive clean `4/10`
- matched-only clean: `0/10`
- control-leak clean: `4/10`
- decision: kill this connector family until the objective changes

The first objective-level rescue also fails on SVAMP32:

- added training-time anti-memory controls against `target_only` and
  `slots_only`
- fixed gates `0.125`, `0.15`, `0.20`: matched-only clean residual IDs `0/6`
- matched-positive clean IDs: `2/6`, but both are explained by zero-source or
  slots-only controls
- mean matched-control delta remains negative at all tested gates
- decision: do not run generation; pivot away from receiver-conditioned
  Perceiver/delta-memory signal formation unless a materially new objective or
  architecture reason appears

The simplest source-only sidecar/router is also negative:

- source-generated numeric residue sidecar with target-side candidate-pool
  decoding
- source numeric coverage: `32/32`
- matched: `4/32`
- target-self preserve: `0/3`
- clean source-necessary IDs: `0/6`
- controls clean union: `0/6`
- decision: kill raw source-generated numeric residue sidecars; the clean
  issue is weak source signal, not target/control leakage

The strongest GSM mechanism clue is `dynalign_module_replace_residrank16`:

- GSM8K32 smoke: `4/32` vs target `2/32`
- GSM8K70 seed 0: `8/70` vs target `4/70`
- seed-0 source controls: zero/shuffle with target fallback retain `0/6` live
  wins
- blocker: seed stability fails; seed 3 is `2/70`, seed 4 is finite but only
  `4/70` with paired `3W/3L/64T`, and seeds 1/2 hit nonfinite checkpoint
  failures

## What Is Done

- Paper/evidence ledgers exist and are useful:
  - `paper/experiment_ledger_20260421.md`
  - `paper/benchmark_expansion_order_20260422.md`
  - `paper/reviewer_feedback.md`
  - SVAMP/GSM per-run memos under `paper/`
- Evaluation machinery is broad:
  - frozen-slice materialization and exact-ID utilities
  - paired/bootstrap uncertainty and oracle/headroom analyzers
  - source controls: zero source, shuffled source, target-only, slots-only, and
    same-norm/noise controls where applicable
  - GSM residual campaign/sweep wrappers with checkpoint health checks
  - SVAMP source/oracle/syndrome analyzers
  - C2C, KVComm, KVPress, and LatentMAS wrapper or matrix support
- Tests are healthy for unit/schema/tooling coverage:
  - `./venv_arm64/bin/python -m pytest -q`
  - result on this review: `668 passed in 26.77s`

## Saturated Or Weakened Branches

- Raw dynalign residual is a mechanism probe, not a paper method, because the
  finite repeat is target-negative and two repeat seeds are nonfinite.
- Selector-gap accept/fallback is killed as a method: fresh zero/shuffle controls
  retain gated wins, so the score is not source-specific.
- Further simple whitening/conditioning is low priority: it can prevent nonfinite
  failures but trades away the seed-0 ceiling.
- Query-pool and ID-weighted SVAMP variants are saturated below the clean gate:
  best rows recover only `1/6` clean residual IDs.
- Perceiver-query, delta-memory, contrastive delta-memory, answer-teacher
  microfit, pooled source-hidden syndrome, and learned source-token syndrome
  probes all fail the source-derived clean gate.
- C2C prefill scalar/residual summary syndrome probes fail the strict SVAMP32
  gate and remain below the target-only decoder floor.
- Perceiver-query answer-teacher plus source-control contrast fails the
  teacher-forced pre-generation gate; target/control memory still explains the
  apparent clean-ID signal.
- Scaling the same Perceiver answer-teacher connector to SVAMP70 also fails
  the teacher-forced pre-generation gate.
- Adding anti-memory target-only/slots-only training controls to that Perceiver
  branch also fails the SVAMP32 teacher-forced pre-generation gate.
- Source-generated numeric residue sidecar/router is killed: it avoids control
  leakage but has no clean source-necessary recovery.
- Source final-answer copying and stronger-source source-margin escalation are
  killed for the current frozen SVAMP32 clean IDs.
- Direct source-hidden syndrome readout is killed, including all-layer pooled
  features: all-layer Qwen2.5 features reach only `9/32`, underperform the
  `14/32` target-only floor, preserve only `2/3` target-self rows, and recover
  `0/6` clean source-necessary IDs.
- The first learned query-bottleneck residue predictor over all-layer summary
  tokens is also negative: it matches the all-layer ridge gate at `9/32`,
  preserves only `2/3` target-self rows, and recovers `0/6` clean
  source-necessary IDs.
- The full all-layer source-token learned syndrome probe is negative:
  matched `7/32`, target-only `14/32`, target-self `2/3`, and clean
  source-necessary `0/6`.
- The process-repair / selector stack is negative on the strict SVAMP32
  source-control surface: process repair selected route reaches only `10/32`
  versus `14/32` target-self repair, recovers `1/6` clean residual IDs, loses
  `2/3` target-self repair wins, and selects the target candidate on `32/32`
  examples.

## Main Gaps

1. No deployable positive method.
   The only clean-passing SVAMP object is the C2C-derived syndrome bound.

2. No stable larger-slice positive row.
   GSM seed 0 is source-dependent, but the method is not seed-stable.

3. Exact reproducibility is incomplete from tracked files alone.
   `results/`, `.debug/`, checkpoint tensors, and external repos are ignored;
   many decisive artifacts are local-only and referenced by hashes or commands.

4. Integration testing is thin.
   The unit suite is strong, but `transformers` and vendor integrations are
   mostly stubbed. Real cached-model integration should be a separate marked
   test lane.

5. Cross-family evidence is intentionally blocked.
   The benchmark order correctly says not to widen until the same-family gate is
   cleared.

## Highest-Priority Next Gate

Pivot to source-surface discovery rather than more SVAMP32 source-state
residue predictors or repair-stack tuning. The next live branch is to convert
the strongest toy interface clue, quotient/GPA sparse dictionaries with
sequence-aligned byte sidecars, into a real cross-family tokenizer/interface
stress gate.

Promotion rule:

- matched `>=14/32`
- target-self `3/3`
- clean source-necessary `>=2/6`
- numeric coverage `>=31/32`
- exact ordered ID parity
- zero-source, shuffled-source, label-shuffle, target-only, and slots-only
  controls have clean union `0/6`

Do not widen to medium, cross-family, or long-context benchmarks until the new
branch clears the same strict small-gate source-control surface.

## Engineering Follow-Ups

- Add `pytest -m integration` for tiny cached HF runs covering calibration,
  evaluation, C2C, and KVComm wrappers.
- Replace first-N slices with manifest-backed exact-ID slices including input
  hash, ordered IDs, command, seed, model revision, and output hashes.
- Make the source-control matrix mandatory for every promoted positive row.
- Unify root/package requirements and default local commands around
  `venv_arm64` on this machine.
- Build one tracked artifact manifest for decisive predictions/checkpoints,
  including external repo commits and model revisions.

## Review Inputs

- Local ledger and memos in `paper/`
- Implementation stack in `latent_bridge/`, `scripts/`, and `tests/`
- Experiment artifacts under `results/`, `checkpoints/`, and `.debug/`
- Reference state under `references/`
- Three folder-level subagent audits: paper/story, code/eval, and artifacts
