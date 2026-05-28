# Phase 9 Positive-Method Funnel Smoke Preregistration

Date: 2026-05-28

## Status

Frozen after the CPU offline filters:

- WJAC is killed offline under the corrected 2-of-4 diagnostic rule.
- LAMBDA survives as `FLAG_DOMINANT_LAYERS`, not as evidence.
- HYST survives as `KEEP/INCONCLUSIVE`, not as evidence.
- Smoke traces are stratified and fixed in `docs/smoke_trace_selection.md`.

This packet runs only the surviving non-WJAC smoke branches on DeepSeek and
Falcon. It is not a full evaluation and cannot by itself support a paper
positive-method claim.

## Models and Smoke Traces

DeepSeek-R1-Distill-Qwen-1.5B:

- prompt indices `[5, 11, 8]`;
- roles: positive-gap trace, no-EOS/high-drift stress trace, representative
  median trace.

Falcon-H1-0.5B-Instruct:

- prompt indices `[7, 1, 11]`;
- roles: positive/high-drift trace, long stress trace, representative median
  trace.

No random smoke-trace selection is allowed.

## Regimes

Reused baselines:

1. BF16 reference.
2. static top-1%.
3. M11b top-10.

New smoke regimes:

1. `mlambda_top10_smoke`: same total protected-channel budget as M11b top-10,
   but allocated across layers by global EMA marginal score over cached
   activations.
2. `hyst_top10_smoke`: flat per-layer M11b top-10 budget with hysteretic
   protected-set updates. A channel enters at top-k and exits only below the
   frozen margin recorded by the CPU gate. The preregistered default is
   top-2k; the Falcon gate in `artifacts/hyst_falcon/` selects a narrower
   5 percentage-point margin and the runner must record that value in its
   config.
3. `random_lambda_top10`: random matched layer allocation/control.
4. `random_hyst_top10`: random matched flat-budget/control.

`lambda_hyst_top10_smoke` is not run in the first smoke packet unless explicitly
requested later. Single factors run before combinations.

## Decision Rules

For each model and method, retain the method for partial evaluation if it has no
catastrophic-negative trace and satisfies at least one rescue gate:

- median recovery beats same-smoke M11b top-10 by at least `0.10`;
- method wins at least 2 of the 3 smoke traces against same-smoke M11b top-10;
- Falcon rescue: median recovery is at least `0.20`;
- DeepSeek rescue: median recovery is greater than the same-smoke M11b top-10
  median and greater than `0.335`.

Catastrophic negative means recovery `< -1.0` on any included positive-gap
smoke trace. No-gap status is determined by the reused static top-1% gap and is
reported, but the method cannot change no-gap status because all regimes share
the same BF16/static references.

## Interpretation

- PASS_SMOKE_SURVIVOR: at least one method/model pair satisfies a rescue gate.
- KILL_SMOKE_NO_SURVIVOR: no method/model pair satisfies any rescue gate.
- FAIL_INFRA_FUNNEL_SMOKE: artifact, scoring, prompt, or control failure.

Smoke PASS only authorizes 6-trace partial evaluation. It is not a paper result.

## Forbidden Actions

- Do not run WJAC endpoint scoring in this packet.
- Do not substitute random traces.
- Do not merge DeepSeek and Falcon medians into a single cross-model score.
- Do not report smoke results as final evidence.
