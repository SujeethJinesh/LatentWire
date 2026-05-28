# Positive-Method Sprint Decisions

Last updated: 2026-05-28T06:23Z

## Current Framing

The paper remains regime-aware: static and hard-switch channel protection fail
under drift; budgeted EMA succeeds in the Nemotron MoE-hybrid regime; rotation
succeeds in the Granite dense-hybrid regime; the protocol should choose,
reject, or defer methods from cheap calibration evidence.

The decision rule is prospective in design and evaluated descriptively on the
current four-model study. It is not described as pre-specified or validated
unless a genuinely held-out frozen-threshold run is executed.

## Live / Killed Branches

| Branch | Decision | Reason |
|---|---|---|
| V1 ParoQuant-on-Nemotron | RUNNING | Baseline-vetting GPU job active. |
| WJAC | KILL | Artifactized prefilter found at least two kill diagnostics on each covered model/slice; DeepSeek/Falcon full cached coverage, Granite/Nemotron representative slice coverage. |
| LAMBDA | SMOKE_ONLY | Artifactized funnel prefilter found meaningful layer heterogeneity on DeepSeek/Falcon, but no causal per-layer marginal recovery curves. |
| HYST | SMOKE_ONLY | Artifactized funnel prefilter found late protected-set churn with locally stable high-score pools on DeepSeek/Falcon. |
| RISKGUARD | RUNNING_CPU_GATE | Calibrate only from deployable cached signals; promote to GPU confirm only if predicted CI lower bound crosses 0 without oracle labels. |
| TRACE-ROUTER | RUNNING_CPU_GATE | Evaluate only as a descriptive calibration helper; Falcon is marked not routable by verified headroom. |
| M-SURFACE | RUNNING_REFRESH | Prior artifact was inconclusive; refresh hook complexity and promote only if a tiny diagnostic can test lower-drift internal surfaces cheaply. |
| M-BRANCH | RUNNING_REFRESH | Prior artifact did not recommend GPU; refresh hook map under Falcon-first funnel and promote only if branch-local drift gate is plausible. |
| LayerKeep | QUEUED_CPU_GATE | Coarse fallback only if layer-level signals separate while channel-level methods remain unstable. |
| M-FISH | DEFERRED | No hybrid Fisher infrastructure; only reconsider for DeepSeek if WJAC/KLLOOK show sensitivity headroom. |
| M-GATE | CONDITIONAL | Only if DeepSeek is close but WJAC/Fisher fail. |
| M-ROUTE | CONDITIONAL | Only if V1 shows M11b beats or matches ParoQuant on Nemotron. |

## Next Decision Gate

After V1 completes:

1. If V1 is headline-changing, pause paper reframing and write a progress note
   before integrating.
2. Otherwise, run Falcon-first LAMBDA then HYST smoke on fixed stratified
   traces `[7, 1, 11]`, using the refreshed C2/C3 configs.
3. Run LAMBDA+HYST only if both single-factor smokes pass and the combination
   beats the better single method.
4. If LAMBDA/HYST smoke both fail, run restricted KLLOOK only if needed to
   decide whether channel-set methods have ceiling headroom.
