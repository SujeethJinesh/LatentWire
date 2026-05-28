# Positive-Method Sprint Decisions

Last updated: 2026-05-28T04:18Z

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
| M-SURFACE | DEFERRED | Hookable but no cached internal activations; requires tiny Granite sanity only after higher-priority smoke or if a finalist needs placement evidence. |
| M-BRANCH | DEFERRED | Falcon branch-local cached activations absent and implementation inspection does not justify a GPU branch run before smoke. |
| M-FISH | DEFERRED | No hybrid Fisher infrastructure; only reconsider for DeepSeek if WJAC/KLLOOK show sensitivity headroom. |
| M-GATE | CONDITIONAL | Only if DeepSeek is close but WJAC/Fisher fail. |
| M-ROUTE | CONDITIONAL | Only if V1 shows M11b beats or matches ParoQuant on Nemotron. |

## Next Decision Gate

After V1 completes:

1. If V1 is headline-changing, pause paper reframing and write a progress note
   before integrating.
2. Otherwise, run only LAMBDA/HYST smoke on the fixed DeepSeek/Falcon traces
   selected in `artifacts/funnel_prefilters/smoke_traces.json`.
3. If LAMBDA/HYST smoke both fail, run restricted KLLOOK only if needed to
   decide whether channel-set methods have ceiling headroom.
