# Positive-Method Sprint Decisions

Last updated: 2026-05-28T06:40Z

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
| LAMBDA | DEFER | Falcon prior reallocates 18.9% of total budget, but the standalone smoke gate is false because causal per-layer headroom is absent. |
| HYST | READY_FOR_SMOKE | Falcon churn/local-pool gate selected margin `m=5`; runner now supports `--methods hyst --hyst-exit-margin-pct-points 5`. |
| RISKGUARD | DEFER | Best cached trigger is in-sample only; leave-one-trace-out CI lower bound collapses to 0 and CVaR remains negative. |
| TRACE-ROUTER | WEAK_OFFLINE_ONLY | Granite/DeepSeek show positive tiny-n CV gain, but this needs a preregistered larger frozen slice before evidence claims. Falcon remains not routable. |
| M-SURFACE | CONDITIONAL_DIAGNOSTIC | Granite `mamba_out_projection_input` hook sanity is cheap; promote only if internal drift is <0.30 or at least 0.15 below same-run post-block. |
| M-BRANCH | DEFER | No cached branch-local Falcon activations and no branch diagnostic is recommended before stronger smoke evidence. |
| LayerKeep | CONDITIONAL_FALLBACK | Falcon late layers `30-35` are a bounded fallback candidate; no-gap detector is not recommended. |
| M-FISH | DEFERRED | No hybrid Fisher infrastructure; only reconsider for DeepSeek if WJAC/KLLOOK show sensitivity headroom. |
| M-GATE | CONDITIONAL | Only if DeepSeek is close but WJAC/Fisher fail. |
| M-ROUTE | CONDITIONAL | Only if V1 shows M11b beats or matches ParoQuant on Nemotron. |

## Next Decision Gate

After V1 completes:

1. If V1 is headline-changing, pause paper reframing and write a progress note
   before integrating.
2. Otherwise, run Falcon HYST-only smoke on fixed stratified traces `[7, 1, 11]`
   with `--methods hyst --hyst-exit-margin-pct-points 5`.
3. Do not run standalone Falcon LAMBDA or LAMBDA+HYST unless new causal
   per-layer evidence reverses the C2 gate.
4. If HYST fails or is ambiguous, run the cheap Granite M-SURFACE 2-trace
   diagnostic or Falcon LayerKeep `30-35`, selecting the one with the clearer
   implementation surface at that point.
5. If all cheap branches fail or remain ambiguous, run restricted Falcon KLLOOK
   to decide whether channel-set methods have ceiling headroom.
