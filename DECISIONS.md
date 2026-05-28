# Positive-Method Sprint Decisions

Last updated: 2026-05-28T04:25Z

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
| WJAC | PROVISIONAL_KILL | Prior CPU doc found 3/4 kill diagnostics on all analyzed surfaces; artifactized prefilter must reproduce this before final skip. |
| LAMBDA | PROVISIONAL_KEEP | Cached activations show late-layer dominance; no causal marginal curves yet. |
| HYST | PROVISIONAL_KEEP | Churn is low on Nemotron after warmup but remains high enough on Granite/DeepSeek/Falcon to justify smoke testing. |
| M-SURFACE | QUEUED | Diagnostic only; needs hook map before GPU sanity. |
| M-BRANCH | QUEUED | Falcon-only diagnostic; needs hook map before GPU sanity. |
| M-FISH | DEFERRED | No hybrid Fisher infrastructure; only reconsider for DeepSeek if WJAC/KLLOOK show sensitivity headroom. |
| M-GATE | CONDITIONAL | Only if DeepSeek is close but WJAC/Fisher fail. |
| M-ROUTE | CONDITIONAL | Only if V1 shows M11b beats or matches ParoQuant on Nemotron. |

## Next Decision Gate

After V1 completes and artifactized CPU filters land:

1. If WJAC remains killed and LAMBDA/HYST survive, run only LAMBDA/HYST smoke
   on fixed DeepSeek/Falcon traces.
2. If all cheap branches are killed by artifactized filters, skip smoke and run
   restricted KLLOOK ceiling test.
3. If V1 is headline-changing, pause paper reframing and write a progress note
   before integrating.
