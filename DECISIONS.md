# Positive-Method Sprint Decisions

Last updated: 2026-05-28T16:55Z

## Current Framing

The working story is now rotation-first. Static and hard-switch channel
protection fail under drift; ParoQuant-style rotation succeeds on Granite and
dominates M11b on Nemotron. ParoQuant is not our method. The live contribution
candidate is DriftRot: drift-aware rotation calibration, surface/branch
selection, or residual correction that beats or robustifies static ParoQuant.
If DriftRot fails, the mechanism/protocol contribution is that channel identity
drifts while rotation removes basis dependence.

The decision rule is prospective in design and evaluated descriptively on the
current four-model study. It is not described as pre-specified or validated
unless a genuinely held-out frozen-threshold run is executed.

## Live / Killed Branches

| Branch | Decision | Reason |
|---|---|---|
| V1 ParoQuant-on-Nemotron | PASS_ROTATION_DOMINATES | ParoQuant median recovery is 1.047 with CI95 [1.007, 1.292], beating Nemotron M11b top-10 by +0.232. This is headline-changing baseline-vetting evidence. |
| DriftRot | LIVE | Primary question: can long-decode drift-aware rotation choices beat or robustify static ParoQuant on held-out traces/seeds? |
| ParoQuant Falcon smoke | PASS_ROTATION_RESCUE | Full 12-trace packet gives median recovery 0.381 with CI95 [0.0645, 0.547], beating Falcon M11b top-10 by +0.337. Falcon channel rescue drops in priority. |
| ParoQuant DeepSeek smoke | PASS_ROTATION_DOMINATES | Full 12-trace packet gives median recovery 0.756 with CI95 [-0.246, 0.855], beating DeepSeek static-top10 by +0.379. Median supports rotation-dominant framing, but the negative lower CI keeps DriftRot tail-control live. |
| Granite clip/CVaR retune | PROMOTE_AFTER_G0_G1 | Only DriftRot config gate promoted by CPU filters; target ParoQuant's Granite tail, not a broad grid. |
| Drift-aware pairing | NEEDS_ACTIVATION_CACHE | Pairings differ from current ParoQuant on block-output proxies, but exact rotation-surface caches are required before GPU. |
| Residual correction | KILL_TOP8X32_MOE_RESIDUAL_DIAGNOSTIC | Residual norms, activation EMA, candidate pool, and DeltaW columns were produced. A fixed Granite prompt-4 diagnostic scored -12.29 recovery versus tight ParoQuant reference 5.94, so this MoE top-8x32 candidate is not promoted. Only reopen with a bounded or KLLOOK-gated design. |
| WJAC | KILL | Artifactized prefilter found at least two kill diagnostics on each covered model/slice; DeepSeek/Falcon full cached coverage, Granite/Nemotron representative slice coverage. |
| LAMBDA | DEFER | Falcon prior reallocates 18.9% of total budget, but the standalone smoke gate is false because causal per-layer headroom is absent. |
| HYST | DEFER_AFTER_FALCON_ROTATION_PASS | Falcon churn/local-pool gate selected margin `m=5`, but ParoQuant now rescues Falcon enough that channel fallback is lower priority. |
| RISKGUARD | DEFER | Best cached trigger is in-sample only; leave-one-trace-out CI lower bound collapses to 0 and CVaR remains negative. |
| TRACE-ROUTER | WEAK_OFFLINE_ONLY | Granite/DeepSeek show positive tiny-n CV gain, but this needs a preregistered larger frozen slice before evidence claims. Falcon remains not routable. |
| M-SURFACE | CONDITIONAL_DIAGNOSTIC | Granite `mamba_out_projection_input` hook sanity is cheap; promote only if internal drift is <0.30 or at least 0.15 below same-run post-block. |
| M-BRANCH | DEFER | No cached branch-local Falcon activations and no branch diagnostic is recommended before stronger smoke evidence. |
| LayerKeep | CONDITIONAL_FALLBACK | Falcon late layers `30-35` are a bounded fallback candidate; no-gap detector is not recommended. |
| M-FISH | DEFERRED | No hybrid Fisher infrastructure; only reconsider for DeepSeek if WJAC/KLLOOK show sensitivity headroom. |
| M-GATE | CONDITIONAL | Only if DeepSeek is close but WJAC/Fisher fail. |
| M-ROUTE | CONDITIONAL | Only if V1 shows M11b beats or matches ParoQuant on Nemotron. |

## Next Decision Gate

After V1 completed, it triggered `PASS_V1_PAROQUANT_NEMOTRON_ROTATION_DOMINATES`.
The Falcon follow-up also passed: `PASS_V1_PAROQUANT_FALCON_ROTATION_RESCUE`.

The next decision gate is rotation-first:

1. Run CPU filters C1-C12 in parallel into disjoint `artifacts/<task_name>/`
   directories.
2. Prepare ParoQuant Falcon and DeepSeek smoke packets.
3. ParoQuant Falcon passed, so Falcon channel rescue drops in priority.
4. ParoQuant DeepSeek passed, so the rotation-dominant four-model story is now
   supported by Granite, Nemotron, Falcon, and DeepSeek.
5. Next run DriftRot Scale/CVaR/Clip, residual correction, pairing,
   BranchRot, M-SURFACE, or Falcon channel fallbacks according to their CPU
   gates.

Do not claim ParoQuant as our method. A new method passes only if it beats
ParoQuant on held-out traces/seeds, improves ParoQuant CI/tail, or rescues a
model where ParoQuant fails.

CPU gates now favor this concrete order:

1. Granite clip/CVaR retune smoke to test whether DriftRot can improve
   ParoQuant tails/CI rather than merely reproduce the rotation baseline.
2. Pairing only after collecting exact rotation-surface caches; residual
   correction is demoted unless a bounded or KLLOOK-gated design is written.
3. Falcon/DeepSeek channel fallbacks only if a later rotation-specific gate
   exposes a weakness requiring architecture-local rescue.
