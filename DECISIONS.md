# Positive-Method Sprint Decisions

Last updated: 2026-05-28T15:18Z

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
| ParoQuant Falcon smoke | NEXT | First rotation-first GPU gate; decides whether Falcon needs channel rescue. |
| ParoQuant DeepSeek smoke | NEXT | Tests whether rotation-dominance extends to the dense Transformer regime. |
| Granite clip/CVaR retune | PROMOTE_AFTER_G0_G1 | Only DriftRot config gate promoted by CPU filters; target ParoQuant's Granite tail, not a broad grid. |
| Drift-aware pairing | NEEDS_ACTIVATION_CACHE | Pairings differ from current ParoQuant on block-output proxies, but exact rotation-surface caches are required before GPU. |
| Residual correction | NEEDS_WEIGHT_RESIDUAL_CACHE | Method is specified, but no candidate pool exists until a ParoQuant run emits residual column norms or equivalent summaries. |
| WJAC | KILL | Artifactized prefilter found at least two kill diagnostics on each covered model/slice; DeepSeek/Falcon full cached coverage, Granite/Nemotron representative slice coverage. |
| LAMBDA | DEFER | Falcon prior reallocates 18.9% of total budget, but the standalone smoke gate is false because causal per-layer headroom is absent. |
| HYST | DEFER_AFTER_ROTATION | Falcon churn/local-pool gate selected margin `m=5`, but channel fallback waits until ParoQuant Falcon and BranchRot are known. |
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

The next decision gate is rotation-first:

1. Run CPU filters C1-C12 in parallel into disjoint `artifacts/<task_name>/`
   directories.
2. Prepare ParoQuant Falcon and DeepSeek smoke packets.
3. Run ParoQuant Falcon smoke first; if it passes, Falcon channel rescue drops
   in priority.
4. Run ParoQuant DeepSeek smoke; if it passes, the rotation-dominant
   four-model story strengthens.
5. Only then run DriftRot Scale/CVaR/Clip, residual correction, pairing,
   BranchRot, M-SURFACE, or Falcon channel fallbacks according to their CPU
   gates.

Do not claim ParoQuant as our method. A new method passes only if it beats
ParoQuant on held-out traces/seeds, improves ParoQuant CI/tail, or rescues a
model where ParoQuant fails.

CPU gates now favor this concrete order:

1. G0 ParoQuant Falcon smoke.
2. G1 ParoQuant DeepSeek smoke.
3. Granite clip/CVaR retune smoke if G0/G1 do not already settle the story.
4. BranchRot/HYST only if Falcon ParoQuant is weak.
5. Pairing/residual correction only after collecting exact rotation-surface or
   residual-column caches.
