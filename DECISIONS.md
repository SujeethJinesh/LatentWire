# Positive-Method Sprint Decisions

Last updated: 2026-05-29T04:05Z

## Current Framing

The final story for this submission cycle is mechanism/regime. Static and
hard-switch channel protection fail under drift; budgeted EMA succeeds only in
the Nemotron regime; ParoQuant-style rotation is the strongest prior-work
baseline/remedy class across the current model set. ParoQuant is not our method.
The paper contribution is long-decode drift measurement, matched-control
failure taxonomy, regime guidance, and a systems-cost envelope for future
residual selectors.

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
| Granite clip/CVaR retune | PASS_GRANITE_ONLY_AMBIG_DEEPSEEK_FALCON | Tight clip `[0.5, 2.0]` fixed the inspected Granite tail, passed held-out positive-gap prompts 1/2/5, and completed all eight recoverable Granite traces. Granite median recovery improves 0.754 -> 0.922 and worst trace improves -26.37 -> 0.601. On DeepSeek, the same clip improves CI lower (-0.246 -> 0.265) but regresses median 0.756 -> 0.518 and worst trace -3.056 -> -3.524. On Falcon, median is essentially flat (0.381 -> 0.390), but CI lower drops 0.0645 -> 0.0019 and worst trace regresses -1.323 -> -1.610. The safe claim is Granite-specific tail control, not a universal retune. |
| Drift-aware pairing | NEEDS_ACTIVATION_CACHE | Pairings differ from current ParoQuant on block-output proxies, but exact rotation-surface caches are required before GPU. |
| Residual correction | KILL_TOP8X32_MOE_RESIDUAL_DIAGNOSTIC | Residual norms, activation EMA, candidate pool, and DeltaW columns were produced. A fixed Granite prompt-4 diagnostic scored -12.29 recovery versus tight ParoQuant reference 5.94, so this MoE top-8x32 candidate is not promoted. Only reopen with a bounded or KLLOOK-gated design. |
| WJAC | KILL | Artifactized prefilter found at least two kill diagnostics on each covered model/slice; DeepSeek/Falcon full cached coverage, Granite/Nemotron representative slice coverage. |
| LAMBDA | DEFER | Falcon prior reallocates 18.9% of total budget, but the standalone smoke gate is false because causal per-layer headroom is absent. |
| HYST | DEFER_AFTER_FALCON_ROTATION_PASS | Falcon churn/local-pool gate selected margin `m=5`, but ParoQuant now rescues Falcon enough that channel fallback is lower priority. |
| RISKGUARD | DEFER | Best cached trigger is in-sample only; leave-one-trace-out CI lower bound collapses to 0 and CVaR remains negative. |
| TRACE-ROUTER | WEAK_OFFLINE_ONLY | Granite/DeepSeek show positive tiny-n CV gain, but this needs a preregistered larger frozen slice before evidence claims. Falcon remains not routable. |
| M-SURFACE | KILL_OR_DEFER_SURFACE_NO_LOWER_DRIFT | Granite two-trace hook sanity measured post-block mean strict leaving 0.578, Mamba `out_proj` input 0.887, and attention `o_proj` input 0.780. The cheap internal hook surfaces drift more than post-block, so do not promote this surface family or run Falcon M-SURFACE without a new local-tensor hypothesis. |
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
3. M-SURFACE cheap module hooks are killed/deferred; only reopen with SSM B/C
   internals or KLLOOK-backed placement evidence.
4. Falcon/DeepSeek channel fallbacks only if a later rotation-specific gate
   exposes a weakness requiring architecture-local rescue.

## DriftRot+ Gate Update

Timestamp: 2026-05-29T02:34Z.

The rotated-basis residual-correction sprint has a stricter CPU gate:

- `artifacts/k_res/`: `KILL_CURRENT_TOP8X32_PROXY_NO_GPU`. The existing
  residual headroom is not actionable for the current proxy. Relative residual
  energy is about 1%, top-128 residual concentration is only a few percent per
  tensor, and the valid top-8x32 MoE correction worsened Granite tail recovery
  (-12.29) compared with tight ParoQuant (5.94). Do not run P1/P2 GPU for this
  proxy.
- `artifacts/k_churn/`: `INCOMPLETE_NO_POSITIONAL_ROTATED_SET_CACHE`. Dynamic
  residual tracking/hysteresis/phasing needs a position-resolved residual-set
  cache before GPU.
- `artifacts/k1_cov/`: `P8_NOT_PROMOTED_NO_TRUE_COVARIANCE`. Clip/config
  retuning is closed as a universal method; online scale-refresh needs true
  covariance evidence.
- `artifacts/msurface/`: cheap projection-input surfaces are a negative screen,
  but SSM input and B/C generation are unmeasured.
- `artifacts/k_branch/`: Falcon BranchRot remains deferred without branch-local
  drift evidence.

Decision: no new GPU job is authorized from this residual family yet. The next
positive-method work must either define a bounded/KLLOOK-gated residual selector
with better CPU evidence, or shift to a different surface/branch gate. ParoQuant
remains a baseline, not our method.

## Final Bounded Gate After K-RES Kill

Timestamp: 2026-05-29T02:57Z.

| Branch | Decision | Reason |
|---|---|---|
| K-RES current top-8x32 proxy | `VALID_IMPLEMENTATION_KEEP_K_RES_PROXY_KILLED` | The audit found the fixed run used the tight ParoQuant reference, same Granite tail trace/window, post-ParoQuant residual columns, valid shape/index sidecars, and exactly top-8 modules x 32 columns. It worsened recovery (-12.29) versus tight ParoQuant (5.94). |
| Residual KLLOOK oracle | `NO_RUN_EXCEEDS_ONE_HOUR_RUNNER_GATE` | The available M-KLLOOK runner is original-basis channel protection, not rotated-basis residual correction. A correct residual KLLOOK runner is not a <=1 hour build, and the K-RES audit already supports the kill. |
| M-SURFACE | `KILL_MSURFACE_NO_LOWER_INTERNAL_SURFACE` | Complete Granite internals run measured SSM input, B/C generation, Mamba out-proj input, attention o-proj input, and post-block output. No internal surface is below 0.30-0.40 or at least 0.15 below post-block. |
| Falcon BranchRot | `DEFERRED_TO_FUTURE_WORK_NOT_RUN_THIS_PASS` | BranchRot was deferred unless M-SURFACE promoted. M-SURFACE did not promote, and only post-block Falcon activations are cached. |

Decision: stop the current positive-method search and move the paper to the
mechanism/regime path. The safe paper claim is that channel identity drifts and
channel-set remedies repeatedly fail or remain unpromoted, while rotation is the
strongest baseline. ParoQuant is prior work and must remain labeled as such.

## Final Mechanism/Regime Submission Framing

Timestamp: 2026-05-29T04:05Z.

Decision: finalize this submission cycle as a mechanism/regime paper. Do not
restart positive-method search for this draft.

Rationale:

- The core measurement result is mature: protected-channel sets drift
  substantially under long reasoning decode across four architectures.
- The matched-control method table is the central contribution: static maps,
  hard switching, binned scales, top-1% EMA, prediction, WJAC, K-RES, tight
  clip, and M-SURFACE either fail, remain ambiguous, or only help in a narrow
  regime.
- ParoQuant-style rotation is the strongest baseline/remedy class and must be
  framed as prior work, not as our method.
- K-RES is elevated to a mechanism finding: valid post-rotation residual
  columns selected by residual energy worsen the Granite tail gate, so residual
  energy is not a sufficient selector.
- The systems contribution is an analytical cost envelope for a future
  residual selector, not a kernel or deployed positive method.

Recommended submission framing: "Long-reasoning W4A16 protected-channel
identity drifts; direct channel-set remedies mostly fail under controls;
rotation removes much of the basis dependence; a descriptive calibration
protocol tells practitioners when to use rotation, when budgeted EMA may help,
and when to reject channel-set protection."
