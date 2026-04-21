# Master Comparison Table (2026-04-19)

This table is the current paper-facing snapshot for the main Qwen same-pair setting.
It is intentionally narrow: exact held-out settings that have direct baseline
comparisons and a tracked readout in `latent_bridge/current_readout_20260418.md`.

## Qwen2.5-0.5B -> Qwen3-0.6B, GSM8K

### GSM30 Stochastic-Route Smoke

| Split | Method | Accuracy | Avg bytes | Notes |
| --- | --- | ---: | ---: | --- |
| `gsm8k_gate_search_30` | target-alone | `0.0667` | `0` | current stochastic-reranker baseline |
| `gsm8k_gate_search_30` | strict stochastic selector | `0.1667` | `-` | best current non-oracle selector over three random route/value candidates |
| `gsm8k_gate_search_30` | target-model listwise verifier | `0.0667` | `-` | selected target-alone on all 30 examples; useful negative selector control |
| `gsm8k_gate_search_30` | shuffled-label target-model verifier | `0.1000` | `-` | target selection drops to `0.2000`, but `Choice A` rate is `0.9667`; position-bias diagnostic, not a solved selector |
| `gsm8k_gate_search_30` | pairwise verifier tournament | `0.0667` | `-` | target selection drops to `0.2000` and seed selection rises to `0.8000`, but selected seeds are not reliably correct |
| `gsm8k_gate_search_30` eval-half | confidence-gated route expansion | `0.2000` | `-` | calibrated on first 15, evaluated on last 15; beats eval-half target-alone `0.0667` and random matched `0.1333`, but ties fixed route budgets |
| `gsm8k_gate_search_30` eval-half | calibrated feature selector | `0.0000` | `-` | calibration-split feature weights overfit and always select seeds; negative selector control |
| `gsm8k_gate_search_30` | strict selector + process repair | `0.2333` | `-` | repairs selected route with target model; pre-repair `0.1667`, help `0.0667`, harm `0.0000`, oracle `0.3000` |
| `gsm8k_gate_search_30` | target-or-seed oracle | `0.3000` | `-` | candidate-quality ceiling, label-leaking |
| `gsm8k_gate_search_30` | C2C native smoke | `0.0667` | `-` | exact Qwen pair through published C2C artifact |
| `gsm8k_gate_search_30` | KVPress none | `0.0667` | `-` | same-model compression control |
| `gsm8k_gate_search_30` | KVPress expected-attention `0.5` | `0.0667` | `-` | same-model compression control |

| Split | Method | Accuracy | Avg bytes | Notes |
| --- | --- | ---: | ---: | --- |
| `gsm8k_eval_70` | `target-alone` | `0.0429` | `0` | no source communication |
| `gsm8k_eval_70` | target candidate, shared-chat repair manifest | `0.0571` | `0` | target-side baseline inside the held-out process-repair route pool |
| `gsm8k_eval_70` | `text-to-text` | `0.1000` | `-` | text communication baseline |
| `gsm8k_eval_70` | fixed head prior | `0.0857` | `151,163.7` | best current internal same-pair branch |
| `gsm8k_eval_70` | shuffled fixed prior | `0.0429` | `151,163.7` | query-blind null |
| `gsm8k_eval_70` | grouped signature transport | `0.0429` | `147,812.6` | best current transport-only branch |
| `gsm8k_eval_70` | grouped subspace transport | `0.0429` | `147,812.6` | tied grouped signature transport |
| `gsm8k_eval_70` | grouped subspace transport + rank-4 residual | `0.0571` | `145,508.8` | best current transport-plus-correction branch |
| `gsm8k_eval_70` | grouped subspace transport + rank-4 residual + bridge-ridge correction | `0.0429` | `295,614.9` | first bridge branch that survives held-out slices, but still below the live internal bars |
| `gsm8k_eval_70` | grouped subspace transport + rank-4 residual + QK-fidelity budget | `0.0429` | `157,989.2` | query-conditioned per-head budget on top of the best transport-plus-correction checkpoint |
| `gsm8k_eval_70` | grouped covariance transport + rank-4 residual | `0.0143` | `146,417.7` | covariance-aware transport-plus-correction failure |
| `gsm8k_eval_70` | grouped template transport + rank-4 residual | `0.0429` | `150,038.8` | attention-template transport-plus-correction probe (`64`-prompt calibration slice) |
| `gsm8k_eval_70` | grouped template-subspace transport + rank-4 residual | `0.0143` | `149,129.8` | stacked grouped-penalty failure (`64`-prompt calibration slice) |
| `gsm8k_eval_70` | broadcast template transport + rank-4 residual | `0.0000` | `149,129.8` | rectangular `2 -> 8` head transport probe (`64`-prompt calibration slice) |
| `gsm8k_eval_70` | broadcast template OT transport + rank-4 residual | `0.0000` | `149,129.8` | rectangular Sinkhorn-style `2 -> 8` head transport probe (`64`-prompt calibration slice) |
| `gsm8k_eval_70` | broadcast peak-template OT transport + rank-4 residual | `0.0143` | `149,129.8` | rectangular Sinkhorn-style `2 -> 8` transport using peak-location templates (`64`-prompt calibration slice) |
| `gsm8k_eval_70` | broadcast retrieval-spectrum OT transport + rank-4 residual | `0.0143` | `625,463.7` | rectangular Sinkhorn-style `2 -> 8` transport using retrieval-weighted key spectra under matched sparse `K-only` evaluation (`64`-prompt calibration slice) |
| `gsm8k_eval_70` | broadcast QK-template OT transport + rank-4 residual | `0.0143` | `625,463.7` | rectangular Sinkhorn-style `2 -> 8` transport using last-token QK logit templates under matched sparse `K-only` evaluation (`64`-prompt calibration slice) |
| `gsm8k_eval_70` | grouped canonical transport | `0.0286` | `149,496.2` | low-rank canonical basis shortcut |
| `gsm8k_eval_70` | selected route, no repair | `0.1286` | `-` | held-out strict selector control on the same route pool; no target-side repair |
| `gsm8k_eval_70` | target self-repair | `0.1714` | `-` | same repair prompt and target decode budget applied to target-alone candidate |
| `gsm8k_eval_70` | strict selector + process repair | `0.2000` | `-` | held-out repair over stochastic route-pool salts `0/1/2`; pre-repair `0.1286`, help `0.0714`, harm `0.0000`, oracle `0.1571`; adds target-side repair compute |
| `gsm8k_eval_70` | `C2C` | `0.1286` | `-` | strongest external baseline so far |
| `gsm8k_eval_70` | lifted `KVComm` replay | `0.0000` | `-` | compatibility-lifted heterogeneous replay |

| Split | Method | Accuracy | Avg bytes | Notes |
| --- | --- | ---: | ---: | --- |
| `gsm8k_100` | `target-alone` | `0.0400` | `0` | no source communication |
| `gsm8k_100` | `text-to-text` | `0.1000` | `-` | text communication baseline |
| `gsm8k_100` | fixed head prior | `0.0700` | `-` | best current internal branch on larger slice |
| `gsm8k_100` | shuffled fixed prior | `0.0400` | `-` | matched null |
| `gsm8k_100` | `C2C` | `0.1100` | `-` | strongest external baseline so far |

## Qwen2.5-0.5B -> Qwen3-0.6B, SVAMP

| Split | Method | Accuracy | Avg bytes | Notes |
| --- | --- | ---: | ---: | --- |
| `svamp_eval_70` | `target-alone` | `0.0714` | `0` | no source communication |
| `svamp_eval_70` | target candidate, shared-chat repair manifest | `0.3000` | `0` | target-side baseline inside the held-out process-repair route pool |
| `svamp_eval_70` | `text-to-text` | `0.4143` | `-` | text communication baseline |
| `svamp_eval_70` | grouped CCA fixed prior | `0.1714` | `-` | best current internal SVAMP branch |
| `svamp_eval_70` | grouped CCA shuffled null | `0.1286` | `-` | matched query-blind null |
| `svamp_eval_70` | selected route, no repair | `0.3571` | `-` | held-out strict selector control on the same route pool; no target-side repair |
| `svamp_eval_70` | target self-repair | `0.5000` | `-` | same repair prompt and target decode budget applied to target-alone candidate |
| `svamp_eval_70` | strict selector + process repair | `0.5429` | `-` | held-out repair over stochastic route-pool salts `0/1/2`; pre-repair `0.3571`, help `0.1857`, harm `0.0000`, oracle `0.5286`; adds target-side repair compute |
| `svamp_eval_70` | `C2C` | `0.4429` | `-` | strongest external baseline so far |

## Current Read

- Best frozen latent-transport-only GSM branch is still the fixed head-prior
  branch, not transport-first.
- `C2C` remains the strongest no-extra-repair direct external comparator, but
  held-out strict selector + process repair now beats its raw accuracy on
  GSM70 and SVAMP70. This is a method candidate, not yet a fair efficiency
  claim, because it adds target-side repair generation and needs matched
  compute/token/byte accounting.
- On the newer GSM30 stochastic-route smoke, the strict selector is the first
  non-oracle internal method to beat target-alone and same-slice C2C/KVPress
  controls (`0.1667` vs `0.0667`), but it is not yet a held-out paper result.
- The naive target-model listwise verifier is a negative control: it chose the
  target candidate on every GSM30 example, so future verifier work needs
  calibration, position randomization, or process-level checks.
- Position randomization confirms the verifier failure is not simply target
  preference: shuffled labels improve only to `0.1000`, while the model chooses
  option `A` on `29/30` examples. Future selector work should avoid raw
  listwise letters and use calibrated pairwise/pointwise checks, answer repair,
  or confidence-gated expansion.
- Pairwise tournament verification removes much of the target/default collapse
  but still ties target-alone at `0.0667`; this points to verifier competence,
  not only position bias, as the next blocker.
- Confidence-gated route expansion is directionally useful on the GSM30
  eval-half (`0.2000` vs target `0.0667` and random matched `0.1333`), but it
  does not beat fixed route-budget controls yet. Treat it as a compute policy
  primitive, not a paper method.
- A transparent calibrated feature selector over format/numeric/completion/
  agreement features collapses on the eval-half (`0.0000`), so simple candidate
  metadata is not enough; the next selector needs process verification or
  repair.
- Strict selector plus process repair is now the strongest real-model method
  lane: GSM30 `0.2333`, held-out GSM70 `0.2000`, and held-out SVAMP70
  `0.5429`, with observed repair harm `0.0000` on all three current slices.
  The next paper risk is fair budget matching, not whether the repair lane has
  any held-out signal.
- Raw stochastic route pools are not stable enough to claim alone: GSM70 salts
  land at `0.0857`, `0.0286`, and `0.0571`; SVAMP70 salts land at `0.3000`,
  `0.3000`, and `0.2571`. The positive result comes from target-side process
  repair on top of candidate generation.
- Same-prompt control arms are now logged on the full frozen route pools.
  Selected-route repair still beats target self-repair on both splits, but by
  modest margins: GSM70 `0.2000` vs `0.1714`, SVAMP70 `0.5429` vs `0.5000`.
  This supports a route-specific gain while keeping target-side self-correction
  as the main fairness control. Bootstrap attribution intervals are now logged
  in `results/process_repair_holdout_20260421/process_repair_attribution_20260421.md`;
  they show GSM70 route-specific delta `+0.0286 [-0.0429, 0.1143]` and SVAMP70
  delta `+0.0429 [0.0000, 0.1000]`.
- The current competitor readout is consolidated in
  `paper/competitor_benchmark_readout_20260421.md`. Direct peers and same-model
  compression controls must stay separated: `C2C` is the direct semantic bar,
  while `KVPress`/`KVzip`/`Quest`/`H2O`/`SnapKV` are cache controls.
- Test-before-repair replay over the held-out repair telemetry shows an
  efficiency path: GSM70 `format_gate` preserves `0.2000` accuracy while saving
  `27.1%` of selected-route repair calls; SVAMP70 `format_delta_gate` preserves
  `0.5429` while saving `15.7%`. This reduces target-side repair budget but
  does not yet increase accuracy.
- Repair-gate feature audit explains why the cheap gate works: format score is
  the strongest safe-skip predictor on GSM70 (selected-correct AUROC `0.8570`)
  and SVAMP70 (`0.7880`), while numeric consistency/count features are too weak
  to trust as standalone process checks. The next selector should add
  process-step or generated-test telemetry, not more numeric metadata.
- Process-gate text audit improves the efficiency primitive: GSM70
  `format_plus_process_score` preserves `0.2000` while saving `32.9%` of repair
  calls, and SVAMP70 `process_completeness_score` preserves `0.5429` while
  saving `22.9%`. This is stronger than the cheap format-only replay, but still
  not an accuracy gain over repair-all.
- The updated test-before-repair replay now includes paired bootstrap accuracy
  CIs and extra repair-cost proxies. GSM70 `format_plus_process_gate` has
  accuracy CI `[0.1143, 0.3000]` and mean extra repair prompt chars `676.9`
  versus `738.0` for format-only; SVAMP70 `process_gate` has CI
  `[0.4286, 0.6571]` and mean extra repair prompt chars `661.1` versus `713.9`
  for format-delta.
- The shared-feature dictionary toy gives the strongest new additive design
  clue: raw residual transport reaches `0.3646`, separate dictionaries reach
  `0.4167`, a shared dictionary/crosscoder reaches `0.5417`, and the oracle is
  `0.5938`. Treat this as a diagnostic-to-promote, not yet a real-model claim.
- The route-atom codebook toy adds a second additive clue: learned shared
  codebooks improve task accuracy (`0.8438` vs raw ridge `0.7812`) despite much
  worse MSE, and protected outlier atoms keep the accuracy gain while reducing
  reconstruction damage. Future bridge tables must report atom/feature recovery
  alongside MSE.
- The feature+atom stack toy is stronger: raw ridge reaches `0.6458`, shared
  feature only drops to `0.4167`, route atom only reaches `0.5833`, but the
  stacked feature+atom interface reaches `0.8542`. This supports testing
  interaction stacks rather than rejecting components from isolated ablations.
- A deterministic LLMLingua-style prompt-compression control is now available:
  it preserves all numeric spans on GSM70/SVAMP70 and saves an estimated
  `123.5` / `71.5` bytes on average, but it is a budget/preservation diagnostic
  only and has no downstream accuracy claim.
- The modern architecture sweep adds four controlled ablation lanes to the
  roadmap: selective-SSM vs attention transport, writable/test-time memory vs
  sliding cache, adaptive compute vs fixed bridge depth, and MQA/GQA-style KV
  sharing. These should be budget-matched before they appear in the main table.
- The ICLR evaluation contract is now explicit: every headline LatentWire row
  needs paired bootstrap CIs, sample-level correctness, token/byte/latency
  ledgers, repair/verifier call counts, frozen prompt manifests, and exact
  comparator budgets.
- The competitor gap plan now adds `LatentMAS` and `LLMLingua` to the watchlist:
  direct communication baselines should be normalized against `C2C`, `KVComm`,
  and `LatentMAS`, while prompt-compression claims need a separate LLMLingua
  control with fixed prompt and answer budgets.
- The verifier/agent-training sweep points to the next route-quality amplifier:
  scalar route scoring should be compared against step-localization,
  critique-plus-repair, pairwise/tournament verification, and verifier-guided
  frontier pruning under matched token budgets.
- Transport-only branches improved from `grouped_transport` to `grouped_signature_transport`, but they plateaued below the fixed-prior branch and well below `C2C`.
- The first transport-plus-correction branch improves over the pure transport family, but it still does not catch the fixed-prior branch or `C2C`.
- The first bridge-style correction branch that actually survives beyond tiny smokes is `bridge_ridge`, but it still trails the grouped-subspace-plus-rank4 checkpoint and the fixed-prior branch.
- A genuinely query-conditioned QK-fidelity budget on top of that same best transport-plus-correction checkpoint recovers only to `0.0429`, so live query-conditioning alone is still not enough.
- A covariance-aware version of that same transport-plus-correction branch falls back to `0.0143`, so covariance geometry is not the next shortcut here.
- A calibration-time attention-template version of that same branch lands at `0.0429`, so light behavior matching inside the current grouped solver is also not enough.
- A hybrid template-plus-subspace version falls further to `0.0143`, so stacking the two best grouped penalties is not the right fix either.
- A finer rectangular `2 -> 8` broadcast-template transport branch falls all the way to `0.0000`, so the grouped family was not failing only because of coarse grouped transport.
- A richer rectangular Sinkhorn-style OT plan in that same attention-template space still lands at `0.0000`, so the remaining issue is not just transport granularity or many-to-many mass assignment.
- Replacing mean attention templates with simple peak-location templates lifts that OT branch to `0.0143`, so representation matters a bit, but the gain is still far below the fixed prior and `C2C`.
- Replacing the retrieval-spectrum descriptor with simple last-token QK logit templates does not move that frontier at all: it ties the retrieval-spectrum OT branch at `0.0143` while staying far less byte-efficient than the live sparse branches.
- The paper can now be framed around a positive repair-and-routing method
  candidate, with blocker/mechanism evidence explaining why frozen transport
  alone saturates. The remaining bar is strengthening the route-specific margin
  under matched compute against text-to-text, `C2C`, target self-repair, and
  test-before-repair controls.
