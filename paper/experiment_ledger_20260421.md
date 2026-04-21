# Experiment Ledger 2026-04-21

Purpose: prevent circular exploration. This ledger separates saturated lanes,
active positive clues, and combinations worth testing next.

## Stop Repeating

| Lane | Tried | Read |
|---|---|---|
| Static grouped transport variants | grouped signature, subspace, covariance, template, template+subspace, canonical, rotational, shared-basis, broadcast, Sinkhorn OT, retrieval-spectrum, QK-template | Offline fit can improve, but held-out reasoning stays below fixed prior/C2C. Do not add another static calibration descriptor unless it changes the runtime interface. |
| Evaluator-only query gates | QK-fidelity budget, attention-fidelity gate, QK-template budget, QK-bank budget, tokenwise gates | Query-conditioning after a frozen weak map is mostly saturated. Move query information into the bridge, route atoms, or repair controller. |
| Tiny local teacher variants on same bridge | affinity, sampled attention KL, CAB, EM-KD, local interaction, likelihood-like losses | Local teacher targets did not stabilize held-out gains. Preference/output-aware targets are less destructive, but still need a different interface or stronger module. |
| Raw listwise verifier | original and shuffled-label GSM30 selector | Position/default bias and weak verifier competence make raw multiple-choice selection unreliable. Keep pairwise/pointwise/process verification only. |
| Confidence-only routing | routed projector toy confidence bank; router-stability confidence router | Target-head confidence routes to the wrong expert frequently (`0.3000` and `0.3688` toy accuracy). Confidence can be a halt or uncertainty signal, not the sole router. |
| Fixed-depth refinement | iterative/refinement-stop toys | Two steps can lower MSE; four steps over-refine and harm accuracy. Never promote fixed latent repair without stop reasons and help/harm counters. |
| Current frontier + stop heuristics | route-conditioned hub sweep over prior/feature/sticky/confidence/random/oracle routers | The present quant-error frontier and verifier stop heuristics are not reusable defaults. Oracle routing lifts the hub base, but frontier/stop still lower accuracy. Change the heuristic or training target before rerunning. |
| Route-/class-calibrated frontier heuristics in isolation | route-class patch-protect / route-class patch-frontier / route-class mode-step sweep | Calibration-aware protection only ties the current quant-error frontier and route-class frontiering still hurts. Do not keep tuning local protection scores without changing the hub/interface or the pruning rule itself. |
| Harness probes as competitor claims | LatentMAS Qwen2.5-0.5B `N=1` baseline/text-MAS probes | These verify wrapper plumbing only. Do not compare them against GSM70 method rows or cite them as fair LatentMAS results. |
| Naive full-stack composition | hub + sticky router + protected mixed-bit frontier + verifier stop toy | Individually plausible pieces can interfere. The leak-free stack scores `0.5938` versus raw pairwise `0.7344`; only oracle routing beats raw at `0.8229`. Validate interfaces before stacking. |

## Keep But Gate

| Lane | Current evidence | Promotion condition |
|---|---|---|
| Strict route selection + process repair | GSM70 `0.2000` vs target self-repair `0.1714`; SVAMP70 `0.5429` vs `0.5000`; zero observed repair harm on current slices | Needs matched token/latency/byte budgets, paired intervals, and LatentMAS/C2C/text controls on exact IDs. |
| Shared feature dictionary / crosscoder | Toy shared dictionary beats raw residual transport; feature+atom stack beats isolated components | Promote only with real route-pool feature IDs, atom recovery, and patch/quant-error calibration. |
| Route atoms / codebooks | Learned shared codebook beats raw ridge despite worse MSE | Require atom recovery, codebook usage/perplexity, dead-code rate, and task delta. |
| Byte/span tokenizer interface | Token-ID reconstruction fails while byte-span reconstruction succeeds on stress split | Run on real tokenizer pairs and include remap coverage before any downstream bridge claim. |
| Quant-error mixed-bit allocation | Recovers uniform-4-bit toy accuracy at lower achieved bpw in isolated toys, but the current hub sweep shows at most `+0.0104` frontier gain and a negative oracle frontier delta | Promote only after the protected set is made route- and class-aware; log bit histogram, outlier protection, help/harm, and false-prune. |
| Feature-routed projector bank | Toy feature routing reaches `0.9187` vs monolithic `0.8687`, close to oracle `0.9688` | Move into route-pool harness with random/confidence/oracle controls and matched bytes. |
| Hub dictionary bridge | Shared hub toy reaches `1.0000` accuracy and atom recovery with fewer adapters than pairwise; random hub fails | Promote only with real route-pool feature IDs, atom recovery, dead-feature rate, and hub-versus-pairwise scaling. |
| Sticky/feature router | Feature router reaches `0.9438`; sticky router keeps accuracy and raises perturb stability to `1.0000` | Promote only with route entropy, perturb stability, load-balance, random/confidence/oracle controls, and matched compute. |
| Verifier/process stop rules | Verifier-harm stop preserves high toy accuracy in isolated repair toys, but the current hub sweep shows stop gain `<= 0.0000` and over-refinement `0.45-0.59` | Promote only with route-/class-calibrated stop features, real repair trajectories, stop reason, halt precision, missed-help, and over-refinement telemetry. |
| LatentMAS comparator | Wrapper exists; cached baseline/text-MAS `N=1` probes run; latent-MAS direct mode is runtime-blocked after MPS fallback | Fix latent-mode runtime and run bounded matched GSM/SVAMP baseline/text-MAS/latent-MAS before head-to-head claims. |
| Stack oracle route ceiling | In the composition toy, oracle routing reaches `0.8229` versus raw pairwise `0.7344` | Use this to debug route assignment and stop policy; do not cite as a method row. |
| Route-/class-calibrated frontier | On the held-out hub sweep, route-class patch-protect ties quant-error (`0.6354` prior, `0.8125` oracle) while route-class frontier remains negative | Promote only if a redesigned pruning rule beats both quant-error and all-low-bit hub baselines; otherwise move to multi-way canonical hubs or tokenizer/interface simplification. |
| Multi-way canonical hub | On the held-out-family toy, `multiway_gpa_canonical` is the best non-oracle/shared-basis method at `1` shot/class (`0.1327` MSE vs few-shot `0.1463`) and crushes global seen-family ridge, but direct held-out family fitting regains the MSE lead by `2+` shots/class | Promote only as a low-shot initializer or regularizer unless it also wins once moderate paired data exists; the next real test should be GPA-initialized shared hub plus sparse dictionary on a held-out-family route split. |
| GPA sparse dictionary hub | On the held-out-family toy, `multiway_gpa_sparse_dictionary` is now the strongest low-shot shared-basis method at `1` shot/class (`0.1171` MSE vs few-shot `0.1825` and canonical `0.2355`), but direct held-out family fitting regains the MSE lead by `2+` shots/class and the verifier-gated repair step stays at `0.0000` accept/help | Promote only as a low-shot shared-basis backbone if the gain survives tokenizer/interface shifts and the dictionary becomes interpretable enough to justify the story; keep repair as a blocker until it actually fires and helps. |
| Real tokenizer pair sweep | On GSM30 prompts, the exact Qwen2.5->Qwen3 pair is effectively tokenizer-identical (`shared decoded = 1.0000`, `boundary F1 = 1.0000`), while Qwen->Mistral and Qwen->Phi3 show real surface and boundary mismatch (`shared decoded ~0.80`, `boundary F1 0.93-0.95`) | Do not treat tokenizer mismatch as the likely blocker for the current same-pair method; use byte/span/vocab controls as a robustness and cross-family lane instead. |

## Next Stack To Test

The next positive-method stack should be additive only after each component has
an interaction control:

1. GPA-initialized shared hub plus sparse shared dictionary instead of plain
   pairwise bridges.
2. Byte/span/vocabulary interface controls instead of assuming tokenizer
   compatibility.
3. Sticky feature-routed projector bank instead of one monolithic bridge or
   confidence-only routing.
4. A genuinely different protected-pruning rule instead of the current quant-error
   or route-class frontier heuristics.
5. A repair step only after it shows nonzero accept/help on held-out toys.
6. Matched comparison against target-alone, target self-repair, text-to-text,
   C2C, LatentMAS, and same-model compression controls.

## Required Telemetry

Every promoted run should emit `example_id`, `method`, `source_model`,
`target_model`, `dataset`, `seed`, `correct`, `baseline_correct`, `route_help`,
`route_harm`, `bytes`, `tokens_in`, `tokens_out`, `latency_ms`, `repair_calls`,
`projector_id`, `gate_entropy`, `route_stability`, `route_atom_ids`,
`atom_recovery`, `bit_allocation`, `stop_reason`, `halt_confidence`,
`over_refinement_flag`, `parse_failure`, and compact trace hashes.

## Current Evidence Ladder

`paper/ablation_evidence_ladder_20260421.md` is the current anti-loop summary
for stack decisions. It keeps toy-positive components, controls, and blockers
in one table so the next real route-pool run must justify each included
component against its logged promotion gate.

`results/query_pool_toy_20260421/hub_router_frontier_sweep_20260421.md` is the
current interface sweep for that stack. Oracle routing lifts the hub base
above raw pairwise, but the current frontier and stop heuristics still hurt,
so both route assignment and later heuristics need redesign.

`results/query_pool_toy_20260421/route_class_frontier_sweep_20260421.md` is
the current follow-up on that redesign attempt. The read is negative: smarter
local protection scores are not enough by themselves, so the next step should
change the shared hub/interface or pruning rule rather than keep tuning the
same frontier family.

`results/query_pool_toy_20260421/multiway_canonical_hub_20260421.md` is the
current interface-first follow-up. The read is mixed but useful: multi-way
canonicalization helps at the true `1`-shot held-out-family point, where
`multiway_gpa_canonical` reaches the lowest non-oracle MSE (`0.1327` vs
`0.1463` for direct few-shot fitting), but it loses back to direct family
fitting as soon as `2+` paired shots/class are available. Treat this as
evidence for low-shot hub initialization, not as a universal replacement for
family-specific fitting.

`results/query_pool_toy_20260421/gpa_sparse_dictionary_hub_20260421.md` is the
current low-shot shared-basis follow-up. The read is sharper: the sparse
dictionary beats both direct few-shot fitting and canonical-only GPA at the
true `1`-shot point, but the gain disappears once moderate paired data exists
and the verifier-gated repair branch still accepts nothing.

`results/query_pool_toy_20260421/real_tokenizer_interface_pair_sweep_20260421.md`
is the current real-tokenizer interface readout. It shows that tokenizer
mismatch is not the blocker for the exact Qwen2.5->Qwen3 pair used in the main
same-pair runs, but it becomes real immediately on broader cross-family pairs
such as Qwen->Mistral and Qwen->Phi3.
