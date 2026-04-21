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
| Byte/span tokenizer interface | Token-ID reconstruction fails while byte-span reconstruction succeeds on stress split, and under strong held-out-family interface corruption the byte/span remap is now the best composed shared-basis variant at `1-2` shots/class (`0.0566`, `0.0570` MSE) | Run on real tokenizer pairs and held-out route pools with remap coverage and interface-noise telemetry before any downstream bridge claim. |
| Byte sidecar interface | Under the same strong held-out-family interface corruption, a tokenizer-agnostic byte sidecar on top of quotient+GPA+sparse-dictionary becomes the best shared-basis variant at `1-2` shots/class (`0.0392`, `0.0394` MSE), beating both the remap-only variant and the oracle-interface latent-only variant, but direct few-shot + remap still wins by `4` shots/class (`0.0238`) | Promote only as the next interface component on top of the current low-shot lane; require survival on real cross-family pairs and a frozen benchmark smoke before promoting it into the main paper story. |
| Quant-error mixed-bit allocation | Recovers uniform-4-bit toy accuracy at lower achieved bpw in isolated toys, but the current hub sweep shows at most `+0.0104` frontier gain and a negative oracle frontier delta | Promote only after the protected set is made route- and class-aware; log bit histogram, outlier protection, help/harm, and false-prune. |
| Feature-routed projector bank | Toy feature routing reaches `0.9187` vs monolithic `0.8687`, close to oracle `0.9688` | Move into route-pool harness with random/confidence/oracle controls and matched bytes. |
| Hub dictionary bridge | Shared hub toy reaches `1.0000` accuracy and atom recovery with fewer adapters than pairwise; random hub fails | Promote only with real route-pool feature IDs, atom recovery, dead-feature rate, and hub-versus-pairwise scaling. |
| Sticky/feature router | Feature router reaches `0.9438`; sticky router keeps accuracy and raises perturb stability to `1.0000` | Promote only with route entropy, perturb stability, load-balance, random/confidence/oracle controls, and matched compute. |
| Verifier/process stop rules | Verifier-harm stop preserves high toy accuracy in isolated repair toys, but the current hub sweep shows stop gain `<= 0.0000` and over-refinement `0.45-0.59` | Promote only with route-/class-calibrated stop features, real repair trajectories, stop reason, halt precision, missed-help, and over-refinement telemetry. |
| LatentMAS comparator | Wrapper exists; cached baseline/text-MAS `N=1` probes run; latent-MAS direct mode is runtime-blocked after MPS fallback | Fix latent-mode runtime and run bounded matched GSM/SVAMP baseline/text-MAS/latent-MAS before head-to-head claims. |
| Stack oracle route ceiling | In the composition toy, oracle routing reaches `0.8229` versus raw pairwise `0.7344` | Use this to debug route assignment and stop policy; do not cite as a method row. |
| Route-/class-calibrated frontier | On the held-out hub sweep, route-class patch-protect ties quant-error (`0.6354` prior, `0.8125` oracle) while route-class frontier remains negative | Promote only if a redesigned pruning rule beats both quant-error and all-low-bit hub baselines; otherwise move to multi-way canonical hubs or tokenizer/interface simplification. |
| Multi-way canonical hub | On the held-out-family toy, `multiway_gpa_canonical` is the best non-oracle/shared-basis method at `1` shot/class (`0.1327` MSE vs few-shot `0.1463`) and crushes global seen-family ridge, but direct held-out family fitting regains the MSE lead by `2+` shots/class | Promote only as a low-shot initializer or regularizer unless it also wins once moderate paired data exists; the next real test should be GPA-initialized shared hub plus sparse dictionary on a held-out-family route split. |
| Gauge-fix / quotient bridge | On the held-out-family symmetry toy, `quotient_match_after_fix` is now the best non-oracle method at `1` shot/class (`0.0796` MSE vs few-shot `0.0985`, no-match gauge-fix `0.1665`, global seen-family ridge `1.7099`) and recovers the true head correspondence (`head_match_accuracy = 1.0000`), but direct held-out-family fitting regains the MSE lead by `2+` shots/class | Promote only as a low-shot gauge/canonical initializer before the shared dictionary lane; require the gain to survive composition with sparse dictionaries and real held-out route pools before elevating it into the main method story. |
| Quotient + GPA sparse dictionary | On the composed held-out-family toy, `quotient_gpa_sparse_dictionary` is best non-oracle at both `1` and `2` shots/class (`0.0568` and `0.0576` MSE), beats direct held-out-family few-shot fitting (`0.1003` and `0.0638`), preserves exact head recovery (`head_match_accuracy = 1.0000`), and under strong interface stress the byte/span-remap variant is best shared-basis at `1-2` shots/class (`0.0566`, `0.0570`) before direct few-shot + remap retakes the lead by `4` shots/class (`0.0238` vs `0.0555`) | Promote as the current best low-shot compositional lane; next require survival on real held-out route pools and matched benchmark contracts before writing a positive-method claim. |
| GPA sparse dictionary hub | On the held-out-family toy, `multiway_gpa_sparse_dictionary` is now the strongest low-shot shared-basis method at `1` shot/class (`0.1171` MSE vs few-shot `0.1825` and canonical `0.2355`), but direct held-out family fitting regains the MSE lead by `2+` shots/class and the verifier-gated repair step stays at `0.0000` accept/help | Promote only as a low-shot shared-basis backbone if the gain survives tokenizer/interface shifts and the dictionary becomes interpretable enough to justify the story; keep repair as a blocker until it actually fires and helps. |
| Real tokenizer pair sweep | On GSM30 prompts, the exact Qwen2.5->Qwen3 pair is effectively tokenizer-identical (`shared decoded = 1.0000`, `boundary F1 = 1.0000`), while Qwen->Mistral and Qwen->Phi3 show real surface and boundary mismatch (`shared decoded ~0.80`, `boundary F1 0.93-0.95`) | Do not treat tokenizer mismatch as the likely blocker for the current same-pair method; use byte/span/vocab controls as a robustness and cross-family lane instead. |

## Next Stack To Test

The next positive-method stack should be additive only after each component has
an interaction control:

1. Quotient-aware matching plus GPA plus a sparse shared dictionary as the
   default low-shot shared-basis lane.
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

`results/query_pool_toy_20260421/gauge_fix_quotient_bridge_20260421.md` is the
current symmetry-aware follow-up. The read is narrower but stronger: once the
head-matching score is made gauge-invariant, quotient-aware matching becomes
the best non-oracle method at the true `1`-shot point (`0.0796` MSE vs
`0.0985` for direct few-shot and `0.1665` for no-match gauge-fix) and
recovers the true head assignment (`head_match_accuracy = 1.0000`), but direct
held-out-family fitting still retakes the MSE lead by `2+` shots/class. Treat
this as evidence for a low-shot gauge/canonical initializer, not as a full
method by itself.

`results/query_pool_toy_20260421/quotient_gpa_sparse_dictionary_20260421.md`
is the current compositional follow-up. The read is the strongest low-shot
shared-basis result so far: quotient-aware matching, GPA canonicalization, and
a sparse shared dictionary now compose positively, giving the best non-oracle
MSE at both `1` and `2` shots/class (`0.0568`, `0.0576`) before direct
held-out-family fitting retakes the lead at `4` shots/class. Treat this as the
current best method-discovery lane, but keep the repair branch blocked because
accepted help is still `0.0000`.

`results/query_pool_toy_20260421/quotient_gpa_sparse_dictionary_interface_stress_20260421.md`
is the current robustness follow-up for that lane. Under strong tokenizer-like
interface corruption, the byte/span-remap variant becomes the best shared-basis
version at `1` and `2` shots/class (`0.0566`, `0.0570` MSE), slightly beating
the token-id interface and matching the oracle-interface ceiling closely, while
direct held-out-family fitting plus the same remap retakes the lead once `4`
paired shots/class are available. Treat this as evidence that the composed
low-shot lane survives interface stress and can benefit from explicit
byte/span controls, not as evidence that remap alone rescues the method.

`results/query_pool_toy_20260421/quotient_gpa_sparse_dictionary_byte_sidecar_20260421.md`
is the current side-channel follow-up for that same lane. Under the same
strong interface corruption, adding a tokenizer-agnostic byte sidecar to the
quotient+GPA+sparse-dictionary bridge lowers MSE sharply at `1` and `2`
shots/class (`0.0392`, `0.0394`), beating both the remap-only branch and the
oracle-interface latent-only branch, but direct held-out-family few-shot
fitting plus remap still wins by `4` shots/class (`0.0238`). Treat this as
the strongest current interface-side additive clue, not as proof that the
paper is benchmark-ready.

`results/query_pool_toy_20260421/quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_20260421.md`
is the current sequence-aware follow-up for that same lane. Under the same
strong interface corruption, the sequence-aligned sidecar branch lowers MSE
again at `1` and `2` shots/class (`0.0360`, `0.0362`), beating the plain byte
sidecar and remap-only shared-basis variants while preserving exact head
matching. It also remains the best shared-basis variant at `4` and `8`
shots/class, even though direct held-out-family fitting plus remap still wins
overall once paired data is abundant. Treat this as the current best
interface-side extension of the low-shot lane, still toy-only and not yet
benchmark evidence.

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
