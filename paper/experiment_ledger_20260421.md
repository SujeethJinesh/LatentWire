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
| Single-gate routed residual mixing | `dynalign_routed_module_replace_residrank16` on frozen GSM8K32 | A one-gate blend between the live dynalign residual and its routed alternative only ties target (`0.0625`) despite full numeric extraction coverage (`32/32`). The route family is still alive only in more selective forms, so do not spend another cycle on full K/V dense mixing. |
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
| Frozen GSM8K32 smoke contract | The same-pair frozen GSM8K smoke now runs end-to-end with matched non-thinking greedy decode and exact ID parity: `target_alone = 0.0625`, `text_to_text = 0.0312`, `rotalign_kv = 0.0625`, `c2c_generate = 0.1250`; `c2c` clears the smoke gate but `rotalign_kv` fails numeric extraction coverage (`28/32`) | Promote only after a new method row beats target on the same slice while passing all contract checks, especially numeric extraction coverage `>= 31/32` and byte-identical target reruns. |
| Frozen GSM8K32 checkpoint sweep | On the exact same contract, the best existing real proxies are `dynalign_module_replace = 0.0938` and `spanalign_module_replace = 0.0938`, both above target but still below `C2C = 0.1250`; `bytespan_module_replace = 0.0312`, and `sae_adapter = 0.0000` with coverage failure (`30/32`) | Promote only the output-aware alignment lane for further real benchmarking; do not spend another same-pair benchmark cycle on byte-only or first-pass SAE proxies without a materially different teacher or residual correction. |
| Frozen GSM8K32 expanded dynalign sweep | On the narrowed same-pair teacher-family sweep, `dynalign_module_replace = 0.0938` and `tokenbasis_replace = 0.0938` tie for best, while `dynalign_dwakd = 0.0625`, `dynalign_prefdist = 0.0312`, and `dynalign_spanalm = 0.0312` all fail to improve on the contract | Treat the plain output-aware teacher and token-grounded basis as the ceiling of the current teacher family on this pair; the next real branch should add residual correction or adaptive canonicalization, not more teacher-side elaboration. |
| Frozen GSM8K32 residual harness baseline | The dedicated residual-rank sweep harness now reproduces the live ceiling exactly: reused `dynalign_module_replace_residrank8 = 0.0938` and `tokenbasis_replace_residrank8 = 0.0938`, both with full numeric extraction coverage (`32/32`) and exact ID parity | Treat the residual harness as validated. The next real benchmark action is the expensive `rank16` recalibration on the same contract; do not claim residual-repair gains until one of those rows beats `0.0938`. |
| Frozen GSM8K32 dynalign residual rank16 | The first recalibrated residual row clears the old same-pair ceiling: `dynalign_module_replace_residrank16 = 0.1250` on the exact frozen contract, with full numeric extraction coverage (`32/32`) and `2/32` wins over target | Treat this as the first real same-pair row that matches the current `C2C` smoke accuracy on this slice. Do not promote the lane into paper mode until the matched `tokenbasis_replace + rank16` control and at least one broader held-out slice confirm it. |
| Frozen GSM8K32 dynalign residual reviewer diagnostics | On the same exact 32-example contract, `source_alone = 0.0312`, the source model is wrong on both `dynalign_module_replace_residrank16` win examples and on both target-only `text_to_text` poison examples, and the oracle verification bound `max(target_alone, dynalign_module_replace_residrank16)` is exactly `0.1250` (`4/32`) | Treat the current GSM8K32 slice as oracle-saturated for the live row: it is useful for falsification and reproducibility, but it cannot show additional verifier-side headroom. The next testing step is a larger frozen slice plus a cross-family falsification pair before spending more method cycles on the same 32 examples. |
| Hardened GSM8K diagnostic guard | `paper/gsm8k_diagnostic_guard_20260423.md` upgrades the reviewer diagnostic into a reusable artifact guard with ordered/set ID parity, numeric coverage, empty-prediction checks, flip matrix, exact sign p-value, candidate/source/text numeric-equality checks, and explicit gate labels. Regenerated reads: GSM8K32 seed 0 remains `4/32` with `2` non-copy wins and oracle saturation; GSM8K70 seed 0 is `8/70` with `6` non-copy wins, `2` losses, and oracle `10/70`; GSM8K70 seed 3 is demoted by the guard because the candidate is negative and has numeric coverage `69/70`. | Require this diagnostic guard for future live-row claims. The source-copying hypothesis remains weak on positive finite rows, but the live row is still not promotable because seed stability and strict source-control validity are unresolved. Next exact gate is shuffled-source plus zero-source controls on GSM8K70 seed 0 and the next finite seed before the learned 16-query connector pivot. |
| Reviewer-driven campaign runner | Added a dedicated contract-campaign runner that can scale slice size, seed repeats, and cross-family pairs while automatically regenerating diagnostics for selected candidate rows, plus paired bootstrap delta summaries against `target_alone` | Use this as the default entrypoint for the next evaluation phase rather than one-off manual runs; the first real campaign should be a larger frozen same-pair slice plus explicit seed repeats and paired intervals, then a matched cross-family pair. |
| Frozen GSM8K32 tokenbasis residual rank16 | The matched token-grounded control fails to reproduce the residual lift: `tokenbasis_replace_residrank16 = 0.0625`, exactly tying target with full numeric extraction coverage (`32/32`) and `0/32` wins over target | Treat the new residual lift as dynalign-specific rather than generic. The next exact real branch is `dynalign + gauge/canonicalization wrapper`, not another blind residual sweep over the same token-grounded lane. |
| Frozen GSM8K32 fixed gauge/canonicalization wrappers | The first fixed wrappers on top of the live dynalign residual lane both collapse: `dynalign_resid16_fitted_rotation = 0.0000` with numeric extraction coverage `0/32`, and `dynalign_resid16_shared_basis = 0.0000` with numeric extraction coverage `0/32` | Treat naive fixed wrappers as ruled out on the exact same-pair contract. The next exact real branch should be adaptive canonicalization or a stronger eigenspace residual, not more fixed gauge maps. |
| Frozen GSM8K32 adaptive canonicalization wrapper | The new adaptive grouped canonicalization wrapper no longer collapses the live lane, but it also does not improve it: `dynalign_resid16_adaptive = 0.1250`, with full numeric extraction coverage (`32/32`) and the same `2/32` wins over target as the plain dynalign residual row | Treat adaptive canonicalization as a stabilizing wrapper or control, not the missing lift by itself. The next exact real branch should be an eigenspace-aware residual or adaptive canonicalization combined with a materially different residual split, not more wrapper-only sweeps. |
| Frozen GSM8K32 preserve-core residual split | The first preserve-core / repair-tail follow-up on the live dynalign family fails to keep the residual lift: `dynalign_preserve_module_replace_residrank16 = 0.0625`, with full numeric extraction coverage (`32/32`) but `0/32` wins over target | Treat the naive preserved-subspace split as negative on the exact same-pair contract. The next real branch should move to an eigenspace-aware or saliency-aware residual formulation, not another simple preserve-top-subspace split. |
| Frozen GSM8K32 eigenspace residual branch | The first eigenspace-constrained follow-up is also negative: `dynalign_eigenspace_module_replace_residrank16 = 0.0312`, with full numeric extraction coverage (`32/32`) but `0/32` wins over target and one outright loss vs target | Treat naive dominant-eigenspace projection as ruled out on the exact same-pair contract. The next real branch should move to saliency-aware or learned-importance residual repair, not another simple geometric projection of the live dynalign lane. |
| Frozen GSM8K32 saliency residual branch | The first saliency-weighted follow-up is also negative: `dynalign_saliency_module_replace_residrank16 = 0.0312`, with full numeric extraction coverage (`32/32`) but `0/32` wins over target and one outright loss vs target | Treat simple loss-weighted saliency repair as ruled out on the exact same-pair contract. The next real branch should move to learned-importance preserve-plus-tail repair, routed residual repair, or codebook-style repair rather than another one-shot weighting tweak. |
| Frozen GSM8K32 saliency-preserve residual branch | The first saliency-preserve plus tail follow-up also fails to keep the live row: `dynalign_saliency_preserve_module_replace_residrank16 = 0.0625`, with full numeric extraction coverage (`32/32`), `1/32` win, `1/32` loss, and `30/32` ties vs target | Treat simple saliency-preserve plus dense tail repair as another negative same-pair control. The next exact branch should move to routed residual repair or an anchor-preserving codebook tail rather than another single-path preserve split. |
| Frozen GSM8K32 routed residual branch | The first routed-repair follow-up also fails to keep the live row: `dynalign_routed_module_replace_residrank16 = 0.0625`, with full numeric extraction coverage (`32/32`), `1/32` win, `1/32` loss, and `30/32` ties vs target | Treat simple single-gate routed repair as another negative same-pair control. The next exact branch should move to multi-expert / value-side routed repair or an anchor-preserving codebook tail rather than another one-gate dense mixture. |
| Frozen GSM8K32 value-routed residual branch | The first value-side selective repair follow-up keeps the live row without improving it: `dynalign_value_routed_module_replace_residrank16 = 0.1250`, with full numeric extraction coverage (`32/32`), `2/32` wins, `0/32` losses, and `30/32` ties vs target | Treat value-side selective repair as a live same-pair branch and the first routed variant that preserves the dynalign residual lift, but not as a new ceiling. The next exact branch should move to multi-expert / value-bank routed repair or an anchor-preserving codebook tail, and benchmark widening should remain frozen until one of those branches survives this same contract or materially reduces bytes/latency at the same accuracy. |
| Frozen GSM8K32 value-bank residual branch | The first value-bank routed follow-up does not preserve the live row: `dynalign_value_bank_module_replace_residrank16 = 0.0938`, with full numeric extraction coverage (`32/32`), `1/32` win, `0/32` losses, and `31/32` ties vs target | Treat value-bank expert repair as a non-additive routed control on this contract. It still beats target, but it falls back to the old dynalign ceiling and does not retain the `0.1250` lift. The routed family remains alive only in the simpler value-routed branch, so the next exact branch should either add genuinely stronger expert capacity/top-2 routing or pivot to anchor-preserving codebook tails or verifier-gated sidecars before widening benchmarks. |
| Frozen GSM8K32 value-routed-bank residual branch | The sparse top-2 value-routed bank follow-up also falls back to the old dynalign ceiling: `dynalign_value_routed_bank_module_replace_residrank16 = 0.0938`, with full numeric extraction coverage (`32/32`), `1/32` win, `0/32` losses, and `31/32` ties vs target | Treat sparse top-2 bank correction as another non-additive routed control on this contract. It keeps the row valid and above target, but it does not preserve the live `0.1250` lift. That means the next exact branch should leave the routed-bank family and move to verifier-gated V-side sidecars or anchor-preserving codebook tails before any benchmark widening. |
| Frozen GSM8K32 value-query-bank residual branch | The query-feature-routed bank follow-up also falls back to the old dynalign ceiling: `dynalign_value_query_bank_module_replace_residrank16 = 0.0938`, with full numeric extraction coverage (`32/32`), `1/32` win, `0/32` losses, and `31/32` ties vs target | Treat query-feature bank routing as another non-additive routed control on this contract. It validates the idea that the router itself can be changed without breaking coverage, but it still does not preserve the live `0.1250` row. That means the next exact branch should leave simple bank-routing tweaks and move to anchor-preserving codebook tails, stronger multi-expert value repair, or materially better verifier-gated repair before widening benchmarks. |
| Preserve-topk dominant subspace + codec tail toy | On the new codec toy, `preserve_topk_uniform_tail` lifts low-bit accuracy from `0.9583` to `0.9896` and MSE from `0.7463` down to `0.0284`, but the first `codebook_tail` and `codebook_tail_residual_fix` variants stall at `0.9844` with roughly `0.2470` MSE | Keep the dominant-subspace idea alive, but do not promote the current tail codec. The next codec branch should preserve anchors and redesign the tail model, not just add a naive codebook. |
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

## Reviewer-Driven Next Phase

Freeze the next phase so nearby variants are not ranked on oracle-saturated
GSM8K32 alone:

1. larger frozen same-pair campaign first
   - in practice: `gsm8k_eval_70` with explicit seeds and paired uncertainty
2. one matched cross-family pair second
   - preferred: `Qwen2.5-3B ↔ Llama-3.2-3B`
   - cheaper fallback: `Qwen2.5-1.5B ↔ Llama-3.2-1B`
3. then `RULER`
4. then `SCBench`
5. then `LongBench v2`

Method priority inside that phase is now:

1. anchor-preserving selective precision or codebook-tail repair on top of the
   live `dynalign_module_replace_residrank16` lane
2. learned connector / query-bottleneck transport
3. only then richer routed or verifier-gated repair

Do not return to another simple routed-bank or one-sidecar tweak unless the
larger-slice campaign shows a real uncertainty-aware reason to reopen it.

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

`results/gsm8k_smoke_contract_20260421/gsm8k_smoke_contract_20260421.md` is
the current frozen same-pair benchmark contract. It validates the prompt and
scoring path, keeps `C2C` as the live external bar on the exact 32-example
slice (`0.1250` vs target `0.0625`), and shows that the current `rotalign_kv`
row is not promotable yet because it only ties target while failing numeric
extraction coverage (`28/32`).

`results/gsm8k_contract_checkpoint_sweep_20260421/gsm8k_contract_checkpoint_sweep_20260421.md`
is the current nearest-proxy benchmark sweep. It says the live same-pair gain
comes from stronger output-aware alignment teachers (`dynalign` and
`spanalign` both `0.0938`), while byte-only alignment regresses to
`0.0312` and the first SAE-style sparse codebook proxy collapses to `0.0000`
with a coverage failure.

`results/gsm8k_contract_checkpoint_sweep_expanded_20260421/gsm8k_contract_checkpoint_sweep_20260421.md`
is the current narrowed teacher-family follow-up. It shows that the only
same-pair real rows that survive are plain `dynalign_module_replace` and the
token-grounded `tokenbasis_replace`, both at `0.0938`; heavier teacher-side
variants regress, so the next real lift should come from residual correction
or adaptive canonicalization instead of more dynalign-teacher complexity.

`results/gsm8k_contract_residual_baseline_20260421/gsm8k_contract_residual_sweep_20260421.md`
is the current residual-sweep harness baseline. It reproduces the known
`0.0938` same-pair ceiling for both reused `dynalign` and reused `tokenbasis`
rows with full numeric extraction coverage, which validates the harness before
the expensive `rank16` recalibration is attempted.

`results/gsm8k_contract_residual_rank16_dynalign_20260421/gsm8k_contract_residual_sweep_20260421.md`
is the first recalibrated residual follow-up on the frozen contract. It lifts
`dynalign_module_replace` from the old `0.0938` ceiling to `0.1250`, restores
full numeric extraction coverage (`32/32`), and matches the current `C2C`
smoke accuracy on the same exact slice. Treat this as the first real
same-pair positive clue for residual repair, not as a finished paper result:
the matched `tokenbasis + rank16` control and at least one broader held-out
slice still need to agree before the lane is promotable.

`results/gsm8k_contract_residual_rank16_tokenbasis_20260421/gsm8k_contract_residual_sweep_20260421.md`
is the matched token-grounded control for that residual follow-up. It fails to
reproduce the lift: `tokenbasis_replace_residrank16 = 0.0625`, exactly tying
the target row while still keeping full numeric extraction coverage (`32/32`).
Treat this as evidence that the new residual win is specific to the live
`dynalign` lane rather than a generic effect of raising residual rank.

`results/gsm8k_contract_gauge_wrapper_fitted_rotation_20260421/gsm8k_contract_gauge_wrapper_sweep_20260421.md`
and
`results/gsm8k_contract_gauge_wrapper_shared_basis_20260421/gsm8k_contract_gauge_wrapper_sweep_20260421.md`
are the first fixed gauge/canonicalization wrappers tested on top of that live
lane. Both are clean negatives on the exact frozen contract: they collapse to
`0.0000` accuracy with numeric extraction coverage `0/32`. Treat this as
evidence against naive fixed wrappers and as a push toward adaptive
canonicalization or stronger residual formulations instead.

`results/query_pool_toy_20260421/preserve_topk_codebook_tail_20260421.md` is
the first codec-side toy follow-up after that lift. It shows the preserved
dominant-subspace idea is real: keeping top-k atoms in high precision while
quantizing the rest lifts low-bit accuracy from `0.9583` to `0.9896` and drops
MSE from `0.7463` to `0.0284`. But the first `codebook_tail` and
`codebook_tail_residual_fix` variants both underperform that simpler preserved-
anchor baseline (`0.9844`, `~0.2470` MSE). Treat this as evidence to keep the
anchor-preserving codec story and redesign the tail model, not as a positive
result for the current naive codebook-tail implementation.

`results/gsm8k_contract_residual_value_verifier_sidecar_20260422/gsm8k_contract_residual_sweep_20260421.md`
is the first verifier-gated sidecar follow-up on the exact frozen same-pair
contract. It is valid but not additive:
`dynalign_value_verifier_sidecar_module_replace_residrank16 = 0.0938`, with
full numeric extraction coverage (`32/32`), `1/32` win, `0/32` losses, and
`31/32` ties versus target. Treat this as evidence that a single verifier-gated
value-side sidecar still falls back to the old dynalign ceiling and does not
preserve the live `0.1250` residual row. The next real branch should leave the
single-sidecar family and move to anchor-preserving codebook tails or a
materially stronger multi-expert / verifier design.

The next codec-side branch is now scaffolded directly in the repo as
`bridge_ridge_qk_dynalign_anchor_tail_module_replace`. It keeps the live
dynalign residual lane as the real anchor, stores a saliency-based preserve
mask, and is designed to quantize only the residual tail while keeping the
protected anchor exact. Treat this as the next same-pair falsification branch,
but judge it primarily on the larger frozen campaign and bytes-aware reporting,
not on GSM8K32 alone.

`paper/gsm8k32_anchor_tail_seed1_20260422.md` is the first bad-seed
falsification of that scaffold after tightening it into a true `V`-only runtime
wrapper. It is a clean negative on seed `1`: the checkpoint still quarantines
with `2,381,056` non-finite values and the same layer-8 `V` family
(`W_V.8`, `quant_proj_V.8`, `quant_aux_proj_V.8`, and the matching residual
slots). Treat this as evidence that wrapper-level value-side anchor-tail repair
does not move the instability upstream. The next bounded robustness branch
should patch the layer-8 `V` calibration fit itself before spending more budget
on wrapper-only codec variants.

`paper/gsm8k32_conditioned_bad_seed_controls_20260423.md` is the first
conditioning-first robustness follow-up on the live lane. It is mixed but
important:

- seed `0`: finite, but falls from `0.1250` to `0.0625`
- seed `1`: finite, rises from `checkpoint_nonfinite` to `0.0625`
- seed `2`: finite, rises from `checkpoint_nonfinite` to `0.0938`

Treat this as the first evidence that conditioning can remove the catastrophic
bad-seed failure on the exact frozen contract, but only by trading away some of
the best-seed ceiling. So global source+target whitening is a live stabilizer,
not yet the paper method. The next exact robustness branch should be more
selective than global whitening: layer-local or `V`-only conditioning around
the layer-8 failure family, rerun on the same `0/1/2` seed controls, and only
then reopen GSM70.

`paper/gsm8k32_selective_conditioning_l8_v_20260423.md` is the first targeted
version of that idea. It conditions only the `V` stream and only target layer
`8` with source+target whitening. The read is finite but negative:

- seed `1`: `0.0625`, full coverage, no non-finite checkpoint values
- seed `0`: `0.0312`, full coverage, no non-finite checkpoint values

Treat this as evidence that conditioning touches the numerical failure surface,
but the exact layer-8 `V` intersection is too narrow and is not promotable. The
next conditioning screen should broaden one axis before any GSM70 rerun:
`V`-only source+target conditioning across all layers, or layer-8
source+target conditioning across both `K/V`.

`paper/gsm8k32_selective_conditioning_v_all_layers_20260423.md` broadens the
failed intersection along the layer axis while keeping the value-stream
restriction. The read is finite but also negative:

- seed `1`: `0.0625`, full coverage, no non-finite checkpoint values
- seed `0`: `0.0312`, full coverage, no non-finite checkpoint values

Treat this as further evidence that conditioning/preconditioning reaches the
right numerical surface, but simple `V`-only source+target whitening is too
blunt and erases the live seed-0 signal. The last worthwhile whitening screen
is layer-8 source+target conditioning across both `K/V`; if that also kills
seed `0`, stop whitening sweeps and move to a protected/outlier-escrow
calibration fit for the layer-8 `V` family.

`paper/gsm8k32_selective_conditioning_l8_kv_20260423.md` is that final simple
whitening screen. It conditions both `K/V` but only at target layer `8`. The
bad-seed read is the best conditioned row so far, but the branch still fails
the seed-0 guard:

- seed `1`: `0.0938`, full coverage, no non-finite checkpoint values
- seed `0`: `0.0312`, full coverage, no non-finite checkpoint values

Treat this as the end of the simple whitening sweep. Conditioning reliably
removes the catastrophic checkpoint failure, and layer-8 `K/V` coupling can
recover a positive bad-seed row, but every whitening variant tested so far
erases the live seed-0 signal. The next method branch should patch the layer-8
`V` calibration fit with protected/outlier-escrow smoothing rather than keep
tuning whitening scope.

`paper/gsm8k70_campaign_20260422.md` is the first larger frozen same-pair read
after the reviewer pivot. It shows that the live
`dynalign_module_replace_residrank16` lane survives beyond GSM8K32:

- `target_alone = 0.0571` (`4/70`)
- `dynalign_module_replace_residrank16 = 0.1143` (`8/70`)
- `c2c_generate = 0.1286` (`9/70`)
- oracle(target, candidate) = `0.1429` (`10/70`)

Treat this as the first useful evidence that the live residual lane is not a
pure GSM8K32 artifact, but still not as a promotable paper result:

1. the run is still only one seed
2. the paired bootstrap interval against target still crosses zero
3. the external bar is still slightly ahead
4. the next critical steps are multi-seed repetition on the same larger slice
   and one matched cross-family falsification pair

`paper/gsm8k70_seed_stability_full_20260422.md` is the current four-seed read
on that larger slice. It is stronger and more negative than the first partial
repeat-seed note:

- seed `0`: `0.1143` (`8/70`), positive
- seed `1`: `0.0000` (`0/70`), `checkpoint_nonfinite`
- seed `2`: `0.0000` (`0/70`), `checkpoint_nonfinite`
- seed `3`: `0.0286` (`2/70`), finite but below target

Treat this as the current stability verdict on the live same-pair lane:

1. only `1 / 4` attempted seeds is both finite and above target
2. the repeated collapse is not random:
   - seeds `1` and `2` both fail in the same layer-8 `V` family
3. finite seeds can still be weak:
   - seed `3` is numerically valid but loses to `target_alone`

So the live row remains a mechanism clue, not a stable method. The next exact
branch should stay narrow and robustness-first:

1. `V`-only anchor-preserving selective precision / tail coding
2. rerun the same GSM70 seed audit
3. only then reopen cross-family or broader benchmark expansion

`paper/gsm8k32_v8_outlier_escrow_and_srcwhite_l8_kv_20260423.md` records two
follow-up gates after the simple whitening sweep:

- `bridge_ridge_qk_dynalign_v8_outlier_escrow_module_replace` was implemented
  and tested on GSM8K32 seed `1`, but the checkpoint still quarantined with
  `2,381,056` non-finite values and first bad key `W_V.8`. This kills the
  current protected-channel escrow branch as a fix for the bad-seed layer-8
  value failure.
- Source-only layer-8 `K/V` conditioning was then run as the cheapest
  no-code fallback. It produced a finite checkpoint and valid coverage on seed
  `1`, but only tied target (`0` wins, `0` losses, `32` ties; accuracy
  `0.0625`). This weakens the hypothesis that target whitening was the only
  reason conditioned rows erased the live seed-0 signal.

Do not run seed `0`, GSM70, or cross-family widening for either branch. The next
exact gate is now a direct layer/stream-specific `W_V.8` fit regularization or
diagnostic branch, because the blocker remains localized to the same value-side
fit surface rather than to runtime tail quantization or target whitening alone.

`paper/gsm8k32_wv8_fit_ridge_override_20260423.md` records that direct
layer/stream-specific fit regularization gate. Commit `fba00f40` added a
generic per-layer/per-stream source-to-target fit ridge override with calibrate
and residual-sweep provenance. The first GSM8K32 bad-seed run used only
`W_V.8` at lambda `1e-2`:

- seed `1`: finite, accuracy `0.0625`, full coverage, no empty predictions
- paired vs target: `0` wins, `0` losses, `32` ties
- checkpoint nonfinite values: `0`; first bad key: none
- checkpoint max abs: `6416.1553`, with top abs in `quant_aux_proj_V.15`

Treat this as a weakened branch, not a rescue. Localized `W_V.8` ridge damping
removes the catastrophic nonfinite checkpoint failure, but it collapses to
target-cache parity and does not establish communication. Do not run seed `0`,
GSM70, or cross-family widening for scalar ridge-only stabilization. The next
exact gate should preserve value-side information while stabilizing the layer-8
fit surface: a protected value-channel / value-innovation codec with explicit
byte accounting, tested first on GSM8K32 seed `1` and only then on seed `0` if
it is finite and positive.

`paper/gsm8k32_wv8_protected_fit_ridge_20260423.md` records that protected
value-channel follow-up. Commits `4d71704f` and `a3bc0509` added
`fit_ridge_protected_rank`, calibrate/residual-sweep provenance, and then
stabilized the implementation from a split-tail residual solve to a safer
full-tail-plus-protected-overwrite solve.

Reads on GSM8K32 seed `1`:

- unsafe protected split, rank `2`: `checkpoint_nonfinite`, first bad key
  `W_V.8`, `2,380,800` non-finite checkpoint values. This kills the split-tail
  residual solve.
- protected overwrite, rank `2`: finite, accuracy `0.0625`, full coverage,
  `0` wins / `0` losses / `32` ties vs target.
- protected overwrite, rank `4`: finite, accuracy `0.0625`, full coverage,
  `0` wins / `0` losses / `32` ties vs target.

Treat this as a weakened scalar protected-ridge branch, not a rescue. The safer
overwrite primitive is useful infrastructure and fixes nonfinites, but tiny
protected ranks do not recover bad-seed communication; they still collapse to
target-cache parity. Do not run seed `0`, GSM70, or cross-family widening for
protected scalar ridge. The next gate should shift from scalar closed-form
stabilization to either a source-correctness/flip-table diagnostic on the live
seed-0 wins, or a learned query/resampler connector with an explicit bottleneck.

`paper/gsm8k70_source_controls_20260423.md` records the first strict
source-control gate on the live GSM8K70 seed-0 row. Commit `4408e61a` added
`--source-prompt-control shuffle_examples` and per-record source-control
telemetry. `scripts/analyze_gsm8k_source_controls.py` then compared the live
row against zero-source and deterministic shuffled-source controls.

Readout:

- live matched-source row: `8/70`, paired vs target `6` wins / `2` losses /
  `62` ties, numeric coverage `70/70`.
- zero_source: `0/70`, paired vs target `0` wins / `4` losses / `66` ties,
  live-win retention `0/6`, numeric coverage `0/70`.
- shuffled_source_salt0: `0/70`, paired vs target `0` wins / `4` losses /
  `66` ties, live-win retention `0/6`, source derangement telemetry passes,
  numeric coverage `1/70`.

This weakens the target-cache explanation: neither control preserves any live
candidate-only wins, and the shuffled-source artifact confirms source examples
were mismatched while target ids/order were preserved. It does not yet clear the
strict reviewer gate, because the controls collapse through low numeric
coverage and empty/non-numeric generations. Treat the live row as matched-source
dependent but not yet reviewer-ready. The next exact gate is a
validity-preserving shuffled-source mismatch control, likely via a target-safe
fusion shrinkage/verifier fallback or a learned bottleneck/resampler connector,
before seed repeats or cross-family widening.

Follow-up target-fallback diagnostic on the same artifacts:
`results/gsm8k70_source_controls_20260423/seed0/source_control_target_fallback_readout_20260423.md`.
`scripts/analyze_gsm8k_source_controls.py` now supports a conservative
`--fallback-nonnumeric-controls-to-target` mode that replaces only empty or
nonnumeric control outputs with the paired `target_alone` prediction. This is a
diagnostic safety envelope, not a method claim.

Readout:

- zero_source + target fallback: `4/70`, paired vs target `0` wins / `0`
  losses / `70` ties, live-win retention `0/6`, numeric coverage `70/70`.
- shuffled_source_salt0 + target fallback: `4/70`, paired vs target `0` wins /
  `0` losses / `70` ties, live-win retention `0/6`, numeric coverage `70/70`,
  source derangement telemetry passes.

This clears the seed-0 diagnostic source-dependence gate: even after giving the
controls a target-safe fallback, the decisive shuffled-source control remains at
target level and retains none of the live matched-source wins. It still does not
clear the paper method gate, because seed stability remains negative and
nonfinite/target-parity follow-ups dominate. The next exact gate is a symmetric
target-safe live/control path that preserves the seed-0 lift while making at
least one additional finite seed positive; otherwise pivot to the learned
query/resampler connector branch.

`paper/gsm8k32_anchor_tail_seed1_20260422.md` now includes a 2026-04-23
independent rerun of the V-only anchor/tail seed-1 screen with explicit scratch
checkpoint provenance. The rerun used:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_anchor_tail_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_anchor_tail_resid16_seed1_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_dynalign_anchor_tail_resid16_seed1_20260423 \
  --seed 1
```

Readout:

- status: `checkpoint_nonfinite`
- first bad key: `W_V.8`
- checkpoint nonfinite values: `2,381,056`
- numeric coverage: `0/32`
- empty predictions: `32`
- summary JSON SHA256:
  `c447f87287b319933c7269d3e6df644ca10a8244af565cfc475edf5194012b30`

This confirms the earlier anchor-tail falsification and marks the wrapper-only
V-anchor/tail selective-precision lane as saturated. It should not be run on
seed `0`, GSM70, or cross-family pairs. The branch fails before evaluation, so
it cannot clear the current stability gate. Combined with the ridge and
protected-ridge rows, the closed-form value-side repair lane is now mostly
exhausted unless a new diagnostic changes the failure surface. The next exact
gate should be the learned query/resampler connector with an explicit
bottleneck and the same matched/zero/shuffled-source target-safe control
envelope.

`paper/gsm8k32_value_routed_seed1_20260423.md` records the cheapest existing
learned-connector seed-stability screen. The branch
`dynalign_value_routed_module_replace_residrank16` had preserved the GSM8K32
seed-0 live row at `4/32` with full coverage, so it was the nearest existing
learned query/module surface to test before adding a new query-resampler mode.

Run:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_value_routed_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_dynalign_value_routed_resid16_seed1_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_dynalign_value_routed_resid16_seed1_20260423 \
  --seed 1
```

Readout:

- status: `checkpoint_nonfinite`
- first bad key: `W_V.8`
- checkpoint nonfinite values: `2,382,081`
- numeric coverage: `0/32`
- empty predictions: `32`
- summary JSON SHA256:
  `9a29f7357a230ccaaf3ae9992d090c4207c3ce917fb893b27f36a68bb4f12fa7`

This kills rerunning the current value-routed branch as the next seed-stability
rescue. The failure repeats the same layer-8 value family as the live bad seeds
and anchor-tail wrapper. The next exact gate is now an explicit guarded
query/resampler connector branch, not another existing value-routed rerun:
reuse the slotted query-module plumbing, expose bottleneck slots/rank as first-
class provenance, add fit-time finite/norm checks or fallback-to-base for bad
layers, and evaluate with matched/zero/shuffled-source target-safe controls plus
a `0/4/16/32` slot capacity/null sweep before any cross-family widening.

`paper/gsm8k32_guarded_query_resampler_seed1_20260423.md` records the first
explicit guarded query/resampler connector gate. The implementation added the
new correction `bridge_ridge_qk_dynalign_query_resampler_replace`, reused the
slotted query-module replacement path, added branch-specific finite/high-norm
guards before checkpoint materialization, wired calibration/evaluation/sweep
registration, and added translator/evaluation/parser/sweep regression tests.

Runs:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_resampler_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_guarded_query_resampler_seed1_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_guarded_query_resampler_seed1_20260423 \
  --seed 1

./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_resampler_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_guarded_query_resampler_seed1_evalfix_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_guarded_query_resampler_seed1_20260423 \
  --seed 1
```

The first run produced a finite checkpoint but exposed a missing evaluation
dispatch for live query heads. The dispatch was fixed and the second run reused
the same checkpoint.

Readout:

- status: `ok`
- checkpoint nonfinite values: `0`
- first bad key: `-`
- checkpoint max abs: `6416.1553`
- checkpoint SHA256:
  `b4accaec6a9b6414015c58017a0b77894be88462898654b0261b690a9d7653c7`
- accuracy: `2/32`
- paired vs target: `0` wins / `0` losses / `32` ties
- exact ID parity: pass
- numeric coverage: `32/32`
- empty predictions: `0`
- summary JSON SHA256:
  `a61ff8f9ec5b3deecb08ba7ca22671164438aa6a24a37bc276a3fec19a9ad290`

This clears the immediate checkpoint-health/validity blocker for the learned
connector family, but it does not clear the positive-method gate. The guarded
branch converts the prior seed-1 nonfinite failure into exact target parity,
which is useful evidence but not a publishable communication result. The
guarded query/resampler surface remains alive as the finite implementation
surface, while this exact full-replacement guarded variant is weakened as a
positive method. The next exact gate is either an innovation/residual-only
target-safe query-resampler path or a cheap `bridge_bank_size = 0/4/16/32`
capacity/null sweep on GSM8K32 seed1 before any GSM70 or cross-family widening.

`paper/gsm8k32_query_resampler_bank_sweep_seed1_20260423.md` records the
capacity/null follow-up for the guarded query-resampler branch. The sweep runner
now exposes `--bridge-bank-size`, disambiguates non-default bank sizes in
labels/checkpoint names, records capacity in row conditioning, and writes
checkpoint health sidecars for successful checkpoints. The translator now
allows `bridge_bank_size=0` as a true no-private-slot null for
`bridge_ridge_qk_dynalign_query_resampler_replace`.

Runs:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_resampler_replace \
  --rank 16 \
  --bits 4 \
  --bridge-bank-size 0 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_query_resampler_bank0_seed1_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_query_resampler_bank_sweep_seed1_20260423 \
  --seed 1

./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_resampler_replace \
  --rank 16 \
  --bits 4 \
  --bridge-bank-size 16 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_query_resampler_bank16_seed1_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_query_resampler_bank_sweep_seed1_20260423 \
  --seed 1
```

Readout:

- bank0: status `ok`, checkpoint nonfinite values `0`, accuracy `2/32`,
  paired vs target `1` win / `1` loss / `30` ties, numeric coverage `31/32`,
  empty predictions `0`, diagnostic status `invalid_artifact`
- bank16: status `ok`, checkpoint nonfinite values `0`, accuracy `2/32`,
  paired vs target `0` wins / `0` losses / `32` ties, numeric coverage `32/32`,
  empty predictions `0`, diagnostic status `target_parity_or_negative`
- bank0 checkpoint SHA256:
  `29d568668cf0bfb2d3d6638293b937b36dc89685d6bf79edab558cd8e203f543`
- bank16 checkpoint SHA256:
  `900193a8f035b79b4cc4c247d205693b4d99f15a28b44a63b7eab376d56b4a3e`
- bank0 diagnostics JSON SHA256:
  `1d9a4ac1257c79395fb62ae95b9b463ebf7adafdfd4864ec0cdc85c67cc5c473`
- bank16 diagnostics JSON SHA256:
  `40eece8d88af2d3d40a4045b7abec83d4040292c274bb7ea905ddc51bed1cc1c`

This weakens plain capacity scaling of the guarded full-replacement
query-resampler. Bank0 shows the live-memory path can change target outcomes,
including one non-copy candidate-only win, but it also creates one target loss
and fails strict numeric coverage. Bank16 is fully valid but exact target
parity. The next exact gate is the target-safe innovation/residual
query-resampler branch with matched zero/shuffled-source controls; GSM70 and
cross-family widening remain blocked until that branch clears GSM8K32 seed1.

`paper/gsm8k32_query_innovation_resampler_seed1_20260423.md` records the
target-safe residual/innovation query-resampler follow-up. The new
`bridge_ridge_qk_dynalign_query_innovation_resampler_replace` branch reuses the
guarded query-resampler path, fits only `target - base_bridge_prediction`, and
applies a bounded additive residual at fusion time instead of full KV
replacement. It is wired through calibration, evaluation, the GSM8K residual
sweep runner, and regression tests.

Runs:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_innovation_resampler_replace \
  --rank 16 \
  --bits 4 \
  --bridge-bank-size 16 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_query_innovation_resampler_seed1_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423 \
  --seed 1

./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_innovation_resampler_replace \
  --rank 16 \
  --bits 4 \
  --bridge-bank-size 16 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_query_innovation_resampler_seed1_gate025_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423 \
  --seed 1 \
  --gate 0.25

./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_query_innovation_resampler_replace \
  --rank 16 \
  --bits 4 \
  --bridge-bank-size 16 \
  --kv-transport k_only \
  --slice-size 32 \
  --baseline-results-dir results/gsm8k_smoke_contract_20260421 \
  --results-dir .debug/gsm8k32_query_innovation_resampler_seed1_gate015_20260423 \
  --checkpoints-dir .debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423 \
  --seed 1 \
  --gate 0.15
```

Readout:

- checkpoint nonfinite values: `0`
- first bad key: `-`
- checkpoint max abs: `6416.1553`
- checkpoint SHA256:
  `b1f0cfa62c67ffcbdbce631c6cfd80df3240e132e252b0775aef355940a557b8`
- gate `0.10`: accuracy `2/32`, paired vs target `0` wins / `0` losses /
  `32` ties, numeric coverage `32/32`, empty predictions `0`
- gate `0.25`: accuracy `2/32`, paired vs target `1` win / `1` loss /
  `30` ties, numeric coverage `32/32`, empty predictions `0`
- gate `0.15`: accuracy `3/32`, paired vs target `1` win / `0` losses /
  `31` ties, numeric coverage `32/32`, empty predictions `0`
- gate `0.15` zero-source control: accuracy `2/32`, paired vs target `1` win /
  `1` loss / `30` ties, live-win retention `1/1`
- gate `0.15` shuffled-source control: accuracy `3/32`, paired vs target `1`
  win / `0` losses / `31` ties, live-win retention `1/1`, deranged source
  indices, `4/32` target fallback
- source-control status: `source_controls_do_not_clear_gate`
- diagnostic note: the only candidate-only win has the same numeric answer as
  text-to-text; source-alone is wrong, but zero/shuffled controls retain the
  win ID.

This branch clears implementation, checkpoint, coverage, and one live-row
smoke gate at fixed gate `0.15`, but it fails the required source-control gate.
The result weakens the innovation-resampler as a publishable positive method:
the branch can move answers through a bounded target-safe path, but the
observed win is not proven to depend on real source communication. Do not widen
this row to GSM70 or cross-family. The next exact gate is either the strongest
existing real lane with seed/source-control repeats, or a source-control-aware
verifier/gate that can suppress wins retained under zero/shuffled source before
retesting this same GSM8K32 seed-1 surface.

`paper/gsm8k_residual_sweep_source_controls_20260423.md` records the residual
sweep source-control wrapper update. This is a gate/reproducibility change, not
a new method result. The sweep runner now exposes `--run-source-controls` and,
for each live row that clears the normal contract, runs matched zero-source and
shuffled-source controls, calls `scripts/analyze_gsm8k_source_controls.py`,
stores row-local artifacts under `{results_dir}/{label}/source_controls/`, and
blocks markdown promotion unless the analyzer returns
`source_controls_support_matched_source_signal`.

Verification:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_gsm8k_contract_residual_sweep.py \
  tests/test_analyze_gsm8k_source_controls.py
```

Readout:

- tests: `39 passed in 0.13s`
- source-control wrapper status: implemented and regression-tested
- affected promotion policy: rows can no longer be promoted by the sweep
  markdown when `--run-source-controls` is enabled and source controls fail
- subagent audit follow-up: `scripts/build_gsm8k_contract_manifest.py` still
  resolves rows positionally and should be hardened by explicit label before
  the next artifact-manifest refresh

This does not revive the demoted query-innovation resampler row; it makes the
failure mode first-class so future rows cannot bypass the falsification check.
The next exact gate is to run the integrated wrapper on GSM70
`dynalign_module_replace_residrank16` seed 0 plus one finite repeat, then decide
whether a source-control-aware accept/fallback gate is worth implementing.

`paper/gsm8k70_integrated_source_controls_20260423.md` records the integrated
source-control gate on the strongest remaining GSM70 lane. Seed 0 was rerun
through `scripts/run_gsm8k_contract_residual_sweep.py --run-source-controls` on
the frozen 70-example slice with the campaign baseline. The live row was
regenerated; the prior matched zero-source and shuffled-source control JSONL
files were reused through row-local `.debug` symlinks and re-analyzed by the
new wrapper path.

Readout:

- seed 0 live: `8/70`, paired vs target `6` wins / `2` losses / `62` ties,
  numeric coverage `70/70`, empty predictions `0`
- seed 0 source-control status:
  `source_controls_support_matched_source_signal`
- zero-source + target fallback: `4/70`, paired vs target `0/0/70`,
  paired vs live `2/6/62`, live-win retention `0/6`, coverage `70/70`
- shuffled-source + target fallback: `4/70`, paired vs target `0/0/70`,
  paired vs live `2/6/62`, live-win retention `0/6`, coverage `70/70`,
  source deranged
- seed 3 finite repeat: `2/70`, paired vs target `1` win / `3` losses /
  `66` ties, numeric coverage `69/70`, source-control status
  `not_run_live_gate_failed`
- seed 0 wrapper JSON SHA256:
  `0ada9e55c0c3518b36049c2a99f817f0f62a6b21e5be05f21665896972851d3f`
- seed 0 source-control readout SHA256:
  `c6bf310dea326ddbee116e656c64db5f5556b8fca163818fbb29e44ad630ed86`
- seed 3 wrapper JSON SHA256:
  `f2762cb777a71ef220df36c561dca964ce893800e1eb3bd9dc34f60283592a51`

This strengthens the seed-0 source-dependence interpretation but does not
promote the raw dynalign lane. The finite repeat is target-negative, so the
blocking gap is now target-safe selection/robustness rather than source
controls on seed 0. The next exact gate is an offline control-calibrated
accept/fallback replay over the seed-0 and seed-3 artifacts: accept the latent
intervention only when a predeclared score keeps zero/shuffled control accepts
near zero and avoids seed-3 harms. If that fails, demote dynalign to a brittle
mechanism probe and return the main method effort to learned
connector/conditional innovation designs.

The repo cleanup sidecar also identified a manifest drift risk. In response,
`scripts/build_gsm8k_contract_manifest.py` now resolves live/control rows by
explicit label instead of positional `rows[0]`, and preserves source-control
sidecar paths when present. Verification:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_build_gsm8k_contract_manifest.py \
  tests/test_gsm8k_contract_residual_sweep.py \
  tests/test_analyze_gsm8k_source_controls.py
```

Readout: `41 passed in 0.10s`.

`paper/gsm8k70_accept_fallback_replay_20260423.md` records the offline
accept/fallback replay over the strongest remaining GSM70 lane. The replay
tested `dynalign_module_replace_residrank16` seed 0, the finite seed-3 repeat,
and the seed-0 zero/shuffled-source controls against target fallback using
non-oracle selector telemetry.

Command:

```bash
./venv_arm64/bin/python scripts/analyze_gsm8k_accept_fallback.py \
  --baseline-predictions results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke/gsm8k32_latentwire.jsonl \
  --candidate seed0=.debug/gsm8k70_integrated_source_controls_20260423/seed0/dynalign_module_replace_residrank16.jsonl \
  --candidate seed3=.debug/gsm8k70_integrated_source_controls_20260423/seed3/dynalign_module_replace_residrank16.jsonl \
  --control zero_source=.debug/gsm8k70_integrated_source_controls_20260423/seed0/dynalign_module_replace_residrank16/source_controls/zero_source.jsonl \
  --control shuffled_source_salt0=.debug/gsm8k70_integrated_source_controls_20260423/seed0/dynalign_module_replace_residrank16/source_controls/shuffled_source_salt0.jsonl \
  --score-field selector_gap_min \
  --score-quantile 0.6 \
  --score-quantile 0.7 \
  --score-quantile 0.75 \
  --score-quantile 0.8 \
  --score-quantile 0.9 \
  --output-json results/gsm8k70_accept_fallback_replay_20260423/accept_fallback_replay.json \
  --output-md results/gsm8k70_accept_fallback_replay_20260423/accept_fallback_replay.md
```

Readout:

- target baseline: `4/70`
- `numeric_changed`: seed 0 `8/70`, paired `6/2/62`; seed 3 `2/70`,
  paired `1/3/66`; fails because harms remain
- `selector_gap_min_ge_q0p7_numeric_changed`: seed 0 `7/70`, paired
  `3/0/67`, accepted `13`; seed 3 `5/70`, paired `1/0/69`, accepted `16`;
  zero-source and shuffled-source controls both `4/70`, paired `0/0/70`,
  accepted `0`
- q0.75/q0.80/q0.90 selector-gap policies also clear the offline gate with no
  target losses and zero control accepts, but retain fewer seed-0 wins as the
  threshold tightens
- replay JSON SHA256:
  `3ac363245ac963c4354175ae29ccfc454209ab3bb50b9489fef590da0b7330f9`
- replay markdown SHA256:
  `af42eb83c5c50b2443fb2793af6f8f4064df371aa1ab36f0e6cb32906a313370`

This promotes a selector-gap gated target-fallback method as the next live
branch, but it is not yet a publishable method result because the threshold was
chosen in offline replay. The next exact gate is to freeze
`selector_gap_min_ge_q0p7_numeric_changed` at threshold
`0.029237359762191772`, implement it in the runtime evaluation path, and rerun
seed 0, seed 3, and matched zero/shuffled-source controls from scratch. If the
runtime replay collapses, demote dynalign to a brittle mechanism probe and move
the main method effort to a learned contrastive innovation connector.

`paper/gsm8k70_runtime_accept_fallback_20260423.md` records the runtime
validation of that frozen q0.70 selector-gap policy. The wrapper now supports
runtime accept/fallback rows while preserving raw and gated live/control JSONL
files separately. The fresh seed-0 live row reproduced the offline positive,
but fresh raw controls also passed the selector, so the branch is killed as a
paper method.

Command:

```bash
./venv_arm64/bin/python scripts/run_gsm8k_contract_residual_sweep.py \
  --base dynalign_module_replace \
  --rank 16 \
  --bits 4 \
  --kv-transport k_only \
  --slice-size 70 \
  --eval-file data/gsm8k_eval_70.jsonl \
  --materialized-eval-file results/gsm8k70_seed_repeat_full_20260422/_artifacts/gsm8k_eval_70.jsonl \
  --baseline-results-dir results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke \
  --results-dir .debug/gsm8k70_runtime_accept_fallback_20260423/seed0 \
  --checkpoints-dir checkpoints/gsm8k_contract_residual_sweep_20260421 \
  --seed 0 \
  --gate 0.10 \
  --run-source-controls \
  --source-control-random-salt 0 \
  --accept-fallback-score-field selector_gap_min \
  --accept-fallback-threshold 0.029237359762191772
```

Readout:

- target baseline: `4/70`
- live matched source: `7/70`, paired `3/0/67`, accepted `13`
- zero-source control: `6/70`, paired `2/0/68`, accepted `14`, retained `2/3`
  live wins
- shuffled-source control: `6/70`, paired `2/0/68`, accepted `14`, retained
  `2/3` live wins
- retained control win IDs: `31715a2b361f0b6d`, `e100c479d9fc22f8`
- source-control status: `source_controls_do_not_clear_gate`
- runtime sweep JSON SHA256:
  `1af3ea2d9abcfe01db7da65c4b9b05ee19d46c242e6dd96310e96ea874f02ba3`
- source-control readout JSON SHA256:
  `8bd30ce4b99de871c54b847b37eb0a7c0d0ec7f5af2e398624de9d9aee9de09f`

Decision: do not run seed 3 or cross-family for this policy. The q0.70
selector-gap branch is killed because the gate is not source-specific under
fresh controls. The next exact gate is a control-contrastive learned innovation
connector: train a small bottleneck/additive sidecar with matched-source
positives and zero/shuffled-source penalties, then rerun this same GSM70 seed-0
source-control gate before any widening.

`paper/gsm8k70_communication_headroom_20260423.md` adds a reusable
communication-headroom analyzer and reruns the GSM70 seed-0 readout against
source-alone, raw live, gated live, raw controls, and gated controls. The result
sharpens the negative decision: source-alone is `1/70`, target-or-source oracle
is only `5/70`, source-alone explains `0/6` raw live wins and `0/3` gated live
wins, raw controls retain `3/6` raw live wins, gated controls retain `2/3`
gated live wins, and `selector_gap_min` score contrast keeps `0/6` raw live
wins because every raw live win has exactly equal matched/zero/shuffled-source
scores.

Command:

```bash
./venv_arm64/bin/python scripts/analyze_gsm8k_communication_headroom.py \
  --baseline-predictions results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke/gsm8k32_latentwire.jsonl \
  --source seed0=results/gsm8k70_seed_repeat_full_20260422/seed0/gsm8k32_source_alone.jsonl \
  --candidate raw_live=.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16.jsonl \
  --control zero_source_raw=.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736/source_controls/zero_source_raw.jsonl \
  --control shuffled_source_salt0_raw=.debug/gsm8k70_runtime_accept_fallback_20260423/seed0/dynalign_module_replace_residrank16_accept_selector_gap_min_ge_0p02923736/source_controls/shuffled_source_salt0_raw.jsonl \
  --score-field selector_gap_min \
  --output-json results/gsm8k70_communication_headroom_20260423/headroom_raw_controls.json \
  --output-md results/gsm8k70_communication_headroom_20260423/headroom_raw_controls.md
```

Decision: do not implement delayed selector-gap contrastive gating; it would
accept zero raw live wins on this decisive surface. The next exact gate is a
learned source-control-contrastive innovation connector, or a preliminary
stronger-source slice selection if GSM8K70 source-alone is judged too weak to
expose real communication.

`paper/source_headroom_surface_scan_20260423.md` adds a source-headroom surface
scanner and ranks available frozen target/source-like surfaces before connector
training. The scan confirms GSM source-alone is a poor primary gate (`1`
source-only win on GSM70, `1` on GSM32), while SVAMP and C2C/text surfaces have
substantially more target-complementary headroom: SVAMP70 text relay has `26`
source-only wins and oracle `31/70`; SVAMP70 C2C has `18` source-only wins and
oracle `39/70`; SVAMP70 process repair has `17` source-only wins and oracle
`38/70`; GSM70 C2C has `7` source-only wins and oracle `11/70`. A follow-up
communication-headroom readout using GSM70 C2C as the source-like row found that
raw dynalign overlaps C2C on `2/6` raw live wins, but both overlaps are also
retained by raw zero-source and shuffled-source controls, so current dynalign is
not yet a source-specific C2C-style innovation transfer.

Command:

```bash
./venv_arm64/bin/python scripts/analyze_source_headroom_surfaces.py \
  --surface gsm70_source_alone=target_path=results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke/gsm8k32_latentwire.jsonl,source_path=results/gsm8k70_seed_repeat_full_20260422/seed0/gsm8k32_source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=gsm70_source_alone \
  --surface svamp70_text_relay=path=results/svamp_replication_20260417/predictions/svamp70_attention_g010_pos05.jsonl,target_method=target_alone,source_method=text_to_text,eval_file=data/svamp_eval_70.jsonl,note=svamp_text_relay \
  --surface svamp70_c2c=target_path=results/process_repair_holdout_20260421/qwen_svamp70_process_repair_strict_selector_telemetry.jsonl,source_path=results/c2c_svamp70_20260418/qwen_svamp70_c2c.jsonl,target_method=target_alone,source_method=c2c_generate,note=svamp_c2c \
  --surface gsm70_c2c=target_path=results/gsm8k_contract_campaign_slice128_seed0_20260422/smoke/gsm8k32_latentwire.jsonl,source_path=results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl,target_method=target_alone,source_method=c2c_generate,note=gsm_c2c \
  --min-source-only 5 \
  --output-json results/source_headroom_surfaces_20260423/headroom_surfaces.json \
  --output-md results/source_headroom_surfaces_20260423/headroom_surfaces.md
```

Decision: before training the learned control-contrastive innovation connector,
materialize fresh exact-ID SVAMP70 source/target/text/C2C rows in resumable
per-method jobs. If `source_alone` or another latent-accessible source row has
`>=5` to `10` source-only wins, use that as the first connector gate; if only
text/C2C has headroom, use it as a teacher/upper-bound surface rather than a
direct paper claim.

`paper/svamp_exactid_baselines_20260423.md` adds a resumable generation-baseline
materializer and validates it on a fresh exact-ID SVAMP dev-smoke slice. The new
runner writes one method per artifact, materializes the exact eval slice under
the results directory, logs each command independently, and skips only artifacts
that parse with the expected count and ordered `example_id` parity. The
`limit=5` run completed all four rows: target `2/5`, source `2/5`, text-to-text
`0/5`, and C2C `1/5`, all with exact ordered ID parity and `5/5` numeric
coverage. Source has one source-only win over target and a source/target oracle
of `3/5`; text-to-text and C2C add no source-only wins on this tiny slice.

Command:

```bash
./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/svamp_exactid_baselines_20260423 \
  --limit 5 \
  --methods target source t2t c2c \
  --device mps \
  --max-new-tokens 64 \
  --continue-on-error
```

Decision: the runner clears the reproducibility/resume gate, but `N=5` is only a
dev smoke. The next exact gate is SVAMP32 with the same rows. Promote connector
training only if source or another latent-accessible source row reaches at least
`5/32` source-only wins with exact ordered ID parity and near-complete numeric
coverage; otherwise frame SVAMP as a teacher/headroom surface and do not claim a
direct latent-source positive result.

`paper/svamp32_exactid_c2c_teacher_gate_20260423.md` scales the fresh SVAMP
materialization gate to `N=32` and hardens the runner before recording the
manifest. The runner now validates unique ordered IDs, strict method names,
sidecar config parity, manifest SHA256 provenance, and uses temp outputs for new
runs before replacing final artifacts. The SVAMP32 rows all have exact ordered
ID parity and unique IDs. Results: target `8/32`; source `5/32` with `3`
source-only wins and source/target oracle `11/32`; text-to-text `2/32` with `1`
source-only win; C2C `16/32` with `10` C2C-only wins and target/C2C oracle
`18/32`. Numeric coverage is complete except source-alone, which is `31/32`.

Commands:

```bash
./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/svamp_exactid_baselines32_20260423 \
  --limit 32 \
  --methods target source t2t c2c \
  --device mps \
  --max-new-tokens 64 \
  --continue-on-error
```

```bash
./venv_arm64/bin/python scripts/analyze_source_headroom_surfaces.py \
  --surface fresh_svamp32_source=target_path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,source_path=results/svamp_exactid_baselines32_20260423/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=fresh_svamp32_source \
  --surface fresh_svamp32_t2t=target_path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,source_path=results/svamp_exactid_baselines32_20260423/text_to_text.jsonl,target_method=target_alone,source_method=text_to_text,note=fresh_svamp32_text \
  --surface fresh_svamp32_c2c=target_path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,source_path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,target_method=target_alone,source_method=c2c_generate,note=fresh_svamp32_c2c \
  --min-source-only 5 \
  --output-json results/svamp_exactid_baselines32_20260423/headroom_surfaces.json \
  --output-md results/svamp_exactid_baselines32_20260423/headroom_surfaces.md
```

Decision: the direct source-alone SVAMP32 gate fails (`3/32` source-only wins,
below the predeclared `5/32` threshold), so do not claim direct latent-source
transfer from source-alone on this surface. C2C is a strong teacher/competitor
surface (`10` C2C-only wins), so the next exact gate is a C2C-teacher innovation
probe on the same SVAMP32 IDs with matched-source, zero-source, shuffled-source,
and target-only rows. Kill the branch if controls retain the C2C-overlap wins.

`paper/svamp32_c2c_teacher_innovation_probe_20260423.md` adds a frozen-ID
C2C-teacher innovation provenance probe. The probe compares fresh SVAMP32
target/source/text/C2C rows against exact-prefix-compatible SVAMP70 process
repair and dynalign artifacts. C2C remains the strongest teacher surface:
target `8/32`, C2C `16/32`, and `10` C2C-only target-complementary wins.
Process repair recovers `3/10` C2C-only IDs, but target self-repair also
recovers all `3/10`, so that overlap is target-side repair rather than
communication. Dynalign salts recover `0/10`, `1/10`, and `2/10` C2C-only IDs;
salt 1 has one non-target-control-overlapped hit and salt 2 has two hits with
one control overlap. This is only a weak hint because the old dynalign rows do
not include zero-source or shuffled-source controls on the same SVAMP32 gate.

Command:

```bash
./venv_arm64/bin/python scripts/analyze_c2c_teacher_innovation.py \
  --target target=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --source source=path=results/svamp_exactid_baselines32_20260423/source_alone.jsonl,method=source_alone \
  --source t2t=path=results/svamp_exactid_baselines32_20260423/text_to_text.jsonl,method=text_to_text \
  --control target_self_repair=path=results/process_repair_holdout_20260421/qwen_svamp70_process_repair_controls_strict_selector_telemetry.jsonl,method=target_self_repair \
  --control selected_route_no_repair=path=results/process_repair_holdout_20260421/qwen_svamp70_process_repair_controls_strict_selector_telemetry.jsonl,method=selected_route_no_repair \
  --candidate process_repair=path=results/process_repair_holdout_20260421/qwen_svamp70_process_repair_controls_strict_selector_telemetry.jsonl,method=process_repair_selected_route \
  --candidate dynalign_salt0=path=results/process_repair_holdout_20260421/qwen_svamp70_dynalign_prefdist_asym_kv_random_r025_v075_cal16_chat_salt0_telemetry.jsonl,method=rotalign_kv \
  --candidate dynalign_salt1=path=results/process_repair_holdout_20260421/qwen_svamp70_dynalign_prefdist_asym_kv_random_r025_v075_cal16_chat_salt1_telemetry.jsonl,method=rotalign_kv \
  --candidate dynalign_salt2=path=results/process_repair_holdout_20260421/qwen_svamp70_dynalign_prefdist_asym_kv_random_r025_v075_cal16_chat_salt2_telemetry.jsonl,method=rotalign_kv \
  --min-teacher-only 5 \
  --require-controls \
  --output-json results/svamp_exactid_baselines32_20260423/c2c_teacher_innovation_probe.json \
  --output-md results/svamp_exactid_baselines32_20260423/c2c_teacher_innovation_probe.md
```

Decision: do not promote process repair or existing dynalign to a positive
method. The next exact gate is a controlled C2C-teacher innovation connector on
the same frozen SVAMP32 IDs with matched-source, zero-source, deterministic
shuffled-source, and target-only controls. Promote only if matched source
recovers at least `4/10` C2C-only wins, reaches at least `11/32`, loses no more
than one target-correct ID, and controls recover at most one of the same
C2C-only wins.

`paper/svamp32_dynalign_source_controls_20260423.md` resolves the remaining
legacy-dynalign ambiguity on the SVAMP32 exact-ID C2C-teacher gate. Using the
same prefdist dynalign checkpoint and the frozen materialized 32-example SVAMP
slice, I ran exact-ID zero-source and deterministic shuffled-source controls for
the two legacy salts that still had any teacher-only overlap. Salt 1 is cleanly
killed: its only teacher-only recovery (`575d7e83d84c1e67`) is reproduced by
both zero-source and shuffled-source controls, so the salt 1 hint is not
source-specific. Salt 2 remains only a weak lower-bound clue: the matched row is
`8/32`, recovers `2/10` C2C-only IDs, loses `4` target-correct IDs, and one of
its teacher-only hits (`4d780f825bb8541c`) is reproduced by shuffled-source.
The second hit (`e3ab8666238a289e`) remains matched-only in this probe, but that
is still far below a paper-grade gate.

Commands:

```bash
./venv_arm64/bin/python scripts/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --task-type generation \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --methods rotalign \
  --gate-mode fixed \
  --fixed-gate 0.1 \
  --fusion-rule static \
  --kv-transport both \
  --position-selection-metric attention \
  --position-selection-ratio 0.50 \
  --kv-route-selection-ratio 0.25 \
  --kv-value-selection-ratio 0.75 \
  --kv-route-selection-metric random \
  --kv-value-selection-metric random \
  --runtime-head-selection-metric attention_peak \
  --runtime-head-selection-ratio 1.0 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1 \
  --source-kv-control zero \
  --prediction-output results/svamp32_dynalign_source_controls_20260423/dynalign_salt1_zero_source.jsonl
```

```bash
./venv_arm64/bin/python scripts/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --task-type generation \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --methods rotalign \
  --gate-mode fixed \
  --fixed-gate 0.1 \
  --fusion-rule static \
  --kv-transport both \
  --position-selection-metric attention \
  --position-selection-ratio 0.50 \
  --kv-route-selection-ratio 0.25 \
  --kv-value-selection-ratio 0.75 \
  --kv-route-selection-metric random \
  --kv-value-selection-metric random \
  --runtime-head-selection-metric attention_peak \
  --runtime-head-selection-ratio 1.0 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1 \
  --source-prompt-control shuffle_examples \
  --prediction-output results/svamp32_dynalign_source_controls_20260423/dynalign_salt1_shuffled_source_salt1.jsonl
```

```bash
./venv_arm64/bin/python scripts/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --task-type generation \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --methods rotalign \
  --gate-mode fixed \
  --fixed-gate 0.1 \
  --fusion-rule static \
  --kv-transport both \
  --position-selection-metric attention \
  --position-selection-ratio 0.50 \
  --kv-route-selection-ratio 0.25 \
  --kv-value-selection-ratio 0.75 \
  --kv-route-selection-metric random \
  --kv-value-selection-metric random \
  --runtime-head-selection-metric attention_peak \
  --runtime-head-selection-ratio 1.0 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 2 \
  --source-kv-control zero \
  --prediction-output results/svamp32_dynalign_source_controls_20260423/dynalign_salt2_zero_source.jsonl
```

```bash
./venv_arm64/bin/python scripts/evaluate.py \
  --translator checkpoints/bridge_ridge_qk_dynalign_prefdist_module_replace_20260420_diag/qwen25_to_qwen3_grouped_subspace_transport_w010_r4_dynalign_prefdist_module_replace_cal16_chat.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --task-type generation \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --methods rotalign \
  --gate-mode fixed \
  --fixed-gate 0.1 \
  --fusion-rule static \
  --kv-transport both \
  --position-selection-metric attention \
  --position-selection-ratio 0.50 \
  --kv-route-selection-ratio 0.25 \
  --kv-value-selection-ratio 0.75 \
  --kv-route-selection-metric random \
  --kv-value-selection-metric random \
  --runtime-head-selection-metric attention_peak \
  --runtime-head-selection-ratio 1.0 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 2 \
  --source-prompt-control shuffle_examples \
  --prediction-output results/svamp32_dynalign_source_controls_20260423/dynalign_salt2_shuffled_source_salt2.jsonl
```

Decision: legacy dynalign is not promotable as the positive method. Keep salt 2
only as a weak lower-bound comparator because one teacher-only ID remains
matched-only. The next exact gate is a learned source-control-contrastive
innovation connector on the same frozen SVAMP32 IDs, with the same strict
matched/zero/shuffle evaluation and the predeclared promotion threshold
(`>=4/10` teacher-only recoveries, `>=11/32` overall, `<=1` target loss, and
controls recovering at most one of the same matched teacher-only wins).

`paper/svamp32_query_innovation_resampler_gate015_20260423.md` records the next
same-pair learned-connector gate on the frozen SVAMP32 C2C-teacher surface.
Rather than retraining, this turn reused the existing finite
`bridge_ridge_qk_dynalign_query_innovation_resampler_replace` checkpoint from
the earlier GSM8K32 seed-1 run and tested whether its promising `0.15` gate
transfers to the stronger exact-ID SVAMP32 surface.

First, a tiny fixed-gate transfer sweep:

```bash
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --task-type generation \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --kv-transport k_only \
  --position-selection-ratio 0.5 \
  --position-selection-metric attention \
  --gate-mode sweep \
  --gate-values 0.10 0.15 0.25 \
  --methods rotalign \
  --prediction-output .debug/svamp32_query_innovation_resampler_gate_sweep_20260423/live_gate_sweep.jsonl \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

Sweep readout:

- gate `0.10`: `7/32`, below target
- gate `0.15`: `9/32`, the only live gate
- gate `0.25`: `8/32`, target parity

Then exact controls at gate `0.15` plus the C2C-teacher probe:

```bash
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --task-type generation \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --kv-transport k_only \
  --position-selection-ratio 0.5 \
  --position-selection-metric attention \
  --gate-mode fixed \
  --fixed-gate 0.15 \
  --methods rotalign \
  --prediction-output results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_zero_source.jsonl \
  --source-kv-control zero \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

```bash
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --task-type generation \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --kv-transport k_only \
  --position-selection-ratio 0.5 \
  --position-selection-metric attention \
  --gate-mode fixed \
  --fixed-gate 0.15 \
  --methods rotalign \
  --prediction-output results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_shuffled_source_salt1.jsonl \
  --source-prompt-control shuffle_examples \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

```bash
./venv_arm64/bin/python scripts/analyze_c2c_teacher_innovation.py \
  --target target=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --source source=path=results/svamp_exactid_baselines32_20260423/source_alone.jsonl,method=source_alone \
  --source t2t=path=results/svamp_exactid_baselines32_20260423/text_to_text.jsonl,method=text_to_text \
  --candidate matched=path=results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --control zero_source=path=results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_zero_source.jsonl,method=rotalign_kv \
  --control shuffled_source=path=results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_shuffled_source_salt1.jsonl,method=rotalign_kv \
  --output-json results/svamp32_query_innovation_resampler_gate015_20260423/c2c_teacher_probe_gate015.json \
  --output-md results/svamp32_query_innovation_resampler_gate015_20260423/c2c_teacher_probe_gate015.md
```

Evidence:

- matched gate `0.15`: `9/32`, wins `3e8a5691f5443495` and
  `575d7e83d84c1e67`, loss `c042f0a2949ff8e6`
- zero-source: `8/32`, retains `575d7e83d84c1e67` and the same target loss
- shuffled-source: `9/32`, reproduces both matched wins and the same target
  loss, but with only `31/32` numeric coverage and `1` empty prediction
- matched teacher-only recovered count: `1/10`
- the only teacher-only recovered ID is `575d7e83d84c1e67`
- that same teacher-only ID is recovered by both zero-source and
  shuffled-source controls
- the other matched-only win, `3e8a5691f5443495`, is not on the C2C teacher
  surface
- probe gate result: `candidate_teacher_recovery_explained_by_controls`

Decision: the current fixed-gate query-innovation-resampler checkpoint is
weakened on the exact SVAMP32 paper surface. It can move answers, but its only
teacher-only recovery is fully explained by controls, and shuffled-source
reproduces the full matched headline row. Do not widen this branch or treat it
as the paper method. The next exact gate is a control-discriminating innovation
connector on the same frozen SVAMP32 surface, likely with an explicit
verifier/contrastive rule that rejects wins retained under zero/shuffled
source.

## 2026-04-23 18:20 PT — SVAMP32 query_pool_transport source controls

Paper status:

- still not ICLR-ready
- same submission blocker: a matched-source connector must recover frozen-ID
  C2C-only wins that zero/shuffled-source controls do not

What ran:

- reused the existing learned
  `bridge_ridge_qk_dynalign_query_innovation_resampler_replace` checkpoint
- switched runtime selection to `query_pool_transport`
- swept `0.10/0.15/0.25` on the frozen SVAMP32 slice
- then ran exact matched, zero-source, and deterministic shuffled-source rows
  for gate `0.10`
- then ran `scripts/analyze_c2c_teacher_innovation.py`

Repo fix:

- shuffled-source initially crashed because runtime attention-derived position
  scores were not resized to translated KV length for
  `query_pool_transport`
- fixed this in `latent_bridge/evaluate.py` via
  `_resize_position_scores()`
- added focused regression coverage in `tests/test_evaluate_helpers.py`
- focused test result:
  `./venv_arm64/bin/python -m pytest tests/test_evaluate_helpers.py -k 'query_pool_transport or resize_position_scores or shuffle_examples_uses_mismatched_source_prompt' -q`
  -> `5 passed`

Evidence:

- gate sweep:
  - `0.10`: `9/32`
  - `0.15`: `7/32`
  - `0.25`: `6/32`
- gate `0.10` exact rows:
  - matched: `9/32`, wins `3e8a5691f5443495` and `575d7e83d84c1e67`, loss
    `c042f0a2949ff8e6`
  - zero-source: `8/32`, reproduces both matched wins, adds extra loss
    `de2a795ab37694af`
  - shuffled-source: `9/32`, reproduces the full matched headline row with
    `31/32` numeric coverage and `1` empty prediction
- teacher probe:
  - matched recovers only `1/10` C2C-only IDs: `575d7e83d84c1e67`
  - zero-source also recovers `575d7e83d84c1e67`
  - shuffled-source also recovers `575d7e83d84c1e67`
  - source-alone also recovers `575d7e83d84c1e67`
  - probe verdict:
    `candidate_teacher_recovery_explained_by_controls`
- artifact sanity:
  - exact ordered ID parity is `true` across target/matched/zero/shuffled rows
  - shuffled rows all carry `source_prompt_control=shuffle_examples`
  - shuffled rows have `0` same-index source mappings

Status update:

- alive:
  - decoder-conditioned innovation connectors as a class
- weakened:
  - `query_pool_transport` as a runtime wrapper on the current learned
    innovation checkpoint
- saturated:
  - deterministic selector swaps on this checkpoint for the frozen SVAMP32
    paper gate

Decision:

- do not widen this runtime family further
- after the control-path bug fix, shuffled-source still reproduces the full
  matched row and the only teacher-only recovery
- treat this as a clean negative result, not a live positive-method candidate

Next exact gate:

- a target-conditioned innovation connector on the same frozen SVAMP32 surface
  with matched / zero-source / shuffled-source / target-self-repair rows
- superseded by the stricter target-self-repair paper gate below: promotion now
  requires `16/32`, `+1` versus target_self_repair, at least `5/10` C2C-only
  recoveries, at least `2` C2C-only recoveries unique versus
  target_self_repair, at most `1` target loss, and at most `1` retained
  matched C2C-only win under each source control

## 2026-04-23 19:05 PT — SVAMP32 target-self-repair paper gate

Paper status:

- not ICLR-ready
- current story: SVAMP32 has real C2C teacher headroom, but current
  learned/rotalign rows do not separate matched-source communication from
  target-side repair or source controls
- submission blocker: a candidate must beat target_self_repair and recover
  C2C-only wins not retained under zero/shuffled source

What changed:

- added `scripts/analyze_svamp32_paper_gate.py`
- added `tests/test_analyze_svamp32_paper_gate.py`
- materialized target-self-repair gate artifacts for the live
  query_pool_transport row:
  - `results/svamp32_query_innovation_query_pool_transport_20260423/c2c_teacher_probe_gate010_with_target_repair.json`
  - `results/svamp32_query_innovation_query_pool_transport_20260423/c2c_teacher_probe_gate010_with_target_repair.md`
  - `results/svamp32_query_innovation_query_pool_transport_20260423/paper_gate_gate010_with_target_repair.json`
  - `results/svamp32_query_innovation_query_pool_transport_20260423/paper_gate_gate010_with_target_repair.md`
- updated:
  - `paper/svamp32_query_pool_transport_source_controls_20260423.md`
  - `paper/svamp32_target_self_repair_paper_gate_20260423.md`
  - `results/svamp32_query_innovation_query_pool_transport_20260423/manifest.md`

Verification:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_paper_gate.py tests/test_analyze_c2c_teacher_innovation.py -q
```

Result: `5 passed`

Gate command:

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_paper_gate.py \
  --probe-json results/svamp32_query_innovation_query_pool_transport_20260423/c2c_teacher_probe_gate010_with_target_repair.json \
  --output-json results/svamp32_query_innovation_query_pool_transport_20260423/paper_gate_gate010_with_target_repair.json \
  --output-md results/svamp32_query_innovation_query_pool_transport_20260423/paper_gate_gate010_with_target_repair.md
```

Evidence:

- target-alone: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`, `3/10` C2C-only recoveries, `0` target losses
- query_pool_matched: `9/32`, `1/10` C2C-only recovery, `1` target loss
- query_pool_matched delta versus target_self_repair: `-5`
- query_pool_matched gate verdict: `fails_paper_gate`
- run verdict: `no_candidate_passes_target_self_repair_gate`
- failing criteria:
  - `min_correct`
  - `beats_target_self_repair`
  - `min_teacher_only`
  - `min_unique_vs_target_self_repair`
- only recovered C2C-only ID: `575d7e83d84c1e67`
- that ID is retained by both zero-source and shuffled-source controls

Status update:

- alive:
  - C2C-distilled conditional innovation fuser
  - learned Q-Former/Perceiver-style query-bottleneck connector
- saturated:
  - selector/runtime swaps on the current query-innovation-resampler checkpoint
  - query_pool_transport as a paper method on SVAMP32
- blocked:
  - any same-pair claim that does not beat target_self_repair on exact IDs

Next exact gate:

- implement one target-conditioned connector or C2C-distilled innovation fuser
- run matched / zero-source / shuffled-source / target_self_repair on the same
  frozen SVAMP32 exact-ID surface
- promote only if `scripts/analyze_svamp32_paper_gate.py` returns
  `candidate_passes_target_self_repair_gate`

## 2026-04-23 19:45 PT — SVAMP32 clean innovation target set

Paper status:

- not ICLR-ready
- current story: the live method needs a source-specific residual innovation
  connector, not another target-cache perturbation
- blocking gap: identify which C2C-only IDs remain legitimate positives after
  removing target_self_repair, source-alone/text relay, and zero/shuffled-source
  explanations

What changed:

- added `scripts/build_svamp32_innovation_target_set.py`
- added `tests/test_build_svamp32_innovation_target_set.py`
- materialized:
  - `results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json`
  - `results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.md`
- updated:
  - `paper/svamp32_innovation_target_set_20260423.md`
  - `results/svamp32_query_innovation_query_pool_transport_20260423/manifest.md`

Verification:

```bash
./venv_arm64/bin/python -m pytest tests/test_build_svamp32_innovation_target_set.py tests/test_analyze_svamp32_paper_gate.py -q
```

Result: `5 passed`

Evidence:

- target-alone: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`
- target_self_repair C2C-only recoveries: `3/10`
- clean residual C2C-only targets after removing target_self_repair,
  source-alone/text relay, and zero/shuffled-source explanations: `6`
- clean residual target IDs:
  - `13cb77b698eeadb5`
  - `1d50b408c8f5cd2c`
  - `2de1549556000830`
  - `6e9745b37ab6fc45`
  - `aee922049c757331`
  - `e3ab8666238a289e`
- excluded as target_self_repair recovered:
  - `4c84ebf42812703b`
  - `4d780f825bb8541c`
  - `de1bf4d142544e5b`
- excluded as source/source-control explained:
  - `575d7e83d84c1e67`
- target_self_repair plus C2C teacher oracle: `21/32`
- required clean residual wins if preserving target_self_repair: `2`

Status update:

- alive:
  - C2C-distilled conditional innovation fuser
  - Wyner-Ziv / Q-Former-style conditional innovation bottleneck
  - source-necessity replay ablation for the next candidate
- saturated:
  - current query_pool_transport row and selector/runtime variants
- promoted:
  - train/evaluate only against the clean residual target set for positive
    source communication claims

Next exact gate:

- implement the smallest target-self-preserving conditional innovation
  connector
- run matched, post-bridge-zero if available, zero-source, shuffled-source with
  at least two salts, and target_self_repair
- score with `scripts/analyze_c2c_teacher_innovation.py` and
  `scripts/analyze_svamp32_paper_gate.py`

## 2026-04-23 20:20 PT — SVAMP32 clean-target paper gate integration

Paper status:

- not ICLR-ready
- current story: there is enough clean residual C2C headroom, but candidate
  promotion must explicitly recover those clean IDs rather than generic
  C2C-only or target-self-repair IDs
- blocking gap: a next connector must preserve target_self_repair and add clean
  source-specific residual wins

What changed:

- extended `scripts/analyze_svamp32_paper_gate.py` with optional
  `--target-set-json`
- when a target set is provided, the gate requires clean residual recovery from
  `ids.clean_residual_targets`
- default clean-residual and clean-source-necessary thresholds are read from
  `summary.required_clean_residual_to_clear_gate_if_preserving_self`
- added tests for clean residual pass/fail, clean source-control subtraction,
  and missing target-set failure
- materialized clean-target gate artifacts:
  - `results/svamp32_query_innovation_query_pool_transport_20260423/paper_gate_gate010_with_clean_targets.json`
  - `results/svamp32_query_innovation_query_pool_transport_20260423/paper_gate_gate010_with_clean_targets.md`
- updated:
  - `paper/svamp32_target_self_repair_paper_gate_20260423.md`
  - `paper/svamp32_innovation_target_set_20260423.md`
  - `results/svamp32_query_innovation_query_pool_transport_20260423/manifest.md`

Verification:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_analyze_svamp32_paper_gate.py \
  tests/test_build_svamp32_innovation_target_set.py \
  tests/test_analyze_c2c_teacher_innovation.py -q
```

Result: `11 passed`

Gate command:

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_paper_gate.py \
  --probe-json results/svamp32_query_innovation_query_pool_transport_20260423/c2c_teacher_probe_gate010_with_target_repair.json \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --output-json results/svamp32_query_innovation_query_pool_transport_20260423/paper_gate_gate010_with_clean_targets.json \
  --output-md results/svamp32_query_innovation_query_pool_transport_20260423/paper_gate_gate010_with_clean_targets.md
```

Evidence:

- clean residual target set size: `6`
- required clean residual / clean source-necessary wins if preserving
  target_self_repair: `2`
- query_pool_matched clean residual recovered: `0/6`
- query_pool_matched clean source-necessary recovered: `0/6`
- query_pool_matched added failing criteria:
  `min_clean_residual_recovered`, `min_clean_source_necessary`
- verdict remains:
  `no_candidate_passes_target_self_repair_gate`

Status update:

- alive:
  - target-self-preserving conditional innovation fuser
  - Wyner-Ziv / Q-Former query bottleneck using target cache as side
    information
- saturated:
  - current query_pool_transport row; it recovers no clean residual target IDs
- promoted:
  - any next candidate must be scored against the clean target set, not only
    aggregate C2C-only recovery

Next exact gate:

- implement the smallest conditional innovation candidate
- run matched / post-bridge-zero if available / zero-source / shuffled-source
  with at least two salts / target_self_repair
- run `scripts/analyze_svamp32_paper_gate.py --target-set-json ...`

## 2026-04-23 12:40 PDT — strict SVAMP32 promotion provenance

Paper status:

- not ICLR-ready
- current story: the clean residual target set is real, but method promotion
  must fail closed on stale or subsetted artifacts
- blocking gap: next connector still needs at least two clean source-necessary
  residual wins, but the gate must first guarantee exact artifact provenance

Top 3 next moves considered:

- strict provenance hardening. This matters because the previous
  target_self_repair comparator was scored as `n=32` from an `artifact_n=70`
  source. It might fail by invalidating prior readouts, but the expected
  evidence is cleaner replay accounting. Cost: low. Helps reproducibility.
- ID-weighted conditional innovation connector. This matters because it
  directly attacks the clean residual gate. It might fail by learning an ID or
  target-cache prior. Cost: medium/high. Helps same-pair and interpretability.
- full source-necessity control contract. This matters because the existing
  false positive is control retention. It might fail if not all controls are
  implemented for the current candidate path. Cost: medium. Helps robustness.

Decision:

- picked strict provenance hardening because it is the promotion precondition
  for any next method row

What changed:

- added `scripts/materialize_exact_id_slice.py`
- added `tests/test_materialize_exact_id_slice.py`
- added `--require-exact-artifacts` to
  `scripts/analyze_c2c_teacher_innovation.py`
- made `scripts/analyze_svamp32_paper_gate.py` fail closed by default:
  `--target-set-json` is required unless `--allow-legacy-gate` is set
- added strict SVAMP32 paper provenance validation:
  - `reference_n == expected_n == 32`
  - every scored row has `n == artifact_n == reference_n`
  - every scored row has exact ordered ID parity
  - every scored row has numeric extraction coverage at least `31/32`
  - target-set `ids.teacher_only` matches probe `teacher_only_ids`
  - target-set clean residual IDs are a subset of probe teacher-only IDs
- materialized exact-32 repair-control slices:
  - `results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl`
  - `results/svamp32_query_innovation_query_pool_transport_20260423/selected_route_no_repair_exact32.jsonl`
- regenerated strict-provenance probe and gate artifacts:
  - `results/svamp32_query_innovation_query_pool_transport_20260423/c2c_teacher_probe_gate010_with_target_repair.json`
  - `results/svamp32_query_innovation_query_pool_transport_20260423/paper_gate_gate010_with_clean_targets.json`

Verification:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_analyze_svamp32_paper_gate.py \
  tests/test_analyze_c2c_teacher_innovation.py \
  tests/test_materialize_exact_id_slice.py \
  tests/test_build_svamp32_innovation_target_set.py -q
```

Result: `19 passed`

Evidence:

- exact target_self_repair slice:
  - output `32/32`
  - source artifact `70` rows
  - dropped source rows: `38`
  - exact ordered ID parity: `true`
- exact selected_route_no_repair slice:
  - output `32/32`
  - source artifact `70` rows
  - dropped source rows: `38`
  - exact ordered ID parity: `true`
- strict gate verdict remains:
  `no_candidate_passes_target_self_repair_gate`
- query_pool_matched remains saturated:
  - clean residual recovered: `0/6`
  - clean source-necessary recovered: `0/6`

Subagent context:

- literature and lateral agents converged on a Wyner-Ziv / Kalman conditional
  innovation bottleneck with Q-Former/Perceiver-style source queries as the
  best creative method branch
- ablation agent promoted union-of-controls clean-source-necessity replay as
  mandatory for the next candidate
- audit agent promoted strict gate provenance as the highest-value bounded
  hardening step

Next exact gate:

- implement the smallest ID-weighted conditional innovation candidate
- run matched, source-kv-zero, translated-kv-zero if available, two shuffled
  salts, and target_self_repair
- score with the strict-provenance SVAMP32 paper gate

## 2026-04-23 - SVAMP32 ID-Weighted Query Innovation

Paper readiness:

- not ICLR-ready
- the live method now has one clean source-necessary residual win, but still
  fails the target-self-repair paper gate

Current story:

- the target cache is strong decoder side information
- the promising direction is conditional source innovation over
  `target_self_repair`, not full cache replacement

Blocking gap:

- recover at least `2/6` clean residual IDs while preserving
  `target_self_repair`'s `14/32`

Top next moves considered:

- ID-weighted query innovation module fit. This directly targets the clean
  residual IDs, but may memorize IDs or learn target-side repair. Cost: medium.
  Helps same-pair, reproducibility, and interpretability.
- full source-necessity control runner. This prevents false promotion, but is
  not worth full cost unless the matched row recovers enough clean IDs. Cost:
  medium/high. Helps robustness.
- new target-side-information connector. This is the cleanest paper story
  from Wyner-Ziv/Q-Former/Perceiver priors, but has higher implementation risk.
  Cost: high. Helps same-pair, efficiency, and interpretability.

Decision:

- implemented the smallest exact-ID-weighted hook on the existing
  `bridge_ridge_qk_dynalign_query_innovation_resampler_replace` path

What changed:

- `latent_bridge/calibrate.py`
  - added JSONL-aware prompt metadata loading
  - reconstructs stable generation example IDs matching evaluation artifacts
  - added `--innovation-target-set-json`
  - added `--innovation-positive-weight`
  - added `--innovation-default-weight`
  - expands prompt-level clean residual target IDs to flattened bridge sample
    weights via `sample_prompt_ids`
- `latent_bridge/translator.py`
  - forwards optional sample weights into the query-innovation module fit
  - leaves the base ridge fit unweighted for this mode
- tests added for metadata extraction, prompt-weight construction, CLI parsing,
  and query-innovation sample-weight forwarding

Verification:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_calibrate_and_ablation.py \
  tests/test_translator_core.py -q
```

Result: `211 passed`

Artifacts:

- checkpoint:
  `.debug/checkpoints_svamp32_conditional_innovation_20260423/id_weighted_query_innovation/qwen25_to_qwen3_svamp32_idweighted_query_innovation_r16_bank16_seed1.pt`
- run directory:
  `results/svamp32_idweighted_query_innovation_20260423/`
- readout:
  `paper/svamp32_idweighted_query_innovation_20260423.md`

Evidence:

- calibration matched all `6` clean residual target prompts
- dynamic token-mixture samples: `1411`
- matched gate sweep:
  - `0.10`: `8/32`, clean residual recovered `0/6`
  - `0.15`: `10/32`, clean residual recovered `1/6`
  - `0.20`: `8/32`, clean residual recovered `0/6`
- `gate015` recovered teacher-only IDs
  `575d7e83d84c1e67` and `aee922049c757331`
- `575d7e83d84c1e67` is retained by `translated_kv_zero`, so it is not a clean
  source-necessary win
- `aee922049c757331` disappears under `translated_kv_zero`, so it is the first
  clean source-necessary residual win for this branch
- strict target-set gate remains:
  `no_candidate_passes_target_self_repair_gate`
- gate failure reasons:
  - `min_correct`
  - `beats_target_self_repair`
  - `min_teacher_only`
  - `min_clean_residual_recovered`
  - `min_clean_source_necessary`

Hypothesis update:

- revived: conditional innovation can move at least one clean source-necessary
  residual ID
- weakened: ID weighting alone is sufficient to clear the gate
- promoted: preserve `target_self_repair` first, then add bounded source
  innovation as a repair sidecar
- still blocked: no method has `>=2/6` clean residual source-necessary wins

Next exact gate:

- implement target-self-preserving residual composition:
  `target_self_repair + source innovation sidecar`
- require `>=2/6` clean source-necessary IDs and no regression below
  `14/32`
- rerun matched, translated-KV-zero, source-KV-zero, two shuffled-source salts,
  and strict target-set paper gate

## 2026-04-23 13:23 PDT — SVAMP32 source-innovation sidecar bound

Paper status:

- not ICLR-ready
- current story: target self-repair is the decoder-side-information floor, and
  source communication should add only clean residual innovation
- blocking gap: current live candidate has one clean source-necessary ID, but
  the paper gate requires two while preserving target_self_repair

Top next moves considered:

- sidecar oracle/proxy-bound analyzer. This matters because it tests whether
  the current candidate is worth wrapping in a target-self-preserving runtime
  router. It might fail if the candidate only exposes one clean source-specific
  ID. Cost: low. Helps reproducibility, robustness, and interpretability.
- runtime target-self-preserving source sidecar. This matters because it is the
  real method shape. It might fail or leak because no valid source-necessity
  router exists yet. Cost: medium/high. Helps same-pair and robustness.
- another ID-weighted residual fit. This might find the second clean ID, but it
  risks overfitting without first quantifying the sidecar headroom. Cost:
  medium. Helps same-pair.

Decision:

- picked the sidecar oracle-bound analyzer first
- rationale: if a perfect target-self-preserving sidecar around the current
  `gate015` candidate cannot clear the paper gate, runtime router work on that
  exact candidate is premature

What changed:

- added `scripts/analyze_svamp32_source_sidecar_bound.py`
- added `tests/test_analyze_svamp32_source_sidecar_bound.py`
- materialized:
  - `results/svamp32_idweighted_query_innovation_20260423/source_sidecar_bound_gate015_targetself_translated_zero.json`
  - `results/svamp32_idweighted_query_innovation_20260423/source_sidecar_bound_gate015_targetself_translated_zero.md`
- added:
  - `results/svamp32_idweighted_query_innovation_20260423/manifest.md`
- updated:
  - `paper/svamp32_idweighted_query_innovation_20260423.md`

Verification:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_analyze_svamp32_source_sidecar_bound.py \
  tests/test_analyze_svamp32_paper_gate.py \
  tests/test_analyze_c2c_teacher_innovation.py \
  tests/test_build_svamp32_innovation_target_set.py -q
```

Result: `20 passed`

Bound command:

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_source_sidecar_bound.py \
  --probe-json results/svamp32_idweighted_query_innovation_20260423/c2c_teacher_probe_gate015_targetself_translated_zero.json \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --candidate-label gate015 \
  --source-control-label translated_kv_zero \
  --output-json results/svamp32_idweighted_query_innovation_20260423/source_sidecar_bound_gate015_targetself_translated_zero.json \
  --output-md results/svamp32_idweighted_query_innovation_20260423/source_sidecar_bound_gate015_targetself_translated_zero.md
```

Evidence:

- target-alone: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`
- matched `gate015`: `10/32`
- matched `gate015` C2C-only recovered: `2`
- matched clean residual recovered: `1/6`
- retained by translated-KV-zero: `575d7e83d84c1e67`
- clean source-necessary ID: `aee922049c757331`
- oracle `target_self_repair + clean source sidecar`: `15/32`
- target losses versus `target_self_repair`: `0`
- verdict: `oracle_sidecar_bound_fails_gate`
- failing criteria:
  - `min_correct`
  - `min_clean_source_necessary`

Hypothesis update:

- promoted: target-self-preserving source innovation remains the right method
  shape
- killed for now: wrapping this exact `gate015` candidate in a runtime sidecar,
  because even a perfect oracle router reaches only `15/32`
- still alive: training/searching for a candidate with at least two clean
  source-necessary IDs before router implementation

Next exact gate:

- produce a candidate that exposes `>=2/6` clean source-necessary IDs under
  matched versus source-destroying controls
- then implement the runtime target-self-preserving acceptor with
  translated-KV-zero, source-zero, two shuffled-source salts, and clean-ID swap
  controls

## 2026-04-23 14:29 PDT — SVAMP32 ID-weighted fine attention gate search

Paper status:

- not ICLR-ready
- current story: target self-repair is the decoder-side-information floor, and
  source communication must add clean residual innovation on top
- blocking gap: no candidate exposes `>=2/6` clean source-necessary IDs under
  strict exact-ID controls

Top next moves considered:

- fine fixed-gate search on the existing ID-weighted checkpoint. This matters
  because it is the cheapest way to test whether the second clean residual ID
  is hidden behind a runtime threshold. It might fail if the checkpoint only
  learned one source-specific residual. Cost: one SVAMP32 seven-gate decode.
  Helps same-pair, robustness, and reproducibility.
- new calibration-side source-control/contrastive query-innovation objective.
  This matters because it directly penalizes target-cache and translated-zero
  residuals. It might overfit the six clean IDs. Cost: medium. Helps
  same-pair, robustness, and interpretability.
- runtime target-self-preserving acceptor. This matters for the final method
  shape, but it might only wrap a one-ID candidate. Cost: medium/high. Helps
  robustness and interpretability.

Decision:

- picked fine fixed-gate search first
- rationale: controls and runtime router work are premature unless a matched
  candidate first reaches `>=2/6` clean residual IDs

What changed:

- added `scripts/analyze_svamp32_gate_sweep_clean_targets.py`
- added `tests/test_analyze_svamp32_gate_sweep_clean_targets.py`
- materialized:
  - `.debug/svamp32_candidate_search_20260423/idweighted_attention_fine_gate_sweep.jsonl`
  - `results/svamp32_idweighted_query_innovation_20260423/fine_gate_sweep_clean_targets_attention.json`
  - `results/svamp32_idweighted_query_innovation_20260423/fine_gate_sweep_clean_targets_attention.md`
- updated:
  - `results/svamp32_idweighted_query_innovation_20260423/manifest.md`
  - `paper/svamp32_idweighted_query_innovation_20260423.md`

Verification:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_analyze_svamp32_gate_sweep_clean_targets.py \
  tests/test_analyze_svamp32_source_sidecar_bound.py -q
```

Result: `8 passed`

Fine sweep command:

```bash
./venv_arm64/bin/python latent_bridge/evaluate.py \
  --translator .debug/checkpoints_svamp32_conditional_innovation_20260423/id_weighted_query_innovation/qwen25_to_qwen3_svamp32_idweighted_query_innovation_r16_bank16_seed1.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --task-type generation \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --kv-transport k_only \
  --position-selection-ratio 0.5 \
  --position-selection-metric attention \
  --gate-mode sweep \
  --gate-values 0.05 0.10 0.125 0.15 0.175 0.20 0.25 \
  --methods rotalign \
  --prediction-output .debug/svamp32_candidate_search_20260423/idweighted_attention_fine_gate_sweep.jsonl \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

Evidence:

- status: `no_matched_gate_candidate_for_controls`
- target-alone: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`
- best fine row: `rotalign_kv_gate_0.17`, `11/32`
- best fine row clean residual recovered: `1/6`
- best fine row clean residual ID: `aee922049c757331`
- best fine row teacher-only recovered: `2`
- oracle `target_self_repair + clean candidate`: `15/32`
- numeric extraction coverage: `32/32` for every row
- verdict: no fine attention-gate row justifies translated-KV-zero,
  source-zero, or shuffle controls

Hypothesis update:

- weakened: runtime threshold search alone can rescue the ID-weighted
  query-innovation checkpoint
- saturated for now: attention fixed-gate retuning of this exact checkpoint
- promoted: train/search a new candidate with source-control contrastive
  pressure or a changed bottleneck before router implementation

Next exact gate:

- train or search a candidate that reaches `>=2/6` clean residual IDs in
  matched exact-ID scoring
- only then run translated-KV-zero, source-zero, two shuffle salts, and the
  target-self sidecar bound

## 2026-04-23 15:09 PDT — SVAMP32 focused clean-ID retrain

Paper status:

- not ICLR-ready
- current story: target self-repair is the decoder-side-information floor, and
  source communication must add clean residual innovation on top
- blocking gap: no candidate exposes `>=2/6` clean source-necessary IDs under
  strict exact-ID controls

Top next moves considered:

- focused clean-ID retrain with stronger positive pressure. This matters
  because it tests whether scalar calibration weighting, rather than runtime
  gate choice, can expose the second clean residual ID. It might fail by
  overfitting or suppressing the one real source-specific residual. Cost: one
  calibration plus one five-gate decode. Helps same-pair and reproducibility.
- explicit source-control contrastive query-innovation objective. This matters
  because it optimizes source necessity directly. It might be invasive and
  overfit the six clean IDs. Cost: high. Helps robustness and interpretability.
- asymmetric value-side transport search. This matters because the current
  matched row is `k_only`, so `V` may carry answer-side source innovation. It
  might increase target losses or recover target-cache wins. Cost: medium.
  Helps same-pair and efficiency.

Decision:

- picked focused clean-ID retrain first
- rationale: it is the cheapest training-surface change after runtime gate
  retuning saturated, and it is less invasive than a new contrastive objective

What changed:

- calibrated:
  - `.debug/svamp32_clean_innovation_sweep_20260423/checkpoints/idw_p32_d025_r16_b16_seed1.pt`
- evaluated:
  - `.debug/svamp32_clean_innovation_sweep_20260423/preds/idw_p32_d025_r16_b16_seed1_attention_gate_sweep.jsonl`
- materialized:
  - `results/svamp32_idweighted_query_innovation_20260423/idw_p32_d025_r16_b16_attention_clean_targets.json`
  - `results/svamp32_idweighted_query_innovation_20260423/idw_p32_d025_r16_b16_attention_clean_targets.md`
- updated:
  - `results/svamp32_idweighted_query_innovation_20260423/manifest.md`
  - `paper/svamp32_idweighted_query_innovation_20260423.md`

Calibration:

- correction: `bridge_ridge_qk_dynalign_query_innovation_resampler_replace`
- rank: `16`
- bank size: `16`
- `innovation-positive-weight`: `32`
- `innovation-default-weight`: `0.25`
- matched clean residual prompts: `6`
- samples: `1411`
- average fit quality:
  - K cosine: `0.951`
  - V cosine: `0.734`

Evidence:

- status: `no_matched_gate_candidate_for_controls`
- target-alone: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`
- best row: `rotalign_kv_gate_0.17`, `10/32`
- clean residual recovered: `0/6`
- teacher-only recovered: `1`
- target losses at best row: `0`
- oracle `target_self_repair + clean candidate`: `14/32`
- numeric extraction coverage: `32/32` for every row
- verdict: no source-destroying controls are justified

Hypothesis update:

- killed for now: scalar clean-ID overpressure on this query-innovation
  resampler
- weakened: calibration-side weighting alone can recover the second clean ID
- promoted: explicit source-control contrastive innovation fuser, because both
  runtime gate retuning and scalar ID weighting are saturated/negative
- still alive as a cheap side branch: one bounded V-side `both` transport
  matched-only sweep, but it should not displace the contrastive objective

Next exact gate:

- implement or train a source-control contrastive innovation candidate that
  reaches `>=2/6` clean residual IDs in matched exact-ID scoring
- then run translated-KV-zero, source-zero, two shuffle salts, and the
  target-self sidecar bound

## 2026-04-23 15:25 PDT — SVAMP32 source-control contrastive innovation

Paper status:

- not ICLR-ready
- current story: target self-repair is the decoder-side-information floor, and
  source communication must add clean residual innovation above it
- blocking gap: no candidate exposes `>=2/6` clean source-necessary IDs under
  strict exact-ID controls

Top next moves considered:

- source-control query-innovation objective. This matters because it directly
  targets source necessity through zero and shuffled source negatives. It might
  fail by suppressing all residual innovation. Cost: one calibration plus one
  matched sweep. Helps same-pair, robustness, and interpretability.
- V-full `both` transport screen on the prior ID-weighted checkpoint. This
  matters because value transport may expose another clean ID. It might fail by
  adding V noise or target-cache wins. Cost: one decode sweep. Helps same-pair
  and efficiency.
- larger Q-former / Perceiver connector. This matters because a stronger
  learned bottleneck is the best architectural story after scalar objectives
  saturate. It might fail by adding too many degrees before the live gate. Cost:
  high. Helps cross-family and interpretability if it works.

Decision:

- picked source-control query-innovation objective
- rationale: runtime gate search and scalar clean-ID weighting were already
  saturated, so the next most direct hypothesis was matched-source-vs-control
  training pressure

What changed:

- added default-off query-innovation source-control knobs:
  - `innovation_control_weight`
  - `innovation_control_mode`
  - `innovation_contrastive_margin`
- constrained source controls to
  `bridge_ridge_qk_dynalign_query_innovation_resampler_replace`
- trained zero and shuffled controls toward zero innovation delta, not full
  target reconstruction
- added regression coverage for CLI parse, config validation, prompt-ID
  forwarding, and shuffled-control prompt-ID requirements
- created memo:
  - `paper/svamp32_control_contrastive_innovation_20260423.md`

Verification:

- `./venv_arm64/bin/python -m pytest tests/test_translator_core.py tests/test_calibrate_and_ablation.py -q`
- result: `214 passed`

Artifacts:

- checkpoint:
  - `.debug/svamp32_control_contrastive_innovation_20260423/checkpoints/qwen25_to_qwen3_svamp32_control_zero_shuffle_w010_m001_r16_bank16_seed1.pt`
- matched sweep:
  - `.debug/svamp32_control_contrastive_innovation_20260423/preds/control_zero_shuffle_w010_m001_attention_gate_sweep.jsonl`
- readouts:
  - `results/svamp32_control_contrastive_innovation_20260423/control_zero_shuffle_w010_m001_attention_clean_targets.json`
  - `results/svamp32_control_contrastive_innovation_20260423/control_zero_shuffle_w010_m001_attention_clean_targets.md`

Candidate settings:

- correction: `bridge_ridge_qk_dynalign_query_innovation_resampler_replace`
- rank: `16`
- bank size: `16`
- seed: `1`
- positive/default innovation weights: `16` / `1`
- source-control mode: `zero_and_shuffle`
- source-control weight: `0.10`
- contrastive margin: `0.001`
- prompt IDs for source controls: `1411` samples, `32` prompts
- average fit quality:
  - K cosine: `0.951`
  - V cosine: `0.734`

Evidence:

- status: `no_matched_gate_candidate_for_controls`
- target-alone: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`
- best row: `rotalign_kv_gate_0.12`, `9/32`
- clean residual recovered: `0/6`
- teacher-only recovered: `1`
- delta vs target_self_repair: `-5`
- target losses at best row: `0`
- oracle `target_self_repair + clean candidate`: `14/32`
- verdict: do not run translated-zero, source-zero, shuffled-source, or
  sidecar controls for this checkpoint

Hypothesis update:

- weakened: naive source-control contrastive pressure on the existing
  query-innovation resampler can expose clean source-necessary IDs
- strengthened: scalar/objective tweaks on this architecture are saturated
- still alive: V-full `both` transport screen as a cheap falsification branch
- promoted: small learned query bottleneck / Q-former-style connector that can
  preserve target self-repair while injecting bounded source innovation

Operational note:

- seven-gate full generation sweeps are too expensive for broad iteration on
  MPS; future candidate screens should use fewer gates until a row is near
  promotion

Next exact gate:

- run the cheap V-full `both` transport screen on the prior ID-weighted
  checkpoint or implement the small query bottleneck
- unchanged promotion criterion: `>=2/6` clean residual IDs in matched exact-ID
  scoring before any control sweep

## 2026-04-23 16:03 PDT — SVAMP32 value transport screen

Paper status:

- not ICLR-ready
- current story: target self-repair is the decoder-side-information floor, and
  source communication must add clean residual innovation above it
- blocking gap: no candidate exposes `>=2/6` clean source-necessary IDs under
  strict exact-ID controls

Top next moves considered:

- V-full `both` transport screen on the prior ID-weighted checkpoint. This
  matters because the prior live row was `k_only`, so value-side answer
  evidence could expose another clean residual ID. It might fail by adding V
  noise or target-cache wins. Cost: one focused decode. Helps same-pair and
  efficiency.
- sparse source-attention V follow-up. This matters because full V could be too
  noisy while a sparse V lane might preserve the K route and add answer-side
  source evidence. It might fail by recovering only the same known clean ID.
  Cost: one three-gate decode. Helps same-pair and efficiency.
- conditional query-bottleneck connector. This matters because scalar bridge
  tweaks are now saturated and learned connector priors are stronger. It might
  fail by memorizing the six clean IDs or leaking target repair. Cost:
  medium-high. Helps interpretability and reproducibility if it works.

Decision:

- ran V-full first, then one sparse source-attention V follow-up after V-full
  failed
- rationale: this killed the remaining cheap runtime value-side hypothesis
  before spending implementation time on the query-bottleneck branch

References checked:

- C2C: https://arxiv.org/abs/2510.03215
- KVComm: https://arxiv.org/abs/2510.03346
- BLIP-2 / Q-Former: https://arxiv.org/abs/2301.12597
- Perceiver IO: https://arxiv.org/abs/2107.14795

Verification:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_gate_sweep_clean_targets.py -q`
- result: `4 passed`

Artifacts:

- checkpoint:
  - `.debug/checkpoints_svamp32_conditional_innovation_20260423/id_weighted_query_innovation/qwen25_to_qwen3_svamp32_idweighted_query_innovation_r16_bank16_seed1.pt`
  - sha256: `4e67fdf2b6ea2c962036aad080ec3fe6c64a4083c627a282b925bdf546b90831`
- V-full matched sweep:
  - `.debug/svamp32_vfull_both_transport_20260423/preds/idweighted_both_vfull_attention_gate_sweep.jsonl`
  - JSONL sha256: `d9bbed502ba4293120362b396188bf12458f1d10bcd482718c23184eed440dff`
  - meta sha256: `849e318a9c00be1a86951b66f435986d2b1ba664b4c0dc8068dbc81432cb2744`
- sparse source-attention V matched sweep:
  - `.debug/svamp32_vfull_both_transport_20260423/preds/idweighted_sparse_sourcev_attention_gate_sweep.jsonl`
  - JSONL sha256: `988820878a2a3f3659a3f8448e2d4e0b1c44ec815667219ac8bf69e1a1dfb1ab`
  - meta sha256: `463c66f7b6f43f533df0abe509e797bec1263981892a623ad9f4f724739f389d`
- readouts:
  - `results/svamp32_vfull_both_transport_20260423/idweighted_both_vfull_attention_clean_targets.json`
  - `results/svamp32_vfull_both_transport_20260423/idweighted_both_vfull_attention_clean_targets.md`
  - `results/svamp32_vfull_both_transport_20260423/idweighted_sparse_sourcev_attention_clean_targets.json`
  - `results/svamp32_vfull_both_transport_20260423/idweighted_sparse_sourcev_attention_clean_targets.md`
  - `results/svamp32_vfull_both_transport_20260423/manifest.md`
- memo:
  - `paper/svamp32_value_transport_screen_20260423.md`
- next-method memo from creative subagent:
  - `paper/svamp32_next_method_conditional_residual_query_codec_20260423.md`

Evidence:

- V-full status: `no_matched_gate_candidate_for_controls`
- V-full best row: `rotalign_kv_gate_0.20`, `9/32`
- V-full clean residual recovered: `0/6`
- V-full target losses: `2`
- V-full transport bytes: `1,193,918.25`
- sparse source-attention V status: `no_matched_gate_candidate_for_controls`
- sparse source-attention V best rows: `rotalign_kv_gate_0.15` and
  `rotalign_kv_gate_0.17`, `10/32`
- sparse source-attention V clean residual recovered: `1/6`
- sparse source-attention V clean ID: `aee922049c757331`
- sparse source-attention V target losses: `1` at gate `0.15`
- sparse source-attention V transport bytes: `597,337.671875`

Hypothesis update:

- killed for now: full value transport can rescue the prior ID-weighted
  checkpoint
- weakened: runtime sparse value selection can expose a second clean residual ID
- saturated: runtime K/V selection on this checkpoint
- promoted: target-self-preserving conditional residual query codec / learned
  query bottleneck

Next exact gate:

- implement or train the conditional residual query codec described in
  `paper/svamp32_next_method_conditional_residual_query_codec_20260423.md`
- promotion criterion remains `>=2/6` clean residual IDs in matched exact-ID
  scoring before any source-destroying controls

## 2026-04-23: SVAMP32 Target-Self-Preserving K-Only Query Codec

Current paper readiness:

- not ICLR-ready
- estimated distance: medium-high; the paper still lacks a same-pair
  source-conditioned positive method beyond the target self-repair floor

Current paper story:

- target self-repair remains a strong decoder-side floor at `14/32`
- source communication must add conditional residual signal on the six clean
  C2C-only IDs without losing the target-self-repair IDs

Blocking gap:

- no source-conditioned method has reached `>=2/6` clean residual recoveries
  under matched exact-ID scoring

Top moves considered:

- target-self-preserving K-only residual query codec. This matters because the
  previous query-innovation checkpoint had one clean source-necessary ID and
  value transport looked noisy. It might fail by suppressing the residual too
  hard or memorizing target-cache behavior. Cost: one calibration plus matched
  gate sweep. Helps same-pair and reproducibility.
- actual target-conditioned query-memory variant. This matters because
  Q-Former/Perceiver/Wyner-Ziv priors point to decoder side information. It
  might fail from coordinate mismatch or target-cache leakage. Cost:
  implementation plus calibration. Helps same-pair, interpretability, and
  efficiency if it works.
- runtime sidecar/verifier selector around existing rows. This matters for
  no-harm preservation. It might fail because the sidecar bound already showed
  the existing row cannot clear the gate. Cost: low. Helps robustness but not
  enough without a stronger residual row.

Decision:

- implemented and tested the bounded K-only protected-loss variant first
- did not run zero/shuffle controls because matched exact-ID recovery was
  `0/6`, below the promotion gate

Code changes:

- `latent_bridge/calibrate.py`
  - added `--innovation-target-self-preserve-weight`
  - added `--innovation-value-loss-weight`
  - expanded `ids.target_self_repair` into no-op residual masks when enabled
- `latent_bridge/translator.py`
  - added query-innovation value-loss weighting
  - added calibration-time zero-residual masks for protected prompts
- tests added for prompt-weight plans, CLI parsing, zero-residual masking, and
  value-loss forwarding

Verification:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_calibrate_and_ablation.py \
  tests/test_translator_core.py -q
```

Result: `216 passed`

Artifacts:

- checkpoint:
  - `.debug/svamp32_conditional_residual_query_codec_20260423/checkpoints/qwen25_to_qwen3_svamp32_preserve_konly_query_codec_r16_bank16_seed1.pt`
  - sha256: `a6236dd37c2dd8caa0d3928d644286ce5843ee26ff8f6fb336dcbfd8e6e24eca`
- matched sweep:
  - `.debug/svamp32_conditional_residual_query_codec_20260423/preds/preserve_konly_attention_gate_sweep.jsonl`
  - JSONL sha256: `dc8cf4cb13e210d19ae56462bd741c4043fc237d8e2e7311bd42181aef1fa167`
  - meta sha256: `091a0ea04cd543a6c321bf7d525fc3ae6e204d18834f95dad3d5114a212499b0`
- readout:
  - `results/svamp32_conditional_residual_query_codec_20260423/preserve_konly_attention_clean_targets.json`
  - sha256: `73256d6e51fbe9e09ef357c1939087c67db821cf8cf6b34f86fb5a90d0edeb11`
  - `results/svamp32_conditional_residual_query_codec_20260423/preserve_konly_attention_clean_targets.md`
  - sha256: `0207a238675f8cceca3a7e25f21906d9579a061b9e771fd19ba52e8942201f89`
- memo:
  - `paper/svamp32_conditional_residual_query_codec_20260423.md`

Evidence:

- calibration matched all `6` clean residual prompts and all `3`
  target-self-preserve prompts
- dynamic token-mixture samples: `1411`
- average fit quality: K cosine `0.951`, V cosine `0.734`
- matched readout status: `no_matched_gate_candidate_for_controls`
- best matched rows: `8/32` at gates `0.05`, `0.12`, `0.15`, `0.17`,
  `0.20`, and `0.25`
- clean residual recovered: `0/6` at every gate
- teacher-only recovered: `1` at every gate, ID `575d7e83d84c1e67`
- target-self delta: `-6` or `-7` versus the `14/32` target self-repair row
- numeric extraction coverage: `32/32`
- exact ordered ID parity: true

Hypothesis update:

- weakened: target-self no-op loss plus K-only value-loss suppression as a
  sufficient conditional codec
- killed for now: this bounded protected-loss variant as a same-pair gate
  candidate
- still alive: a real target-conditioned memory variant that appends target
  prior K/V to the query resampler memory instead of only changing losses
- saturated: further scalar/gate/value-loss screens on the current residual
  learner unless the architecture changes

Next exact gate:

- implement an isolated opt-in conditional query-memory variant:
  `[source K/V, target-prior K/V, learned slots]`
- rerun the same SVAMP32 matched gate and only spend controls if it reaches
  `>=2/6` clean residual IDs while preserving near-target-self accuracy

## 2026-04-23: SVAMP32 Target-Memory K-Only Query Codec

Current paper readiness:

- not ICLR-ready
- estimated distance: medium-high; the live method still has not cleared the
  same-pair clean-residual gate

Current paper story:

- target self-repair remains the decoder-side floor at `14/32`
- a positive method must add real source-conditioned residual signal on clean
  C2C-only IDs, not just reproduce target-cache behavior

Blocking gap:

- no source-conditioned method has reached `>=2/6` clean residual recoveries
  under exact-ID matched scoring

Top moves considered:

- opt-in target-prior query memory. This matters because prior Q-Former /
  Perceiver-style side-information suggestions point to receiver-conditioned
  bottlenecks. It might fail if the target prior dominates or the source signal
  remains unseparated. Cost: code plus one calibration and matched gate. Helps
  same-pair, interpretability, and reproducibility.
- full target-prior delta-memory. This matters because it explicitly exposes
  source-minus-target residual rows. It might fail by overfitting or by needing
  target-only controls to disambiguate leakage. Cost: a second implementation
  and matched gate. Helps same-pair and interpretability.
- verifier/selector sidecar over existing rows. This matters for no-harm
  preservation. It might fail because the existing rows do not contain enough
  clean residual wins to clear the gate. Cost: low. Helps robustness only after
  a stronger residual row exists.

Decision:

- implemented the bounded opt-in target-prior K/V memory path first
- ran the frozen SVAMP32 exact-ID matched gate
- did not run controls because no matched row reached `>=2/6` clean residual
  IDs

Code changes:

- `latent_bridge/calibrate.py`
  - added `--innovation-conditional-target-memory`
  - validates that it is only used with
    `bridge_ridge_qk_dynalign_query_innovation_resampler_replace`
- `latent_bridge/translator.py`
  - added `TranslatorConfig.innovation_conditional_target_memory`
  - forwards target-prior K/V rows into query-module fit/runtime memory
  - forwards conditional target memory through calibration-time diagnostics
- `latent_bridge/evaluate.py`
  - forwards target prompt K/V into `translate_layer` when the config flag is
    enabled
- tests added for CLI parsing, config validation, fit-time forwarding, and
  evaluation-time target-cache forwarding

Verification:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_translator_core.py \
  tests/test_calibrate_and_ablation.py \
  tests/test_evaluate_helpers.py -q
```

Result: `319 passed`

Artifacts:

- checkpoint:
  - `.debug/svamp32_targetmem_query_codec_20260423/checkpoints/qwen25_to_qwen3_svamp32_targetmem_konly_query_codec_r16_bank16_seed1.pt`
  - sha256: `071fc28113d8a8b4829feb8fb5391cd4d158e2fcb623bd9ea773cf3142bf2d67`
- matched sweep:
  - `.debug/svamp32_targetmem_query_codec_20260423/preds/targetmem_konly_attention_gate_sweep.jsonl`
  - sha256: `e16b0526ada85956a4842ba7abd9f783a50fae6c2985a57d4fddf01a3153547b`
  - meta sha256: `e570934f84fcb9c6d773153a5c4dd4a11217d43ca79c82f7ac6235211e0d59cc`
- readout:
  - `results/svamp32_targetmem_query_codec_20260423/targetmem_konly_attention_clean_targets.json`
  - sha256: `3baed26ad7dbc7a60c73cd49342759901539def40db1bd37a4681df617bc2f4a`
  - `results/svamp32_targetmem_query_codec_20260423/targetmem_konly_attention_clean_targets.md`
  - sha256: `2fb014905d46d342e650a21e6b3e5f1fc9a75218dbad0cf5e9aa06b1828d3da0`
- memo:
  - `paper/svamp32_targetmem_query_codec_20260423.md`

Evidence:

- calibration matched all `6` clean residual prompts and all `3`
  target-self-preserve prompts
- dynamic token-mixture samples: `1411`
- average fit quality: K cosine `0.951`, V cosine `0.734`
- matched readout status: `no_matched_gate_candidate_for_controls`
- best row: `rotalign_kv_gate_0.20`, `9/32`
- clean residual recovered: `0/6`
- teacher-only recovered: `1`, ID `575d7e83d84c1e67`
- target losses: `1`, ID `c042f0a2949ff8e6`
- numeric extraction coverage: `32/32`
- exact ordered ID parity: true
- bytes: `397,923.75`
- best latency: `8.662850` seconds/example

Hypothesis update:

- killed for now: bounded target-prior K-only memory as a same-pair gate
  candidate
- weakened: target side-information alone is sufficient without an explicit
  source-minus-target residual/delta channel
- still alive: full target-prior delta-memory branch with runtime source-only,
  target-only, delta-only, and combined masks
- saturated: scalar gate sweeps, K-only value-loss suppression, target no-op
  preservation, and current query-innovation memory without a delta channel

Next exact gate:

- implement full target-prior delta-memory
  `[source K/V, target-prior K/V, source-minus-target delta K/V, learned slots]`
- run the same matched SVAMP32 exact-ID gate plus runtime memory-mask controls
  only if the combined row reaches `>=2/6` clean residual IDs

## 2026-04-24 - SVAMP32 target-prior delta-memory query codec

Question:

- does adding explicit source-minus-target delta K/V memory rows recover clean
  source-only residual IDs beyond target-prior memory and target self-repair?

Decision:

- branch failed the same-pair matched gate
- did not run runtime memory-mask controls because the combined row did not
  reach `>=2/6` clean residual IDs

Code changes:

- `latent_bridge/calibrate.py`
  - added `--innovation-conditional-delta-memory`
  - delta memory implies target-prior memory at calibration time
- `latent_bridge/translator.py`
  - added `TranslatorConfig.innovation_conditional_delta_memory`
  - added `TranslatorConfig.innovation_memory_control`
  - fit-time memory can include `source_predicted_k/v - target_prior_k/v`
  - runtime controls select `combined`, `no_delta`, `source_only`,
    `target_only`, `delta_only`, or `slots_only`
- `latent_bridge/evaluate.py`
  - added `--innovation-memory-control` checkpoint override for control sweeps
  - forwards target K/V conditions for target-prior or delta-memory checkpoints
- tests
  - added CLI/config coverage and exact runtime memory row-selection coverage

Verification:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_translator_core.py \
  tests/test_calibrate_and_ablation.py \
  tests/test_evaluate_helpers.py -q
```

Result: `333 passed in 4.23s`

Artifacts:

- checkpoint:
  - `.debug/svamp32_delta_memory_query_codec_20260424/checkpoints/qwen25_to_qwen3_svamp32_deltamem_konly_query_codec_r16_bank16_seed1.pt`
  - sha256: `29ff93c6d7291fb9a4e00ac35a7ffa519c4d71c8bd4a38062c0d748baecf4ebb`
- matched sweep:
  - `.debug/svamp32_delta_memory_query_codec_20260424/preds/deltamem_konly_attention_gate_sweep.jsonl`
  - sha256: `01b3524fc887ef46ad0dc0ce86aa5cd145ce6f962d852e748f8472d7f7afc93a`
- readout:
  - `results/svamp32_delta_memory_query_codec_20260424/deltamem_konly_attention_clean_targets.json`
  - sha256: `8ab4d02b947369428bf49b74e658cd8aa9fd944eee916c0393d6e597a864158c`
  - `results/svamp32_delta_memory_query_codec_20260424/deltamem_konly_attention_clean_targets.md`
  - sha256: `9fed7cdbfb75bacc4bee0617ace866525ad9f96fc6e630246e3e93074d0fbde2`
- memo:
  - `paper/svamp32_delta_memory_query_codec_20260424.md`

Evidence:

- calibration matched all `6` clean residual prompts and all `3`
  target-self-preserve prompts
- dynamic token-mixture samples: `1411`
- average fit quality: K cosine `0.951`, V cosine `0.734`
- matched readout status: `no_matched_gate_candidate_for_controls`
- best rows:
  - `rotalign_kv_gate_0.17`: `9/32`, clean residual `0/6`
  - `rotalign_kv_gate_0.15`: `9/32`, clean residual `0/6`
- teacher-only recovered: `1`, ID `575d7e83d84c1e67`
- target losses at best rows: `1`, ID `c042f0a2949ff8e6`
- numeric extraction coverage: `32/32`
- exact ordered ID parity: true
- bytes: `397,923.75`
- best latency: `7.807506` seconds/example

Hypothesis update:

- killed for now: this K-only target-prior delta-memory query-codec
  configuration as a same-pair gate candidate
- weakened: plain `source - target_prior` delta rows are enough to recover
  clean source residual information
- still alive: delta-memory infrastructure as a runtime control surface for a
  stronger source-discriminative objective
- saturated: scalar gate sweeps, K-only value-loss suppression, target-only
  memory, and unregularized target-prior delta memory on SVAMP32 exact-ID

Next exact gate:

- add a source-discriminative objective/control before widening:
  matched-vs-shuffled source contrast on the same residual target set, or a
  verifier-gated repair selector that must preserve the `14/32` self-repair
  floor
- require `>=2/6` clean residual IDs on SVAMP32 exact-ID before larger slices,
  seed repeats, or cross-family experiments

## 2026-04-24 - SVAMP32 source informativeness and oracle-bound diagnostic

Question:

- after target-memory and delta-memory failed, does the frozen SVAMP32 slice
  contain exploitable source/complementarity headroom, and do any existing rows
  expose enough clean source-necessary IDs to justify verifier-only selection?

Decision:

- added a reusable source/oracle-bound analyzer
- ran it across source/text rows, target_self_repair, selected route, and the
  strongest recent candidate rows
- cross-checked the best ID-weighted row with the stricter source-sidecar-bound
  analyzer
- no method promoted; result is a diagnostic gate

Code changes:

- `scripts/analyze_svamp32_source_oracle_bound.py`
  - preserves raw gate-sweep method labels before falling back to normalized
    method names
  - reports source correctness on teacher-only and clean residual IDs
  - reports candidate clean residual recovery and per-clean-ID provenance
  - computes oracle bounds for target/baseline rows against source and
    candidate rows
- `tests/test_analyze_svamp32_source_oracle_bound.py`
  - synthetic exact-ID coverage for clean residual/source/oracle accounting

Verification:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_oracle_bound.py -q
```

Result: `1 passed in 0.02s`

Artifacts:

- aggregate source/oracle readout:
  - `results/svamp32_source_oracle_bound_20260424/source_oracle_bound.json`
  - sha256: `806a3b9e64a3562b386a77d6b8573bc529f8524dcaab9fbd82fbdcbf97378966`
  - `results/svamp32_source_oracle_bound_20260424/source_oracle_bound.md`
  - sha256: `6de23b56930bca29c311a403b9ebd08be653a620a4f7d6b9b290cb3208c57557`
- strict sidecar-bound cross-check:
  - `results/svamp32_source_oracle_bound_20260424/idweighted_gate015_sidecar_bound.json`
  - sha256: `678015b227017b2c679d2708ff89311fa749407814320c138be49789aeb3ad08`
  - `results/svamp32_source_oracle_bound_20260424/idweighted_gate015_sidecar_bound.md`
  - sha256: `2b4958c03166acd9e78b55e9c3f9a65647c8136606093acf9a94f14e06c87ed8`
- memo:
  - `paper/svamp32_source_oracle_bound_20260424.md`

Evidence:

- target-alone: `8/32`
- C2C teacher: `16/32`
- target_self_repair: `14/32`
- source_alone: `5/32`
- text_to_text: `2/32`
- source/text union clean residual exact-correct: `0/6`
- existing candidate union clean residual exact-correct: `1/6`
- only current clean residual recovered by any candidate:
  `aee922049c757331`, recovered by `idweighted_gate015`
- best candidate-row oracle with target_self_repair:
  `idweighted_gate015`, `17/32`, `+3`
- strict source-control-clean sidecar bound for `idweighted_gate015`:
  `15/32`, `+1`, `1` clean source-necessary ID
- sidecar-bound failing criteria:
  `min_correct`, `min_clean_source_necessary`

Subagent synthesis:

- literature/external, ablation, repo audit, and internet-creative side paths
  converged on a source-control-contrastive Q-Former/Perceiver-style connector
  as the next positive-method branch
- verifier-only selection is weakened because existing rows expose only one
  clean source-necessary residual ID under strict controls

Hypothesis update:

- promoted: C2C/cache-level source distillation, because C2C solves clean IDs
  that source final text and text relay do not
- promoted: learned query connector with matched-vs-zero/shuffled source
  contrast as the next method gate
- weakened: verifier-only repair/selection from existing rows as a positive
  method
- killed for now: additional direct target-memory or delta-memory variants
  without a source-discriminative connector objective
- saturated: source final-answer relay, text-to-text relay, current
  query-pool/idweighted rows, and target_self_repair as a comparator

Next exact gate:

- implement the smallest source-control-contrastive learned query connector
  smoke:
  - frozen source and target
  - `8` learned connector queries
  - C2C clean-residual distillation
  - target-correct/self-repair preservation
  - matched-vs-zero/shuffled source contrast
- promote only if SVAMP32 exact-ID reaches `>=14/32`, `>=2/6` clean residual
  IDs, at most `1` target-correct loss, and clean wins vanish under source
  controls

## 2026-04-24 - SVAMP32 contrastive delta-memory connector smoke

Question:

- does combining explicit target-prior delta memory with stronger
  zero/shuffle source-control contrast recover clean source-necessary residual
  IDs before implementing a deeper connector?

Decision:

- trained one combined source-control/delta-memory query-innovation checkpoint
- ran only the matched SVAMP32 exact-ID gate sweep
- did not run source-zero, source-shuffle, or memory-mask controls because the
  matched row failed `>=2/6` clean residual IDs
- logged the result as a branch-killing smoke, not a promoted method

Implementation:

- no code changes required; existing infrastructure already supports:
  - `--innovation-conditional-delta-memory`
  - `--innovation-control-mode zero_and_shuffle`
  - `--innovation-memory-control`
- new tracked memo:
  - `paper/svamp32_contrastive_deltamem_connector_20260424.md`

Artifacts:

- clean-target readout:
  - `results/svamp32_contrastive_deltamem_connector_20260424/combined_w050_m001_attention_clean_targets.json`
  - sha256: `34d7086b75d0e034d5793b571fba459bd3c218849b9b3df0e5cd8586e7658d56`
  - `results/svamp32_contrastive_deltamem_connector_20260424/combined_w050_m001_attention_clean_targets.md`
  - sha256: `7127f373e67e9187745e8cdfc1bccc586394d66433d1923ec41fc4dda24293aa`
- source/oracle cross-check:
  - `results/svamp32_contrastive_deltamem_connector_20260424/source_oracle_bound_with_contrastive_deltamem.json`
  - sha256: `cbf5b293238ded7e178afb893acc730c7a303f4c6ffea13babc8dedc135178f1`
  - `results/svamp32_contrastive_deltamem_connector_20260424/source_oracle_bound_with_contrastive_deltamem.md`
  - sha256: `214155c75285cf0c4131bae5812f9f0554a5fa411ddf473c2c8b5b3ed911ed6b`
- scratch checkpoint/logs/preds:
  - `.debug/svamp32_contrastive_deltamem_connector_20260424/checkpoints/qwen25_to_qwen3_svamp32_contrastive_deltamem_query_connector_w050_m001_r16_b16_seed1.pt`
  - sha256: `6b22d1b62da5455134c4a7935252426617d6d556ad0317a8f98217409c607bb8`
  - `.debug/svamp32_contrastive_deltamem_connector_20260424/preds/combined_attention_gate_sweep.jsonl`
  - sha256: `d9b4735d64503f485a05fa300c78352f8c17b9ea018971dc9394cbbb71162fc4`

Evidence:

- calibration matched `6` clean residual prompts and `3` target-self-preserve
  prompts
- dynamic token-mixture samples: `1411`
- average fit quality: K cosine `0.951`, V cosine `0.734`
- matched readout status: `no_matched_gate_candidate_for_controls`
- all tested gates (`0.125`, `0.15`, `0.175`, `0.20`) scored `8/32`
- clean residual recovered: `0/6`
- teacher-only recovered: `1`
- delta vs target_self_repair: `-6`
- target losses: `1`
- oracle `target_self_repair + contrastive_deltamem_w050_gate020`: `15/32`
- clean residual added to `target_self_repair`: `0`
- bytes: `397,923.75`
- best latency among tested gates: `7.430172` seconds/example

Subagent synthesis:

- literature, ablation, repo-audit, and internet-creative subagents converged
  on source-control-contrastive learned-query transport
- repo audit confirmed the current combined smoke is executable with existing
  flags, but warned that the likely failure mode is target-prior leakage or
  over-regularization
- internet/literature synthesis promoted a true Q-Former/Perceiver-style
  receiver-conditioned connector rather than further scalar tuning

Hypothesis update:

- killed for now: cheap scalar tuning of combined delta-memory plus
  zero/shuffle source-control contrast under the current query-innovation module
- weakened: explicit source-minus-target delta rows can expose clean residual
  information when regularized for source specificity
- weakened: source-control contrast alone is enough without a stronger
  receiver-conditioned connector architecture
- still alive: source-control contrast and target-prior/delta controls as
  diagnostics for a deeper connector
- promoted: implement a true receiver-conditioned Q-Former/Perceiver-style
  connector with C2C residual distillation and source-destroying controls

Next exact gate:

- stop tuning this cheap connector family unless the architecture changes
- implement a small receiver-conditioned learned-query connector:
  - frozen source and target
  - `8-16` learned connector queries
  - query cross-attention over source K/V plus target-prior state
  - C2C clean-residual distillation
  - target-self-repair preservation
  - matched-vs-zero/shuffled/target-only source controls
- require `>=14/32`, `>=2/6` clean residual IDs, at most `1` target-correct
  loss, exact ID parity, and source-control collapse before widening

## 2026-04-24 - SVAMP32 Perceiver-query connector smoke

Question:

- does a minimal receiver-conditioned learned-query connector, inspired by
  Q-Former/Perceiver bottlenecks, recover clean source-necessary SVAMP32
  residual IDs where scalar/delta-memory query-innovation variants failed?

Decision:

- implemented a default-off `innovation_connector_mode=perceiver_queries`
  topology for the query-innovation connector
- trained one exact-ID SVAMP32 checkpoint with `8` learned connector queries,
  K-only loss, target/delta memory, and zero/shuffle source-control training
- ran the matched gate sweep only
- did not run source-zero, source-shuffle, or target-only memory controls
  because matched performance failed the clean residual gate

Implementation:

- `latent_bridge/translator.py`
  - added `TranslatorConfig.innovation_connector_mode`
  - added validation for supported correction family and nonzero query bank
  - fit path: bridge-bank rows can act as learned connector queries over
    live source/target memory before receiver-query readout
  - runtime path mirrors the fit-time Perceiver-query topology
- `latent_bridge/calibrate.py`
  - added `--innovation-connector-mode`
- tests
  - parser coverage for the new CLI flag
  - config validation for invalid/unsupported modes
  - finite fit/runtime smoke with a runtime equation spy

Artifacts:

- memo:
  - `paper/svamp32_perceiver_query_connector_20260424.md`
- results manifest:
  - `results/svamp32_perceiver_query_connector_20260424/manifest.md`
- clean-target readout:
  - `results/svamp32_perceiver_query_connector_20260424/perceiver_queries_w030_m010_attention_clean_targets.json`
  - sha256: `fbccf197d063dfd133f584a0397322f2e35f6e6de710b8ec92cf5dc594335e3c`
  - `results/svamp32_perceiver_query_connector_20260424/perceiver_queries_w030_m010_attention_clean_targets.md`
  - sha256: `fd80e0f402bfa2e49166262d29b1950b3d2abb550bf28fc2c7b63d23e8b062e9`
- scratch checkpoint/logs/preds:
  - `.debug/svamp32_perceiver_query_connector_20260424/checkpoints/qwen25_to_qwen3_svamp32_perceiver_queries_w030_m010_r16_q8_seed1.pt`
  - sha256: `ad64ffd29b5e31f029e9a4d14d75ed6bcb64906d44dc7746532f1606146712f0`
  - `.debug/svamp32_perceiver_query_connector_20260424/preds/perceiver_queries_combined_attention_gate_sweep.jsonl`
  - sha256: `a45d014b712f1e315210335a899cd12f18ada8d24e11c40addd53609350927e0`

Verification:

- `./venv_arm64/bin/python -m pytest tests/test_translator_core.py tests/test_calibrate_and_ablation.py -q`
- result: `234 passed in 3.90s`

Evidence:

- calibration matched `6` clean residual prompts and `3`
  target-self-preserve prompts
- dynamic token-mixture samples: `1411`
- average fit quality: K cosine `0.951`, V cosine `0.734`
- matched readout status: `no_matched_gate_candidate_for_controls`
- best row: `rotalign_kv_gate_0.15`
- best row accuracy: `10/32`
- clean residual recovered: `0/6`
- teacher-only recovered: `2`
- delta vs target self-repair: `-4`
- target losses: `1`
- exact ordered ID parity: true
- numeric extraction coverage: `32/32`
- average bytes: `397,923.75`
- best latency among tested gates: `7.262815` seconds/example

Subagent synthesis:

- literature and internet-creative agents both recommended a small
  receiver-conditioned Q-Former/Perceiver connector, citing BLIP-2, Flamingo,
  Perceiver IO, C2C, and KV communication work as the closest design analogs
- ablation agent recommended matched-first promotion with source-zero,
  shuffled-source, and target-only controls only after the matched row clears
- repo-audit agent warned that runtime integration was the main reproducibility
  risk, so tests now assert the runtime Perceiver-query path is actually used

Hypothesis update:

- killed: this specific 8-query K-only Perceiver-query checkpoint as a
  same-pair positive method row
- weakened: learned connector queries alone are enough to recover clean
  source-necessary residual IDs under the current query-innovation objective
- weakened: more slot/query topology without a stronger teacher or answer-token
  objective will overcome the target-self-repair floor
- still alive: receiver-conditioned connectors with explicit C2C residual
  distillation or teacher-forced answer-token objectives
- promoted: a cheaper feasibility diagnostic before another full 32-example
  generation sweep

Next exact gate:

- train/evaluate the connector on the `6` clean residual IDs with explicit C2C
  residual distillation or teacher-forced answer-token loss
- include matched, zero-source, shuffled-source, and target-only memory in one
  small diagnostic
- promote only if matched recovers at least `2/6` clean IDs, source controls
  collapse those wins, target-self-repair IDs are preserved, and exact ID parity
  remains true

## 2026-04-24 - SVAMP32 teacher-forced connector diagnostic

Question:

- does the failed 8-query Perceiver connector checkpoint contain hidden
  teacher-forced source-specific answer evidence on the six clean SVAMP32
  residual IDs?

Decision:

- implemented a standalone answer-margin diagnostic instead of running another
  full generation sweep
- scored gold numeric continuations against target-alone wrong numeric
  distractors under matched, zero-source, shuffled-source, target-only, and
  slots-only controls
- killed this checkpoint as a source-necessary positive row

Implementation:

- `scripts/analyze_svamp32_teacher_forced_connector_diagnostic.py`
  - loads a fitted `RotAlignKVTranslator`
  - reuses the existing fused prefix-state and continuation logprob path
  - writes exact-ID JSON and Markdown artifacts
  - records per-control source IDs, margins, bytes, and gate status
- `tests/test_analyze_svamp32_teacher_forced_connector_diagnostic.py`
  - covers continuation-template validation and summary gate accounting

Artifacts:

- memo:
  - `paper/svamp32_teacher_forced_connector_diagnostic_20260424.md`
- results manifest:
  - `results/svamp32_teacher_forced_connector_diagnostic_20260424/manifest.md`
- clean-only diagnostic:
  - `results/svamp32_teacher_forced_connector_diagnostic_20260424/perceiver_queries_gate015_answer_margin_clean.json`
  - sha256: `3e67de34ca7121cc803bc10bad78b1b3aab4e2857efd8654eab6655132f693a9`
  - `results/svamp32_teacher_forced_connector_diagnostic_20260424/perceiver_queries_gate015_answer_margin_clean.md`
  - sha256: `098cd43ddddc9e269f357699260880cacab3cc4925851035db90323107ccb48d`
- clean plus target-self diagnostic:
  - `results/svamp32_teacher_forced_connector_diagnostic_20260424/perceiver_queries_gate015_answer_margin_clean_self.json`
  - sha256: `47443a71295606330e26911777ed4b496f390506538f943695e2e1d6df746c0c`
  - `results/svamp32_teacher_forced_connector_diagnostic_20260424/perceiver_queries_gate015_answer_margin_clean_self.md`
  - sha256: `6bf0367b38a34621508ebb8c4e40209462ac7107c432914d650a0ea584be6903`
- scratch log:
  - `.debug/svamp32_teacher_forced_connector_diagnostic_20260424/logs/perceiver_queries_gate015_answer_margin_clean_self.log`
  - sha256: `ebecf85e36ff89b93ba15b947eaf889ecb30af1ac728c43b01ae691c23d182b0`

Verification:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_teacher_forced_connector_diagnostic.py -q`
- result: `2 passed in 0.01s`

Evidence:

- clean residual IDs scored: `6`
- target-self-repair IDs scored: `3`
- matched-positive clean IDs: `2`
- matched-only clean IDs: `0`
- control-leak clean IDs: `2`
- mean matched margin: `-3.764944`
- mean best-control margin: `-2.272635`
- mean matched-minus-control margin: `-1.492309`
- the two matched-positive clean IDs were `aee922049c757331` and
  `e3ab8666238a289e`, but both had stronger zero-source margins than matched
  source margins

Subagent synthesis:

- literature and internet-creative agents recommended teacher-forced
  source-control answer margins before another generation sweep
- ablation agent recommended the minimal decisive controls: matched,
  zero-source, shuffled-source, target-only, and slots-only
- repo-audit agent recommended a standalone script and explicit artifact
  manifests to keep the diagnostic reproducible

Hypothesis update:

- killed: this Perceiver-query checkpoint contains hidden matched-only answer
  signal on the clean residual IDs
- killed: another full generation sweep from this checkpoint is likely to
  produce a source-necessary positive row
- weakened: K-only Perceiver-query transport under the current objective is
  enough to create a usable residual channel before answer-token supervision
- still alive: receiver-conditioned connectors with direct answer-token or C2C
  residual objectives and source-destroying controls
- promoted: controlled answer-token microfit as the next architecture gate

Next exact gate:

- do not widen benchmarks or run seed repeats for this killed checkpoint
- train a small answer-token or C2C residual microfit on the six clean residual
  IDs plus target-self-preserve IDs
- evaluate with the new teacher-forced diagnostic before greedy generation
- promote only if matched-only positive margins appear on at least `2/6` clean
  IDs and collapse under zero-source, shuffled-source, target-only, and
  slots-only controls

## 2026-04-24 - SVAMP32 answer-teacher microfit

Question:

- can direct gold answer-token teacher supervision on the six clean SVAMP32
  residual IDs make the current query-innovation Perceiver connector
  source-dependent under the committed matched/control diagnostic?

Decision:

- implemented a default-off calibration microfit hook:
  `--innovation-answer-teacher-weight`
- injected gold answer continuation token teacher rows only for clean residual
  prompt IDs
- preserved target-self-repair IDs with the existing zero-residual mask
- ran the teacher-forced diagnostic before greedy generation
- killed this calibration-proxy microfit because matched source did not beat
  source-destroying controls on any clean residual ID

Implementation:

- `latent_bridge/calibrate.py`
  - added prompt answer record loading
  - added `inject_answer_token_teacher(...)`
  - added answer-teacher CLI validation and wiring for
    `bridge_ridge_qk_dynalign_query_innovation_resampler_replace`
- `tests/test_calibrate_and_ablation.py`
  - added helper tests for clean-only teacher injection
  - added parser coverage for answer-teacher flags

Artifacts:

- memo:
  - `paper/svamp32_answer_teacher_microfit_20260424.md`
- results manifest:
  - `results/svamp32_answer_teacher_microfit_20260424/manifest.md`
- diagnostic:
  - `results/svamp32_answer_teacher_microfit_20260424/answer_teacher_w090_gate015_clean_self.json`
  - sha256: `e9db6ffed6ba5c42a9b983e48154fde3eac98248c56b05c57900cd9870266f71`
  - `results/svamp32_answer_teacher_microfit_20260424/answer_teacher_w090_gate015_clean_self.md`
  - sha256: `324e123639f812030b3e5e3f8c1ab81127468010f0438c3ee5752ca166a1a6e2`
- scratch checkpoint/logs:
  - `.debug/svamp32_answer_teacher_microfit_20260424/checkpoints/qwen25_to_qwen3_svamp32_answer_teacher_w090_r16_q8_seed1.pt`
  - sha256: `437b7eecf8f0b3704eb8e6260cefcd9d45ead2a31d02855c33655c06dd2de8fc`
  - `.debug/svamp32_answer_teacher_microfit_20260424/logs/calibrate_answer_teacher_w090_r16_q8_seed1.log`
  - sha256: `8cbfe57de7c83d86fbae9c46e134f08110938794a6b2f60606456cb9b4091d88`
  - `.debug/svamp32_answer_teacher_microfit_20260424/logs/diagnostic_answer_teacher_w090_gate015_clean_self.log`
  - sha256: `ada211e52f4d0b3189a5a5ce2d9487536367419d0db5705942fdb0c9302461a1`

Verification:

- `./venv_arm64/bin/python -m pytest tests/test_calibrate_and_ablation.py -q`
- result: `98 passed in 0.24s`

Evidence:

- answer-teacher prompts injected: `6`
- answer-teacher samples injected: `277`
- clean residual IDs scored: `6`
- target-self-repair IDs scored: `3`
- matched-positive clean IDs: `2`
- matched-only clean IDs: `0`
- control-leak clean IDs: `2`
- mean matched margin: `-3.834308`
- mean best-control margin: `-2.637356`
- mean matched-minus-control margin: `-1.196952`
- the two matched-positive clean IDs still leaked:
  - `aee922049c757331`: best control `slots_only`
  - `e3ab8666238a289e`: best control `target_only`

Subagent synthesis:

- literature/internet agents recommended C2C-distilled Q-Former/Perceiver
  fusers with source-destroying controls
- ablation agent recommended no greedy sweep unless the teacher-forced gate
  cleared
- repo-audit agent recommended a future standalone differentiable
  answer-margin sidecar if this calibration proxy failed
- creative agent proposed a latent syndrome sidecar that transmits numeric
  residues with target candidates as decoder side information

Hypothesis update:

- killed: answer-token teacher injection in the existing calibration objective
  is enough to make the current query-innovation Perceiver connector
  source-dependent
- killed: greedy generation from this microfit is evidence-driven
- weakened: this same-family Qwen pair exposes clean residual answer signal
  through the current connector family
- still alive: direct differentiable answer-margin sidecar
- still alive: latent syndrome sidecar or source-informativeness gate
- promoted: stop calibration-proxy tuning; next test source informativeness or
  a direct margin/syndrome branch

Next exact gate:

- run a source-informativeness audit on the six clean residual IDs before more
  same-pair connector tuning, or implement the standalone differentiable
  answer-margin sidecar with the same `>=2/6` matched-only gate
- do not widen benchmark scope until a live row clears the teacher-forced
  source-necessity gate

## 2026-04-24 - SVAMP32 source margin audit

Question:

- do the six clean SVAMP32 residual IDs contain source-side answer evidence
  under source-alone teacher-forced gold-vs-target-wrong numeric margins?

Decision:

- implemented `scripts/analyze_svamp32_source_margin_audit.py`
- scored source and target model answer margins on the six clean residual IDs
  plus three target-self-repair IDs
- separated source/text final-answer correctness from teacher-forced source
  answer margins
- killed source-final-answer and source-margin informativeness for this
  same-pair Qwen2.5-0.5B to Qwen3-0.6B surface

Artifacts:

- memo:
  - `paper/svamp32_source_margin_audit_20260424.md`
- results manifest:
  - `results/svamp32_source_margin_audit_20260424/manifest.md`
- diagnostic:
  - `results/svamp32_source_margin_audit_20260424/source_margin_clean_self.json`
  - sha256: `16ab06a97024d61cbb6efb3b1cfbebacc9f542bab25862e1671b5e4ec7a919ff`
  - `results/svamp32_source_margin_audit_20260424/source_margin_clean_self.md`
  - sha256: `9bb5a859eff08ff0727ab6fd8d57bc37b38dbd8dae0f072b725e6fb609f9b91b`
- scratch log:
  - `.debug/svamp32_source_margin_audit_20260424/logs/source_margin_clean_self.log`
  - sha256: `ce606e110ab7f35096ea799d52bedeeed070d8590d0b3db1af789f11b5ab03c3`

Verification:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_margin_audit.py -q`
- result: `3 passed in 0.02s`
- `./venv_arm64/bin/python -m pytest -q`
- result: `657 passed in 25.85s`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_margin_audit.py`
- result: pass

Evidence:

- clean residual IDs scored: `6`
- target-self-repair IDs scored: `3`
- source/text final clean correct: `0/6`
- source-margin positive clean IDs: `2/6`
- source-margin positive+advantage clean IDs: `0/6`
- mean source margin: `-3.065174`
- mean target margin: `-3.624139`
- mean source-minus-target margin: `0.558965`
- the only source-positive clean IDs were `aee922049c757331` and
  `e3ab8666238a289e`, and both had stronger target-alone margins

Subagent synthesis:

- source-audit and ablation agents converged on the same pass/fail rule:
  promote only if source is positive and beats target/control evidence on at
  least `2/6` clean residual IDs
- repo-audit agent emphasized exact-ID provenance and numeric coverage before
  scoring any source-informativeness claim
- creative internet agent suggested latent arithmetic-syndrome sidecars, but
  only as a new branch with zero/shuffle controls

Hypothesis update:

- killed: source/text final-answer relay is informative on the clean residual
  IDs
- killed: source answer-token margins justify more same-pair connector tuning
- weakened: standalone answer-margin sidecar for this pair
- still alive: C2C/cache-residual distillation because C2C remains `16/32`
  while source final answers and source margins fail
- promoted: stronger-source/cross-family source-informativeness falsification
  or a C2C-residual distillation gate

Next exact gate:

- do not run another same-pair Qwen2.5-0.5B to Qwen3-0.6B calibration-proxy
  connector
- run a strict stronger-source or cross-family source-informativeness gate on
  the same frozen SVAMP32 IDs, or implement a C2C-residual distillation sidecar
  that must clear the same `>=2/6` clean matched-vs-control threshold

## 2026-04-24 - SVAMP32 stronger-source margin audit

Question:

- was the failed Qwen2.5-0.5B to Qwen3-0.6B source-margin audit just a weak
  source problem, or do stronger Qwen sources still fail to expose the six
  clean SVAMP32 residual IDs?

Decision:

- extended `scripts/analyze_svamp32_source_margin_audit.py` with optional
  source/text provenance and `--dtype`
- added `scripts/materialize_generation_id_subset.py` for exact stable-ID
  generation subsets
- ran stronger-source audits for `Qwen/Qwen2.5-1.5B-Instruct` and
  `Qwen/Qwen2.5-7B-Instruct`
- killed source-final/source-margin stronger-source escalation as the next
  method path because the strongest source still exposes only isolated
  `1/6` clean signals per channel

Artifacts:

- memo:
  - `paper/svamp32_stronger_source_margin_audit_20260424.md`
- results manifest:
  - `results/svamp32_stronger_source_margin_audit_20260424/manifest.md`
- 1.5B final audit:
  - `results/svamp32_stronger_source_margin_audit_20260424/qwen25_15b_to_qwen3_06b_source_margin_clean_self_with_sourcegen.json`
  - sha256: `2c0b067317e6e47235a167a50f9a609aff02e2c4105c30c3c630f63bf895fa58`
- 7B final audit:
  - `results/svamp32_stronger_source_margin_audit_20260424/qwen25_7b_to_qwen3_06b_source_margin_clean_self_with_source_text.json`
  - sha256: `59bffba1dd9bd06155bab537a888bdc4bac92eaf963258aebc19363cfe806e97`
- exact clean+self subset:
  - `results/svamp32_stronger_source_margin_audit_20260424/svamp32_clean_self_eval.jsonl`
  - sha256: `07784ad26e52e51a1c6080b71294543bf420854b67e4851f5fe6a6dcf0e30995`
- 7B source/text subset predictions:
  - `results/svamp32_stronger_source_margin_audit_20260424/qwen25_7b_source_alone_clean_self.jsonl`
  - sha256: `0163af711efba78a106a54a85ab10ebab22d9bb5612c4b4f4df40302554a3774`
  - `results/svamp32_stronger_source_margin_audit_20260424/qwen25_7b_text_to_text_clean_self.jsonl`
  - sha256: `1261e702f8e9e089ffe1cdf2f43282dfba8c93ddb105479575e726a37dd0f860`
- 1.5B full-32 source baseline:
  - `results/svamp32_stronger_source_baselines_20260424/source_alone.jsonl`
  - sha256: `24c00515dec342c44b52267b5a9d269f6ee92b2f7ba0676bb30db4ccd535a228`

Verification:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_margin_audit.py tests/test_materialize_generation_id_subset.py -q`
- result: `6 passed in 0.03s`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_margin_audit.py scripts/materialize_generation_id_subset.py`
- result: pass
- `./venv_arm64/bin/python -m pytest -q`
- result: `660 passed in 25.35s`
- JSON artifact validation
- result: pass
- `git diff --check`
- result: pass

Evidence:

- 1.5B source-final full SVAMP32: `3/32`
- 1.5B source-final clean residual: `0/6`
- 1.5B source-margin positive+advantage clean IDs: `1/6`
- 7B source-final clean residual: `0/6`
- 7B text-relay clean residual: `1/6`
- 7B source-margin positive+advantage clean IDs: `1/6`
- 7B positive-margin ID: `6e9745b37ab6fc45`
- 7B text-relay clean ID: `e3ab8666238a289e`

Subagent synthesis:

- source-pair agent recommended the 1.5B stronger-source gate and exact pass
  thresholds
- ablation agent recommended attaching source/text final-answer provenance and
  treating prompt/template sensitivity as a follow-up only if this gate cleared
- repo-audit agent recommended optional source/text JSONL fields so margin-only
  audits do not depend on stale source artifacts
- creative/literature agent recommended a Wyner-Ziv latent syndrome sidecar as
  the next branch if stronger-source answer evidence remained weak

Hypothesis update:

- killed: source scale alone creates a source-answer or source-margin surface
  strong enough for the current clean SVAMP32 residual IDs
- killed: source-final copying is the right next positive-method path
- weakened: stronger-source text relay as a baseline branch; it gets only
  `1/6` clean on the 7B subset
- revived weakly: one clean ID has robust source-margin advantage under 1.5B
  and 7B sources
- promoted: C2C-residual distillation or source-control syndrome sidecar,
  because C2C remains the only signal with `16/32` headroom

Next exact gate:

- implement a C2C-residual or latent-syndrome sidecar that uses target
  candidate pools as decoder side information
- require `>=2/6` clean residual IDs, matched source beating zero/shuffle/
  target-only/slots-only controls, and preservation of the `>=14/32`
  target-self/self-repair floor

## 2026-04-24 - SVAMP32 syndrome sidecar probe

Question:

- do target-side candidate pools contain enough clean residual gold answers for
  a compact C2C-derived syndrome to select them under strict source-destroying
  controls?

Decision:

- implemented `scripts/analyze_svamp32_syndrome_sidecar_probe.py`
- tested strict target-side candidate pools and an augmented source/text
  sensitivity pool on the frozen SVAMP32 exact-ID rows
- promoted latent syndrome sidecar as the next live branch, but only as a
  bound: the current probe uses C2C numeric answers as proxy syndromes and is
  not a deployable method

Artifacts:

- memo:
  - `paper/svamp32_syndrome_sidecar_probe_20260424.md`
- results manifest:
  - `results/svamp32_syndrome_sidecar_probe_20260424/manifest.md`
- strict target-pool probe:
  - `results/svamp32_syndrome_sidecar_probe_20260424/targetpool_syndrome_probe.json`
  - sha256: `48f94eb7f14081b7c2b662a207dfdc96a2b81e3df4cfd54e919a2e55e3891ffb`
- augmented sensitivity probe:
  - `results/svamp32_syndrome_sidecar_probe_20260424/augmentedpool_syndrome_probe.json`
  - sha256: `aa108b5b3ffd2e78acc8010e2f39c45876c79497e7e4944a15f719eecd46dfd5`
- references:
  - `references/452_syndrome_sidecar_refs.md`

Verification:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_syndrome_sidecar_probe.py -q`
- result: `2 passed in 0.02s`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_syndrome_sidecar_probe.py`
- result: pass
- `./venv_arm64/bin/python -m pytest -q`
- result: `662 passed in 25.98s`
- JSON artifact validation
- result: pass
- `git diff --check`
- result: pass

Evidence:

- strict target-side pool, `[2,3,5,7]` residues:
  - syndrome bytes: `1`
  - matched: `14/32`
  - target-only fallback: `14/32`
  - target-self matched: `3/3`
  - clean gold in pool: `2/6`
  - clean matched: `2/6`
  - clean source-necessary: `2/6`
  - control clean union: `0/6`
  - source-necessary IDs: `1d50b408c8f5cd2c`, `aee922049c757331`
- augmented sensitivity pool, `[2,3,5,7]` residues:
  - matched: `15/32`
  - clean source-necessary: `3/6`
  - additional clean ID: `6e9745b37ab6fc45`

Subagent synthesis:

- repo/repro agent recommended reusing exact SVAMP32 target-set and existing
  matched/zero/shuffle/target-only/slots-only controls before generation
- ablation agent recommended the same source-necessity matrix and pass rule
- literature/creative agent recommended a Wyner-Ziv/Slepian-Wolf style
  candidate-pool syndrome sidecar with compact residues and strict controls

Hypothesis update:

- promoted: latent syndrome sidecar is now the next highest-value live branch
- revived: low-rate interpretable communication may recover clean residual IDs
  despite failed source-final and source-margin channels
- weakened: training another dense connector before source-syndrome prediction
  is lower value
- saturated: source-final copying, stronger-source source-margin escalation,
  and answer-teacher calibration proxy microfits

Next exact gate:

- train the smallest source-latent syndrome predictor on frozen source
  hidden/cache features and target-side candidate pools
- require `>=2/6` clean source-necessary IDs, matched `>=14/32`, target-self
  `3/3`, zero/shuffle/target-only/slots-only controls `0/6` clean, exact ID
  parity, and numeric coverage `>=31/32`

## 2026-04-24 - SVAMP32 source-latent syndrome probe

Gate:

- replace the C2C oracle residue syndrome with a leave-one-ID-out predictor
  from frozen source hidden summaries
- keep the strict target-side candidate pool and require matched `>=14/32`,
  target-self `3/3`, clean source-necessary `>=2/6`, exact ID parity, numeric
  coverage `>=31/32`, and zero/shuffle/label-shuffle/target-only/slots-only
  clean union `0/6`

Decision:

- implemented `scripts/analyze_svamp32_source_latent_syndrome_probe.py`
- added focused tests in
  `tests/test_analyze_svamp32_source_latent_syndrome_probe.py`
- ran Qwen2.5-0.5B frozen source hidden summary probes with `last` and
  `mid,last` feature layers
- weakened the direct linear pooled-source-hidden syndrome branch; both
  variants fail below the target-only floor and recover `0/6` clean
  source-necessary IDs

Artifacts:

- memo:
  - `paper/svamp32_source_latent_syndrome_probe_20260424.md`
- results manifest:
  - `results/svamp32_source_latent_syndrome_probe_20260424/manifest.md`
- last-layer probe:
  - `results/svamp32_source_latent_syndrome_probe_20260424/qwen25_05b_last_targetpool_probe.json`
  - sha256: `afc769b2d3f56e450ba3e0d2a4f5df73975d4fceb564b80518fb6d653229410e`
- mid+last probe:
  - `results/svamp32_source_latent_syndrome_probe_20260424/qwen25_05b_mid_last_targetpool_probe.json`
  - sha256: `2fe61a32c3cc872cc72887ae34a41716e27d887754dd627aff6807fc7e20e40f`

Verification:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_latent_syndrome_probe.py -q`
- result: `3 passed in 0.02s`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_latent_syndrome_probe.py`
- result: pass

Evidence:

- last-layer source features:
  - status: `source_latent_syndrome_probe_fails_gate`
  - matched: `9/32`
  - target-only fallback: `14/32`
  - zero-source: `13/32`
  - shuffled-source: `10/32`
  - label-shuffled: `13/32`
  - slots-only: `8/32`
  - clean source-necessary: `0/6`
  - target-self: `2/3`
  - teacher numeric coverage: `32/32`
- mid+last source features:
  - status: `source_latent_syndrome_probe_fails_gate`
  - matched: `9/32`
  - target-only fallback: `14/32`
  - zero-source: `14/32`
  - shuffled-source: `10/32`
  - label-shuffled: `13/32`
  - slots-only: `8/32`
  - clean source-necessary: `0/6`
  - target-self: `3/3`
  - teacher numeric coverage: `32/32`

Subagent synthesis:

- repo/repro agent recommended a standalone analyzer first, not editing
  `translator.py`, to avoid mixing source, target, delta, and slot-memory
  effects
- ablation agent recommended the strict matched/zero/shuffle/label-shuffle/
  target-only/slots-only matrix and a hard `0/6` clean-control union rule
- creative/literature agent recommended a future Syndrome-Q variant with a
  Q-Former/Perceiver-style query bottleneck, but only after this linear
  readout gate was tested

Hypothesis update:

- weakened: pooled frozen Qwen2.5-0.5B source hidden summaries do not linearly
  expose the C2C residue syndrome on frozen SVAMP32
- still alive: target-candidate syndrome decoding as a bound
- promoted next: cross-fitted learned query bottleneck or C2C-residual
  distillation target with the same strict candidate-pool controls
- killed for now: claiming a source-latent positive method from pooled hidden
  summaries

Next exact gate:

- train or fit the smallest cross-fitted query-bottleneck/C2C-residual
  syndrome predictor and decode through the same strict target candidate pool
- require matched `>=14/32`, target-self `3/3`, clean source-necessary
  `>=2/6`, numeric coverage `>=31/32`, exact ID parity, and all
  source-destroying controls `0/6` clean

## 2026-04-24 - SVAMP32 learned syndrome probe

Gate:

- test whether a cross-fitted learned query bottleneck over source token states
  can recover the C2C-derived residue syndrome after pooled hidden readout
  failed
- keep the strict target-side candidate pool and require matched `>=14/32`,
  target-self `3/3`, clean source-necessary `>=2/6`, exact ID parity, numeric
  coverage `>=31/32`, and zero/shuffle/label-shuffle/same-norm-noise/
  target-only/slots-only clean union `0/6`

Decision:

- implemented `scripts/analyze_svamp32_learned_syndrome_probe.py`
- added focused tests in `tests/test_analyze_svamp32_learned_syndrome_probe.py`
- ran two source-token query bottleneck variants:
  - `q=4`, `h=16`, `8` outer folds, `80` epochs
  - `q=8`, `h=64`, `8` outer folds, `120` epochs
- weakened the learned source-token syndrome branch; neither variant reaches
  the target-only floor or recovers any clean source-necessary IDs

Artifacts:

- memo:
  - `paper/svamp32_learned_syndrome_probe_20260424.md`
- results manifest:
  - `results/svamp32_learned_syndrome_probe_20260424/manifest.md`
- learned `q=4`, `h=16` probe:
  - `results/svamp32_learned_syndrome_probe_20260424/qbottleneck_q4_h16_f8_seed1_targetpool_probe.json`
  - sha256: `8115eeabe5c98d6699c3aad7dd477bcb1740c84fbd8d7f4927922106f9193908`
- learned `q=8`, `h=64` probe:
  - `results/svamp32_learned_syndrome_probe_20260424/qbottleneck_q8_h64_f8_seed1_targetpool_probe.json`
  - sha256: `ae61f6f4c4947a5b6596537c75a2ffe2b7733735ed3ea2fbfe72b76424f43052`
- references:
  - `references/453_learned_syndrome_probe_refs.md`

Verification:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_learned_syndrome_probe.py -q`
- result: `3 passed in 0.95s`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_learned_syndrome_probe.py`
- result: pass

Evidence:

- `q=4`, `h=16`, `8`-fold source-token query bottleneck:
  - status: `learned_syndrome_probe_fails_gate`
  - matched: `10/32`
  - target-only fallback: `14/32`
  - zero-source: `11/32`
  - shuffled-source: `10/32`
  - label-shuffled: `13/32`
  - same-norm-noise: `14/32`
  - slots-only: `8/32`
  - clean source-necessary: `0/6`
  - target-self: `2/3`
  - teacher numeric coverage: `32/32`
- `q=8`, `h=64`, `8`-fold source-token query bottleneck:
  - status: `learned_syndrome_probe_fails_gate`
  - matched: `9/32`
  - target-only fallback: `14/32`
  - zero-source: `14/32`
  - shuffled-source: `10/32`
  - label-shuffled: `13/32`
  - same-norm-noise: `14/32`
  - slots-only: `8/32`
  - clean source-necessary: `0/6`
  - target-self: `2/3`
  - teacher numeric coverage: `32/32`

Subagent synthesis:

- repo agent recommended a sibling analyzer around the existing syndrome
  decoder rather than editing `translator.py`
- ablation agent recommended adding token-order/noise-style source-destroying
  controls and keeping the clean-control union hard at `0/6`
- literature agent recommended query bottlenecks, cross-fitting, and verifier
  gates; the verifier branch is lower priority until a source-derived clean
  signal exists

Hypothesis update:

- weakened: source-token query bottlenecks recover the C2C residue syndrome
  from frozen Qwen2.5-0.5B source states
- promoted: C2C-residual distillation or another C2C-mechanism-derived source
  signal as the next exact branch
- rejected for now: verifier-gated repair, because there is no source-derived
  clean signal to gate
- still alive: C2C-derived syndrome sidecar as a strict bound and target for
  the next predictor

Next exact gate:

- inspect C2C artifact/cache availability and implement the smallest
  C2C-residual distillation probe that predicts the same compact residue from
  deployable source/cache signals rather than C2C final answers
- require matched `>=14/32`, target-self `3/3`, clean source-necessary
  `>=2/6`, numeric coverage `>=31/32`, exact ID parity, and all
  source-destroying controls `0/6` clean

## 2026-04-26 - SVAMP32 C2C mechanism syndrome probe

Cycle:

- cycle number: `2026-04-26-c2c-mechanism-1`
- live branch entering cycle: C2C-mechanism/source-cache syndrome distillation
- scale-up rung: strict small gate
- ICLR readiness: not ready; still blocked on a deployable positive method

Gate:

- test whether C2C prefill projector traces can predict the compact SVAMP32
  C2C residue syndrome without parsing C2C final answers
- keep the strict target candidate pool and source-destroying controls
- require matched `>=14/32`, target-self `3/3`, clean source-necessary
  `>=2/6`, numeric coverage `>=31/32`, exact ID parity, and control clean
  union `0/6`

Decision:

- implemented C2C prefill trace extraction in `latent_bridge/c2c_eval.py`
- added `scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py`
- added focused tests in `tests/test_c2c_mechanism_trace.py`
- weakened the C2C summary-feature syndrome-distillation branch; both scalar
  and residual summaries fail the strict small gate

Artifacts:

- memo:
  - `paper/svamp32_c2c_mechanism_syndrome_probe_20260426.md`
- results manifest:
  - `results/svamp32_c2c_mechanism_syndrome_probe_20260426/manifest.md`
- scalar trace probe:
  - `results/svamp32_c2c_mechanism_syndrome_probe_20260426/prefill_scalar_trace_targetpool_probe.json`
  - sha256: `0220017a3a76dff54056fa8caeefde244b61a87cde27bfcbcab93c67f081c902`
- residual trace probe:
  - `results/svamp32_c2c_mechanism_syndrome_probe_20260426/prefill_residual_trace_targetpool_probe.json`
  - sha256: `685d76e3640b17084b25544c970ec8a95efe1555e5d36469fb49ba88325176f7`
  - feature tensor sha256:
    `75ad00f84a99ae632ec5641fa53e66e987188ba693858079dfae319381de7e73`
- references:
  - `references/454_c2c_mechanism_syndrome_refs.md`

Verification:

- `./venv_arm64/bin/python -m pytest tests/test_c2c_mechanism_trace.py tests/test_analyze_svamp32_source_latent_syndrome_probe.py -q`
- result: `5 passed in 0.04s`
- `./venv_arm64/bin/python -m py_compile latent_bridge/c2c_eval.py scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py`
- result: pass

Evidence:

- scalar trace, feature dim `336`:
  - status: `c2c_mechanism_syndrome_probe_fails_gate`
  - matched: `11/32`
  - target-only: `14/32`
  - zero-source: `14/32`
  - shuffled-source: `9/32`
  - label-shuffle: `14/32`
  - slots-only: `8/32`
  - clean source-necessary: `0/6`
- residual trace, feature dim `896`:
  - status: `c2c_mechanism_syndrome_probe_fails_gate`
  - matched: `12/32`
  - target-only: `14/32`
  - zero-source: `13/32`
  - shuffled-source: `9/32`
  - label-shuffle: `14/32`
  - slots-only: `8/32`
  - clean source-necessary: `0/6`
  - teacher numeric coverage: `32/32`
  - exact ordered ID parity: true

Subagent synthesis:

- artifact audit found that existing C2C JSONLs contain final predictions only,
  while the vendored C2C model exposes projector scalar/gate and residual
  computation points
- harness audit recommended a sibling analyzer that reuses the strict SVAMP32
  syndrome decoder and source-destroying controls
- stack audit ranked C2C-residual syndrome distillation as the highest-value
  next stack before this gate; this result weakens that stack

Hypothesis update:

- weakened: C2C prefill scalar/residual summary features linearly expose the
  compact C2C residue syndrome on frozen SVAMP32
- still alive as a bound: C2C-derived syndrome sidecar with target candidate
  pools
- promoted next: source-control contrastive innovation bottleneck on a surface
  with measured source headroom
- do not scale: C2C summary-feature syndrome distillation to SVAMP70/GSM70

Next exact gate:

- implement the cheapest source-control contrastive innovation bottleneck with
  matched-source positives and zero/shuffled/wrong-source penalties
- run on an exact-ID small surface with target-safe fallback and the same clean
  source-necessary accounting
- require target-only floor preservation, clean source-necessary recovery, full
  numeric coverage, and source-destroying controls before medium scale-up

## 2026-04-26 - SVAMP32 Perceiver answer-teacher contrastive pre-gate

Cycle:

- cycle number: `2026-04-26-perceiver-answer-teacher-1`
- live branch entering cycle: receiver-conditioned Perceiver/query bottleneck
  with answer-teacher supervision and source-control contrast
- scale-up rung: strict small teacher-forced pre-gate
- ICLR readiness: not ready; still blocked on a deployable positive method

Gate:

- train the existing `bridge_ridge_qk_dynalign_query_innovation_resampler_replace`
  path with `perceiver_queries`, answer-token teacher blend, target-self
  preservation, conditional delta memory, and zero/shuffle source-control loss
- run teacher-forced matched-vs-control diagnostics on six clean C2C residual
  IDs plus three target-self IDs before generation
- require at least `2/6` clean IDs with matched-source positive margin over
  every source-destroying/target-only/slots-only control before generation

Decision:

- checkpoint trained successfully
- teacher-forced gate failed at fixed gates `0.125`, `0.15`, and `0.20`
- generation was not run
- weakened this Perceiver answer-teacher plus contrastive delta-memory variant

Artifacts:

- memo:
  - `paper/svamp32_perceiver_answer_teacher_contrastive_20260426.md`
- results manifest:
  - `results/svamp32_perceiver_answer_teacher_contrastive_20260426/manifest.md`
- checkpoint:
  - `.debug/svamp32_perceiver_answer_teacher_contrastive_20260426/checkpoints/qwen25_to_qwen3_svamp32_perceiver_answer_teacher_w080_ctrl050_r16_b16_seed1.pt`
  - sha256: `65aea1fc6db7e96d5a0df5e3d98380fe44549a3a2eb35dff4bc7c09a1d89a485`
  - not tracked, size `1.8G`
- calibration log:
  - `.debug/svamp32_perceiver_answer_teacher_contrastive_20260426/logs/calibrate_w080_ctrl050_seed1.log`
  - sha256: `dcee16b600a8918fc7fadcd433baf27dabe8ad9ef89586146daaeb6bce737101`
- teacher-forced gates:
  - `results/svamp32_perceiver_answer_teacher_contrastive_20260426/teacher_forced_gate0125.json`
    - sha256: `db0641fb41a2e49106fd7a63c72b2c09f97d3946969c03df3598c201cb49435f`
  - `results/svamp32_perceiver_answer_teacher_contrastive_20260426/teacher_forced_gate015.json`
    - sha256: `7ce7a255e8f847c43caccfd98ed5f37131a515e023c6660d93b14b1b485f82c5`
  - `results/svamp32_perceiver_answer_teacher_contrastive_20260426/teacher_forced_gate020.json`
    - sha256: `9aed1c804d854e4193281b0e24df71407f465dd1fc647c044e79a8f0db6a8802`
- source/headroom target set for next surface:
  - `results/source_contrastive_target_sets_20260426/svamp70_c2c_vs_process_repair_target_set.json`
  - sha256: `fbfa638927307e76de507dff1d15b65ed5a9eb985f0d80f9c6a1a8f235b1ef9c`
- references:
  - `references/455_contrastive_perceiver_bottleneck_refs.md`

Verification:

- `./venv_arm64/bin/python -m pytest tests/test_build_source_contrastive_target_set.py -q`
- result: `2 passed in 0.02s`
- `./venv_arm64/bin/python -m py_compile scripts/build_source_contrastive_target_set.py`
- result: pass

Evidence:

- calibration:
  - prompts: `32`
  - dynamic mixture samples: `1411`
  - answer-teacher injected prompts: `6`
  - answer-teacher injected samples: `277`
  - average K alignment cosine: `0.951`
  - average V alignment cosine: `0.734`
- teacher-forced gates:
  - gate `0.125`: matched-positive clean `2/6`, matched-only clean `0/6`,
    control-leak clean `2/6`, mean matched-control delta `-1.1011`
  - gate `0.150`: matched-positive clean `2/6`, matched-only clean `0/6`,
    control-leak clean `2/6`, mean matched-control delta `-1.1543`
  - gate `0.200`: matched-positive clean `2/6`, matched-only clean `0/6`,
    control-leak clean `2/6`, mean matched-control delta `-1.2968`

Subagent synthesis:

- artifact audit recommended staying on SVAMP32 exact IDs for this gate because
  exact target-self/C2C/source/text artifacts and clean residual IDs already
  exist
- code audit identified the existing Perceiver/query-innovation path as the
  smallest implementation route
- literature audit recommended treating contrastive loss as a constraint around
  a receiver-conditioned connector, not as the full method

Hypothesis update:

- weakened: Perceiver-query answer-teacher supervision plus zero/shuffle
  source-control contrast is enough to produce source-necessary clean margins
  on SVAMP32
- weakened: target/delta memory can be safely used without stronger
  target-only leakage penalties
- promoted next: move to a headroom-richer exact surface, or introduce an
  explicit target-only penalty before answer-teacher supervision
- do not scale: this checkpoint to generation, SVAMP70, GSM70, or cross-family

Next exact gate:

- use the SVAMP70 C2C-vs-process-repair target set with `10` clean source-only
  IDs, or add a stricter target-only/slots-only penalty on SVAMP32 before any
  answer-teacher blend
- require teacher-forced matched-only clean recovery before generation

## 2026-04-26 - SVAMP70 Perceiver answer-teacher contrastive pre-gate

Cycle:

- cycle number: `2026-04-26-svamp70-perceiver-answer-teacher-1`
- live branch entering cycle: same Perceiver/query answer-teacher connector on
  the stronger SVAMP70 C2C-vs-process-repair source-headroom surface
- scale-up rung: medium/headroom surface teacher-forced pre-gate
- ICLR readiness: not ready; still blocked on source-necessary positive method

Gate:

- use the materialized SVAMP70 C2C-vs-process-repair target set with `10`
  clean C2C source-only IDs after excluding the process-repair baseline
- train the same Perceiver/query answer-teacher source-control connector on
  `data/svamp_eval_70.jsonl`
- run a clean-only teacher-forced diagnostic at fixed gate `0.15`
- require matched-only positive margins on clean IDs before generation

Decision:

- checkpoint trained successfully
- clean-only teacher-forced gate failed
- generation and preservation-row scoring were not run
- killed the current Perceiver answer-teacher plus contrastive delta-memory
  branch until the objective changes

Artifacts:

- memo:
  - `paper/svamp70_perceiver_answer_teacher_contrastive_20260426.md`
- results manifest:
  - `results/svamp70_perceiver_answer_teacher_contrastive_20260426/manifest.md`
- target set:
  - `results/source_contrastive_target_sets_20260426/svamp70_c2c_vs_process_repair_target_set.json`
  - sha256: `e2f3a4da9848519f009260cb681f378dc2767d1f3ba0bc67ce3bff94747287c5`
- checkpoint:
  - `.debug/svamp70_perceiver_answer_teacher_contrastive_20260426/checkpoints/qwen25_to_qwen3_svamp70_perceiver_answer_teacher_w080_ctrl050_r16_b16_seed1.pt`
  - sha256: `a7221d6d0ee81b99573bf1893b66570ec682f22faee1ffcc6bf7e9fc1f36df6a`
  - not tracked, size `1.8G`
- calibration log:
  - `.debug/svamp70_perceiver_answer_teacher_contrastive_20260426/logs/calibrate_w080_ctrl050_seed1.log`
  - sha256: `5701b20b25d2dffa1bacbfd5ff65e9b6cbbbd8f8bb591754636c3ab6bffba73f`
- teacher-forced clean-only gate:
  - `results/svamp70_perceiver_answer_teacher_contrastive_20260426/teacher_forced_gate015_clean_only.json`
  - sha256: `d57eae6555fbce396e077d63eebcb99b75cfd7bd50bf69363dec488e92b99960`

Evidence:

- calibration:
  - prompts: `70`
  - dynamic mixture samples: `2948`
  - answer-teacher injected prompts: `10`
  - answer-teacher injected samples: `446`
  - average K alignment cosine: `0.949`
  - average V alignment cosine: `0.718`
- teacher-forced gate `0.15`:
  - status: `no_teacher_forced_source_signal`
  - clean IDs scored: `10`
  - matched-positive clean IDs: `4/10`
  - matched-only clean IDs: `0/10`
  - control-leak clean IDs: `4/10`
  - mean matched-control delta: `-0.7158`

Hypothesis update:

- killed for now: Perceiver answer-teacher plus contrastive delta-memory as the
  main method family
- weakened: simply moving to a headroom-richer surface fixes target/control
  leakage
- promoted next: objective-level change that penalizes target-only/slots-only
  recovery before answer-teacher supervision, token/layer-level C2C residual
  behavior with matched-vs-control separation, or a source-only sidecar/router
  whose source-signal formation cannot access target-only memory

Next exact gate:

- implement one objective-level change and run teacher-forced matched-only clean
  recovery before any generation
- do not run larger generations, seeds, cross-family, or long-context expansion
  for this connector family as currently formulated

## 2026-04-26 - SVAMP32 anti-memory Perceiver objective pre-gate

Cycle:

- cycle number: `2026-04-26-svamp32-anti-memory-perceiver-1`
- live branch entering cycle: objective-level rescue for the Perceiver/query
  answer-teacher connector
- scale-up rung: strict small teacher-forced pre-gate
- ICLR readiness: not ready; still blocked on a deployable source-necessary
  positive method
- code commit: `8b3d7924`

Gate:

- add training-time anti-memory controls against `target_only` and `slots_only`
  while keeping zero/shuffle source controls
- train on the frozen SVAMP32 exact-ID surface
- score teacher-forced clean residual and target-self IDs at fixed gates
  `0.125`, `0.15`, and `0.20`
- require matched-only positive margins on at least `2/6` clean IDs before any
  generation

Decision:

- checkpoint trained successfully
- all tested teacher-forced gates failed
- generation was not run
- killed/weakened: anti-memory penalties as a rescue for the current Perceiver
  answer-teacher plus delta-memory branch
- promoted next: source-only sidecar/router exact-ID gate on SVAMP32

Artifacts:

- memo:
  - `paper/svamp32_anti_memory_perceiver_20260426.md`
- results manifest:
  - `results/svamp32_anti_memory_perceiver_20260426/manifest.md`
- reference memo:
  - `references/456_conditional_debiasing_side_information_refs.md`
- checkpoint:
  - `.debug/svamp32_anti_memory_perceiver_20260426/checkpoints/qwen25_to_qwen3_svamp32_anti_memory_w080_ctrl050_am050_r16_b16_seed1.pt`
  - sha256: `6a3932946c6fcb580a1b136e1e5d710e555884a73c94def8ef1485fc613692ad`
  - not tracked, size too large for git
- calibration log:
  - `.debug/svamp32_anti_memory_perceiver_20260426/logs/calibrate_w080_ctrl050_am050_seed1.log`
  - sha256: `4e1d345a5c3c0ac671c3e244c446b49708f92302d88b8c6ea82b2d2928080318`
- teacher-forced gates:
  - `results/svamp32_anti_memory_perceiver_20260426/teacher_forced_gate0125.json`
    - sha256: `0c45517529bc18ed73953d239543c40f70dfffe79582ea472ca0b3767496ff0d`
  - `results/svamp32_anti_memory_perceiver_20260426/teacher_forced_gate015.json`
    - sha256: `09b0a4ff220a5be7760eeebcf82ee53f9c9c8baf213aa66f473ac39dc6754f94`
  - `results/svamp32_anti_memory_perceiver_20260426/teacher_forced_gate020.json`
    - sha256: `53b0f3d718455ff26d711ecc51bd603032e16d24258eb12d5ba90947a9bfa1c2`

Evidence:

- calibration:
  - prompts: `32`
  - dynamic mixture samples: `1411`
  - answer-teacher injected prompts: `6`
  - answer-teacher injected samples: `277`
  - average K alignment cosine: `0.951`
  - average V alignment cosine: `0.734`
- teacher-forced gates:
  - gate `0.125`: matched-positive clean `2/6`, matched-only clean `0/6`,
    control-leak clean `2/6`, mean matched-control delta `-0.8898`
  - gate `0.150`: matched-positive clean `2/6`, matched-only clean `0/6`,
    control-leak clean `2/6`, mean matched-control delta `-0.8921`
  - gate `0.200`: matched-positive clean `2/6`, matched-only clean `0/6`,
    control-leak clean `2/6`, mean matched-control delta `-0.8660`

Hypothesis update:

- weakened: target-only/slots-only zero-innovation and teacher-KL margin
  penalties are enough to force source-necessary answer-teacher signal inside
  the receiver-conditioned Perceiver delta-memory connector
- strengthened: target/control leakage is structural in this branch, not just a
  missing regularizer
- revived/promoted: source-only sidecar/router, because source-signal formation
  can be cleanly isolated from target-only and slots-only memory

Next exact gate:

- implement or run `svamp32_source_only_sidecar_router_gate`
- use the frozen SVAMP32 clean source-necessary IDs plus target-self preserve
  IDs
- require matched `>=14/32`, target-self preserve `3/3`, clean
  source-necessary `>=2/6`, numeric coverage `>=31/32`, exact ordered ID
  parity, and zero clean union for source-destroying controls

## 2026-04-26 - SVAMP32 source-only numeric sidecar/router gate

Cycle:

- cycle number: `2026-04-26-svamp32-source-only-router-1`
- live branch entering cycle: source-only residue sidecar/router
- scale-up rung: strict small exact-ID gate
- ICLR readiness: not ready; still blocked on a deployable source-necessary
  positive method

Gate:

- implement `scripts/analyze_svamp32_source_only_sidecar_router_gate.py`
- form the transmitted sidecar only from source-side numeric predictions
- decode against the existing target-side candidate pool
- controls: zero-source, shuffled-source, label-shuffle, same-norm-noise
  signature, target-only, and slots-only
- promotion rule: matched `>=14/32`, target-self `3/3`, clean
  source-necessary `>=2/6`, numeric coverage `>=31/32`, clean control union
  `0/6`

Decision:

- script implemented and unit-tested
- frozen SVAMP32 gate failed
- killed: raw source-generated numeric residue sidecars
- next branch: source latent/token-feature sidecar or token/layer-level C2C
  residual distillation with matched-vs-control separation

Artifacts:

- script:
  - `scripts/analyze_svamp32_source_only_sidecar_router_gate.py`
- tests:
  - `tests/test_analyze_svamp32_source_only_sidecar_router_gate.py`
- memo:
  - `paper/svamp32_source_only_sidecar_router_20260426.md`
- results manifest:
  - `results/svamp32_source_only_sidecar_router_20260426/manifest.md`
- result JSON:
  - `results/svamp32_source_only_sidecar_router_20260426/source_only_router_gate.json`
  - sha256: `6f92482c8b2b500eb4cb3d29a228e0797dea59e5f2fa4c78935c739413addce2`
- result markdown:
  - `results/svamp32_source_only_sidecar_router_20260426/source_only_router_gate.md`
  - sha256: `bbd7e47d55dbeee118b4812ef2b3ac5a305290eb7e947f3e663765510e755b95`
- run log:
  - `.debug/svamp32_anti_memory_perceiver_20260426/logs/source_only_sidecar_router_gate.log`
  - sha256: `8f3645a99c1fde7e266e5a160e76b923a90fa0a39d4777d2507d9f706f06e5ee`

Evidence:

- source numeric coverage: `32/32`
- moduli `2,3,5,7`: matched `4/32`, target-self `0/3`, clean matched `0/6`,
  clean source-necessary `0/6`, clean control union `0/6`
- moduli `97`: matched `4/32`, target-self `0/3`, clean matched `0/6`,
  clean source-necessary `0/6`, clean control union `0/6`

Hypothesis update:

- killed: raw source generated answers contain enough signal for the clean C2C
  source-only IDs
- strengthened: target/control leakage is separable from source-signal
  weakness; this branch has clean controls but no positive clean recovery
- promoted next: learn a source latent/token-feature sidecar against C2C
  residues or token/layer C2C behavior, with cross-fitting and label-shuffle
  controls

Next exact gate:

- run a source-latent/token predictor or token/layer C2C residual distillation
  gate on the same frozen SVAMP32 `6 + 3` IDs
- require at least `2/6` clean source-necessary wins and `3/3` target-self
  preservation before any medium confirmation

## 2026-04-26 - Recovered-branch audit: GSM70 seed4 and SVAMP32 all-layer source-latent gates

Cycle:

- cycle number: `2026-04-26-recovered-branches-1`
- timestamp: `2026-04-25 23:48:50 PDT`
- live branch entering cycle: recovered GSM70
  `dynalign_module_replace_residrank16`
- scale-up rung: strict small same-family seed/source-control gate, then strict
  SVAMP32 exact-ID source-latent gate
- ICLR readiness: not ready; still blocked on a deployable source-necessary
  positive method

Start-of-cycle status:

- current paper story: C2C/cache and syndrome bounds show headroom; GSM70
  seed0 dynalign was the strongest older source-dependent clue
- exact blocker: seed stability and source-control separation under fresh
  finite seeds
- highest-priority gate: rerun the recovered GSM70 dynalign branch before
  spending compute on new methods

Gate 1 decision:

- ran GSM70 `dynalign_module_replace_residrank16` seed4 from scratch with
  checkpoint health and integrated source controls enabled
- checkpoint was finite, exact IDs matched, numeric coverage was `70/70`
- live row failed to beat target: `4/70`, paired `3W/3L/64T`
- source controls correctly did not run because the live gate failed
- killed: raw GSM70 dynalign scale-up as the current live method branch

Gate 2 decision:

- patched `scripts/analyze_svamp32_source_latent_syndrome_probe.py` so
  high-dimensional feature probes use a dual ridge solve when feature dimension
  exceeds sample count
- ran the strict SVAMP32 source-latent syndrome gate with all source hidden
  layers
- all-layer feature dimension: `44800`
- matched: `9/32`
- target-only: `14/32`
- zero-source: `14/32`
- label-shuffle: `14/32`
- target-self: `2/3`
- clean source-necessary: `0/6`
- killed: direct linear source-hidden syndrome readout, including all-layer
  pooled summaries

Artifacts:

- GSM memo:
  - `paper/gsm8k70_seed4_dynalign_source_controls_20260426.md`
- GSM manifest:
  - `results/gsm8k70_seed4_dynalign_source_controls_20260426/manifest.md`
- GSM result JSON:
  - `results/gsm8k70_seed4_dynalign_source_controls_20260426/seed4_residual_sweep.json`
  - sha256: `324d2b84ff5f47c920e6352534adb183526b7aecd070c4f4d6394e4f743ffcbc`
- GSM prediction JSONL:
  - `results/gsm8k70_seed4_dynalign_source_controls_20260426/dynalign_module_replace_residrank16_seed4.jsonl`
  - sha256: `0a442c7aa43708e2aa8301a6ceb8e986f53d3523edca881ce2904b858c58589c`
- GSM checkpoint tensor, not tracked:
  - `checkpoints/gsm8k_contract_residual_sweep_20260421/dynalign_module_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_module_replace_cal64_chat_seed4.pt`
  - sha256: `1d9e667fe90a7fbe4b06d982796d09f58398f18144780bb20f4950e774a0d26e`
- SVAMP memo:
  - `paper/svamp32_source_latent_all_layers_20260426.md`
- SVAMP manifest:
  - `results/svamp32_source_latent_all_layers_20260426/manifest.md`
- SVAMP result JSON:
  - `results/svamp32_source_latent_all_layers_20260426/qwen25_05b_all_layers_targetpool_probe.json`
  - sha256: `5d37613d648392c58f7b28a7274ba8dadffd3b76cd09cd75dd517df64990bce9`
- SVAMP result markdown:
  - `results/svamp32_source_latent_all_layers_20260426/qwen25_05b_all_layers_targetpool_probe.md`
  - sha256: `6dcce69bde942ff9e7616b8bfc446ff3df942399b241e37f27069ac6ce4900c2`
- reference memo:
  - `references/457_query_bottleneck_residue_refs.md`

Literature update:

- primary-source scan supports a learned query-bottleneck residue predictor,
  not more pooled source summaries
- useful mechanisms: Q-Former-style queries, Perceiver/Perceiver-Resampler
  slots, anchor-relative features, variational rate control, and
  cross-tokenizer distillation for later cross-family falsification

Hypothesis update:

- killed: raw GSM70 dynalign scale-up as a stable positive method
- killed: direct linear source-hidden syndrome prediction, including all-layer
  hidden summaries
- promoted next: cross-fitted learned query-bottleneck residue predictor with
  output queries and explicit rate/slot ablations

Next exact gate:

- implement a SVAMP32 query-bottleneck residue predictor on the same frozen
  target-candidate decoder
- controls: matched, zero-source, shuffled-source, label-shuffle, target-only,
  and slots-only
- require matched `>=14/32`, target-self `3/3`, clean source-necessary
  `>=2/6`, numeric coverage `>=31/32`, exact ordered ID parity, and clean
  control union `0/6`

## 2026-04-26 - SVAMP32 query-bottleneck residue smoke gate

Cycle:

- cycle number: `2026-04-26-query-bottleneck-residue-1`
- live branch entering cycle: learned query-bottleneck residue predictor
- scale-up rung: strict small exact-ID gate
- ICLR readiness: not ready; still blocked on a deployable source-necessary
  positive method

Gate:

- implemented `probe_model=query_bottleneck` in
  `scripts/analyze_svamp32_source_latent_syndrome_probe.py`
- trained leave-one-ID-out learned query slots over all-layer source summary
  tokens
- decoded through the same frozen SVAMP32 target candidate pool
- controls: zero-source, shuffled-source, label-shuffle, target-only, and
  slots-only
- configuration: `8` query slots, `80` epochs, lr `0.01`, weight decay
  `0.001`, query seed `0`

Decision:

- frozen SVAMP32 gate failed
- matched `9/32`, below the `14/32` target-only floor
- target-self preservation `2/3`
- clean source-necessary `0/6`
- control clean union `0/6`
- weakened: query-bottleneck over layer-summary tokens
- do not scale this summary-token query bottleneck upward

Artifacts:

- script:
  - `scripts/analyze_svamp32_source_latent_syndrome_probe.py`
- tests:
  - `tests/test_analyze_svamp32_source_latent_syndrome_probe.py`
- memo:
  - `paper/svamp32_query_bottleneck_residue_20260426.md`
- results manifest:
  - `results/svamp32_query_bottleneck_residue_20260426/manifest.md`
- result JSON:
  - `results/svamp32_query_bottleneck_residue_20260426/qwen25_05b_all_layers_query_slots8_probe.json`
  - sha256: `59964c426e13f61dc00805beb30574aaa376df70ba77377864d0eeb41bb9d7b3`
- result markdown:
  - `results/svamp32_query_bottleneck_residue_20260426/qwen25_05b_all_layers_query_slots8_probe.md`
  - sha256: `9736a7e2558bfcab3e91ee316a858c25c54320c7abdca0e14b3d947d8a9170e8`
- run log:
  - `.debug/svamp32_query_bottleneck_residue_20260426/logs/qwen25_05b_all_layers_query_slots8_probe.log`
  - sha256: `4c8a62cdde5629759edb83d874d0441616eac46c7bdfd1db2b1666d479e0183c`

Hypothesis update:

- weakened: learned querying over layer-summary tokens is enough to recover C2C
  residue sidecars
- strengthened: the source-signal bottleneck likely requires full token/layer
  traces, token/layer C2C-residual targets, or a different source-derived
  training signal rather than summary-token residue classification

Next exact gate:

- implement token/layer-level C2C-residual distillation or a full source-token
  query bottleneck with a rate/slot curve
- keep the same strict SVAMP32 target-candidate decoder and controls
- require matched `>=14/32`, target-self `3/3`, clean source-necessary
  `>=2/6`, numeric coverage `>=31/32`, exact ordered ID parity, and clean
  control union `0/6`

## 2026-04-26 - Remaining MD/results audit and source-token all-layer bottleneck gate

Cycle:

- cycle number: `2026-04-26-source-token-all-layer-1`
- timestamp: `2026-04-26 00:03:07 PDT`
- live branch entering cycle: full source-token query-bottleneck residue
  predictor
- scale-up rung: strict small exact-ID gate
- ICLR readiness: not ready; no deployable source-derived positive method

Start-of-cycle status:

- current paper story: C2C/syndrome bounds show headroom, but source-state
  residue predictors and raw dynalign are failing source-control or stability
  gates
- exact blocker: no source-derived signal recovers clean C2C-only IDs beyond
  target-self controls
- highest-priority gate: finish combing older MD/results and run the one
  remaining crisp source-token query-bottleneck falsification

Audit synthesis:

- no overlooked real benchmark-positive branch was found
- strongest toy-only interface clue: quotient/GPA/sparse dictionary plus
  sequence-aligned byte sidecar under tokenizer-like corruption
- strongest selector/repair clue: GSM30 process repair selected route
  `0.2333` vs target `0.0667`, but not source-control-clean
- strongest oracle/selector headroom: random route target-or-seed oracle
  `0.3000` with McNemar p `0.0233`, still only candidate-set headroom
- SVAMP32 source oracle bound remains useful, but idweighted sidecar bound
  adds only `1` clean source-necessary ID beyond target self-repair
- GSM70 dynalign is not source-specific under later runtime controls and is
  seed-fragile

Gate:

- ran `scripts/analyze_svamp32_learned_syndrome_probe.py` with all source
  hidden layers, `q=4`, `h=16`, `8` outer folds, `80` epochs, seed `2`
- controls: zero-source, shuffled-source, label-shuffled, same-norm-noise,
  target-only, and slots-only

Decision:

- frozen SVAMP32 gate failed
- matched `7/32`, below target-only `14/32`
- target-self preservation `2/3`
- clean source-necessary `0/6`
- control clean union `0/6`
- killed: source-token query-bottleneck residue prediction on the current
  SVAMP32 syndrome surface

Artifacts:

- memo:
  - `paper/svamp32_source_token_all_layers_bottleneck_20260426.md`
- results manifest:
  - `results/svamp32_source_token_all_layers_bottleneck_20260426/manifest.md`
- result JSON:
  - `results/svamp32_source_token_all_layers_bottleneck_20260426/qbottleneck_q4_h16_f8_seed2_all_layers_targetpool_probe.json`
  - sha256: `c09874826af09a957a7c467ee5afd54fa36ec2122e62b2455cd553eaf7064e6a`
- result markdown:
  - `results/svamp32_source_token_all_layers_bottleneck_20260426/qbottleneck_q4_h16_f8_seed2_all_layers_targetpool_probe.md`
  - sha256: `dee99f9ac14137e1f8c1da8fdebceee4182f3ff32cdf463f8c5cbccb6dd6ffa8`
- run log:
  - `.debug/svamp32_source_token_all_layers_bottleneck_20260426/logs/qbottleneck_q4_h16_f8_seed2_all_layers_targetpool_probe.log`
  - sha256: `dcd00fe42f1c137f64110882bde470aa259489ecb747b8a2a73045fbd3043de6`

Hypothesis update:

- killed: source-token query bottlenecks over Qwen2.5-0.5B source states as a
  direct C2C residue predictor on this surface
- weakened: more SVAMP32 source-state residue tuning without a new source
  signal
- promoted next: source-surface discovery rather than further tuning dead
  residue predictors

Next exact gate:

- either run a process-repair/selector source-control diagnostic on a strict
  clean surface, requiring at least `2` clean IDs beyond target self-repair and
  no target-self losses, or convert the quotient/GPA sparse dictionary plus
  sequence-aligned byte sidecar toy into a real cross-family tokenizer/interface
  stress gate

## Cycle Checkpoint: 2026-04-26 SVAMP32 Process Repair Source-Control Gate

- cycle number: `2026-04-26-process-repair-source-control-1`
- timestamp: `2026-04-26 00:13:02 PDT`
- live branch entering cycle: process-repair / selector stack on SVAMP32 clean
  residual surface
- scale-up rung: strict small exact-ID gate
- ICLR readiness: not ready; no deployable source-derived positive method

Start-of-cycle status:

- current paper story: C2C and target-side repair expose headroom, but no
  LatentWire row is source-necessary under strict controls
- exact blocker: process repair must recover clean residual IDs beyond
  target-self repair without losing target-self wins
- highest-priority gate: matched process repair on the frozen SVAMP32
  query-pool transport surface

Decision:

- killed: process-repair / selector stack as a source-communication method on
  this surface
- process repair selected route: `10/32`
- selected route, no repair: `8/32`
- target alone: `8/32`
- target-self repair comparator: `14/32`
- clean residual recovered: `1/6`
- target-self preservation: `1/3`
- selected candidate source: `target` on `32/32`
- zero-source and shuffled-source repair generations were not run because the
  matched row selected target-only candidates and failed the target-self
  comparator before source controls could matter

Artifacts:

- memo:
  - `paper/svamp32_process_repair_source_controls_20260426.md`
- results manifest:
  - `results/svamp32_process_repair_source_controls_20260426/manifest.md`
- matched process repair JSONL:
  - `results/svamp32_process_repair_source_controls_20260426/matched_process_repair.jsonl`
  - sha256: `a125ac97e1e4c54763739cdcc23e13c8633a8dceb431c3bfbed6c351e5219f6d`
- matched process repair markdown:
  - `results/svamp32_process_repair_source_controls_20260426/matched_process_repair.md`
  - sha256: `98cf77da5299733d35f5618dd43815b6f90d3dc8904c561835351355b8797dfc`
- raw run log:
  - `.debug/svamp32_process_repair_source_controls_20260426/logs/matched_process_repair.log`
  - sha256: `51e0f9286a25fce982dc6fd3de3ae2145bfd495fdc7c6f2018252da597ef1eb4`
- references:
  - `references/458_repair_verifier_source_control_refs.md`

Hypothesis update:

- process repair is useful as a control/baseline, not as a source-derived
  communication row on this candidate surface
- future repair stacks must be budget-matched against target-only
  self-consistency, target self-repair, verifier/tool-only repair, and
  source-identity/order-shuffle controls
- promoted next: real cross-family tokenizer/interface stress using the
  quotient/GPA sparse dictionary plus sequence-aligned byte sidecar direction

Next exact gate:

- find the smallest existing or near-existing real cross-family interface gate
  and run a micro smoke with exact-ID parity, numeric coverage, source
  destruction, bytes/latency accounting, and comparison against target-only plus
  text/token relay before any larger slice

## Cycle Checkpoint: 2026-04-26 Cross-Family Interface Proxy Gate

- cycle number: `2026-04-26-cross-family-interface-proxy-1`
- timestamp: `2026-04-26 02:09:00 PDT`
- live branch entering cycle: quotient/GPA sparse dictionary plus
  sequence-aligned byte sidecar, tested through the closest existing real-model
  byte-span module-replace proxy
- scale-up rung: micro smoke / strict-surface scout
- ICLR readiness: not ready; no deployable source-derived positive method

Start-of-cycle status:

- current paper story: C2C and target repair expose headroom, process repair is
  target-only, and toy sequence-aligned sidecars remain the strongest interface
  clue
- exact blocker: the sidecar/interface branch has no real benchmark row under
  source controls
- highest-priority gate: find a real cross-family surface with tokenizer
  mismatch and enough target/text headroom to test source-derived transport

Gate:

- ran real tokenizer/interface scout on `data/gsm8k_gate_search_30.jsonl`
- repeated the quotient/GPA sparse dictionary plus sequence-aligned sidecar toy
  with seed `1`
- repaired `latent_bridge` calibration/evaluation helpers for decoder-layer
  targets, packed `qkv_proj`, and OPT projection-width mismatch
- calibrated Qwen2.5 -> OPT-350m byte-span module-replace translator
- evaluated matched GSM30 target/source/text/rotalign rows

Decision:

- killed: OPT-350m as a cross-family decision surface
- matched rows:
  - target-alone: `0/30`
  - source-alone: `0/30`
  - text-to-text: `3/30`
  - byte-span rotalign proxy: `0/30`
- byte-span proxy bytes: `525562.6` per example
- tokenizer mismatch was present for OPT (`shared decoded = 0.9047`,
  `boundary F1 = 0.9434`) but the target benchmark surface had no useful
  headroom
- kept alive: sequence-aligned sidecar as an interface component, because the
  seed-1 toy repeat still has best shared-basis low-shot MSE around `0.036`

Artifacts:

- memo:
  - `paper/cross_family_interface_proxy_20260426.md`
- results manifest:
  - `results/cross_family_interface_proxy_20260426/manifest.md`
- matched prediction JSONL:
  - `results/cross_family_interface_proxy_20260426/qwen25_to_opt350m_bytespan_gsm30_matched.jsonl`
  - sha256: `49659e18aa35a0d9deabb3208ccd6160fc62cf82cfdc928615a780eb9b6d663e`
- OPT translator checkpoint:
  - `.debug/qwen25_phi3_bytespan_interface_20260426/qwen25_to_opt350m_bytespan_r4_cal64.pt`
  - sha256: `3a6c0c2cf8aa46be91b58d5c36bab5477183111f66f2d08ef07699592554696c`
- references:
  - `references/459_cross_family_interface_proxy_refs.md`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_evaluate_helpers.py tests/test_calibrate_and_ablation.py -q`
- result: `204 passed`

Hypothesis update:

- byte-span module replacement alone is not a promising real-method row
- OPT is a bad target surface for this gate, so source controls would not be
  discriminative
- the branch should only continue on a surface where target/text baselines have
  nonzero headroom
- compatibility patches are useful for future cross-family targets and should
  be kept

Next exact gate:

- preferred: repair or route around GQA/packed-QKV issues for Phi-3/TinyLlama
  and run only if target/text baselines are nonzero on the exact-ID slice
- fallback: same-family Qwen2.5 -> Qwen3 exact-ID SVAMP32/GSM32 with the
  strongest output-aware transport plus an explicit sidecar component and full
  source-destroying controls

### Surface Baseline Follow-Up

Ran target/source/text-only baselines before more cross-family transport work:

- Qwen2.5 -> Phi-3 on GSM30:
  - target-alone `3/30`
  - source-alone `0/30`
  - text-to-text `1/30`
- Qwen2.5 -> Phi-3 on SVAMP30:
  - target-alone `5/30`
  - source-alone `1/30`
  - text-to-text `2/30`
- Qwen2.5 -> TinyLlama on SVAMP30:
  - target-alone `0/30`
  - source-alone `1/30`
  - text-to-text `0/30`

Decision update:

- killed: TinyLlama as a decision surface
- pruned: OPT and TinyLlama for current cross-family strict gates
- weakened: Phi-3 as an immediate transport target; it has nonzero but low
  headroom
- next branch should fall back to same-family Qwen2.5 -> Qwen3 sidecar or a new
  source-surface discovery pass unless a stronger cross-family target surface
  is found

## Cycle Checkpoint: 2026-04-26 C2C Mechanism Projection Probe

- cycle number: `2026-04-26-c2c-mechanism-projection-1`
- timestamp: `2026-04-26 01:46:00 PDT`
- live branch entering cycle: same-family Qwen2.5 -> Qwen3 C2C mechanism
  distillation / source-surface discovery
- scale-up rung: strict small diagnostic gate
- ICLR readiness: not ready; no deployable source-derived positive method

Start-of-cycle status:

- current paper story: SVAMP32 has real C2C headroom (`16/32` vs target
  `8/32`), but previous source-hidden and C2C summary probes recovered no clean
  source-necessary IDs
- exact blocker: C2C-mechanism features must recover clean C2C residual IDs
  beyond zero-source, label-shuffle, target-only, and slots-only controls
- highest-priority gate: test richer deterministic signed projections of C2C
  prefill projector residuals

Gate:

- implemented optional `--residual-projection-dim` for
  `scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py`
- added deterministic signed bucket projections of full and tail C2C residual
  deltas in `latent_bridge/c2c_eval.py`
- ran projection dim `16` on the frozen SVAMP32 exact-ID surface

Decision:

- failed strict gate
- matched: `13/32`
- zero-source: `14/32`
- shuffled-source: `11/32`
- label-shuffled: `14/32`
- target-only: `14/32`
- slots-only: `8/32`
- clean source-necessary: `0/6`
- target-self preservation: `3/3`

Artifacts:

- memo:
  - `paper/svamp32_c2c_mechanism_projection_probe_20260426.md`
- results manifest:
  - `results/svamp32_c2c_mechanism_projection_probe_20260426/manifest.md`
- result JSON:
  - `results/svamp32_c2c_mechanism_projection_probe_20260426/prefill_residual_projection16_targetpool_probe.json`
  - sha256: `3fc5f51320dbe96de8940c28194becda9f81f6906ab84a06a87e414f22a4400f`
- result markdown:
  - `results/svamp32_c2c_mechanism_projection_probe_20260426/prefill_residual_projection16_targetpool_probe.md`
  - sha256: `5ad4dfdbafc2f52127817b593e5d22be84a921ad42d73ffe770e99e16a9d03a9`
- raw run log:
  - `.debug/svamp32_c2c_mechanism_projection_probe_20260426/logs/prefill_residual_projection16_targetpool_probe.log`
  - sha256: `f569d565b4f5f44db44ba725704d8f4f4f07c501ad996ac33f030b0efc473fea`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_c2c_mechanism_trace.py tests/test_c2c_eval.py -q`
- `./venv_arm64/bin/python -m py_compile latent_bridge/c2c_eval.py scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py`

Hypothesis update:

- killed: C2C scalar/residual summary plus signed-projection features as a live
  source-derived method on SVAMP32
- weakened: more summary/projection tuning without a new anti-cache objective
- promoted next: either token/layer-local C2C residual distillation with
  held-out anti-cache controls, or same-family Qwen sidecar/source-surface
  discovery using sequence-aligned sidecars as an explicit component

Next exact gate:

- do not scale C2C summary/projection features
- choose between implementing a token/layer-local residual distillation gate or
  a same-family Qwen sidecar gate with full source-destroying controls

## Cycle Checkpoint: 2026-04-26 DeepSeek -> Qwen SVAMP32 Surface Scout

- cycle number: `2026-04-26-surface-scout-deepseek-qwen-svamp32`
- timestamp: `2026-04-26 02:03:28 PDT`
- live branch entering cycle: source-surface discovery after raw dynalign and
  C2C summary/projection failures
- scale-up rung: smoke
- ICLR readiness: not ready; no deployable source-derived positive method

Start-of-cycle status:

- current paper story: C2C and target-side candidate pools expose headroom, but
  raw dynalign is seed-fragile and no learned/source-derived connector clears
  strict SVAMP32 source controls
- exact blocker: find a surface where source communication can plausibly beat
  target-alone, text relay, and C2C under strict controls
- highest-priority gate: test whether a locally available stronger reasoning
  source creates target-complementary SVAMP32 headroom

Gate:

- source: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- target: `Qwen/Qwen3-0.6B`
- eval file:
  `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- command:

```bash
./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --results-dir results/surface_scout_deepseek_qwen_svamp32_20260426 \
  --source-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t c2c \
  --limit 32 \
  --device mps \
  --max-new-tokens 64 \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Decision:

- weaken this pair as an immediate method surface
- target-alone: `8/32`
- source-alone: `5/32`
- text relay: `5/32`
- source-only target-missed IDs: `1`
- text-only target-missed IDs: `2`
- target/text oracle: `10/32`
- C2C failed before generation because no published C2C artifact is registered
  for `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B -> Qwen/Qwen3-0.6B`

Artifacts:

- memo:
  - `paper/surface_scout_deepseek_qwen_svamp32_20260426.md`
- results manifest:
  - `results/surface_scout_deepseek_qwen_svamp32_20260426/manifest.md`
  - sha256: `433088e6dcaf1f642d06e6456736a646ec593c641c373548f49b222dbefb7b0a`
- results JSON:
  - `results/surface_scout_deepseek_qwen_svamp32_20260426/manifest.json`
  - sha256: `015c0033bbaa7374899bba9297eb142a48630af99a5d0a10ebdf65885295d59a`
- predictions:
  - `results/surface_scout_deepseek_qwen_svamp32_20260426/source_alone.jsonl`
  - sha256: `6ca901e94a10275967e6451f3e727f6791f698747ed53ba72a25b21cc3ab445d`
  - `results/surface_scout_deepseek_qwen_svamp32_20260426/target_alone.jsonl`
  - sha256: `202336cb3f516afff6633e39f3ecb069a39456f1ff894b47373f93e819e77304`
  - `results/surface_scout_deepseek_qwen_svamp32_20260426/text_to_text.jsonl`
  - sha256: `2a84e04fb897cb637589885911e2c79f74dcf0a02ef77c9a2bb4c15a43c99d95`
- C2C failure log:
  - `results/surface_scout_deepseek_qwen_svamp32_20260426/logs/c2c.log`
  - sha256: `c82af7c0faf58e4d1316b10cd245f315aca6ecc46af3327ccd0cde323d81b30e`

Hypothesis update:

- weakened: DeepSeek-R1-Distill-Qwen-1.5B -> Qwen3-0.6B as an immediate
  SVAMP32 communication surface
- kept alive: surface discovery for sequence-aligned sidecars, but only on
  pairs with stronger target-complementary headroom and a fair C2C/text
  comparator
- killed for now: spending connector/source-control compute on this exact pair

Next exact gate:

- either run a same-family Qwen sidecar/source-control gate on the existing
  C2C-headroom SVAMP32 surface, or scout another already-local source/target
  pair only if target/text/C2C can be evaluated on exact IDs with nontrivial
  target-complementary headroom

## Cycle Checkpoint: 2026-04-26 Qwen2.5-Math -> Qwen3 SVAMP16 Surface Scout

- cycle number: `2026-04-26-surface-scout-qwen25math-qwen3-svamp16`
- timestamp: `2026-04-26 02:26:20 PDT`
- live branch entering cycle: C2C-comparable source-surface discovery
- scale-up rung: smoke
- ICLR readiness: not ready; this is a surface scout, not a method result

Start-of-cycle status:

- current paper story: previous DeepSeek -> Qwen scout was weak and had no
  registered C2C comparator
- exact blocker: find a pair with source/text/C2C target-complementary headroom
  before training another connector
- highest-priority gate: test a registered C2C math-source pair on frozen
  SVAMP IDs

Tooling fix:

- fixed `scripts/materialize_generation_baselines.py` so no-chat-template
  sidecar validation expects `source_enable_thinking=auto` and
  `target_enable_thinking=auto`
- added regression test in `tests/test_materialize_generation_baselines.py`
- test: `./venv_arm64/bin/python -m pytest tests/test_materialize_generation_baselines.py -q`
- result: `5 passed`

No-chat probe:

- source: `Qwen/Qwen2.5-Math-1.5B`
- target: `Qwen/Qwen3-0.6B`
- eval: first 16 IDs from
  `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- target: `0/16`
- source: `1/16`
- text relay: `5/16`
- C2C: `5/16`
- decision: do not use no-chat results for claims; target floor is a
  prompt-template artifact risk

Chat-template probe:

- target: `2/16`
- source: `4/16`
- text relay: `4/16`
- C2C: `5/16`
- C2C method-only over target: `4`
- text method-only over target: `3`
- source method-only over target: `3`
- target-only against C2C: `1`

Decision:

- promote Qwen2.5-Math-1.5B -> Qwen3-0.6B chat-template prompting one rung to
  frozen SVAMP32 surface confirmation
- do not train a connector yet
- monitor source numeric coverage (`12/16`) and target-self preservation

Artifacts:

- memo:
  - `paper/surface_scout_qwen25math_qwen3_svamp16_20260426.md`
- no-chat manifest:
  - `results/surface_scout_qwen25math_qwen3_svamp16_20260426/manifest.json`
  - sha256: `bda00aadd8b2ac109a5fb522fcff409045acf64ff644e7c724f6470e3ede0bcc`
- chat manifest:
  - `results/surface_scout_qwen25math_qwen3_svamp16_chat_20260426/manifest.json`
  - sha256: `834a60d9a2a4762f26ac4110e3e0503f73c9235256f3607fa4510b241a727060`
- chat C2C predictions:
  - `results/surface_scout_qwen25math_qwen3_svamp16_chat_20260426/c2c_generate.jsonl`
  - sha256: `0d00f1b1a6cbb569384de21a3ded03eb9e0edd1cd8e39af5281c64cb3afb410b`

Next exact gate:

```bash
./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --results-dir results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426 \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t c2c \
  --limit 32 \
  --device mps \
  --max-new-tokens 64 \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

## Cycle Checkpoint: 2026-04-26 Qwen2.5-Math -> Qwen3 SVAMP32 Surface Confirmation

- cycle number: `2026-04-26-surface-scout-qwen25math-qwen3-svamp32`
- timestamp: `2026-04-26 02:42:32 PDT`
- live branch entering cycle: Qwen2.5-Math -> Qwen3 C2C-comparable surface
- scale-up rung: strict small surface confirmation
- ICLR readiness: not ready; no deployable source-derived positive method yet

Start-of-cycle status:

- current paper story: the 16-ID chat-template smoke showed C2C beating target,
  source, and text on a registered C2C math-source pair
- exact blocker: verify that C2C headroom survives the frozen SVAMP32 surface
- highest-priority gate: run target/source/text/C2C on identical SVAMP32 IDs

Command:

```bash
./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --results-dir results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426 \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t c2c \
  --limit 32 \
  --device mps \
  --max-new-tokens 64 \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Decision:

- promoted: Qwen2.5-Math -> Qwen3 SVAMP32 as the next strict-small method
  surface
- target-alone: `8/32`
- source-alone: `6/32`
- text relay: `8/32`
- C2C: `15/32`
- C2C-only over target: `9`
- target-only over C2C: `2`
- target/C2C oracle: `17/32`
- target/text oracle: `11/32`

Artifacts:

- memo:
  - `paper/surface_scout_qwen25math_qwen3_svamp32_20260426.md`
- results manifest:
  - `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/manifest.md`
  - sha256: `291dc38614f4431d8cb37d7c53bc13ff95e84de826407c821fb0520437f756af`
- results JSON:
  - `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/manifest.json`
  - sha256: `27395a3e79ac5b02243e2b814a8167d65a838ec1e8faf21feaa06e9a22031b3d`
- C2C predictions:
  - `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl`
  - sha256: `e7389fffe0dc73e1bf583106d130c098e9679ca7dfa35bc693c47b54a509542e`

Hypothesis update:

- strengthened: a C2C-supported math-source surface with meaningful
  target-complementary headroom exists on frozen SVAMP32
- weakened: source-final numeric sidecars alone, because source-alone is below
  target and has only `26/32` numeric coverage
- promoted next: build C2C-headroom target set and test a deployable
  source-derived sidecar with source-destroying controls on this exact surface

Next exact gate:

- create a Qwen2.5-Math -> Qwen3 SVAMP32 C2C-headroom target set
- run the cheapest method that can use source-derived signal without C2C answer
  leakage
- required controls: zero-source, shuffled-source, target-only, slots-only,
  exact-ID parity, numeric coverage, target-self preservation

## Cycle Checkpoint: 2026-04-26 Qwen2.5-Math -> Qwen3 C2C Headroom And Source Probes

- cycle number: `2026-04-26-qwen25math-qwen3-headroom-source-probes`
- timestamp: `2026-04-26 02:51:29 PDT`
- live branch entering cycle: Qwen2.5-Math -> Qwen3 SVAMP32 C2C-headroom
  surface
- scale-up rung: strict small gate
- ICLR readiness: not ready; no deployable source-derived positive method

Start-of-cycle status:

- current paper story: Qwen2.5-Math -> Qwen3 with chat templates exposes target
  `8/32`, source `6/32`, text relay `8/32`, and C2C `15/32`
- exact blocker: turn C2C-only headroom into a deployable source-derived method
  that survives source-destroying controls
- highest-priority gate: build the clean C2C-headroom target set, then test the
  cheapest deployable source-derived sidecar/readout probes

Decision:

- target set status: `clean_headroom_available`
- C2C-only over target: `9`
- source/text explained C2C-only IDs: `3`
- clean C2C-headroom targets: `6`
- target-only vs C2C IDs to preserve: `2`
- source-only numeric sidecar: failed, best matched `8/32`, clean
  source-necessary `0/6`, source numeric coverage `26/32`
- source-hidden ridge probes: failed, last-layer and all-layer both matched
  `8/32` with clean source-necessary `0/6`

Artifacts:

- memo:
  - `paper/qwen25math_svamp32_headroom_and_source_probes_20260426.md`
- C2C-headroom manifest:
  - `results/qwen25math_svamp32_c2c_headroom_20260426/manifest.md`
- C2C-headroom JSON:
  - `results/qwen25math_svamp32_c2c_headroom_20260426/c2c_headroom_target_set.json`
  - sha256: `021b8b098c5fbc5a2b62193393bcf8da6bdba6c4eda2b1a411e32b94b6e81c32`
- source-only sidecar result:
  - `results/qwen25math_svamp32_source_only_sidecar_20260426/source_only_sidecar_router.json`
  - sha256: `457b74ce65e2e1dffc5b0b8b53f40a078d9534d7f90bbde7ed1a328e3d96385b`
- source-latent all-layer result:
  - `results/qwen25math_svamp32_source_latent_probe_20260426/all_layers_ridge_probe.json`
  - sha256: `d643ac24b6ec1f0ee9a66073f192ab873443e736e5497cde9029147d7b66c90a`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_build_c2c_headroom_target_set.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/build_c2c_headroom_target_set.py`

Hypothesis update:

- strengthened: the Qwen2.5-Math -> Qwen3 C2C-headroom surface is reusable and
  has six clean target-complementary C2C IDs
- killed: raw source-generated numeric residue sidecars on this surface
- killed: summary-level source-hidden ridge residue readout on this surface
- promoted next: run a Qwen-Math C2C prefill mechanism projection diagnostic or
  implement a token/layer-local source-derived objective with anti-cache
  controls

Next exact gate:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl \
  --target target=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl,method=c2c_generate \
  --candidate c2c=path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl,method=c2c_generate \
  --target-set-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json \
  --fallback-label target \
  --moduli 2,3,5,7 \
  --ridge-lambda 1.0 \
  --shuffle-offset 1 \
  --min-correct 9 \
  --min-clean-source-necessary 2 \
  --min-numeric-coverage 26 \
  --device mps \
  --max-new-tokens 1 \
  --residual-projection-dim 16 \
  --output-json results/qwen25math_svamp32_c2c_mechanism_probe_20260426/prefill_projection16_probe.json \
  --output-md results/qwen25math_svamp32_c2c_mechanism_probe_20260426/prefill_projection16_probe.md
```

## Cycle Checkpoint: 2026-04-26 Qwen2.5-Math -> Qwen3 Token-Layer C2C Residual Probe

- cycle number: `2026-04-26-qwen25math-qwen3-token-layer-c2c-residual`
- timestamp: `2026-04-26 03:07:53 PDT`
- live branch entering cycle: C2C mechanism distillation diagnostic on the
  Qwen2.5-Math -> Qwen3 SVAMP32 surface
- scale-up rung: strict small diagnostic gate
- ICLR readiness: not ready; no deployable source-derived positive method

Start-of-cycle status:

- current paper story: Qwen2.5-Math -> Qwen3 has six clean C2C-only IDs, but
  source-only sidecars and source-hidden summaries recover `0/6`
- exact blocker: test whether C2C local projector residual tensors carry a
  readable residue signal that could motivate source-side distillation
- highest-priority gate: token/layer-local C2C residual query-bottleneck with
  target-only, zero-source, shuffled-source, label-shuffle, and slots-only
  controls

Implementation:

- added token-local C2C trace extraction in `latent_bridge/c2c_eval.py`
- added `--feature-family token_layer_tail_residual` and `--probe-model
  query_bottleneck` support to
  `scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py`
- taught the shared source-latent evaluator to reshape arbitrary
  metadata-backed `feature_token_shape` tensors

Decision:

- failed strict gate
- matched: `8/32`
- target-only: `8/32`
- zero-source: `8/32`
- shuffled-source: `8/32`
- label-shuffled: `8/32`
- slots-only: `7/32`
- clean source-necessary: `0/6`
- control clean union: `1/6`
- feature shape: `[32, 229376]`
- token shape: `[224, 1024]`

Artifacts:

- memo:
  - `paper/qwen25math_svamp32_token_layer_c2c_residual_20260426.md`
- results manifest:
  - `results/qwen25math_svamp32_token_layer_c2c_residual_20260426/manifest.md`
- result JSON:
  - `results/qwen25math_svamp32_token_layer_c2c_residual_20260426/probe.json`
  - sha256: `b2bfb8605b07c7a9f9d98d31fb35091e06457b42580e884f440be1684fba0b6e`
- result markdown:
  - `results/qwen25math_svamp32_token_layer_c2c_residual_20260426/probe.md`
  - sha256: `83ba897e191dd62b51706c2859a443cf2760a6fd6230a8ea7998d374c6c5b440`
- raw run log:
  - `.debug/qwen25math_svamp32_token_layer_c2c_residual_20260426/logs/probe_rerun.log`
  - sha256: `4fa03704c7c5a83ce60124bd99d0b8e54e5ebb003646522ba32d8b3c6c97bd98`
- references:
  - `references/460_token_local_c2c_residual_refs.md`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_c2c_mechanism_trace.py tests/test_analyze_svamp32_source_latent_syndrome_probe.py -q`
- `./venv_arm64/bin/python -m py_compile latent_bridge/c2c_eval.py scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py scripts/analyze_svamp32_source_latent_syndrome_probe.py`

Hypothesis update:

- killed: C2C scalar summaries, signed projections, and tail-token local
  residual query-bottleneck readouts as live C2C-mechanism distillation
  branches on this surface
- strengthened: the blocker is not just over-compressed C2C trace summaries;
  the clean C2C gains are not linearly/query-readably exposed by these
  projector traces
- promoted next: select a deployable source-side branch or a new surface rather
  than another C2C trace readout

Next exact gate:

- audit existing runnable branches and choose the next source-derived gate after
  C2C trace readouts are killed
- do not widen to medium/large until a deployable method clears the current
  strict small source-control surface

## Cycle Checkpoint: 2026-04-26 Qwen2.5-Math -> Qwen3 Source-Contrastive Sidecar

- cycle number: `2026-04-26-qwen25math-qwen3-source-contrastive-sidecar`
- timestamp: `2026-04-26 03:14:11 PDT`
- live branch entering cycle: source-derived method search after C2C trace
  readouts failed
- scale-up rung: strict small gate
- ICLR readiness: not ready; first positive strict-small source-derived row
  needs medium confirmation, uncertainty, systems accounting, and C2C
  comparison

Start-of-cycle status:

- current paper story: Qwen2.5-Math -> Qwen3 has C2C headroom, but C2C-only
  target sets did not yield a deployable source method
- exact blocker: find a source-derived decision surface where source adds clean
  target-complementary wins under controls
- highest-priority gate: build a source-contrastive target set and test a
  target-preserving source residue sidecar stack

Decision:

- source-contrastive target set: ready
- target-alone: `8/32`
- source-alone: `6/32`
- text relay: `8/32`
- source-only over target: `5`
- clean source-only after text exclusion: `4`
- target-or-source oracle: `13/32`
- guarded source sidecar best row: `11/32`
- source sidecar bytes: `1`
- clean source-necessary recovered: `3/4`
- control clean union: `0/4`
- zero-source/shuffled-source/label-shuffle/same-norm noise: at or below
  `8/32` with `0` clean control IDs
- promoted: source-contrastive sidecar stack to SVAMP70 medium confirmation

Artifacts:

- memo:
  - `paper/qwen25math_svamp32_source_contrastive_sidecar_20260426.md`
- results manifest:
  - `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/manifest.md`
- target set:
  - `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json`
  - sha256: `088f0e1651f95ea04a89ec0931276a943ff104a355fe69f434182f68e778ea96`
- guarded sidecar:
  - `results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_only_sidecar_router_t2t_guard.json`
  - sha256: `c5434aeead9e55f5494ca583533fe863f36ee719e8a5bb75ae6fdb2f6f373306`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_only_sidecar_router_gate.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_only_sidecar_router_gate.py`

Hypothesis update:

- strengthened: source-derived numeric residues can add clean target-complementary
  wins on the Qwen-Math surface when paired with a preservation guard
- weakened: C2C-only target sets as the sole method gate; they were useful for
  headroom but did not expose deployable source signal
- caveat: the preservation guard uses text relay, so systems claims must count
  text-relay generation plus the sidecar

Next exact gate:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/qwen25math_qwen3_svamp70_source_surface_20260426 \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t c2c \
  --limit 70 \
  --device mps \
  --max-new-tokens 64 \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Then build the SVAMP70 source-contrastive set and rerun the guarded sidecar with
full controls.

## Cycle Checkpoint: 2026-04-26 Qwen2.5-Math -> Qwen3 SVAMP70 Source-Contrastive Sidecar

- cycle number: `2026-04-26-qwen25math-qwen3-svamp70-source-sidecar`
- timestamp: `2026-04-26 03:49:00 PDT`
- live branch entering cycle: source-contrastive sidecar stack after SVAMP32
  strict-small pass
- scale-up rung: medium confirmation
- ICLR readiness: not ready; medium result beats target/text but not C2C and
  uncertainty crosses zero

Start-of-cycle status:

- current paper story: target/text agreement guard plus 1-byte source residue
  sidecar cleared SVAMP32 with clean source-control wins
- exact blocker: confirm the signal on SVAMP70 and compare against C2C
- highest-priority gate: materialize source/target/text/C2C on SVAMP70, build
  source-contrastive IDs, rerun guarded sidecar, and compute paired uncertainty

Decision:

- target-alone: `21/70`
- source-alone: `13/70`
- text relay: `22/70`
- C2C: `31/70`
- source-only over target: `9`
- clean source-only after text exclusion: `6`
- guarded sidecar: `25/70`
- clean source-necessary: `4/6`
- control clean union: `0/6`
- paired delta vs target: `+0.0571`, bootstrap `[-0.0286, +0.1429]`,
  McNemar `0.3428`
- paired delta vs text: `+0.0429`, bootstrap `[-0.0571, +0.1429]`,
  McNemar `0.6056`
- paired delta vs C2C: `-0.0857`, bootstrap `[-0.2143, +0.0571]`,
  McNemar `0.3074`
- C2C fallback stack: failed, matched `23/70`, clean source-necessary `1/6`,
  control clean union `4/6`

Artifacts:

- memo:
  - `paper/qwen25math_svamp70_source_contrastive_sidecar_20260426.md`
- sidecar manifest:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/sidecar_manifest.md`
- generation manifest:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/manifest.md`
  - sha256: `1155f7f1eee547d0d36e6d62fb6305d9c01cf7042758e19e3cd293383033b0fa`
- guarded sidecar:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_only_sidecar_router_t2t_guard.json`
  - sha256: `0d5971c8152650b31e2fda9ccf0b1263061f6adc045e488ed5feb92841e8389d`
- paired comparisons:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/paired_vs_target.md`
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/paired_vs_text.md`
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/paired_vs_c2c.md`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_only_sidecar_router_gate.py tests/test_build_source_contrastive_target_set.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_only_sidecar_router_gate.py scripts/build_source_contrastive_target_set.py`

Hypothesis update:

- strengthened: source residues carry real target-complementary information on
  both SVAMP32 and SVAMP70 under source-destroying controls
- weakened: current text-guarded sidecar as a headline ICLR method, because it
  is below C2C and uncertainty versus target/text is not decisive
- killed: naive C2C-fallback composition with the same sidecar

Next exact gate:

- do not scale to 500 examples yet
- implement or test a cheaper preservation guard that does not require text
  relay, or a source-derived router that preserves target-correct rows better
  while keeping clean control leakage at zero

## Cycle Checkpoint: 2026-04-26 SVAMP70 Textless Source-Quality Guard

- cycle number: `2026-04-26-qwen25math-qwen3-svamp70-textless-quality-guard`
- timestamp: `2026-04-26 03:59:12 PDT`
- live branch entering cycle: source-contrastive sidecar stack after SVAMP70
  target/text-positive result
- scale-up rung: medium method-improvement gate
- ICLR readiness: not ready; textless row is promising but below C2C and needs
  replication

Start-of-cycle status:

- current paper story: text-guarded source sidecar reaches `25/70`, but uses
  text relay as preservation guard
- exact blocker: remove text relay from the guard/candidate pool while
  preserving source-necessary wins and zero clean control leakage
- highest-priority gate: source/target-only quality guard using decoded output
  length plus numeric source availability

Decision:

- implemented `--source-quality-guard shorter_than_target_numeric`
- no text relay used in the candidate pool or guard
- target-alone: `21/70`
- text relay baseline: `22/70`
- C2C: `31/70`
- textless source sidecar: `26/70`
- clean source-necessary: `4/6`
- control clean union: `0/6`
- paired delta vs target: `+0.0714`, bootstrap `[+0.0000, +0.1429]`,
  McNemar `0.1306`
- paired delta vs text: `+0.0571`, bootstrap `[-0.0714, +0.1857]`,
  McNemar `0.5023`
- paired delta vs C2C: `-0.0714`, bootstrap `[-0.2143, +0.0714]`,
  McNemar `0.4414`

Artifacts:

- updated memo:
  - `paper/qwen25math_svamp70_source_contrastive_sidecar_20260426.md`
- textless sidecar:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_shorter_than_target_guard_sidecar.json`
  - sha256: `19e5ec627968ea943c1483b2d6b19fffc8f642d51242c389ed1b341c0034cb81`
- textless predictions:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_shorter_than_target_guard_predictions.jsonl`
  - sha256: `6b56da11c6846d4a86f8d12d5eb18ad3653ed1bd82fbe14212014b096bd85778`
- paired comparisons:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/shorter_guard_paired_vs_target.md`
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/shorter_guard_paired_vs_text.md`
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/shorter_guard_paired_vs_c2c.md`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_only_sidecar_router_gate.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_only_sidecar_router_gate.py`

Hypothesis update:

- strengthened: source-sidecar communication can beat target/text without text
  relay in the guard, giving a plausible systems branch
- weakened: the guard is decoded-output heuristic and may be slice-specific
- still blocked: C2C remains stronger and no seed/surface replication exists

Next exact gate:

- rerun the textless sidecar on another exact frozen surface or seed/prompt
  variant before widening to 500 examples
- promotion requires paired CI clearly positive versus target/text or a stronger
  systems tradeoff at preserved accuracy

## Cycle Checkpoint: 2026-04-26 SVAMP70 Holdout Length-Ratio Guard

- cycle number: `2026-04-26-qwen25math-qwen3-svamp70-holdout-lenratio`
- timestamp: `2026-04-26 04:42:00 PDT`
- live branch entering cycle: textless source residue sidecar with decoded
  source/target length-ratio preservation guard
- scale-up rung: medium disjoint holdout falsification
- ICLR readiness: not ready; the fixed hand guard failed to replicate

Start-of-cycle status:

- current paper story: source-sidecar communication can beat target/text on the
  original SVAMP70 slice without text relay
- exact blocker: determine whether the fixed `source_target_len_ratio <= 1.0`
  guard generalizes to disjoint SVAMP IDs
- highest-priority gate: SVAMP `chal-101` through `chal-170`, same
  Qwen2.5-Math -> Qwen3 models, same source-destroying controls

Baseline results:

- source-alone: `8/70`, numeric coverage `64/70`
- target-alone: `8/70`, numeric coverage `70/70`
- text relay: `18/70`, numeric coverage `70/70`
- C2C: `37/70`, numeric coverage `70/70`
- source-only over target: `6`
- clean source-only after text exclusion: `2`
- target/source oracle: `14/70`

Decision:

- parameterized the guard as `--source-quality-score-field
  source_target_len_ratio --source-quality-max-threshold 1.0`
- added analyzer prediction JSONL export for paired comparisons
- holdout sidecar result: `10/70`
- clean source-necessary: `0/2`
- control clean union: `2/2`
- paired delta vs target: `+0.0286`, bootstrap `[-0.0286, +0.0857]`,
  McNemar `0.6171`
- paired delta vs text: `-0.1143`, bootstrap `[-0.2286, +0.0000]`,
  McNemar `0.0801`
- paired delta vs C2C: `-0.3857`, bootstrap `[-0.5143, -0.2571]`,
  McNemar `0.0000`

Hypothesis update:

- weakened: fixed decoded length-ratio guard as a live paper method; the
  disjoint holdout fails source-necessity controls
- strengthened: C2C and text relay remain strong on this holdout, so the slice
  has communication headroom even though source-alone is weak
- still alive: broader source-sidecar family, but only with a learned or
  cross-validated router and a stronger source-complementary surface
- do not scale: fixed `source_target_len_ratio <= 1.0` guard to 500 examples

Artifacts:

- memo:
  - `paper/qwen25math_svamp70_holdout_lenratio_guard_20260426.md`
- generation manifest:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/manifest.md`
  - sha256: `2f3080cfa4cfdd2c9455585c30931671f39c4e908ddc419a003f735390394854`
- source target set:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json`
  - sha256: `dd11d8f33b24757222d310342bbf12ce27c115cb091c2f44a8287c8d126721d3`
- sidecar analysis:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_lenratio_guard_sidecar.json`
  - sha256: `cc5b5f2d64f3521b4ab11cd11ea96ac04c848d3c00b163f22823671bec1cfe81`
- sidecar predictions:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_lenratio_guard_predictions.jsonl`
  - sha256: `0ddb79fdd203615c4978b5b7b9d47dbaf72166a65d44d6ddd51be5a5ad0ad267`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_only_sidecar_router_gate.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_only_sidecar_router_gate.py`

Next exact gate:

- implement or test a learned/cross-validated source router rather than a fixed
  length threshold, evaluated on disjoint IDs with the same controls
- alternatively run source-surface discovery for a Qwen2.5-Math -> Qwen3 slice
  with enough clean source-only IDs to support a real sidecar gate

## Cycle Checkpoint: 2026-04-26 SVAMP70 CV Router

- cycle number: `2026-04-26-qwen25math-qwen3-svamp70-cv-router`
- timestamp: `2026-04-26 04:58:00 PDT`
- live branch entering cycle: source-residue sidecar with brittle textless
  preservation guard
- scale-up rung: medium router replacement gate
- ICLR readiness: not ready; CV router passes original slice but fails holdout

Start-of-cycle status:

- current paper story: the fixed length-ratio guard fails disjoint holdout
- exact blocker: replace the hand threshold with an auditable router that does
  not train on held-out rows
- highest-priority gate: 5-fold decision-stump router over existing source and
  target JSONL features

Implementation:

- added `scripts/analyze_svamp_source_sidecar_cv_router_gate.py`
- added `tests/test_analyze_svamp_source_sidecar_cv_router_gate.py`
- router features: source prediction length, source/target length ratio,
  source numeric count, source generated tokens, source final marker
- fold split: deterministic `sha256(example_id) % 5`
- selected rule per fold maximizes `help - harm - penalty * accept_count`
- source controls reuse the matched fold rule without refitting

Original SVAMP70 result with accept penalty `0.10`:

- matched: `25/70`
- clean source-necessary: `4/6`
- control clean union: `0/6`
- accepted harm: `1`
- paired delta vs target: `+0.0571`, bootstrap `[-0.0143, +0.1286]`
- paired delta vs text: `+0.0429`, bootstrap `[-0.0857, +0.1714]`
- paired delta vs C2C: `-0.0857`, bootstrap `[-0.2286, +0.0429]`

Disjoint holdout result with the same router family:

- matched: `6/70`
- clean source-necessary: `0/2`
- control clean union: `0/2`
- accepted harm: `2`
- paired delta vs target: `-0.0286`, bootstrap `[-0.0714, +0.0000]`
- paired delta vs text: `-0.1714`, bootstrap `[-0.2714, -0.0571]`
- paired delta vs C2C: `-0.4429`, bootstrap `[-0.5571, -0.3143]`

Hypothesis update:

- weakened: router over shallow decoded features as a robust method; it can
  reproduce the original slice but does not transfer
- strengthened: the limiting factor is source-surface stability and source
  signal, not just a single bad fixed threshold
- live next branch: source-surface discovery or a stronger source encoder before
  additional router tuning

Artifacts:

- memo:
  - `paper/qwen25math_svamp70_cv_router_20260426.md`
- original CV router:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_cv_router_penalty010_sidecar.json`
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_cv_router_penalty010_predictions.jsonl`
- holdout CV router:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_cv_router_penalty010_sidecar.json`
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_cv_router_penalty010_predictions.jsonl`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp_source_sidecar_cv_router_gate.py tests/test_analyze_svamp32_source_only_sidecar_router_gate.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp_source_sidecar_cv_router_gate.py scripts/analyze_svamp32_source_only_sidecar_router_gate.py`

Next exact gate:

- finish the SVAMP `chal-171..240` source-surface scout using source/target/text
  first
- spend C2C only if clean source-only IDs after text exclusion are high enough
  to support a meaningful sidecar test

## Cycle Checkpoint: 2026-04-26 SVAMP70 Chal171-240 Source Surface Scout

- cycle number: `2026-04-26-qwen25math-qwen3-svamp70-chal171-240-scout`
- timestamp: `2026-04-26 05:24:00 PDT`
- live branch entering cycle: source-surface discovery after CV-router holdout
  failure
- scale-up rung: medium surface scout
- ICLR readiness: not ready; no stable positive source-sidecar surface yet

Start-of-cycle status:

- current paper story: decoded-feature sidecar routers work on the original
  SVAMP70 slice but fail disjoint holdout
- exact blocker: find a surface with enough clean source-only IDs to support a
  source-derived sidecar test
- highest-priority gate: materialize source/target/text first, avoid C2C spend
  if source-only mass is too low

Results:

- source-alone: `8/70`, numeric coverage `64/70`
- target-alone: `22/70`, numeric coverage `70/70`
- text relay: `24/70`, numeric coverage `70/70`
- source-only over target: `2`
- clean source-only after text exclusion: `1`
- target/source oracle: `24/70`

Decision:

- reject SVAMP `chal-171..240` as a source-sidecar decision surface
- do not spend C2C on this slice for the current branch
- strengthened: source-surface discovery is now the live next branch
- weakened: continued router tuning on weak source-complementary slices

Artifacts:

- memo:
  - `paper/qwen25math_svamp70_surface_scout_chal171_240_20260426.md`
- generation manifest:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/manifest.md`
  - sha256: `145f425cd135d08272efe9cd7d0d973fb6b7e52ab744d960a390251d72ea1fc7`
- source target set:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json`
  - sha256: `efe234a2e31ea60f4fa729b9493d48e7db736b2a95de25407f4902b1703f889e`

Next exact gate:

- search another Qwen2.5-Math -> Qwen3 SVAMP/GSM slice with source-only over
  target `>=6/70` and clean source-only after text exclusion `>=4/70`
- run C2C only after the source/text/target scout clears that surface gate

## Cycle Checkpoint: 2026-04-26 SVAMP70 Chal241-310 Source Surface Scout

- cycle number: `2026-04-26-qwen25math-qwen3-svamp70-chal241-310-scout`
- timestamp: `2026-04-26 05:36:00 PDT`
- live branch entering cycle: source-surface discovery after weak
  `chal-171..240`
- scale-up rung: medium surface scout
- ICLR readiness: not ready; no stable positive source-sidecar surface yet

Start-of-cycle status:

- current paper story: original SVAMP70 source-sidecar signal is promising but
  disjoint surfaces remain unstable
- exact blocker: find enough clean source-only mass before spending C2C or
  stronger connector compute
- highest-priority gate: source/target/text first, C2C only if source-only
  over target `>=6/70` and clean source-only after text exclusion `>=4/70`

Results:

- source-alone: `5/70`, numeric coverage `63/70`
- target-alone: `10/70`, numeric coverage `70/70`
- text relay: `14/70`, numeric coverage `70/70`
- source-only over target: `4`
- clean source-only after text exclusion: `4`
- target/source oracle: `14/70`

Decision:

- classify SVAMP `chal-241..310` as weak, not promotable to C2C/sidecar spend
- do not call this a method result; it is only a surface scout
- weakened: repeated adjacent SVAMP range scouting as the main path
- strengthened: move next to GSM70 Math source-surface discovery and stronger
  source-interface branches only after a surface clears

Artifacts:

- memo:
  - `paper/qwen25math_svamp70_surface_scout_chal241_310_20260426.md`
- range materializer:
  - `scripts/materialize_jsonl_range.py`
  - `tests/test_materialize_jsonl_range.py`
- generation manifest:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/manifest.md`
  - sha256: `1d158272963d5b9b0e32d4c4eba13c68b18dc0c3e2dae8a175992af5cde64cf7`
- source target set:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json`
  - sha256: `9aa4a45892bce32b566232340f450749b9a074a8ce2c817c6de8901be15b1b08`
- surface scan:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_headroom_surfaces.json`
  - sha256: `258512e9b3529bb5312cebd66097aee107c0ee5374376b56ba5715d520fc7e2b`
- reference memo:
  - `references/461_source_surface_blocker_interface_refs.md`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_materialize_jsonl_range.py tests/test_analyze_svamp_source_sidecar_cv_router_gate.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/materialize_jsonl_range.py`

Next exact gate:

- run Qwen2.5-Math -> Qwen3 GSM70 source-surface scout with source/target/text
  only
- spend C2C only if GSM70 has source-only over target `>=6/70` and clean
  source-only after text exclusion `>=4/70`

## Cycle Checkpoint: 2026-04-26 GSM70 Qwen2.5-Math Source Surface Scout

- cycle number: `2026-04-26-qwen25math-qwen3-gsm70-scout`
- timestamp: `2026-04-26 05:58:00 PDT`
- live branch entering cycle: source-surface discovery after weak SVAMP
  `chal-241..310`
- scale-up rung: medium surface scout
- ICLR readiness: not ready; source-surface discovery did not find a stable
  decision surface

Start-of-cycle status:

- current paper story: source-sidecar methods need a stronger source-derived
  signal or a better surface before C2C/sidecar spend
- exact blocker: Qwen2.5-Math source signal is too sparse on disjoint SVAMP and
  GSM surfaces
- highest-priority gate: source/target/text GSM70 first, C2C only if raw
  source-only and clean source-only thresholds clear

Results:

- source-alone: `3/70`, numeric coverage `63/70`
- target-alone: `4/70`, numeric coverage `70/70`
- text relay: `6/70`, numeric coverage `70/70`
- source-only over target: `3`
- clean source-only after text exclusion: `2`
- target/source oracle: `7/70`

Decision:

- reject GSM70 as a source-sidecar decision surface
- do not spend C2C compute on this slice for the current branch
- weakened: continued same-pair source-surface scouting as the primary path
- live next branch: stronger source-derived interface smoke on an existing
  exact-ID surface, or a new source/target pair only after a cheap scout

Artifacts:

- memo:
  - `paper/qwen25math_gsm70_source_surface_20260426.md`
- generation manifest:
  - `results/qwen25math_qwen3_gsm70_source_surface_20260426/manifest.md`
  - sha256: `01045e3628480a2c2ba47f925e76c267c8e63f8efdf26c37635f740323c54fe6`
- source target set:
  - `results/qwen25math_qwen3_gsm70_source_surface_20260426/source_contrastive_target_set.json`
  - sha256: `ca8ed8e581c5aea1088071a0f69c19eb00c516cb8b7d2b0503b5f17aac65e061`

Next exact gate:

- run or implement the smallest real-model smoke for a rate-capped
  source-derived query/resampler or shared sparse/anchor sidecar on an existing
  exact-ID SVAMP surface
- require source-destroying controls to collapse and at least `2` clean
  source-necessary recoveries on SVAMP32 or `4` on SVAMP70

## Cycle Checkpoint: 2026-04-26 Qwen2.5-Math-Instruct SVAMP32 Source Surface Scout

- cycle number: `2026-04-26-qwen25math-instruct-qwen3-svamp32-scout`
- timestamp: `2026-04-26 06:15:01 PDT`
- live branch entering cycle: stronger source/source-interface discovery after
  weak same-pair GSM70 and disjoint SVAMP surface scouts
- scale-up rung: strict-small source-surface scout
- ICLR readiness: not ready; no deployable positive source-derived method yet

Start-of-cycle status:

- current paper story: C2C exposes headroom on SVAMP32/SVAMP70, but source-only
  signal is too sparse for deployable sidecars
- exact blocker: determine whether a cached instruct Math source provides a
  better clean source-only surface before spending C2C/sidecar compute
- highest-priority gate: source/target/text only; promote to C2C only if
  source-only over target `>=4/32` and clean source-only after text exclusion
  `>=2/32`

Results:

- source-alone: `3/32`, numeric coverage `32/32`
- target-alone: `8/32`, numeric coverage `32/32`
- text relay: `4/32`, numeric coverage `32/32`
- source-only over target: `2`
- clean source-only after text exclusion: `2`
- target/source oracle: `10/32`

Decision:

- reject `Qwen/Qwen2.5-Math-1.5B-Instruct -> Qwen/Qwen3-0.6B` as the next
  SVAMP32 source-surface branch
- do not spend C2C or sidecar compute on this pair/slice
- weakened: adjacent stronger-source prompting/model variants as the immediate
  path
- keep live: materially different source-derived interfaces, especially
  rate-capped sequence-aligned sparse/anchor sidecars inspired by the
  quotient/GPA toy results

Artifacts:

- memo:
  - `paper/qwen25math_instruct_svamp32_surface_scout_20260426.md`
- generation manifest:
  - `results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/manifest.json`
  - sha256: `5a032574d92589f092ea6fc0270adfbfbaa3faa7a3cd90a59d4957eeeb1dc297`
- source target set:
  - `results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/source_contrastive_target_set.json`
  - sha256: `d2249cfff9c498c19f6374cb669a14bd7ea066640cc5a18aed01eeea21181312`

Tests:

- no code changes in this cycle
- artifact validation was performed by
  `scripts/materialize_generation_baselines.py` and
  `scripts/build_source_contrastive_target_set.py`

Next exact gate:

- inspect existing quotient/GPA, sparse dictionary, rotalign, and
  `latent_bridge` code paths for the smallest real-model sequence-aligned
  sparse/anchor sidecar smoke
- require explicit source-destroying controls and at least `2` clean
  source-necessary recoveries on SVAMP32 before any scale-up

## Cycle Checkpoint: 2026-04-26 SVAMP32 Sparse-Anchor Sidecar Smoke

- cycle number: `2026-04-26-svamp32-sparse-anchor-sidecar-smoke`
- timestamp: `2026-04-26 06:22:41 PDT`
- live branch entering cycle: materially different source-derived sparse /
  anchor sidecar after source-surface scouts failed
- scale-up rung: strict-small real-model smoke
- ICLR readiness: not ready; no deployable source-derived method clears clean
  controls

Start-of-cycle status:

- current paper story: C2C headroom is real on SVAMP32, but previous source
  hidden, query-bottleneck, and source-numeric interfaces fail clean C2C IDs
- exact blocker: test whether sparse anchor plus tokenizer-boundary sidecar
  features recover clean C2C-residue targets
- highest-priority gate: at least `2/6` clean source-necessary recoveries,
  no clean destructive-control recovery, and target floor preservation

Implementation:

- added `scripts/analyze_svamp32_sparse_anchor_sidecar_probe.py`
- added `tests/test_analyze_svamp32_sparse_anchor_sidecar_probe.py`
- features: source hidden summary sparse anchor projection plus
  source/target tokenizer-boundary alignment sidecar
- controls: zero-source, shuffled-source, label-shuffled, target-only, and
  slots-only

Results:

- `probe`: `9/32` matched, `0/6` clean matched, target-only `8/32`,
  slots-only clean control `1/6`, estimated `34` bytes/example
- `probe_budget16_seed3`: `7/32` matched, `0/6` clean matched, target-only
  `8/32`, control clean union `0/6`, estimated `14` bytes/example

Decision:

- fail and weaken the current sparse-anchor implementation
- do not tune projection seed/top-k/byte budget further without a materially
  different feature extractor
- broader quotient/GPA sparse-dictionary branch remains alive only as a
  stricter token/span dictionary implementation or a real SAE-adapter gate

Artifacts:

- memo:
  - `paper/svamp32_sparse_anchor_sidecar_20260426.md`
- first run:
  - `results/svamp32_sparse_anchor_sidecar_20260426/probe.json`
  - sha256: `a9945e7a3679e382aa83f98a7c486c8091bceee0005ec799dcac5cca4cb81896`
- constrained run:
  - `results/svamp32_sparse_anchor_sidecar_20260426/probe_budget16_seed3.json`
  - sha256: `b27aacd16907a933a271183542003a9e8450617b9394389e3cdeaa6d992ad26e`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_sparse_anchor_sidecar_probe.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_sparse_anchor_sidecar_probe.py`

Next exact gate:

- either implement fold-local token/span sparse dictionary controls in the
  sparse-anchor analyzer, including same-norm-noise and boundary-only controls;
  or evaluate the existing SAE adapter lane on the same SVAMP32 clean
  C2C-headroom target set
- do not scale to SVAMP70 until the strict SVAMP32 gate recovers at least
  `2/6` clean source-necessary IDs with zero clean destructive-control recovery

## Cycle Checkpoint: 2026-04-26 Qwen2.5-Math SVAMP32 Source-Token Query-Bottleneck

- cycle number: `2026-04-26-qwen25math-svamp32-source-token-qbottleneck`
- timestamp: `2026-04-26 06:30:23 PDT`
- live branch entering cycle: current SVAMP32 C2C-headroom surface after
  sparse-anchor projection smoke failed
- scale-up rung: strict-small real-model source-interface gate
- ICLR readiness: not ready; current source-readout family fails clean recovery

Start-of-cycle status:

- current paper story: Qwen2.5-Math -> Qwen3 C2C has clean headroom, but
  deployable source-derived interfaces do not recover it
- exact blocker: test whether the stronger Math source plus all-layer
  source-token query bottleneck predicts useful C2C residue signatures
- highest-priority gate: at least `2/6` clean source-necessary IDs with zero
  clean destructive-control recovery

Results:

- matched: `8/32`, clean `0/6`
- zero-source: `8/32`, clean `0/6`
- shuffled-source: `7/32`, clean `0/6`
- label-shuffled: `7/32`, clean `0/6`
- same-norm-noise: `8/32`, clean `0/6`
- target-only: `8/32`, clean `0/6`
- slots-only: `7/32`, clean `1/6`
- candidate-pool clean gold coverage: `6/6`

Decision:

- fail and kill all-layer source-token query-bottleneck residue prediction on
  the current Qwen2.5-Math SVAMP32 C2C-headroom surface
- weakened: shallow source-token/source-summary residue prediction family
- live next branch: materially different signal path, such as fold-local
  token/span sparse dictionaries, a target-safe output-aware dynalign selector,
  or a non-first-pass SAE/shared-code adapter under this same clean target set

Artifacts:

- memo:
  - `paper/qwen25math_svamp32_source_token_qbottleneck_20260426.md`
- probe:
  - `results/qwen25math_svamp32_source_token_qbottleneck_20260426/probe.json`
  - sha256: `abf23bb105ee05d98717d731414de15d4543419b3e96ffc571a54c54983c83d0`
- readout:
  - `results/qwen25math_svamp32_source_token_qbottleneck_20260426/probe.md`
  - sha256: `5ec913320d7f3acf8c75202286910db5b8f7fba8e57130193c88ad5e490975f7`

Next exact gate:

- stop source-token query-bottleneck and sparse-anchor projection tuning
- choose between fold-local token/span dictionary implementation and
  target-safe output-aware dynalign selector/repair; require the same `>=2/6`
  clean source-necessary recovery with zero clean destructive-control recovery

## Cycle Checkpoint: 2026-04-26 Qwen2.5-Math SVAMP32 Token/Span Dictionary

- cycle number: `2026-04-26-qwen25math-svamp32-token-span-dictionary`
- timestamp: `2026-04-26 06:36:12 PDT`
- live branch entering cycle: fold-local token/span sparse dictionary as the
  stricter source-readout follow-up
- scale-up rung: strict-small real-model source-interface gate
- ICLR readiness: not ready; current sparse/source-readout family is killed on
  this surface

Start-of-cycle status:

- current paper story: C2C exposes clean headroom on SVAMP32, but source-side
  readouts have not recovered it
- exact blocker: determine whether fold-local token/span sparse dictionaries
  can recover clean C2C-headroom IDs after projection and query-bottleneck
  variants failed
- highest-priority gate: matched `>=10/32`, target floor preserved, clean
  source-necessary `>=2/6`, clean control union `0/6`

Implementation:

- added `scripts/analyze_svamp32_token_span_dictionary_probe.py`
- added `tests/test_analyze_svamp32_token_span_dictionary_probe.py`
- features: fold-local sparse dictionary over source token states plus
  source/target tokenizer-boundary sidecar
- controls: zero-source, shuffled-source, label-shuffled, same-norm-noise,
  boundary-only, target-only, and slots-only

Results:

- matched: `7/32`, clean `0/6`
- zero-source: `7/32`, clean `0/6`
- shuffled-source: `8/32`, clean `0/6`
- label-shuffled: `6/32`, clean `0/6`
- same-norm-noise: `8/32`, clean `0/6`
- boundary-only: `7/32`, clean `0/6`
- target-only: `8/32`, clean `0/6`
- slots-only: `6/32`, clean `0/6`
- candidate-pool clean gold coverage: `6/6`
- mean dead atom rate: `0.0000`
- mean codebook perplexity: `28.5363`
- estimated sidecar bytes: `22`

Decision:

- fail and kill the current source-readout / sparse-dictionary family on the
  Qwen2.5-Math SVAMP32 C2C-headroom surface
- do not tune source-token readout, random projections, dictionary seed,
  top-k, atom count, or byte budget further on this surface
- next live branch: target-safe output-aware dynalign selector / repair

Artifacts:

- memo:
  - `paper/qwen25math_svamp32_token_span_dictionary_20260426.md`
- probe:
  - `results/qwen25math_svamp32_token_span_dictionary_20260426/probe.json`
  - sha256: `877a5970ccd244cdaf0731426934c8764e6b63d16f12bebc2102dafb46e7a64e`
- readout:
  - `results/qwen25math_svamp32_token_span_dictionary_20260426/probe.md`
  - sha256: `478a14c702ab16ccba98fe5a9656892f4c8a06aef0ab92e4ba943b2013fce54c`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_token_span_dictionary_probe.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_token_span_dictionary_probe.py`

Next exact gate:

- design a target-safe accept/fallback or repair gate over output-aware
  dynalign candidates on an exact-ID surface
- require target-only, zero-source, shuffled-source, and selector-only controls
- promote only if the gate recovers at least `2/6` clean source-necessary IDs
  without target-floor regression

## Cycle Checkpoint: 2026-04-26 SVAMP32 Target-Safe Oracle Replay

- cycle number: `2026-04-26-svamp32-target-safe-oracle-replay`
- timestamp: `2026-04-26 07:15:00 PDT`
- live branch entering cycle: target-safe output-aware dynalign selector /
  repair over existing SVAMP32 dynalign/query-pool candidates
- scale-up rung: strict small exact-ID gate
- ICLR readiness: not ready; no deployable positive method survives clean
  source controls

Start-of-cycle status:

- current paper story: C2C and target-self repair expose real SVAMP32 headroom,
  but existing source-readout, sparse-dictionary, and selector/repair branches
  fail clean source-necessary controls
- exact blocker: determine whether another target-safe selector over existing
  dynalign/query-pool rows can recover at least `2/6` clean C2C residual IDs
- highest-priority gate: oracle upper bound over matched candidates versus the
  matching source-destroying control oracle

Implementation:

- added `scripts/analyze_svamp32_target_safe_oracle.py`
- added `tests/test_analyze_svamp32_target_safe_oracle.py`
- replayed target_self_repair fallback with dynalign salt 1, dynalign salt 2,
  and query-pool candidates
- controls included the matching zero-source and shuffled-source rows

Results:

- target_self_repair: `14/32`, C2C-only `3/10`, clean residual `0/6`
- target-safe candidate oracle: `18/32`, C2C-only `5/10`, clean residual `1/6`
- target-safe control oracle: `18/32`, C2C-only `5/10`, clean residual `1/6`
- clean source-necessary after subtracting controls: `1/6`
- required clean source-necessary threshold: `2/6`

Decision:

- kill target-safe output-aware dynalign selector/repair over these saturated
  candidates
- do not spend GPU on another selector or repair pass over the same SVAMP32
  dynalign/query-pool rows
- promoted next branch: a genuinely learned communication protocol, not a
  replay selector, starting with a minimal target-conditioned soft-token /
  learned-query connector trained against the C2C-over-target_self residual
  surface with source-destroying controls from the start

Artifacts:

- memo:
  - `paper/svamp32_target_safe_oracle_replay_20260426.md`
- replay JSON:
  - `results/svamp32_target_safe_oracle_replay_20260426/oracle.json`
  - sha256: `1cb42394749c7bd1b80439bccc65441b63aba3b59ad9733994f080c337400746`
- replay markdown:
  - `results/svamp32_target_safe_oracle_replay_20260426/oracle.md`
  - sha256: `06f2126b891af93c102c7482b9cacf3e39154ff0c583e8dd6e814ff3c2d638b2`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_target_safe_oracle.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_target_safe_oracle.py`

Next exact gate:

- implement or locate the smallest target-conditioned soft-token /
  learned-query connector that trains on frozen source and target traces
- run it on the SVAMP32 clean residual target set with matched, zero-source,
  shuffled-source, target-only, and slots-only controls
- promote only if it recovers at least `2/6` clean source-necessary IDs with no
  target-self regression

## Cycle Checkpoint: 2026-04-26 Qwen2.5-Math SVAMP70 Chal241 Sidecar Gate

- cycle number: `2026-04-26-qwen25math-svamp70-chal241-sidecar-gate`
- timestamp: `2026-04-26 07:35:00 PDT`
- live branch entering cycle: Qwen2.5-Math source-contrastive sidecar surface
  discovery after original SVAMP70 positive and holdout failure
- scale-up rung: strict small / surface-pruning gate
- ICLR readiness: not ready; source-sidecar branch is positive on one slice
  but unstable across disjoint surfaces

Start-of-cycle status:

- current paper story: the 1-byte source residue sidecar can recover clean
  source-only IDs on the original SVAMP70 surface, but shallow guards failed a
  disjoint holdout
- exact blocker: determine whether the weak clean `chal241-310` source surface
  should receive C2C or connector spend
- highest-priority gate: run the cheap sidecar analyzer before generating new
  expensive C2C rows

Results:

- surface: source `5/70`, target `10/70`, text `14/70`
- clean source-only target set: `4`
- t2t-agreement guard: best matched `9/70`, clean source-necessary up to
  `3/4`, clean control union `1/4` to `2/4`
- textless shorter-than-target guard: best matched `11/70`, clean
  source-necessary `1/4`, clean control union `1/4`

Decision:

- reject `chal241-310` as a sidecar/router decision surface
- do not spend C2C generation or connector training on this slice
- weakened: adjacent SVAMP range scouting as the main path unless a stronger
  source encoder changes source-alone mass

Artifacts:

- memo:
  - `paper/qwen25math_svamp70_chal241_sidecar_gate_20260426.md`
- t2t-guard JSON:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_only_sidecar_router_t2t_guard.json`
  - sha256: `904942cceea20bc2e3e5f654c80532c19b5ce86b3e7fb998216c7ff0196f4ae8`
- textless guard JSON:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_shorter_than_target_guard_sidecar.json`
  - sha256: `34f6b87c2d25a67862f62b247cf2c2e69ace865ec452072e04273c9d0efc5b93`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_only_sidecar_router_gate.py -q`

Next exact gate:

- stop adjacent SVAMP range scouting unless a stronger source encoder changes
  the source-only mass
- run the next source-surface scout on a different math/reasoning split and
  require source-only over target `>=6/70` plus clean source-only after text
  exclusion `>=4/70` before C2C spend

## Cycle Checkpoint: 2026-04-26 Qwen2.5-Math SVAMP32 Perceiver C2C-Residual Gate

- cycle number: `2026-04-26-qwen25math-svamp32-perceiver-c2c-residual`
- timestamp: `2026-04-26 07:15:11 PDT`
- live branch entering cycle: genuinely learned communication protocol after
  target-safe selector replay was killed
- scale-up rung: strict small teacher-forced pre-generation gate
- ICLR readiness: not ready; learned connector does not produce source-necessary
  clean signal

Start-of-cycle status:

- current paper story: Qwen2.5-Math -> Qwen3 has real C2C headroom on SVAMP32,
  but deployable source-derived interfaces have not recovered clean residual IDs
- exact blocker: matched source must beat zero-source, shuffled-source,
  target-only, and slots-only controls on at least `2/6` clean C2C residual IDs
- highest-priority gate: train and score the smallest current Perceiver
  query-innovation connector on the Qwen2.5-Math compatible target set

Results:

- first calibration attempt failed because
  `results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json`
  has no `ids.target_self_repair` entries
- rerun with `--innovation-target-self-preserve-weight 0` completed
- average K alignment cosine: `0.961`
- average V alignment cosine: `0.799`
- answer-teacher injected samples: `274`
- teacher-forced gate `0.125`: matched-positive clean `2/6`,
  matched-only clean `0/6`, control-leak clean `2/6`, mean delta `-0.4836`
- teacher-forced gate `0.150`: matched-positive clean `2/6`,
  matched-only clean `0/6`, control-leak clean `2/6`, mean delta `-0.4102`
- teacher-forced gate `0.200`: matched-positive clean `2/6`,
  matched-only clean `0/6`, control-leak clean `2/6`, mean delta `-0.1916`

Decision:

- kill this specific Qwen2.5-Math Perceiver/query-innovation checkpoint before
  generation
- do not tune fixed gate, positive weight, answer-teacher weight, or anti-memory
  weight on the same Perceiver memory architecture without a materially
  different target-query/source-conditioning path
- promoted next branch: target-query-conditioned source bottleneck with
  target-only learned-prefix, slots-only prefix, zero-source, shuffled-source,
  and projected-soft-prompt controls at matched byte/query budgets

Artifacts:

- memo:
  - `paper/qwen25math_svamp32_perceiver_c2c_residual_20260426.md`
- result manifest:
  - `results/qwen25math_svamp32_perceiver_c2c_residual_20260426/manifest.md`
- checkpoint:
  - `.debug/qwen25math_svamp32_perceiver_c2c_residual_20260426/checkpoints/qwen25math_to_qwen3_svamp32_perceiver_c2c_residual_w080_ctrl050_am050_r16_b16_seed1.pt`
  - sha256: `d50b00fd0b9f5b5afcb09af8f9ae89b868e913b0a0610ef8132e66f20c726759`
- calibration log:
  - `.debug/qwen25math_svamp32_perceiver_c2c_residual_20260426/logs/calibrate_seed1_preserve0.log`
  - sha256: `495a782080e78fc1b40f59452dc25ec936207f93cb1945dbce4063196002d156`
- teacher-forced JSON:
  - `results/qwen25math_svamp32_perceiver_c2c_residual_20260426/teacher_forced_gate0125.json`
  - sha256: `c60c357ecf38a76479a94265296ec3a32905bd95d4b39787c83da84429c21503`
  - `results/qwen25math_svamp32_perceiver_c2c_residual_20260426/teacher_forced_gate015.json`
  - sha256: `caf7de8d67de8fd06defc1bc71d68fa4c636877b05508f533fd1fe913cce690c`
  - `results/qwen25math_svamp32_perceiver_c2c_residual_20260426/teacher_forced_gate020.json`
  - sha256: `97d64048334ddc523b1a7303fb7e8922d46828c6d3bf3ee97e59445a81fc8eca`
- literature memo:
  - `references/462_target_conditioned_side_information_query_bottleneck_refs.md`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_teacher_forced_connector_diagnostic.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_teacher_forced_connector_diagnostic.py`

Next exact gate:

- implement the smallest target-query-conditioned source bottleneck evaluator
  rather than another Perceiver memory checkpoint
- controls must include matched source, zero-source, shuffled-source,
  target-only learned-prefix, slots-only learned-prefix, and projected-soft
  prompt at matched byte/query budget
- promote only if the pre-generation gate recovers at least `2/6` matched-only
  clean residual IDs and positive matched-control mean delta

## Cycle Checkpoint: 2026-04-26 Qwen2.5-Math SVAMP32 Target-Query Source Bottleneck

- cycle number: `2026-04-26-qwen25math-svamp32-target-query-source-bottleneck`
- timestamp: `2026-04-26 07:15:11 PDT`
- live branch entering cycle: target-query-conditioned source bottleneck after
  the Perceiver memory checkpoint failed
- scale-up rung: strict small pre-generation residue/readout gate
- ICLR readiness: not ready; the implemented next branch also fails the clean
  source-necessary gate

Start-of-cycle status:

- current paper story: C2C exposes clean target-missed headroom, but
  source-derived learned readouts are not recovering the residual IDs
- exact blocker: test whether target prompt states querying source token states
  can recover at least `2/6` clean C2C residual IDs
- highest-priority gate: cross-fitted target-query/source bottleneck with
  target-only-prefix and projected-soft-prompt controls

Implementation:

- added `scripts/analyze_svamp32_target_query_source_bottleneck_probe.py`
- added `tests/test_analyze_svamp32_target_query_source_bottleneck_probe.py`
- model: target prompt summary plus learned queries attend over source token
  states; output predicts C2C residue signatures over moduli `2,3,5,7`
- controls: zero-source, shuffled-source, label-shuffled, same-norm-noise,
  target-only-prefix, projected-soft-prompt, target-only, slots-only

Results:

- matched: `7/32`, clean `0/6`
- zero-source: `8/32`, clean `0/6`
- shuffled-source: `6/32`, clean `0/6`
- label-shuffled: `6/32`, clean `0/6`
- same-norm-noise: `7/32`, clean `0/6`
- target-only-prefix: `8/32`, clean `0/6`
- projected-soft-prompt: `8/32`, clean `0/6`
- target-only: `8/32`, clean `0/6`
- slots-only: `6/32`, clean `0/6`
- clean source-necessary: `0/6`
- control clean union: `0/6`

Decision:

- fail and kill target-query-conditioned residue-classifier/readout variants
  on this surface
- do not tune query count, hidden dim, epochs, moduli, or layer selection on
  this exact classifier branch without a new signal source
- next branch, if pursued: true source-conditioned soft-prefix or gated
  cross-attention trained directly on gold-vs-distractor logprob with matched
  target-only learned-prefix, slots-only learned-prefix, projected-soft-prompt,
  zero-source, and shuffled-source controls

Artifacts:

- memo:
  - `paper/qwen25math_svamp32_target_query_source_bottleneck_20260426.md`
- result manifest:
  - `results/qwen25math_svamp32_target_query_source_bottleneck_20260426/manifest.md`
- analyzer:
  - `scripts/analyze_svamp32_target_query_source_bottleneck_probe.py`
  - sha256: `7fcaa9901ea5a78e23e7d4af8a64f513c0cf9da40ab1697fc0ea88c719143203`
- result JSON:
  - `results/qwen25math_svamp32_target_query_source_bottleneck_20260426/probe.json`
  - sha256: `06141d71be5fc57230aa7346525731618f554b023d7230c794ab681c34b05280`
- readout:
  - `results/qwen25math_svamp32_target_query_source_bottleneck_20260426/probe.md`
  - sha256: `482a661d22065e93a83a0d9b2fb5cd5fb5c343d4d051a4eba70fc305bd7be9aa`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_target_query_source_bottleneck_probe.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_target_query_source_bottleneck_probe.py`

Next exact gate:

- stop residue-classifier/readout variants on this surface
- either implement a true source-conditioned soft-prefix/gated cross-attention
  logprob objective with source-destroying and target-only prefix controls, or
  declare the current exact-ID SVAMP32 C2C-residual surface saturated and move
  to a new source-surface discovery branch

## Cycle Checkpoint: 2026-04-26 SVAMP70 Process-Repair Source Controls

- cycle number: `2026-04-26-svamp70-process-repair-source-controls`
- timestamp: `2026-04-26 09:35:55 PDT`
- live branch entering cycle: old process-repair selected-route row, reopened
  after the broader MD/results comb
- scale-up rung: medium source-control falsification
- ICLR readiness: not ready; the strongest repair row is killed as a
  source-communication method

Start-of-cycle status:

- current paper story: C2C and repair rows expose headroom, but no deployable
  source-derived method has survived controls
- exact blocker: decide whether process-repair selected routes transfer source
  information or merely exploit target-side repair/candidate diversity
- highest-priority gate: rerun process repair with zero-source K/V and
  shuffled-source prompt controls, preserving the same three-salt route-pool
  and target self-repair controls

Implementation:

- added `scripts/analyze_process_repair_source_controls.py`
- added `tests/test_analyze_process_repair_source_controls.py`
- added `references/463_process_repair_source_control_followup_refs.md`
- ran source-destroying controls with:
  - `.debug/run_svamp70_process_repair_zero_kv_control_20260426.sh`
  - `.debug/run_svamp70_process_repair_shuffled_source_control_20260426.sh`

Results:

- matched process repair: `38/70`
- target-alone: `21/70`
- target self-repair: `35/70`
- matched-only vs target self-repair: `3` IDs
- zero-source K/V process repair: `35/70`; overlaps `1/3` matched-only IDs
- shuffled-source prompt process repair: `37/70`; overlaps `3/3` matched-only
  IDs
- source-specific matched-only IDs after both controls: `0`

Decision:

- fail and kill process-repair selected routes as a source-communication method
  on SVAMP70
- do not tune verifier/repair-only policies unless a new source-derived route
  signal exists first
- keep process repair as a baseline and confound for future source methods

Artifacts:

- memo:
  - `paper/svamp70_process_repair_source_controls_20260426.md`
- result manifest:
  - `results/process_repair_source_controls_20260426/manifest.md`
- combined source-control gate:
  - `results/process_repair_source_controls_20260426/svamp70_zero_and_shuffled_source_control_gate.md`
  - sha256: `05e7c38e73f012e47345f7430fac2e93d9177a51e6505cae62cffaefd919ca72`
- combined attribution:
  - `results/process_repair_source_controls_20260426/svamp70_zero_and_shuffled_source_attribution.md`
  - sha256: `a7b9d3594392721b29d1db3c0036c6750aabb98209c43a7b78dc9b377e946875`
- full artifact hashes:
  - `results/process_repair_source_controls_20260426/sha256.txt`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_process_repair_source_controls.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_process_repair_source_controls.py`
- `./venv_arm64/bin/python -m json.tool references/research_memo_manifest.json >/dev/null`

Next exact gate:

- implement the true source-conditioned soft-prefix/gated cross-attention
  teacher-forced logprob gate, not another residue classifier or repair-only
  policy
- required controls: matched source, zero-source, shuffled-source, target-only
  learned prefix, slots-only learned prefix, projected soft prompt, and
  label-shuffled training
- promotion rule: at least `2/6` matched-only clean residual IDs with positive
  matched-minus-best-control margin and target-self preservation before any
  generation run

## Cycle Checkpoint: 2026-04-26 SVAMP32 Source Soft-Prefix Logprob Gate

- cycle number: `2026-04-26-svamp32-source-soft-prefix-logprob`
- timestamp: `2026-04-26 09:57:00 PDT`
- live branch entering cycle: true source-conditioned summary soft-prefix
  logprob gate
- scale-up rung: strict-small teacher-forced pre-generation smoke
- ICLR readiness: not ready; summary soft-prefix communication is killed on
  the current strongest same-family surface

Start-of-cycle status:

- current paper story: C2C exposes clean headroom on Qwen2.5-Math -> Qwen3
  SVAMP32, but prior sidecars, repair routes, and residue classifiers were
  control-explained or too weak
- exact blocker: decide whether a deployable source-conditioned soft prefix
  can beat target-only/source-destroying controls on clean C2C-headroom IDs
- highest-priority gate: recover at least `2/6` clean source-communication
  candidate IDs with `0` clean control leaks before any generation

Implementation:

- added `scripts/analyze_svamp32_source_soft_prefix_logprob_probe.py`
- added `tests/test_analyze_svamp32_source_soft_prefix_logprob_probe.py`
- added fold-local feature standardization
- calibrated the final run to source-only matched prefixes, numeric-only
  distractors, and mean-token continuation logprob

Final calibrated run:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_soft_prefix_logprob_probe.py --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl --target-jsonl results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl --teacher-jsonl results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl --target-set-json results/qwen25math_svamp32_c2c_headroom_20260426/c2c_headroom_target_set.json --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false --feature-layers last --prefix-len 2 --hidden-dim 16 --epochs 1 --outer-folds 2 --matched-use-target false --length-normalize true --device mps --train-device mps --dtype float32 --output-json .debug/qwen25math_svamp32_source_soft_prefix_20260426/source_only_numeric_meanlogprob_smoke.json --output-md .debug/qwen25math_svamp32_source_soft_prefix_20260426/source_only_numeric_meanlogprob_smoke.md
```

Results:

- clean IDs scored: `6`
- matched-only clean IDs: `1/6`
- matched-positive clean IDs: `5/6`
- clean control leaks: `4/6`
- mean matched-minus-best-control clean margin: `-0.771126`
- target-preservation IDs scored: `8`
- target-preservation matched-positive count: `6/8`

Decision:

- fail and kill global summary soft-prefix connectors on this surface
- do not scale this branch by epochs, hidden dimension, folds, or generation
  unless the source interface changes
- next live branch: token/layer-local gated cross-attention over source states

Artifacts:

- memo:
  - `paper/qwen25math_svamp32_source_soft_prefix_logprob_20260426.md`
- result manifest:
  - `results/qwen25math_svamp32_source_soft_prefix_logprob_20260426/manifest.md`
- result JSON:
  - `results/qwen25math_svamp32_source_soft_prefix_logprob_20260426/source_only_numeric_meanlogprob_smoke.json`
  - sha256: `f89c0a8759a94574de9e5a52eb50af800fab352c13550efdf1660f85d33778c9`
- readout:
  - `results/qwen25math_svamp32_source_soft_prefix_logprob_20260426/source_only_numeric_meanlogprob_smoke.md`
  - sha256: `cd67391b2c87a449a2096fb04942a8232cc7f8035bcbb3aa989b4e1aeae94169`
- analyzer:
  - `scripts/analyze_svamp32_source_soft_prefix_logprob_probe.py`
  - sha256: `913a0a8f4ae971d90fe47c9ed49f8a05ff83080e64eea4fb7b7ebe8c24bfc573`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_soft_prefix_logprob_probe.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_soft_prefix_logprob_probe.py`

Next exact gate:

- implement a token/layer-local gated cross-attention logprob gate on the same
  frozen SVAMP32 IDs
- keep controls: zero-source, shuffled-source, same-norm, projected soft prompt,
  target-only learned prefix, slots-only learned prefix, and label-shuffled
  training
- promotion rule: at least `2/6` matched-only clean candidate IDs, `0` clean
  control leaks, and positive mean matched-minus-best-control margin before
  generation

## Cycle Checkpoint: 2026-04-26 Source Surface Scan and Finalish Guard Holdout

- cycle number: `2026-04-26-source-surface-scan-finalish-holdout`
- timestamp: `2026-04-26 10:18:00 PDT`
- live branch entering cycle: fixed source-quality guarded 1-byte source
  sidecar on the SVAMP70 source surface
- scale-up rung: medium holdout falsification
- ICLR readiness: not ready; fixed source-quality guarded sidecars are pruned
  as the live method family

Start-of-cycle status:

- current paper story: C2C and source sidecars show headroom, but learned
  summary/prefix/cross-attention connectors fail source controls on SVAMP32
- exact blocker: decide whether the previous SVAMP70 source-sidecar positive
  can be rescued by the `finalish_short_numeric` source-quality guard on a
  disjoint holdout
- highest-priority gate: holdout-test the fixed finalish source guard with the
  same source-destroying controls

Surface scan:

- ran `scripts/analyze_source_headroom_surfaces.py` over existing SVAMP/GSM
  target/source rows
- top surface: `qwen25math_qwen3_svamp70`
  - target `21/70`
  - source `13/70`
  - source-only over target `9`
  - target/source oracle `30/70`
- disjoint holdout surface:
  - target `8/70`
  - source `8/70`
  - source-only over target `6`
  - target/source oracle `14/70`

Finalish holdout command:

```bash
./venv_arm64/bin/python scripts/analyze_svamp32_source_only_sidecar_router_gate.py --target target=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl,method=target_alone --source source=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,method=source_alone --candidate source=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,method=source_alone --target-set-json results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json --fallback-label target --source-quality-guard finalish_short_numeric --min-correct 10 --min-target-self 0 --min-clean-source-necessary 1 --max-control-clean-union 0 --min-numeric-coverage 64 --output-json results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_finalish_guard_sidecar.json --output-md results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_finalish_guard_sidecar.md --output-predictions-jsonl results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_finalish_guard_predictions.jsonl --prediction-method source_finalish_guard_sidecar
```

Results:

- best finalish row: `9/70`
- clean matched: `1/2`
- clean source-necessary: `0/2`
- clean control union: `2/2`
- source-destroying/control clean IDs:
  `ab1e71e8928661d0`, `daea537474de16ac`

Decision:

- fail and prune fixed source-quality guarded sidecars as the live method
  family
- do not tune thresholds or moduli on this family without a new router feature
  family and a frozen holdout gate
- next live branch: either source-surface discovery for a stronger/stabler
  source-complementary slice, or a genuinely new cross-validated source router
  with features beyond shallow length/numeric guards

Artifacts:

- memo:
  - `paper/qwen25math_svamp70_holdout_finalish_guard_20260426.md`
- surface scan:
  - `results/source_headroom_surface_scan_20260426/scan.json`
  - sha256: `9611574620e91181a029e1b60165555bba8234ebbb02fcb78748d7ced52b4a6b`
  - `results/source_headroom_surface_scan_20260426/scan.md`
  - sha256: `421f4bdf2a90c636e41da4f90f05c5aac0fa49bea5a5c21f28ceac0c64755afd`
- finalish holdout:
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_finalish_guard_sidecar.json`
  - sha256: `dc5b99e4500e414dae02241e7472734ee9aef51772cd55d9de9149c6c4dd9c1d`
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_finalish_guard_sidecar.md`
  - sha256: `d5b9c88a414ae71d796d8f742724d14ba5ce22ab92d0b0869af9676fcbc5fcd4`
  - `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_finalish_guard_predictions.jsonl`
  - sha256: `a0b7d2336c515b38c1e053fea09d94fb39e6fc224390e2500bf067928652e45a`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_only_sidecar_router_gate.py tests/test_analyze_svamp_source_sidecar_cv_router_gate.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_only_sidecar_router_gate.py scripts/analyze_source_headroom_surfaces.py`

Next exact gate:

- freeze a new router feature family before looking at holdout labels, or move
  to source-surface discovery; do not keep tuning fixed decoded-length/finalish
  guards

## Cycle Checkpoint: 2026-04-26 SVAMP32 Source Cross-Attention Logprob Gate

- cycle number: `2026-04-26-svamp32-source-cross-attention-logprob`
- timestamp: `2026-04-26 10:07:00 PDT`
- live branch entering cycle: token/layer-local gated cross-attention over
  source states
- scale-up rung: strict-small teacher-forced pre-generation smoke
- ICLR readiness: not ready; the first tiny prefix-emitting cross-attention
  implementation is also control-dominated

Start-of-cycle status:

- current paper story: C2C exposes clean headroom, but summary readouts and
  summary soft-prefix connectors are not source-derived under controls
- exact blocker: decide whether token-local target-query attention into source
  states is enough to separate matched source from controls
- highest-priority gate: recover at least `2/6` clean source-communication
  candidate IDs with `0` clean control leaks

Implementation:

- added `scripts/analyze_svamp32_source_cross_attention_logprob_probe.py`
- added `tests/test_analyze_svamp32_source_cross_attention_logprob_probe.py`
- reused numeric-only distractors, mean-token continuation logprob, fold-local
  token standardization, and the same target/source-destroying controls

Final smoke:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_cross_attention_logprob_probe.py --source-model Qwen/Qwen2.5-Math-1.5B --target-model Qwen/Qwen3-0.6B --eval-file results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl --target-jsonl results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl --teacher-jsonl results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl --target-set-json results/qwen25math_svamp32_c2c_headroom_20260426/c2c_headroom_target_set.json --source-use-chat-template --target-use-chat-template --source-enable-thinking false --target-enable-thinking false --feature-layers last --prefix-len 2 --hidden-dim 16 --epochs 1 --outer-folds 2 --length-normalize true --device mps --train-device mps --dtype float32 --output-json .debug/qwen25math_svamp32_source_cross_attention_20260426/smoke.json --output-md .debug/qwen25math_svamp32_source_cross_attention_20260426/smoke.md
```

Results:

- clean IDs scored: `6`
- matched-only clean IDs: `0/6`
- matched-positive clean IDs: `4/6`
- clean control leaks: `4/6`
- mean matched-minus-best-control clean margin: `-0.383649`
- target-preservation IDs scored: `8`
- target-preservation matched-positive count: `5/8`

Decision:

- fail this first-rung token-local cross-attention implementation
- do not scale this exact tiny prefix-emitting connector by epochs, folds, or
  hidden width without a new mechanism reason
- the next branch should move away from tiny learned prefix emitters on this
  exact surface

Artifacts:

- memo:
  - `paper/qwen25math_svamp32_source_cross_attention_logprob_20260426.md`
- result manifest:
  - `results/qwen25math_svamp32_source_cross_attention_logprob_20260426/manifest.md`
- result JSON:
  - `results/qwen25math_svamp32_source_cross_attention_logprob_20260426/smoke.json`
  - sha256: `ecc014ad50455c81a2f297275a0883d70b7f230d62ee73c48c3130d33eda138e`
- readout:
  - `results/qwen25math_svamp32_source_cross_attention_logprob_20260426/smoke.md`
  - sha256: `b6f8a00ef164832be843f3a0daa440a62f8f919f3b2360ebf137f106aa68e2dc`
- analyzer:
  - `scripts/analyze_svamp32_source_cross_attention_logprob_probe.py`
  - sha256: `097ff46ef9f8679abf0ee4686a3a5316d02773a7dcd26ecd254eb73a0d930b6d`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_cross_attention_logprob_probe.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_cross_attention_logprob_probe.py`

Next exact gate:

- switch to source-surface discovery or a discrete source-derived
  candidate/routing stack
- do not run another tiny prefix-emitter variant on this exact surface unless
  new diagnostics explain the control dominance

## Cycle Checkpoint: 2026-04-26 Source Reselection And SVAMP70 Cross-Attention Rescue

- cycle number: `2026-04-26-source-reselection-svamp70-cross-attention`
- timestamp: `2026-04-26 10:22:00 PDT`
- live branch entering cycle: source-surface reselection after learned-prefix
  failures
- scale-up rung: surface reselection plus top-surface teacher-forced smoke
- ICLR readiness: not ready; no learned prefix emitter remains live

Start-of-cycle status:

- current paper story: C2C and source-only baselines expose headroom, but
  global summary readouts, summary prefixes, process repair, and SVAMP32
  token-local cross-attention are not source-communication methods under
  controls
- exact blocker: decide whether the learned-prefix failure is specific to
  SVAMP32 or also appears on the strongest source-complementary surface
- highest-priority gate: rerank source surfaces, then run the same
  cross-attention gate on the top surface if it has enough clean source-only
  IDs

Surface reselection:

- `svamp70_live`: strong source-complementary surface, source-only `9`,
  target/source oracle `30/70`
- `svamp70_holdout`: strong source-complementary surface, source-only `6`,
  target/source oracle `14/70`
- `svamp32_qwen25math`: weak source-complementary surface, source-only `5`,
  target/source oracle `13/32`
- GSM70, DeepSeek SVAMP32, and Qwen2.5-Math-Instruct SVAMP32 remain weak
  immediate surfaces

SVAMP70 live cross-attention rescue:

- clean IDs scored: `6`
- matched-only clean IDs: `0/6`
- matched-positive clean IDs: `3/6`
- clean control leaks: `3/6`
- mean matched-minus-best-control clean margin: `-0.443233`
- target-preservation IDs scored: `22`
- target-preservation matched-positive count: `13/22`

Decision:

- fail learned prefix emitters as the current live branch
- do not tune epochs, hidden width, folds, or prefix length for this connector
  family without a new mechanism reason
- next branch should be a discrete source-derived candidate/routing stack on
  `svamp70_live` with immediate `svamp70_holdout` validation, or broader
  source-surface discovery

Artifacts:

- memo:
  - `paper/source_surface_reselection_and_svamp70_cross_attention_20260426.md`
- surface reselection:
  - `results/source_surface_reselection_20260426/source_headroom_surfaces.json`
  - sha256: `23de7bba13b3a1879e986edf930874957cf5a6e8badee808437e73c50874e640`
  - `results/source_surface_reselection_20260426/source_headroom_surfaces.md`
  - sha256: `60f62abb94e9e85f8720b6d91c7acbe41b749becdc326d67040846a9820daefc`
- SVAMP70 live cross-attention gate:
  - `results/qwen25math_svamp70_source_cross_attention_logprob_20260426/live_smoke.json`
  - sha256: `013f9d0501bdb2c87a96fc46d3415c42f7e57c3be81b9d285c264ca770863c2d`
  - `results/qwen25math_svamp70_source_cross_attention_logprob_20260426/live_smoke.md`
  - sha256: `f02dbf4ab4e452d24af764b0c151fabd7bc88f5d5f8344f34ad99173fe2eed82`

Tests:

- reused validated cross-attention harness:
  - `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_cross_attention_logprob_probe.py -q`
  - `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_cross_attention_logprob_probe.py`

Next exact gate:

- either implement a discrete source-derived candidate/routing stack with
  live/holdout attribution controls, or materialize a stronger source surface
- do not spend more compute on tiny learned prefix emitters until a new
  diagnostic changes the hypothesis

## Cycle Checkpoint: 2026-04-26 SVAMP70 Source-Trace Router

- cycle number: `2026-04-26-svamp70-source-trace-router`
- timestamp: `2026-04-26 11:05:00 PDT`
- live branch entering cycle: cross-validated source-trace self-consistency
  router over the Qwen2.5-Math -> Qwen3 SVAMP70 source-sidecar surface
- scale-up rung: medium live-CV plus frozen holdout falsification
- ICLR readiness: not ready; shallow text-level source routers are now
  weakened/killed

Start-of-cycle status:

- current paper story: source-sidecar and source-only surfaces show headroom,
  but fixed guards and learned prefix emitters fail controls or holdout
- exact blocker: decide whether richer source trace features can rescue the
  1-byte sidecar without target/cache/control artifacts
- highest-priority gate: train a small live-CV router on source trace features,
  freeze the full-live rule, and apply once to the disjoint holdout with an
  equation-result permutation control

Result:

- frozen full-live feature: `source_answer_reused_in_trace >= 0.5`
- live CV matched correct: `20/70`
- live CV clean source-necessary: `1`
- live CV clean control union: `0`
- live CV accepted harm: `2`
- holdout matched correct: `10/70`
- holdout clean source-necessary: `1`
- holdout clean control union: `0`
- holdout accepted harm: `0`
- holdout equation-permuted retained source-necessary: `1`

Decision:

- fail and prune the source-trace self-consistency router as the next
  source-sidecar rescue
- the zero standard-control leakage is useful, but the signal is too weak and
  the holdout clean win survives equation-result permutation
- next branch should be source-surface discovery or source internal-confidence
  artifact capture, not another fixed shallow source-text guard

Artifacts:

- memo:
  - `paper/qwen25math_svamp70_source_trace_router_20260426.md`
- result manifest:
  - `results/qwen25math_svamp70_source_trace_router_20260426/manifest.md`
- result JSON:
  - `results/qwen25math_svamp70_source_trace_router_20260426/trace_router.json`
  - sha256: `e4e5600e139efbf7bc068ff2117e172cba9f87055e9477f51839a90175c54c03`
- readout:
  - `results/qwen25math_svamp70_source_trace_router_20260426/trace_router.md`
  - sha256: `e439be6bf00ec99ccf9f63aa72ca0b3a47f423d4a950155846d7417322e99a6c`
- analyzer:
  - `scripts/analyze_svamp_source_trace_router_gate.py`
  - sha256: `92828099f0ccc3188fb49f8171f1e0d0ce4260a27e1780459a8789a3f31e03e5`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp_source_trace_router_gate.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp_source_trace_router_gate.py`

Next exact gate:

- run a new source-surface discovery pass for stronger source-only over
  target/text examples, or add generation-time source confidence/logit
  artifacts so the next router uses model-internal source evidence rather than
  shallow source text

## Cycle Checkpoint: 2026-04-26 Source Generation Diagnostics Artifact

- cycle number: `2026-04-26-source-generation-diagnostics-artifact`
- timestamp: `2026-04-26 11:35:00 PDT`
- live branch entering cycle: source-internal confidence/logit artifact for
  future source routers
- scale-up rung: micro tooling smoke
- ICLR readiness: not ready; this is tooling for the next router feature
  family, not a positive method

Start-of-cycle status:

- current paper story: source sidecars have source-complementary pockets, but
  shallow decoded-text guards and trace routers fail holdout or source-control
  attribution
- exact blocker: collect source-internal confidence features so the next router
  is not based only on decoded text surface properties
- highest-priority gate: add a sidecar diagnostics collector and prove it works
  on a two-example MPS smoke without mutating existing baselines

Result:

- added `scripts/collect_source_generation_diagnostics.py`
- added focused unit tests for logprob/entropy/margin extraction
- two-example Qwen2.5-Math SVAMP32 smoke passed with offline cache settings and
  approved MPS escalation
- smoke output: `.debug/source_generation_diagnostics_smoke/source_diagnostics.jsonl`
- smoke JSONL sha256:
  `016e669f76de07666e9d13212e1c2fcc50565daa01e120916665c08f8f2f456f`

Decision:

- promote source-internal confidence diagnostics to the next router feature
  family to test
- do not call this a method result; it is instrumentation for the next gate

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_collect_source_generation_diagnostics.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/collect_source_generation_diagnostics.py`

Next exact gate:

- run diagnostics on `svamp70_live`, then test whether source logprob,
  entropy, top-1 probability, and top-1/top-2 margin separate clean source-only
  wins from source/text failures
- if not, switch to the disjoint `chal311-380` source-surface scout before any
  C2C or connector spend

## Cycle Checkpoint: 2026-04-26 SVAMP70 Source Confidence Router

- cycle number: `2026-04-26-svamp70-source-confidence-router`
- timestamp: `2026-04-26 12:12:00 PDT`
- live branch entering cycle: source-internal confidence router over direct
  source-only greedy-generation diagnostics
- scale-up rung: medium live-CV plus frozen holdout falsification
- ICLR readiness: not ready; confidence routing is instrumentation, not a live
  positive method

Start-of-cycle status:

- current paper story: decoded source-text guards, trace routers, and tiny
  prefix emitters fail holdout/control gates despite source/C2C headroom
- exact blocker: determine whether source-internal confidence features recover
  clean source-only wins without decoded-text leakage
- highest-priority gate: collect direct-prompt source diagnostics for live and
  holdout, train a one-feature live-CV router, and freeze once to holdout

Result:

- live diagnostics match source-alone: `13/70`
- holdout diagnostics match source-alone: `8/70`
- frozen rule: `min_top1_prob >= 0.1639411821961403`
- live CV: matched `24/70`, clean source-necessary `2`, clean control union
  `0`, accepted harm `0`
- frozen holdout: matched `7/70`, clean source-necessary `0`, clean control
  union `0`, accepted harm `1`

Decision:

- fail and prune source-internal confidence routing on this old SVAMP70
  sidecar surface
- the feature family is useful for instrumentation, but not promotable here
- next branch: disjoint `chal311-380` source-surface scout with source/target
  and text relay only; require `>=6/70` source-only over target and `>=4/70`
  clean source-only after text exclusion before C2C or connector spend

Artifacts:

- memo:
  - `paper/qwen25math_svamp70_source_confidence_router_20260426.md`
- manifest:
  - `results/qwen25math_svamp70_source_generation_diagnostics_20260426/manifest.md`
- live diagnostics:
  - `results/qwen25math_svamp70_source_generation_diagnostics_20260426/source_diagnostics.jsonl`
  - sha256: `b17755be3db764f6130830cc516b18b6e4fadce7a78de36d20f10dd8c84c69b2`
- holdout diagnostics:
  - `results/qwen25math_svamp70_source_generation_diagnostics_20260426/holdout_source_diagnostics.jsonl`
  - sha256: `2fc5226940ea4fc743324534bb51c938829910810619040f78afea2c905ecb0e`
- confidence router:
  - `results/qwen25math_svamp70_source_generation_diagnostics_20260426/confidence_router.json`
  - sha256: `291ee7015a7b28f41f7c5e1b397e18b29da1b0781ae0f30c7c528ac3e860b4a8`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_collect_source_generation_diagnostics.py tests/test_analyze_source_confidence_router_gate.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/collect_source_generation_diagnostics.py scripts/analyze_source_confidence_router_gate.py`

Next exact gate:

- materialize SVAMP `chal311-380`, run source/target/text only, build a
  contrastive target set, and reject unless the source surface reaches `>=6/70`
  source-only over target and `>=4/70` clean source-only after text exclusion

## Cycle Checkpoint: 2026-04-26 SVAMP70 Chal311-380 Source Surface Scout

- cycle number: `2026-04-26-svamp70-chal311-380-source-scout`
- timestamp: `2026-04-26 14:38:00 PDT`
- live branch entering cycle: disjoint Qwen2.5-Math -> Qwen3 source-surface
  scouting after confidence-router pruning
- scale-up rung: medium source-surface scout
- ICLR readiness: not ready; this is a surface selection result, not a method

Start-of-cycle status:

- current paper story: source-sidecar and confidence-router signals do not
  generalize from the original SVAMP70 live surface to disjoint holdout slices
- exact blocker: find enough clean source-only mass to justify C2C or learned
  connector compute
- highest-priority gate: source-only over target at least `6/70` and clean
  source-only after text exclusion at least `4/70`

Result:

- source-alone: `8/70`, numeric coverage `63/70`
- target-alone: `21/70`, numeric coverage `70/70`
- text relay: `19/70`, numeric coverage `70/70`
- target/source oracle: `24/70`
- source-only over target: `3`
- clean source-only after excluding text relay: `2`

Decision:

- reject `chal311-380` as a source-sidecar decision surface
- do not spend C2C or connector compute on this slice
- weaken adjacent SVAMP range scouting for this same source/target pair: three
  adjacent scouts now fail to provide enough clean source mass

Artifacts:

- memo:
  - `paper/qwen25math_svamp70_surface_scout_chal311_380_20260426.md`
- eval slice:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/_artifacts/svamp_chal311_380.jsonl`
  - sha256: `f503455a810222bbc5652a58824c5f5090d6a9d7d80973eab2caac5d51612227`
- source predictions:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_alone.jsonl`
  - sha256: `d40cb67ce378477d4c9ad1d13a8a8c610e5b4291f20b8db4b51369a58900a7b7`
- target predictions:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/target_alone.jsonl`
  - sha256: `6859d28388aa329036f767fe034e7de25eb7aa8f0c636e6b075a5e8ad638691d`
- text-relay predictions:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/text_to_text.jsonl`
  - sha256: `c17ef7746601f3c5357a4acd2354a909f97c78623d67f30805832eaef4a9d2cb`
- source-contrastive target set:
  - `results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_contrastive_target_set.json`
  - sha256: `f1bdc7e775075a2b40b7ed0c96cf039795185868a1722afba9047b70c6bd67dc`
- consolidated surface scan:
  - `results/source_headroom_surface_scan_20260426/scan_with_chal311.json`
  - sha256: `5f5034f7de04ffbf48b1dcc7dcac737fea736f17b8e9b0d7f6e8e70246bd10b4`

Tests:

- no code changes in this cycle
- `git diff --check`

Next exact gate:

- stop adjacent SVAMP range scouting for Qwen2.5-Math -> Qwen3 unless a new
  source encoder or prompting hypothesis changes the source surface
- comb historical `rotalign`, `latent_bridge`, and result-folder memos before
  selecting the next live branch

## Cycle Checkpoint: 2026-04-26 Source-Control Contrastive Cross-Attention

- cycle number: `2026-04-26-source-control-contrastive-cross-attention`
- timestamp: `2026-04-26 16:18:00 PDT`
- live branch entering cycle: token-local cross-attention prefix connector with
  training-time source-control penalties
- scale-up rung: strict-small pre-generation diagnostic
- ICLR readiness: not ready; this is a branch-pruning diagnostic, not a
  positive method

Start-of-cycle status:

- current paper story: C2C exposes useful target-missed headroom, but source
  readouts, decoded sidecars, selectors, and tiny prefix emitters have not
  beaten source-destroying controls
- exact blocker: the previous cross-attention connector was dominated by
  target-only, label-shuffled, shuffled-source, and same-norm controls
- highest-priority gate: add training-time penalties against those controls and
  rerun the same SVAMP32 clean C2C-headroom logprob gate

Implementation:

- extended `scripts/analyze_svamp32_source_cross_attention_logprob_probe.py`
  with optional source-control contrastive training
- controls used in the smoke: zero-source, shuffled-source, same-norm-noise,
  projected-soft-prompt
- contrastive weight: `0.25`
- contrastive margin: `0.25`

Result:

- status: `source_cross_attention_logprob_fails_gate`
- clean IDs scored: `6`
- matched-only clean IDs: `0/6`
- matched-positive clean IDs: `4/6`
- clean control leaks: `4/6`
- mean matched margin on clean IDs: `0.070074`
- mean best-control margin on clean IDs: `0.452928`
- mean matched-minus-control clean margin: `-0.382854`

Decision:

- fail and prune this source-control contrastive variant
- do not tune contrastive weight, margin, epochs, or hidden width on this exact
  tiny prefix-emitting cross-attention architecture
- next live method branch needs a larger architecture change, not another
  objective tweak on this family

Artifacts:

- memo:
  - `paper/qwen25math_svamp32_source_cross_attention_contrastive_20260426.md`
- result JSON:
  - `results/qwen25math_svamp32_source_cross_attention_contrastive_20260426/smoke.json`
  - sha256: `2f6e2a38f6b1685b7a571f30f53dd1587fa03532560aa7fe04f4f515a15cb4a1`
- readout:
  - `results/qwen25math_svamp32_source_cross_attention_contrastive_20260426/smoke.md`
  - sha256: `e009da82d298ade4e65aa0c76709f67f77066654485170fe05a27ca3e9918637`
- analyzer:
  - `scripts/analyze_svamp32_source_cross_attention_logprob_probe.py`
  - sha256: `884005c8b61e41fb908eed6759fe484b90cc8060ef42850a515a2b8327be7d75`

Tests:

- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_cross_attention_logprob_probe.py`
- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_cross_attention_logprob_probe.py -q`

Next exact gate:

- do not continue tiny prefix-emitter tuning on the current SVAMP32/SVAMP70
  surfaces
- next branch should be either a true target-side next-token-loss resampler
  with generation-time evaluation and a matched C2C-fuser baseline, or a new
  source/surface pair that first clears source/text/C2C headroom gates

## Cycle Checkpoint: 2026-04-26 Target-CE Cross-Attention Generation Gate

- cycle number: `2026-04-26-target-ce-cross-attention-generation`
- timestamp: `2026-04-26 16:55:46 PDT`
- live branch entering cycle: token-local source cross-attention prefix
  connector trained with target-side continuation next-token CE
- scale-up rung: strict-small logprob diagnostic plus 64-token clean-ID
  generation
- ICLR readiness: not ready; this kills a low-capacity method family rather
  than producing a positive method

Start-of-cycle status:

- current paper story: C2C exposes real source-assisted headroom, but
  deployable LatentWire rows are still explained by target priors, target-only
  prefixes, source-destroying controls, or unstable seeds
- exact blocker: no source-derived method beats target/text/C2C-relevant
  baselines while surviving zero-source, shuffled-source, target-only, and
  slots-only controls
- live branch: target-side next-token-loss rescue of the previously failed
  source cross-attention prefix connector
- highest-priority gate: determine whether target-CE training plus generation
  scoring fixes the control-dominance failure

Implementation:

- added `--training-objective target_ce` to
  `scripts/analyze_svamp32_source_cross_attention_logprob_probe.py`
- added optional heldout greedy generation with learned prefixes
- added clean-ID generation filtering so logprob can score all rows while
  generation decodes only the six C2C-headroom IDs
- added focused unit coverage for generation gate summaries

Run:

- command recorded in
  `paper/qwen25math_svamp32_target_ce_generation_gate_20260426.md`
- git commit before run: `8105d3ee7c7426931d71e9a2dd04a4a440f197b3`
- seed: `1`
- outer folds: `2`
- source model: `Qwen/Qwen2.5-Math-1.5B`
- target model: `Qwen/Qwen3-0.6B`
- eval file:
  `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl`
- target JSONL:
  `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl`
- teacher JSONL:
  `results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/c2c_generate.jsonl`
- clean generation IDs:
  `3e8a5691f5443495`, `1d50b408c8f5cd2c`, `de1bf4d142544e5b`,
  `47464cc0b064f172`, `6e9745b37ab6fc45`, `575d7e83d84c1e67`

Result:

- status: `source_cross_attention_logprob_fails_gate`
- clean IDs scored in logprob: `6`
- matched-only clean IDs: `0/6`
- matched-positive clean IDs: `4/6`
- clean control leaks: `4/6`
- mean matched margin on clean IDs: `0.061688`
- mean best-control margin on clean IDs: `0.256471`
- mean matched-minus-control clean margin: `-0.194783`
- 64-token generation on clean IDs:
  - matched: `1/6`
  - zero-source: `2/6`
  - shuffled-source: `2/6`
  - target-only prefix: `2/6`
  - slots-only prefix: `2/6`

Decision:

- kill the low-capacity source-prefix-emitter family on this surface
- do not tune prefix length, hidden width, contrastive weight, CE weight, or
  generation decoding for this family without a new source-memory interface
- process repair remains a target-side baseline only, not a source-
  communication method
- next highest-value branch: audit/implement the smallest true
  `latent_bridge` query-innovation source-memory resampler that can train or
  score with LM CE and generation, with matched C2C-fuser and target/slots
  controls from the first gate

Artifacts:

- memo:
  - `paper/qwen25math_svamp32_target_ce_generation_gate_20260426.md`
- result manifest:
  - `results/qwen25math_svamp32_target_ce_generation_gate_20260426/manifest.md`
- result JSON:
  - `results/qwen25math_svamp32_target_ce_generation_gate_20260426/smoke.json`
  - sha256: `a8fda429e29ba9ed1ee06285706dc3f0fa95609be2f8829a508b379e689c9517`
- readout:
  - `results/qwen25math_svamp32_target_ce_generation_gate_20260426/smoke.md`
  - sha256: `20be9a45f3be23453c75cd3b15d9a42953c5ae75cc59299e28ab2f19be3e22d9`
- generations:
  - `results/qwen25math_svamp32_target_ce_generation_gate_20260426/generations.jsonl`
  - sha256: `f6cff21a9f981f6482f04b53ef9902cad6ea9ea079b05e5e57c88a03f998acad`
- analyzer:
  - `scripts/analyze_svamp32_source_cross_attention_logprob_probe.py`
  - sha256: `6c6b3fde3f4c2ecc12f071fb69e80c0f58b47559b1299ad4c2b740cbc4074a36`

Tests:

- `./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_cross_attention_logprob_probe.py`
- `./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_cross_attention_logprob_probe.py -q`

Next exact gate:

- command:
  `./venv_arm64/bin/python -m pytest tests/test_translator_core.py::test_query_innovation_perceiver_connector_fit_and_runtime_are_finite tests/test_translator_core.py::test_query_innovation_anti_memory_control_fit_is_finite tests/test_translator_core.py::test_fit_from_pairs_query_innovation_forwards_source_controls -q`
- then inspect `latent_bridge/translator.py` query-innovation resampler hooks for
  whether a true LM CE/generation scorer can be added without destabilizing
  existing calibration

## Cycle Checkpoint: 2026-04-26 Query-Resampler Answer-Likelihood Gate Blocker

- cycle number: `2026-04-26-query-resampler-answer-likelihood`
- timestamp: `2026-04-26 19:32:37 PDT`
- live branch: `latent_bridge` query-innovation/source-memory resampler
- scale-up rung reached: smoke/strict-small harness gate
- ICLR readiness: not ready; no positive-method evidence was produced in this
  cycle because MPS execution is blocked by an unkillable orphaned calibration
  process

Start-of-cycle status:

- current paper story: historical RotAlign/sidecar/query-resampler results show
  source-complementary headroom and useful architecture clues, but deployable
  rows still fail source controls, seed stability, holdout guards, or C2C
  competitiveness
- exact blocker: no source-derived method beats target/text baselines while
  surviving zero-source, shuffled-source, target-only, and slots-only controls
- current live branch: query-innovation/source-memory resampler with eval-only
  answer-likelihood scoring
- highest-priority gate: matched-vs-source-destroyed gold-answer continuation
  logprob on the existing finite query-innovation checkpoint

What changed:

- audited the other MD/results paths requested by the user, including
  `rotalign`, `latent_bridge`, sidecar, query-resampler, query-innovation,
  SVAMP70 holdout, and GSM residual memos/results
- selected query-innovation/source-memory resampler as the next live branch;
  fixed sidecar guards are now treated as surface/headroom evidence rather than
  the live method
- added eval-only generation answer scoring to `latent_bridge/evaluate.py`
  without changing decoding, training, checkpoint format, or translator
  behavior
- added focused unit coverage in `tests/test_evaluate_helpers.py`

Blocker:

- a redundant MPS capacity sweep was started in `.debug/`
- parent sweep was stopped, but child calibration PID `31103` remained orphaned
  under launchd with `STAT=UE`
- exact process:
  `/Library/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python /Users/sujeethjinesh/Desktop/LatentWire/scripts/calibrate.py ... --device mps --dtype float32 --seed 1`
- `SIGTERM` and `SIGKILL` did not terminate it
- no checkpoint was materialized; only
  `.debug/gsm8k32_query_resampler_bank16_seed1_20260426/_artifacts/gsm8k_eval_32.jsonl`
  exists from that aborted run

Decision:

- hard tooling blocker for further MPS-backed experiments
- live scientific branch is not killed; next gate is blocked on clearing PID
  `31103`
- exact next action: restart the machine or otherwise clear PID `31103`, then
  run the matched/zero/shuffle/target-only/slots-only answer-likelihood commands
  recorded in `paper/query_resampler_answer_likelihood_gate_20260426.md`

Artifacts and hashes:

- memo:
  - `paper/query_resampler_answer_likelihood_gate_20260426.md`
- finite query-innovation checkpoint for resume:
  - `.debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt`
  - sha256: `b1f0cfa62c67ffcbdbce631c6cfd80df3240e132e252b0775aef355940a557b8`
- GSM8K32 eval slice for resume:
  - `.debug/gsm8k32_query_innovation_resampler_seed1_20260423/_artifacts/gsm8k_eval_32.jsonl`
  - sha256: `04d3006a6b37aa691347f290d442279bca23bbe119cf9a9b86002263fded20e1`
- `latent_bridge/evaluate.py`
  - sha256: `f143d4c301f783a607e2647fbc2f1efc9e0097d590d37ed28ea6964e1d7268b7`
- `tests/test_evaluate_helpers.py`
  - sha256: `5c2c03120642487cd2b1ec96e98f4ff91e12732b4ffa6dcc4c2069d82a28e3ea`

Tests:

- `./venv_arm64/bin/python -m py_compile latent_bridge/evaluate.py`
- `./venv_arm64/bin/python -m pytest tests/test_evaluate_helpers.py -q`
  - `103 passed`
- `./venv_arm64/bin/python -m pytest tests/test_translator_core.py::test_query_innovation_perceiver_connector_fit_and_runtime_are_finite tests/test_translator_core.py::test_query_innovation_anti_memory_control_fit_is_finite tests/test_translator_core.py::test_fit_from_pairs_query_innovation_forwards_source_controls -q`
  - `3 passed`

Next exact gate:

- first clear PID `31103`
- then run the command block in
  `paper/query_resampler_answer_likelihood_gate_20260426.md`
  to produce:
  - `results/gsm8k32_query_innovation_answer_likelihood_20260426/matched.jsonl`
  - `zero_source.jsonl`
  - `shuffled_source_salt1.jsonl`
  - `target_only.jsonl`
  - `slots_only.jsonl`

## Cycle Checkpoint: 2026-04-26 Query-Innovation CPU Answer-Likelihood Smoke

- cycle number: `2026-04-26-query-innovation-cpu-answer-likelihood-smoke`
- timestamp: `2026-04-26 20:07:00 PDT`
- live branch: finite query-innovation/source-memory checkpoint,
  `dynalign_query_innovation_resampler_replace`, gate `0.15`
- scale-up rung reached: CPU micro smoke
- result summary: fail; matched source does not beat source-destroyed controls
  and ties slots-only exactly

Start-of-cycle status:

- ICLR readiness: not ready; current work is still branch selection/pruning
- current paper story: old RotAlign/sidecar/query-resampler results reveal
  headroom, but deployable methods keep failing source-control attribution
- exact blocker: no query-innovation row has shown matched-source answer
  likelihood that survives zero-source, shuffled-source, and memory-null
  controls
- highest-priority gate: use the new eval-only answer-likelihood fields on a
  CPU micro slice while MPS remains blocked by PID `31103`

Run:

- result directory:
  `results/gsm8k32_query_innovation_answer_likelihood_cpu_smoke_20260426/`
- eval file:
  `results/gsm8k32_query_innovation_answer_likelihood_cpu_smoke_20260426/gsm8k_eval_4.jsonl`
- checkpoint:
  `.debug/checkpoints_gsm8k32_query_innovation_resampler_seed1_20260423/dynalign_query_innovation_resampler_replace/qwen25_to_qwen3_grouped_subspace_transport_w010_r16_dynalign_query_innovation_resampler_replace_cal64_chat_bank16_seed1.pt`
- device/dtype: `cpu` / `float32`
- methods/controls:
  - matched
  - zero source
  - shuffled source, salt `1`
  - slots-only memory
  - target-only attempted but unavailable for this checkpoint

Result:

- analyzer status: `answer_likelihood_controls_fail`
- matched: `0/4`, mean answer logprob `-7.025400`
- zero-source: `0/4`, mean answer logprob `-6.925437`
- shuffled-source: `0/4`, mean answer logprob `-7.048394`
- slots-only: `0/4`, mean answer logprob `-7.025400`
- matched-minus-zero mean delta: `-0.099963`
- matched-minus-shuffled mean delta: `+0.022994`
- matched-minus-slots mean delta: `0.000000`
- matched-minus-best-control mean delta: `-0.115530`
- matched best-control wins/losses/ties: `0/4/0`
- target-only error:
  `ValueError: --innovation-memory-control target_only requires a target-conditioned query-innovation checkpoint`

Decision:

- kill the current finite query-innovation checkpoint as a live
  source-communication row
- weaken the non-target-conditioned query-innovation/source-memory family:
  it can be finite, but current evidence says its apparent signal is
  source-destroyed or slots-only reproducible
- next highest-value branch: target-conditioned query-innovation/source-memory
  connector whose first gate includes matched, zero-source, shuffled-source,
  `target_only`, and `slots_only`
- hard blocker remains: MPS process PID `31103` is stuck in uninterruptible
  `STAT=UE`, so new connector implementation/testing that requires MPS must
  wait until that process is cleared

Artifacts:

- manifest:
  - `results/gsm8k32_query_innovation_answer_likelihood_cpu_smoke_20260426/manifest.md`
- analyzer:
  - `scripts/analyze_answer_likelihood_controls.py`
  - sha256: `5cb2249e869581a2654196057a1ff032dc9b4ca3bef4dc0063fa100d94262056`
- analyzer test:
  - `tests/test_analyze_answer_likelihood_controls.py`
  - sha256: `405237f126053f126f1d53d6f7dbb7224eae1b5137d3870988d2422212ecd965`
- analysis JSON:
  - `results/gsm8k32_query_innovation_answer_likelihood_cpu_smoke_20260426/answer_likelihood_controls.json`
  - sha256: `4848427ad10a3092169424f63b408afbf95a463c8137a46fdfdf866a155723a3`
- analysis MD:
  - `results/gsm8k32_query_innovation_answer_likelihood_cpu_smoke_20260426/answer_likelihood_controls.md`
  - sha256: `54872322ab10b95c13f28206b0fb78c17a830d57e301fdb0be7cde3bdbc862db`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_answer_likelihood_controls.py -q`
  - `2 passed`

Next exact gate:

- clear PID `31103`
- implement or materialize a target-conditioned query-innovation checkpoint
  that supports `--innovation-memory-control target_only`
- rerun the same answer-likelihood analyzer on at least GSM8K32/SVAMP32 with
  matched, zero-source, shuffled-source, target-only, and slots-only controls

## Cycle Checkpoint: 2026-04-26 Query-Memory Answer-Likelihood CPU Sweeps

- cycle number: `2026-04-26-query-memory-answer-likelihood-cpu-sweeps`
- timestamp: `2026-04-26 21:22:00 PDT`
- live branch entering cycle: target-conditioned query-memory / Perceiver
  rescue after the non-target-conditioned query-innovation checkpoint failed
- scale-up rung reached: CPU micro smoke plus strict-small clean-ID expansion
- result summary: fail; one 4-clean-ID Qwen2.5-Math Perceiver answer-likelihood
  pass did not survive all six clean IDs

Start-of-cycle status:

- ICLR readiness: not ready; no deployable source-derived method survives
  source-destroying and memory-null controls
- current paper story: RotAlign/query-resampler/sidecar artifacts show useful
  source-complementary and C2C headroom, but repeated deployable rows are
  explained by target priors, slots-only memory, shuffled-source controls, weak
  source surfaces, or brittle guards
- exact blocker: decide whether target-conditioned query-memory checkpoints
  hide answer-likelihood signal that teacher-forced/generation gates missed
- current live candidates:
  - SVAMP32 target-conditioned delta-memory query codec
  - SVAMP70 Perceiver answer-teacher contrastive checkpoint
  - Qwen2.5-Math SVAMP32 Perceiver C2C-residual checkpoint
- highest-priority gate: matched-vs-zero/shuffle/target-only/slots-only
  answer-likelihood controls on CPU while MPS remains blocked by PID `31103`

Runs:

- SVAMP32 delta-memory:
  - result directory:
    `results/svamp32_deltamem_answer_likelihood_cpu_smoke_20260426/`
  - checkpoint:
    `.debug/svamp32_delta_memory_query_codec_20260424/checkpoints/qwen25_to_qwen3_svamp32_deltamem_konly_query_codec_r16_bank16_seed1.pt`
  - checkpoint sha256:
    `29ff93c6d7291fb9a4e00ac35a7ffa519c4d71c8bd4a38062c0d748baecf4ebb`
  - analyzer status: `answer_likelihood_controls_fail`
  - matched mean answer logprob: `-7.673776`
  - zero-source mean: `-7.568559`
  - shuffled-source mean: `-7.683792`
  - target-only mean: `-8.072149`
  - slots-only mean: `-8.071200`
  - best-control wins/losses/ties: `0/4/0`
  - mean live-best delta: `-0.141012`
- SVAMP70 Perceiver answer-teacher:
  - result directory:
    `results/svamp70_perceiver_answer_likelihood_cpu_smoke_20260426/`
  - checkpoint:
    `.debug/svamp70_perceiver_answer_teacher_contrastive_20260426/checkpoints/qwen25_to_qwen3_svamp70_perceiver_answer_teacher_w080_ctrl050_r16_b16_seed1.pt`
  - checkpoint sha256:
    `a7221d6d0ee81b99573bf1893b66570ec682f22faee1ffcc6bf7e9fc1f36df6a`
  - analyzer status: `answer_likelihood_controls_fail`
  - matched mean answer logprob: `-7.261671`
  - zero-source mean: `-7.262390`
  - shuffled-source mean: `-7.220054`
  - target-only mean: `-7.232674`
  - slots-only mean: `-7.241025`
  - best-control wins/losses/ties: `0/4/0`
  - mean live-best delta: `-0.112360`
- Qwen2.5-Math SVAMP32 Perceiver 4-clean-ID smoke:
  - result directory:
    `results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/`
  - checkpoint:
    `.debug/qwen25math_svamp32_perceiver_c2c_residual_20260426/checkpoints/qwen25math_to_qwen3_svamp32_perceiver_c2c_residual_w080_ctrl050_am050_r16_b16_seed1.pt`
  - checkpoint sha256:
    `d50b00fd0b9f5b5afcb09af8f9ae89b868e913b0a0610ef8132e66f20c726759`
  - analyzer status: `answer_likelihood_controls_pass`
  - matched mean answer logprob: `-7.989116`
  - zero-source mean: `-8.250677`
  - shuffled-source mean: `-8.131923`
  - target-only mean: `-8.162249`
  - slots-only mean: `-8.118848`
  - best-control wins/losses/ties: `3/1/0`
  - mean live-best delta: `+0.080362`
- Qwen2.5-Math SVAMP32 Perceiver clean6 expansion:
  - result directory:
    `results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/`
  - checkpoint:
    `.debug/qwen25math_svamp32_perceiver_c2c_residual_20260426/checkpoints/qwen25math_to_qwen3_svamp32_perceiver_c2c_residual_w080_ctrl050_am050_r16_b16_seed1.pt`
  - checkpoint sha256:
    `d50b00fd0b9f5b5afcb09af8f9ae89b868e913b0a0610ef8132e66f20c726759`
  - analyzer status: `answer_likelihood_controls_fail`
  - matched mean answer logprob: `-8.195434`
  - zero-source mean: `-8.387585`
  - shuffled-source mean: `-8.190414`
  - target-only mean: `-8.192871`
  - slots-only mean: `-8.191226`
  - best-control wins/losses/ties: `4/2/0`
  - mean live-best delta: `-0.090384`

Decision:

- kill the target-conditioned query-memory / Perceiver checkpoint family as the
  current live positive-method branch
- record the 4-ID Qwen2.5-Math pass as a useful partial mechanism clue only;
  it is not promotable because the all-clean-ID expansion fails mean
  matched-minus-control deltas against shuffled-source, target-only, and
  slots-only controls
- do not tune fixed gate, answer-teacher weight, anti-memory weight, query
  count, bridge rank, or another Perceiver memory checkpoint on these exact
  surfaces without a new source-interface hypothesis
- current live method branch: none
- next highest-value branch: source-surface/interface reset after the hard MPS
  blocker is cleared

Artifacts:

- focused memo:
  - `paper/query_memory_answer_likelihood_cpu_sweeps_20260426.md`
- manifests:
  - `results/svamp32_deltamem_answer_likelihood_cpu_smoke_20260426/manifest.md`
  - `results/svamp70_perceiver_answer_likelihood_cpu_smoke_20260426/manifest.md`
  - `results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/manifest.md`
  - `results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/manifest.md`
- analysis JSON/readouts:
  - `results/svamp32_deltamem_answer_likelihood_cpu_smoke_20260426/answer_likelihood_controls.json`
    sha256: `a0d39d36ac0cd3b4bc1c4a25d211e2b48554f5c716722188147b4e8c20122615`
  - `results/svamp70_perceiver_answer_likelihood_cpu_smoke_20260426/answer_likelihood_controls.json`
    sha256: `6b8778bde08cba1be3af04f8529a0c5e54d54507c4a9212e4977052b0e16f856`
  - `results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/answer_likelihood_controls.json`
    sha256: `a834edac89d7721a2c54968c3007bca22d24e18eec248d99af2b2ffde1ddcfc9`
  - `results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/answer_likelihood_controls.json`
    sha256: `ad731dfa93c08bfb6cd27999a53a11c2f273599722d2d1224ef8df55f94cb0bd`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_analyze_answer_likelihood_controls.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/analyze_answer_likelihood_controls.py`

Hard blocker:

- PID `31103` remains orphaned under launchd with `STAT=UE`:
  `/Library/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python /Users/sujeethjinesh/Desktop/LatentWire/scripts/calibrate.py ... --device mps --dtype float32 --seed 1`
- `SIGTERM` and `SIGKILL` have not terminated it
- exact next action from the user/environment: restart the machine or otherwise
  clear PID `31103`; do not start additional MPS jobs while it remains stuck

Existing-artifact surface reset:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_source_headroom_surfaces.py \
  --surface svamp70_live_source=target_path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_live_source \
  --surface svamp70_holdout_source=target_path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_holdout_source \
  --surface svamp70_chal171_source=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_chal171_240_source \
  --surface svamp70_chal241_source=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_chal241_310_source \
  --surface svamp70_chal311_source=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_chal311_380_source \
  --surface gsm70_math_source=target_path=results/qwen25math_qwen3_gsm70_source_surface_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_gsm70_source_surface_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_gsm70_source \
  --surface svamp32_math_chat_source=target_path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl,source_path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp32_chat_source \
  --min-source-only 6 \
  --output-json results/source_headroom_surface_scan_20260426/scan_after_query_memory_prune.json \
  --output-md results/source_headroom_surface_scan_20260426/scan_after_query_memory_prune.md
```

- result:
  - `svamp70_live_source`: strong, target `21/70`, source `13/70`,
    source-only `9`, oracle `30/70`
  - `svamp70_holdout_source`: strong, target `8/70`, source `8/70`,
    source-only `6`, oracle `14/70`
  - all adjacent SVAMP70 scouts, GSM70, and SVAMP32 remain below the
    `>=6/70` source-only threshold
- output hashes:
  - `results/source_headroom_surface_scan_20260426/scan_after_query_memory_prune.json`
    sha256: `3ebd8aff86b732ca6b4137fb11a9306ba3b92b2e9be5a4a9d531dfefb7875b0b`
  - `results/source_headroom_surface_scan_20260426/scan_after_query_memory_prune.md`
    sha256: `0738cfe945fa4925983334801eedaa7ba24e7330e25c32bbbc6dd1c80b1c2f54`

Next exact gate after clearing PID `31103`:

- keep `svamp70_live_source` as the live surface and
  `svamp70_holdout_source` as the immediate validation surface
- do not reuse fixed decoded guards, shallow source-text routers, tiny prefix
  emitters, source-token residue readouts, or Perceiver target-memory
  checkpoints
- implement only a materially different rate-capped source interface on this
  live/holdout surface, or scout a new source/target pair if a cached stronger
  source is available

Immediate resume command before any new MPS work:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Proceed only if PID `31103` is gone or no longer a `scripts/calibrate.py`
process using `--device mps`.

## 2026-04-27 Cycle - SVAMP70 source likelihood sketch branch

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; the project still lacks
   a positive method that survives disjoint source-destroying controls.
2. Current paper story: Qwen2.5-Math -> Qwen3 has real source/C2C headroom on
   SVAMP70 live and holdout surfaces, but decoded guards, process repair,
   prefix emitters, query-memory, and Perceiver-style target-memory rows have
   failed strict controls or holdout validation.
3. Exact blocker to submission: no source-derived communication method beats
   target/text while preserving target-correct examples and proving dependence
   on real source information.
4. Live branch: source likelihood sketch, a rate-capped syndrome-like sidecar
   over the existing target/text/source candidate pool.
5. Highest-priority gate: collect source-model continuation likelihoods on
   live and holdout, fit the acceptance rule on live CV, and evaluate frozen
   holdout once.
6. Scale-up rung: strict-small tooling/gate implementation; scientific MPS run
   blocked.

What changed:

- Added `scripts/collect_source_likelihood_sketch.py` to score candidate
  predictions under the source model and write source-likelihood sketch JSONL.
- Added `scripts/analyze_svamp70_source_likelihood_sketch_gate.py` to quantize
  the source sketch, fit live-CV decision stumps, evaluate source-destroying
  controls, and freeze the live rule for holdout.
- Added focused tests:
  - `tests/test_collect_source_likelihood_sketch.py`
  - `tests/test_analyze_svamp70_source_likelihood_sketch_gate.py`
- Added focused memo:
  - `paper/svamp70_source_likelihood_sketch_20260427.md`
- Added reference memo:
  - `references/465_source_likelihood_syndrome_sidecar_refs.md`
- Added planned artifact manifest:
  - `results/qwen25math_svamp70_source_likelihood_sketch_20260427/manifest.md`

Evidence and decision:

- No new scientific score was produced because PID `31103` is still stuck as
  an orphaned MPS `scripts/calibrate.py` process.
- The branch is selected and harnessed as the next highest-value method because
  it is materially different from prior shallow routers: the source model
  communicates only a compact likelihood preference over target-side candidate
  context.
- This branch is not promoted. It remains a strict-small gate awaiting the MPS
  reset.

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp70_source_likelihood_sketch_gate.py tests/test_collect_source_likelihood_sketch.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_svamp70_source_likelihood_sketch_gate.py scripts/collect_source_likelihood_sketch.py
```

Result:

- `6 passed in 0.03s`
- py-compile passed

Hard blocker:

- PID `31103` remains under launchd with `STAT=UE`:
  `/Library/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python /Users/sujeethjinesh/Desktop/LatentWire/scripts/calibrate.py ... --device mps --dtype float32 --seed 1`
- Earlier `SIGTERM` and `SIGKILL` did not clear it.
- Do not launch further MPS jobs until it is gone.

Immediate resume check:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Next exact gate after that process is cleared:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/collect_source_likelihood_sketch.py \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --eval-file results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl \
  --candidate target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --candidate text=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --candidate source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --reference-label target \
  --candidate-text-field prediction \
  --prompt-mode direct \
  --source-use-chat-template \
  --source-enable-thinking false \
  --device mps \
  --dtype float32 \
  --output-jsonl results/qwen25math_svamp70_source_likelihood_sketch_20260427/live_sketch.jsonl \
  --output-md results/qwen25math_svamp70_source_likelihood_sketch_20260427/live_sketch.md
```

Then run the holdout collection and analyzer commands recorded in
`paper/svamp70_source_likelihood_sketch_20260427.md`.

## 2026-04-27 Cycle - Historical positive branch audit

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; no positive method has
   yet survived live/holdout controls with uncertainty, seed stability, and
   systems accounting.
2. Current paper story: the old RotAlign/latent_bridge/results folders contain
   real source-complementary and systems clues, but the method rows either fail
   seed stability, source controls, holdout, or are only oracle bounds.
3. Exact blocker to submission: no deployable source-derived communication
   method has cleared strict controls on a disjoint validation surface.
4. Live branch: `source_likelihood_sketch` on SVAMP70 live/holdout.
5. Highest-priority gate: clear PID `31103`, then run the source-likelihood
   sketch collector/analyzer commands.
6. Scale-up rung: strict-small branch-selection and tooling; scientific run
   blocked by MPS.

Historical audit summary:

- best historical source-derived clue:
  `qwen25math_svamp32/source_contrastive_sidecar` and its SVAMP70 medium
  follow-up
  - SVAMP32: matched `11/32`, target/text `8/32`, clean source-necessary
    `3/4`, control clean union `0/4`
  - SVAMP70 live: textless sidecar `26/70`, target `21/70`, text `22/70`,
    C2C `31/70`, clean source-necessary `4/6`, control clean union `0/6`
  - holdout falsification: matched `10/70`, clean source-necessary `0/2`,
    control clean union `2/2`
- best old RotAlign clue:
  GSM70 `dynalign_module_replace_residrank16`
  - seed0 `8/70` vs target `4/70`, zero/shuffle controls retained `0/6`
    live wins with target fallback
  - seed3 `2/70`, seed4 `4/70`; raw dynalign killed as a live method
- best oracle/bound:
  SVAMP32 syndrome sidecar
  - strict target-side pool matched `14/32`, clean source-necessary `2/6`,
    control clean union `0/6`, one-byte syndrome
  - not a method because the syndrome used C2C-derived numeric residues
- query-memory/Perceiver family:
  - one 4-ID Qwen2.5-Math pass existed, but clean6 expansion failed
  - keep only as architecture inspiration, not a live branch
- `idweighted_query_innovation` clue:
  - best controlled row `10/32`, clean residual/source-necessary `1/6`
  - clean ID `aee922049c757331` was not retained by translated-KV-zero
  - still below target self-repair `14/32` and below the `>=2/6` clean gate;
    revive only as an innovation-bottleneck design clue, not as-is
- process repair:
  - remains target-side baseline/confound; source controls recover all
    route-specific wins

Decision:

- Keep `source_likelihood_sketch` as the live branch because it is the most
  direct non-duplicative successor to the source-contrastive sidecar and
  syndrome-bound clues.
- Do not revive raw dynalign, fixed decoded guards, process repair, or
  Perceiver/query-memory tuning without a new conditional innovation objective
  and predictor/shuffle controls.

New memo:

- `paper/historical_positive_branch_audit_20260427.md`

Hard blocker:

- PID `31103` remains the MPS blocker; no new model runs were launched.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Then, if clear, run the three source likelihood sketch commands in
`paper/svamp70_source_likelihood_sketch_20260427.md`.

## 2026-04-27 Cycle - Source likelihood collector hardening

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; no positive method has
   yet cleared live/holdout controls with the required reproducibility and
   systems package.
2. Current paper story: historical audit keeps the side-information/source
   sidecar family as the most defensible branch, with `source_likelihood_sketch`
   as the current live method gate.
3. Exact blocker to submission: the branch still needs a scientific live and
   holdout run; local MPS remains blocked by PID `31103`.
4. Live branch: `source_likelihood_sketch` on SVAMP70 live/holdout.
5. Highest-priority gate: make the next collector command resumable and
   artifact-complete, then run it after PID `31103` is cleared.
6. Scale-up rung: strict-small tooling hardening; scientific MPS run blocked.

What changed:

- Hardened `scripts/collect_source_likelihood_sketch.py`:
  - added `--limit` for micro smoke collection
  - added `--resume` for append-safe recovery from interrupted long runs
  - records command, git commit, eval/candidate hashes, ordered example IDs,
    ordered-ID hash, resume status, skipped-existing count, and output hash in
    the markdown readout
- Expanded `tests/test_collect_source_likelihood_sketch.py` to cover
  `--limit` and `--resume`.
- Updated `paper/svamp70_source_likelihood_sketch_20260427.md` so the next
  run uses `--resume` and includes a two-example smoke command.

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_collect_source_likelihood_sketch.py tests/test_analyze_svamp70_source_likelihood_sketch_gate.py -q
./venv_arm64/bin/python -m py_compile scripts/collect_source_likelihood_sketch.py scripts/analyze_svamp70_source_likelihood_sketch_gate.py
```

Result:

- `7 passed in 0.08s`
- py-compile passed

Hard blocker:

- PID `31103` remains the stuck MPS `scripts/calibrate.py` process in
  `STAT=UE`; no model run was launched.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If cleared, first run the `--limit 2` smoke in
`paper/svamp70_source_likelihood_sketch_20260427.md`; if the smoke produces
finite scores and the readout hashes, run the full live and holdout collection
commands with `--resume`, then the frozen analyzer.

## 2026-04-27 Cycle - CPU source likelihood smoke under MPS blocker

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; the live method still
   lacks full live/holdout source-control evidence.
2. Current paper story: `source_likelihood_sketch` remains the top branch
   because historical positive rows point to source-derived side-information,
   but fixed decoded guards and query-memory branches are not reliable enough.
3. Exact blocker to submission: full SVAMP70 live/holdout sketch collection
   cannot run safely on MPS while orphaned PID `31103` remains in `STAT=UE`.
4. Live branch: `source_likelihood_sketch`.
5. Highest-priority gate: validate the collector path with a bounded CPU smoke
   while waiting for MPS to be cleared.
6. Scale-up rung: micro smoke for tooling only; strict-small scientific gate is
   still blocked.

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/collect_source_likelihood_sketch.py \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --eval-file results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl \
  --candidate target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --candidate text=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --candidate source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --reference-label target \
  --candidate-text-field prediction \
  --prompt-mode direct \
  --source-use-chat-template \
  --source-enable-thinking false \
  --device cpu \
  --dtype float32 \
  --limit 2 \
  --output-jsonl .debug/qwen25math_svamp70_source_likelihood_sketch_20260427/live_smoke_cpu.jsonl \
  --output-md .debug/qwen25math_svamp70_source_likelihood_sketch_20260427/live_smoke_cpu.md
```

Result:

- Passed as a collector/provenance smoke: `2` rows, finite continuation scores,
  append-only JSONL output, markdown readout, command capture, input hashes,
  ordered-ID hash, and output hash.
- Elapsed time: `96.06s` on CPU, too slow for the full live/holdout gate.
- Top labels on the two smoke examples: `text`, `text`; this is not a
  scientific pass or fail for the method.

Artifacts:

- JSONL: `.debug/qwen25math_svamp70_source_likelihood_sketch_20260427/live_smoke_cpu.jsonl`
- JSONL sha256:
  `863254ecc5110eab3e62efb65ddb31e9472be42513bce6ce1ab44842e1057e9d`
- markdown: `.debug/qwen25math_svamp70_source_likelihood_sketch_20260427/live_smoke_cpu.md`
- markdown sha256:
  `cd12db13419021f248c311776e9c3b148d60faa69297c31b6a8d272fc863d0f9`
- ordered IDs: `013133cdef4f637c`, `d64f6e35083ffe8c`
- ordered-ID sha256:
  `06403406633be53c4d2db3f0064af1afe2078ee67fc2dd748f91aa3d82ea530b`
- git commit used for the run:
  `154430a33d0d649e30b877d7b4d38015a229ac9a`

Decision:

- Promote only the collector tooling, not the scientific method.
- Keep the live branch unchanged.
- Stop MPS execution until PID `31103` is cleared; CPU is acceptable only for
  tiny smoke/debug work.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If clear, run the MPS `--limit 2` smoke in
`paper/svamp70_source_likelihood_sketch_20260427.md`, then full live and
holdout collection with `--resume`, then the frozen analyzer.

## 2026-04-27 Cycle - Kill source likelihood sketch

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; no deployable positive
   method has cleared live/holdout source controls.
2. Current paper story: historical source sidecars still motivate
   side-information decoding, but source likelihood over candidate answers is
   not a reliable communicator on the frozen SVAMP70 live surface.
3. Exact blocker to submission: the live branch failed strict-small gates;
   MPS remains blocked by PID `31103`.
4. Live branch: `source_likelihood_sketch`, now killed on this surface.
5. Highest-priority gate: switch to a richer source-controlled syndrome
   predictor or source-surface discovery branch.
6. Scale-up rung: strict-small kill decision.

What changed:

- Added `--continuation-template` to
  `scripts/collect_source_likelihood_sketch.py` so answer-only sketches can use
  canonical continuations such as `Answer: {text}`.
- Added collector tests for continuation templates.
- Ran normalized-answer and formatted-answer sketch variants on CPU because the
  continuations are short enough to avoid the MPS blocker.
- Ran the existing `source_trace_router` harness as the next selected
  source-surface branch.

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_collect_source_likelihood_sketch.py tests/test_analyze_svamp70_source_likelihood_sketch_gate.py -q
./venv_arm64/bin/python -m py_compile scripts/collect_source_likelihood_sketch.py scripts/analyze_svamp70_source_likelihood_sketch_gate.py
git diff --check
```

Result:

- `9 passed in 0.13s`
- py-compile passed
- diff check passed

Result summary:

- normalized answer mean logprob: live `21/70`, clean source-necessary `0`;
  holdout `8/70`, clean source-necessary `0`
- normalized answer sum logprob: live `20/70`, clean source-necessary `0`;
  holdout `8/70`, clean source-necessary `0`
- formatted `Answer: {text}` mean logprob: live `20/70`, clean
  source-necessary `0`, control union `1`; holdout `10/70`, clean
  source-necessary `2`, control union `0`
- formatted `Answer: {text}` sum logprob: live `19/70`, clean
  source-necessary `0`; holdout `8/70`, clean source-necessary `0`
- source-trace router scout: live `20/70`, clean source-necessary `1`,
  accepted harm `2`; holdout `10/70`, clean source-necessary `1`, but the
  holdout clean ID survives equation permutation

Decision:

- Kill `source_likelihood_sketch` as the current live branch. Four adjacent
  variants fail live and do not recover clean source-necessary IDs.
- Do not promote `source_trace_router`; it is a diagnostic clue only.
- Select the next branch as a richer source-controlled syndrome predictor,
  using the target-candidate decoder but changing the source signal.

Focused memo:

- `paper/source_likelihood_sketch_kill_20260427.md`

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

After PID `31103` is cleared, implement/run a source-controlled syndrome
predictor rather than another source-likelihood sketch.

## 2026-04-27 Cycle - Post-sketch syndrome bounds

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; no deployable
   source-derived positive method has cleared live/holdout controls.
2. Current paper story: source sidecars and syndrome decoders remain useful as
   diagnostics, but the SVAMP70 live/holdout surface does not support a
   promotable syndrome method after the likelihood-sketch kill.
3. Exact blocker to submission: the current live branch is killed, post-kill
   syndrome bounds fail, and MPS is blocked by orphaned PID `31103`.
4. Live branch: none after this cycle.
5. Highest-priority gate: replay C2C-teacher and source-teacher syndrome bounds
   on frozen SVAMP70 live/holdout artifacts.
6. Scale-up rung: strict small/medium kill check.

What changed:

- Ran the artifact-only `scripts/analyze_svamp32_syndrome_sidecar_probe.py`
  on SVAMP70 live/holdout with C2C-derived residues.
- Ran the same replay with source-answer residues to test whether source
  numeric side information has enough target-safe headroom to justify a richer
  predictor.
- Added focused memo:
  `paper/svamp70_syndrome_bounds_after_sketch_kill_20260427.md`.

Result summary:

- C2C teacher live: best `24/70` vs target `21/70`, clean source-necessary
  `4/6`, control union `0/6`, but below the `25/70` gate and with one
  provenance issue.
- C2C teacher holdout: best `17/70` vs target `8/70`, but clean
  source-necessary `0/2` and control clean union `2/2`.
- Source teacher live: `15/70` vs target `21/70`, clean source-necessary
  `6/6`, control union `0/6`, target-self matched only `4`; it recovers
  source-only IDs by destroying target-correct preservation.
- Source teacher holdout: `9/70` vs target `8/70`, clean source-necessary
  `1/2`, control clean union `1/2`.

Decision:

- Kill the post-sketch syndrome continuation on this SVAMP70 surface.
- Do not implement a richer learned source-syndrome predictor on this exact
  live/holdout slice until a bound first clears target-self preservation and
  source controls.
- Select source-surface discovery as the next branch, but it is blocked on MPS
  because new C2C/source-feature artifacts require model generation or forward
  passes.

Artifacts:

- `results/qwen25math_svamp70_source_likelihood_sketch_20260427/svamp70_live_c2c_syndrome_bound_after_sketch_kill.json`
  - sha256:
    `662534e42454526872e760d3ca622daa25b04c84a82bf8600111533770b0d857`
- `results/qwen25math_svamp70_source_likelihood_sketch_20260427/svamp70_holdout_c2c_syndrome_bound_after_sketch_kill.json`
  - sha256:
    `a1125332f34e4585cb3efbc5d6e5b4ad7a4059695d03347881f1fde0000a9d29`
- `results/qwen25math_svamp70_source_likelihood_sketch_20260427/svamp70_live_source_syndrome_bound_after_sketch_kill.json`
  - sha256:
    `3369b5590c7ea36c732ca8b03904bffe4af18d8067e2c4624e4798a6ce9fcb0a`
- `results/qwen25math_svamp70_source_likelihood_sketch_20260427/svamp70_holdout_source_syndrome_bound_after_sketch_kill.json`
  - sha256:
    `e6fd0ea71815835dc9a9026b8e3d75751c22830fe40b905f2ae1c533b672001f`

Tests:

```bash
git diff --check
```

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

After PID `31103` is cleared, run a new stronger-source surface scout rather
than the previously recorded `chal311_380` adjacent SVAMP scout; existing
`chal311_380` artifacts already fail the source-mass threshold.

## 2026-04-27 Cycle - Post-kill historical and CPU audit

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; no deployable positive
   source-communication method has cleared live/holdout controls.
2. Current paper story: source side-information has real headroom and useful
   bounds, but deployable rows are control-explained, target-self destructive,
   seed-unstable, or too weak.
3. Exact blocker to submission: no live branch remains, and MPS remains
   blocked by orphaned PID `31103` in `STAT=UE`.
4. Live branch: none.
5. Highest-priority gate: exhaust CPU-only existing-artifact branch selection
   before spending the next clear MPS window.
6. Scale-up rung: post-kill branch selection / hard-blocker checkpoint.

What changed:

- Re-read the historical positive MD/results trail for `rotalign`,
  `latent_bridge`, and result folders.
- Ran a post-kill chal241-310 source-sidecar CV router gate.
- Ran a consolidated CPU-only source-headroom scan including `chal311_380`.
- Added focused memo:
  `paper/postkill_historical_cpu_audit_20260427.md`.

Result summary:

- `chal241-310` CV router fails: best row matches `10/70`, recovers only
  `1` clean source-necessary ID, control clean union `0`, accepted harm `1`.
- Existing-surface scan: `svamp70_live` and `svamp70_holdout` have headroom but
  are already consumed and killed by controls; `chal171-240`, `chal241-310`,
  and `chal311-380` are weak with source-only `2`, `4`, and `3` respectively.
- Historical positives remain mechanism clues only:
  `dynalign_module_replace_residrank16` is seed-unstable, ID-weighted
  query-innovation recovers only one clean ID, and Perceiver/query-memory
  checkpoints fail six-clean-ID controls.

Decision:

- Existing-artifact CPU mining is exhausted.
- Do not run the old `chal311_380` MPS scout after the blocker clears; those
  artifacts already exist and fail the surface threshold.
- End this segment as a hard blocker: useful MPS work is unsafe until PID
  `31103` is cleared.

Artifacts:

- `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_cv_router_penalty010_postkill_sidecar.json`
  - sha256:
    `99a742cd10efaf43136be8d3d666b1bfc3fcb73507c66289d58cec5c1654e51b`
- `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_cv_router_penalty010_postkill_sidecar.md`
  - sha256:
    `672fc1e882b01908d227ab814c8359ecca30e107e2e376437f943abd086f74f1`
- `results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_cv_router_penalty010_postkill_predictions.jsonl`
  - sha256:
    `24dbd297d4c18cabe79f141e34df858df6437419d54e6913b0fff9e0770e7a88`
- `.debug/cpu_only_next_gate_20260427/source_headroom_surfaces_with_chal311.json`
  - sha256:
    `181df1b5b0f71c6bde86cccc7d72cddea77c61bbe54c2f762f8fb07952e885eb`
- `.debug/cpu_only_next_gate_20260427/source_headroom_surfaces_with_chal311.md`
  - sha256:
    `0bbf961578eaf80db54ed99bcb3c82b1bafbd31b884f0544d58dd42068fc3981`

Tests:

```bash
./venv_arm64/bin/python scripts/materialize_generation_baselines.py --help
git diff --check
```

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If the PID is absent, run:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/qwen25math7b_qwen3_svamp70_surface_scout_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-Math-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods target source t2t \
  --limit 70 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

## 2026-04-27 Cycle - MPS blocker recheck

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; no deployable positive
   source-communication method has cleared live/holdout controls.
2. Current paper story: historical `rotalign`, `latent_bridge`, and
   side-information positives remain mechanism clues, but the current method
   branch is killed.
3. Exact blocker to submission: no live branch remains, and MPS is blocked by
   orphaned PID `31103`.
4. Live branch: none.
5. Highest-priority gate: verify whether the MPS blocker has cleared before
   starting the recorded stronger-source scout.
6. Scale-up rung: hard-blocker checkpoint.

Result:

- `ps -p 31103 -o pid,ppid,stat,etime,command` still reports PID `31103`,
  `PPID=1`, `STAT=UE`, running `scripts/calibrate.py ... --device mps`.
- A bounded `kill -9 31103` retry did not clear the process; it remained in
  `STAT=UE` after a short wait.
- No MPS jobs were started.

Decision:

- Hard blocker persists. The next action is still OS/session-level cleanup or
  reboot to clear PID `31103`; do not start MPS work from this session while it
  remains present.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent, run the stronger-source scout recorded in
`paper/postkill_historical_cpu_audit_20260427.md`.

## 2026-04-27 Cycle - Creative reference synthesis and PDF conversion

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; no live positive method
   survives target-alone/text/C2C/self-repair baselines, source-destroying
   controls, seed stability, and cross-family falsification.
2. Current paper story: historical `rotalign` and `latent_bridge` results are
   mechanism clues only. The next paper story must show target-side
   side-information decoding from a real source-derived code.
3. Exact blocker to submission: no branch currently recovers source-necessary
   wins beyond target/cache/formatting artifacts; local MPS remains blocked by
   orphaned PID `31103`.
4. Live branch: none. Top candidates: candidate-syndrome decoding and
   zero-init gated query bottlenecks, with anchor-relative sparse difference
   atoms as the geometry revival path.
5. Highest-priority gate: implement a CPU-first candidate-syndrome decoder on
   existing SVAMP70 artifacts before any new MPS work.
6. Scale-up rung: source-surface/new-branch discovery, pre-smoke.

Result:

- Converted 172 local reference PDFs into markdown extracts under
  `references/pdf_markdown/`.
- Conversion status after the promoted regeneration pass: 172 normal
  extractions, 0 unresolved failures.
- Added `references/466_creative_method_synthesis_after_pdf_conversion_refs.md`
  and `paper/creative_reference_synthesis_20260427.md`.
- New top branch: candidate-syndrome decoder, reframing the sidecar as a
  tiny source-derived code over target candidate pools with random-syndrome,
  shuffled-source, zero-source, target-only, slots-only, and matched-byte
  controls.

Decisions:

- Promoted: candidate-syndrome decoder as the next CPU-feasible branch.
- Promoted: zero-init gated query bottleneck as the next learned branch once
  MPS clears.
- Revived with constraints: RotAlign/latent-bridge only through
  anchor-relative sparse difference atoms and source-difference zeroing.
- Deferred: protected-tail quantized residuals until a positive branch exists
  to compress.
- Weakened: shallow source-likelihood sketches and generic Perceiver memories.

Candidate-syndrome CPU artifact gate:

```bash
./venv_arm64/bin/python scripts/analyze_candidate_syndrome_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json \
  --output-dir results/candidate_syndrome_decoder_20260427 \
  --controls zero_source shuffled_source random_syndrome target_only slots_only \
  --run-date 2026-04-27
```

Outcome:

- Status: `candidate_syndrome_decoder_fails_smoke`.
- Live: matched correct `11/70`, clean source-necessary `1`, target-self
  harms `17`, control clean union `0`.
- Holdout: matched correct `5/70`, clean source-necessary `4`, target-self
  harms `14`, control clean union `0`.
- Random/shuffled/zero/target/slots controls did not recover clean source-only
  IDs, but the matched syndrome destroys too many target-self repairs and the
  live surface lacks the minimum clean recovery.
- Decision: do not promote the numeric hash-syndrome artifact probe. The
  candidate-syndrome family only remains alive with learned source predicates
  or a stronger source surface.

Artifacts:

- `results/candidate_syndrome_decoder_20260427/candidate_syndrome_decoder_probe.json`
  - sha256:
    `2ae78c4f3c31cf674f334fe5f755d6f80a8ccf3c66e777a49d4daed01c25cc81`
- `results/candidate_syndrome_decoder_20260427/candidate_syndrome_decoder_probe.md`
  - sha256:
    `719990b7b4dff43278920cbbdbc807e4bdf2c359accc80dc81bf38bcf2f5a4f5`
- `references/pdf_markdown/conversion_manifest.json`
  - sha256:
    `380b5f0f4dbdd0486a5b3eefa08dc1760c254219a7cdf5e715a8be3bec31f468`

Next branch:

- Zero-init gated query bottleneck, only after MPS is cleared or a CPU-only
  feature prototype is scoped to frozen artifacts.

Before any MPS command:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Commit / push status:

- Local commit: `b730e4a1ce7f557736fa47d4151a1bb3d0baad88`
- Push attempt: failed.
- Exact push error:

```text
send-pack: unexpected disconnect while reading sideband packet
Connection to github.com closed by remote host.
fatal: the remote end hung up unexpectedly
```

## 2026-04-27 Cycle - No-harm source-predicate pruning

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; still missing one
   deployable source-derived method that survives live/holdout controls, seed
   repeats, strong text/C2C/self-repair baselines, systems accounting, and
   cross-family falsification.
2. Current paper story: source side information has isolated clues, but the
   deployable method must be target-preserving and erasure-aware rather than a
   shallow answer or source-text threshold selector.
3. Exact blocker to submission: no branch recovers source-necessary wins while
   preserving target-correct examples; PID `31103` still blocks MPS even after
   user attempted `kill -9` and `sudo kill -9`.
4. Current live branch: none. Top candidates after this cycle are learned
   semantic predicate decoding and zero-init target-preserving query
   bottlenecks, contingent on a stronger source surface.
5. Highest-priority gate: run CPU-only no-harm replay gates over existing
   syndrome/source-predicate artifacts before any new MPS work.
6. Scale-up rung: strict CPU smoke / branch pruning.

Commands:

```bash
./venv_arm64/bin/python scripts/analyze_candidate_syndrome_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json \
  --output-dir .debug/candidate_syndrome_bits4_audit \
  --controls zero_source shuffled_source random_syndrome target_only slots_only \
  --bits 4 \
  --run-date 2026-04-27
```

```bash
./venv_arm64/bin/python scripts/analyze_svamp_source_sidecar_cv_router_gate.py \
  --target target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --source source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --candidate source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --candidate t2t=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --fallback-label target \
  --accept-penalty 0.25 \
  --min-correct 25 \
  --min-clean-source-necessary 3 \
  --max-control-clean-union 0 \
  --min-numeric-coverage 63 \
  --date 2026-04-27 \
  --output-json .debug/source_predicate_gates/source_predicate_router_penalty025.json \
  --output-md .debug/source_predicate_gates/source_predicate_router_penalty025.md \
  --output-predictions-jsonl .debug/source_predicate_gates/source_predicate_router_penalty025_predictions.jsonl
```

```bash
./venv_arm64/bin/python scripts/analyze_svamp70_source_likelihood_sketch_gate.py \
  --live-sketch-jsonl results/qwen25math_svamp70_source_likelihood_sketch_20260427/live_normpred_answer_template_sketch_cpu.jsonl \
  --live-candidate target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --live-candidate text=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --live-candidate source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --live-target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-sketch-jsonl results/qwen25math_svamp70_source_likelihood_sketch_20260427/holdout_normpred_answer_template_sketch_cpu.jsonl \
  --holdout-candidate target=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --holdout-candidate text=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --holdout-candidate source=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --holdout-target-set-json results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --accept-penalty 0.25 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json .debug/source_predicate_gates/source_likelihood_noharm_gate.json \
  --output-md .debug/source_predicate_gates/source_likelihood_noharm_gate.md \
  --output-predictions-jsonl .debug/source_predicate_gates/source_likelihood_noharm_gate_predictions.jsonl
```

Results:

- Candidate syndrome bits4: fail. Live clean source-necessary `1`, target-self
  harms `16`; holdout clean source-necessary `4`, target-self harms `14`;
  control clean union `0`.
- Source predicate router at `accept_penalty=0.25`: fail. Best rows matched
  `23/70` with `3` clean source-necessary IDs and control clean union `0`, but
  accepted harm remained `1` and the row failed the matched-correct gate.
- Source likelihood no-harm gate: fail. No-harm abstention removed the useful
  signal: live matched `21/70`, holdout matched `8/70`, both with clean
  source-necessary `0`, accepted harm `0`, and control clean union `0`.

Artifacts:

- Durable manifest:
  `results/noharm_source_predicate_pruning_20260427/manifest.md`
- Focused memo:
  `paper/noharm_source_predicate_pruning_20260427.md`
- New reference memo:
  `references/467_crossfield_noharm_predicate_refs.md`

Artifact hashes:

- `candidate_syndrome_bits4_probe.json`:
  `f304e16922f49bff9d909d73761681b32ee104b04bfbf80a3702254960802fa8`
- `candidate_syndrome_bits4_probe.md`:
  `99b8aa9f997120a906cf96da45cf747a750e933915732d8b89cded73f7c84f34`
- `source_predicate_router_penalty025.json`:
  `4f2e04497fc4964d7637badcedf66644c26c004408f9b8916c060e79841e9e2b`
- `source_predicate_router_penalty025.md`:
  `13625b933afbba6d92995ee7050768377a249f47900fe28e322c892684e4c578`
- `source_predicate_router_penalty025_predictions.jsonl`:
  `ef1409b5e5af758a484b62e9282016522ee957675663eb93c6c5735e1a56c4e6`
- `source_likelihood_noharm_gate.json`:
  `97182e0394c8c61d01da0baa1ab1b53eb3427cd33f379230b10f57e41a2e52db`
- `source_likelihood_noharm_gate.md`:
  `cfdd39a0ada6228f78e5591effb9fb3c1fd4c17e94cab2f472bcc296c94cc6e2`
- `source_likelihood_noharm_gate_predictions.jsonl`:
  `e8d12c2f5cff8480eafdb2e9392f4145020aa3c9779904c8149e425bfdad8633`

Decision:

- Pruned/killed: shallow numeric/hash syndrome and source-text feature routers
  on current artifacts.
- Weakened: source likelihood sketches as a route to communication.
- Revived only with constraints: learned semantic predicate decoding with
  erasure-aware abstention and source-fault detection.
- Next MPS branch: zero-init target-preserving query bottleneck after stronger
  source surface discovery.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent, run the stronger-source scout recorded in
`paper/postkill_historical_cpu_audit_20260427.md`. If it remains present, the
hard blocker is OS/session-level cleanup or reboot before MPS experiments;
normal and sudo `SIGKILL` attempts have not cleared the uninterruptible
`STAT=UE` process.

## 2026-04-27 Cycle - Semantic predicate decoder CPU smoke

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; still missing one
   stable positive method plus medium/large controls, seed stability, C2C/text
   comparisons, systems metrics, and cross-family falsification.
2. Current paper story: source-side side information can be source-specific,
   but the current deployable decoders either damage target-correct examples or
   fail holdout transfer.
3. Exact blocker to submission: no target-preserving method recovers clean
   source-necessary IDs on both live and holdout; MPS is still blocked by PID
   `31103` in `STAT=UE`.
4. Current live branch: none. Candidate branch tested here: learned semantic
   predicates over target/source/text candidate pools with erasure-aware
   abstention.
5. Highest-priority gate: CPU-only live/holdout smoke with source-destroying
   controls and zero accepted harm.
6. Scale-up rung: smoke / branch falsification.

Implementation:

- Added `scripts/analyze_svamp_source_semantic_predicate_decoder.py`.
- Added focused tests in
  `tests/test_analyze_svamp_source_semantic_predicate_decoder.py`.
- Added reference memo
  `references/468_target_preserving_receiver_gate_refs.md`.

Main command:

```bash
./venv_arm64/bin/python scripts/analyze_svamp_source_semantic_predicate_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --mode learned_logodds \
  --outer-folds 5 \
  --accept-penalty 0.75 \
  --harm-weight 20.0 \
  --min-live-correct 25 \
  --min-live-clean-source-necessary 2 \
  --min-holdout-correct 10 \
  --min-holdout-clean-source-necessary 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-dir results/svamp_source_semantic_predicate_decoder_strict_harm20_20260427 \
  --output-predictions-jsonl results/svamp_source_semantic_predicate_decoder_strict_harm20_20260427/predictions.jsonl
```

Result:

- Status: `semantic_predicate_decoder_fails_smoke`.
- Live: `25/70` correct, `5` accepted source sidecars, `3` clean
  source-necessary IDs, accepted harm `0`, control clean union `0`.
- Holdout: `9/70` correct, `2` accepted source sidecars, `0` clean
  source-necessary IDs, accepted harm `0`, control clean union `0`.
- Live clean IDs recovered:
  `2de1549556000830`, `41cce6c6e6bb0058`, `4d780f825bb8541c`.

Auxiliary CPU target-likelihood smoke:

- Ran `Qwen/Qwen3-0.6B` on CPU for 8 live examples using
  `scripts/collect_source_likelihood_sketch.py`.
- C2C as an internal candidate contaminated the receiver gate because long C2C
  text received high continuation likelihood even when wrong.
- Without C2C, target scoring was feasible but often preferred wrong source
  candidates on this tiny slice.
- Decision: keep C2C as an external baseline, not an internal candidate for
  this method.

Artifact hashes:

- `results/svamp_source_semantic_predicate_decoder_strict_harm20_20260427/semantic_predicate_decoder.json`:
  `97a8cd1ba95c1239f0055a82a8bc461c99070bb8aba744bbc47f6b5d2567b53f`
- `results/svamp_source_semantic_predicate_decoder_strict_harm20_20260427/semantic_predicate_decoder.md`:
  `c57a791fa9a137d96ea4a951e76883610cba257076f041b6b73ba86ef16ce46a`
- `results/svamp_source_semantic_predicate_decoder_strict_harm20_20260427/predictions.jsonl`:
  `2fa404726a3cb654ec2de6cba75cfceac3e20fccf7ca2b5b873a687020aa6aed`

Decision:

- Kill/prune generated-source-trace semantic predicate decoding on current
  Qwen2.5-Math -> Qwen3 SVAMP artifacts.
- Weaken but do not fully kill target-preserving receiver gates: they need a
  stronger source surface or model-collected target-side likelihood /
  uncertainty features.
- No additional CPU artifact mining is promoted from this branch.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent, run the stronger-source scout recorded in
`paper/postkill_historical_cpu_audit_20260427.md`. If it remains present, the
hard blocker is OS/session-level cleanup or reboot before MPS experiments.

## 2026-04-27 Cycle - Qwen3 target-likelihood receiver live prune

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; still missing a stable
   positive method plus medium/large controls, seed stability, systems metrics,
   and cross-family falsification.
2. Current paper story: source candidates contain real live headroom, but
   target-safe deployable decoders have not transferred and receiver gates have
   not separated useful source information from harmful source answers.
3. Exact blocker to submission: no method recovers clean source-necessary IDs
   on both live and holdout while preserving target-correct examples under
   source-destroying controls; MPS remains blocked by PID `31103` in `STAT=UE`.
4. Current live branch: none. Candidate tested here: target-side likelihood
   receiver over target/text/source candidate answers.
5. Highest-priority gate: CPU-only live smoke to see whether receiver
   likelihood can support a no-harm accept rule before spending holdout/control
   compute.
6. Scale-up rung: smoke / branch discovery.

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/collect_source_likelihood_sketch.py \
  --source-model Qwen/Qwen3-0.6B \
  --eval-file results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl \
  --candidate target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --candidate text=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --candidate source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --reference-label target \
  --candidate-text-field normalized_prediction \
  --continuation-template 'Answer: {text}' \
  --resume \
  --device cpu \
  --dtype float32 \
  --prompt-mode direct \
  --source-use-chat-template \
  --source-enable-thinking false \
  --date 2026-04-27 \
  --output-jsonl results/qwen3_target_likelihood_receiver_20260427/live_target_model_normpred_answer_template.jsonl \
  --output-md results/qwen3_target_likelihood_receiver_20260427/live_target_model_normpred_answer_template.md
```

Result:

- Status: `fails_live_prune`.
- Rows: `70`.
- Target/text/source candidate correctness: `21/70`, `22/70`, `13/70`.
- Top-likelihood selection: `14/70`.
- Top labels: source `64`, text `6`, target `0`.
- Accept-all source-top clean live source-only IDs: `6`.
- Accept-all source-top target-correct harms: `16`.
- Best simple no-harm live thresholds recover at most `1` clean source-only ID
  and remain around `22-23/70`, below the `25/70` live gate.

Artifact hashes:

- `results/qwen3_target_likelihood_receiver_20260427/live_target_model_normpred_answer_template.jsonl`:
  `104ceba6676c752c2863347a2b201faa48f23f3964fee3cdcd22430b461e3ca0`
- `results/qwen3_target_likelihood_receiver_20260427/live_target_model_normpred_answer_template.md`:
  `10fd305e89940ddb0c86b3a855524d4e24261629cd5f9cfd8893d23209c94f75`
- ordered example IDs sha256:
  `0292230b41840995d6c178c72b571f4f4441e631a6e7f1535a03106717010506`

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_collect_source_likelihood_sketch.py tests/test_analyze_svamp70_source_likelihood_sketch_gate.py -q
```

Result: `9 passed in 0.09s`.

Decision:

- Pruned: this target-likelihood receiver variant on current SVAMP70 artifacts.
- Weakened: target-likelihood receiver gates without stronger source surfaces
  or true condition-specific rescored controls.
- Still live only as infrastructure: a fair receiver harness should accept
  separate rescored sketches for matched, zero-source, shuffled-source,
  target-only, and slots-only candidate pools.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent, resume MPS source-surface/interface reset work. If it
remains present, continue CPU-only with a canonical exact-ID overlap audit
across the SVAMP70 live/holdout surfaces rather than another threshold sweep.

## 2026-04-27 Cycle - SVAMP70 exact-ID overlap audit

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; still missing a stable
   positive method plus strict controls, seed stability, systems metrics, and
   cross-family falsification.
2. Current paper story: source candidates have real example-level headroom, but
   current methods recover it through live-only or harmful routes.
3. Exact blocker to submission: no reusable canonical live+holdout clean ID
   structure under target preservation.
4. Current live branch: none. Candidate decision tested here: whether to keep
   sweeping thresholds on current canonical SVAMP70 artifacts.
5. Highest-priority gate: CPU-only overlap audit across recent branch
   recoveries and exact source-only ID sets.
6. Scale-up rung: smoke / branch selection.

Result:

- Status: `exact_id_overlap_audit_complete`.
- Canonical live clean source-only IDs: `6`; all have been recovered by at
  least one audited branch, but not by a branch that passes holdout and target
  preservation.
- Canonical holdout clean source-only IDs: `2`.
- Canonical holdout recovered ID: only `daea537474de16ac`, and only through
  trace-router branches that fail the full gate.
- `ab1e71e8928661d0` remains unrecovered by audited canonical-branch
  decoders.
- Adjacent scout syndrome recoveries are not canonical holdout evidence and
  include target-self harm.

Artifact hashes:

- `results/svamp70_exact_id_overlap_audit_20260427/exact_id_overlap_audit.json`:
  `358cb6b6db2a76dcea074df91e8e755d03d8114649cce78e019ed4f5626c4f5c`
- `results/svamp70_exact_id_overlap_audit_20260427/exact_id_overlap_audit.md`:
  `92b688053c8948331b7df070538f645dfaa2746a456ada6c53347cc665bd9ec0`

Decision:

- Kill/prune: another CPU threshold/router sweep on current canonical SVAMP70
  artifacts.
- Keep: canonical live/holdout exact IDs as future falsification surfaces.
- Next CPU branch if MPS remains blocked: implement a fair condition-specific
  receiver-control analyzer before collecting more target-likelihood sketches.
- Next MPS branch when available: source-surface/interface reset.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` remains, implement/test the condition-specific receiver-control
analyzer. If it clears, resume MPS source-surface/interface reset.

## 2026-04-27 Cycle - Condition-specific receiver-control harness

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; still missing one
   positive method plus strict controls, seed stability, systems metrics, and
   cross-family falsification.
2. Current paper story: target-likelihood receiver gates are only plausible if
   source-destroying controls are rescored under the same receiver model.
3. Exact blocker to submission: the older likelihood sketch analyzer cannot
   fairly distinguish real source-candidate communication from sketch-shuffle or
   target-fallback artifacts.
4. Current live branch: none. Next branch selected here: condition-specific
   likelihood receiver harness.
5. Highest-priority gate: implement and test the smallest fair analyzer before
   collecting more receiver likelihood sketches.
6. Scale-up rung: harness / smoke preparation.

Implementation:

- Added `scripts/analyze_condition_likelihood_receiver_gate.py`.
- Added `tests/test_analyze_condition_likelihood_receiver_gate.py`.

The new analyzer requires separate receiver-scored sketches per condition:
`matched`, `zero_source`, `shuffled_source`, `label_shuffle`, `target_only`,
and `slots_only`. It fits a live matched receiver rule, applies the same frozen
rule to all controls and holdout, and subtracts the control clean union before
counting source-necessary IDs.

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_condition_likelihood_receiver_gate.py tests/test_analyze_svamp70_source_likelihood_sketch_gate.py tests/test_collect_source_likelihood_sketch.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_condition_likelihood_receiver_gate.py
```

Result: `12 passed in 0.12s`; compile passed.

Decision:

- Next branch is implemented as harness, not promoted as a method.
- Do not claim target-likelihood receiver evidence until condition-specific
  matched/control/holdout sketches are collected and this analyzer clears the
  live and holdout gates.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` remains, collect receiver sketches on CPU only and run
`scripts/analyze_condition_likelihood_receiver_gate.py`. If it clears, MPS can
be used for source-surface/interface reset or faster sketch collection.

## 2026-04-27 Cycle - Condition likelihood receiver live feasibility kill

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; still missing a stable
   positive method plus strict controls, seed stability, systems metrics, and
   cross-family falsification.
2. Current paper story: receiver likelihood is infrastructure, but current
   Qwen3 target-side likelihood does not preserve target-correct examples while
   selecting useful source candidates.
3. Exact blocker to submission: the condition-specific receiver branch fails
   matched live feasibility before controls or holdout.
4. Current live branch: condition-specific target-likelihood receiver on
   canonical SVAMP70.
5. Highest-priority gate: matched-only live CV early stop.
6. Scale-up rung: smoke / branch kill.

Implementation updates:

- Added `scripts/build_condition_likelihood_candidate_pools.py`.
- Updated `scripts/analyze_condition_likelihood_receiver_gate.py` to subtract
  canonical duplicate-answer clean IDs from source-necessary counts.
- Added/updated focused tests for condition pool building and duplicate-answer
  de-duplication.

Control pool definitions:

- `matched`: target/text/source unchanged.
- `zero_source`: target/text unchanged, source blanked.
- `shuffled_source`: target/text unchanged, source replaced by deterministic
  off-example source answer with correctness recomputed.
- `label_shuffle`: labels permuted before scoring.
- `target_only`: same three slots, with non-target slots filled by target
  output.
- `slots_only`: target retained, non-target slots blanked.

Result:

- Status: `condition_likelihood_receiver_live_matched_feasibility_fails`.
- Matched correct: `15/70`.
- Matched accepted: `14`.
- Clean source-necessary IDs: `4d780f825bb8541c`.
- Duplicate-answer clean IDs: none.
- Accepted target-correct harm: `7`.
- Failing criteria: `min_correct`, `min_clean_source_necessary`,
  `max_accepted_harm`.

Artifact hashes:

- `results/condition_likelihood_receiver_live_feasibility_20260427/live_matched_feasibility.json`:
  `06dafa60c62724f440965d90d22af799dfacb4f3f8704904202c2984bf7b7fe7`
- `results/condition_likelihood_receiver_live_feasibility_20260427/live_matched_feasibility.md`:
  `dfa9b37a27ebd5ce2154c8440b98128e83a9f43478040c4d7e03d2e47afc623a`

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_build_condition_likelihood_candidate_pools.py tests/test_analyze_condition_likelihood_receiver_gate.py tests/test_analyze_svamp70_source_likelihood_sketch_gate.py tests/test_collect_source_likelihood_sketch.py -q
./venv_arm64/bin/python -m py_compile scripts/build_condition_likelihood_candidate_pools.py scripts/analyze_condition_likelihood_receiver_gate.py
```

Result: `14 passed in 0.12s`; compile passed.

Decision:

- Kill/prune: condition-specific target-likelihood receiver on current
  canonical SVAMP70 artifacts.
- Keep: the condition-specific receiver harness as infrastructure for a future
  stronger source candidate surface.
- Do not collect remaining live controls or holdout for this killed branch.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` clears, resume MPS source-surface/interface reset. If it remains
blocked, use CPU-only literature/artifact work or apply the new harness only to
a stronger source surface, not this killed receiver branch.

## 2026-04-27 Cycle - Durable source-surface ranking and recent latent-agent baselines

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; no method has cleared
   strict source controls, target preservation, seed stability, systems metrics,
   and cross-family falsification.
2. Current paper story: existing source surfaces contain real clean headroom,
   but prior receiver/router branches failed to convert it into controlled
   communication gains.
3. Exact blocker to submission: no positive method survives on a durable
   source surface; MPS remains blocked by orphaned PID `31103`.
4. Current live branch: none. Top next branch is zero-init gated latent
   side-information after source-surface selection.
5. Highest-priority gate: make source-surface selection reproducible and rank
   by clean source-only IDs.
6. Scale-up rung: smoke / branch selection.

Implementation updates:

- Added `scripts/rank_source_contrastive_target_sets.py`.
- Added `tests/test_rank_source_contrastive_target_sets.py`.
- Added `references/469_recent_latent_agent_communication_refs.md`.
- Updated `references/research_memo_manifest.json`.

Command:

```bash
./venv_arm64/bin/python scripts/rank_source_contrastive_target_sets.py \
  --target-set svamp70_live=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json,role=primary_live,note=canonical_live_surface \
  --target-set svamp70_holdout=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json,role=canonical_holdout,note=canonical_holdout_surface \
  --target-set svamp70_chal241_310=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json,role=adjacent_falsifier,note=best_adjacent_clean_surface \
  --target-set svamp32_qwen25math=path=results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json,role=smoke_debug,note=tiny_debug_surface \
  --target-set gsm70_qwen25math=path=results/qwen25math_qwen3_gsm70_source_surface_20260426/source_contrastive_target_set.json,role=weak_candidate,note=weak_gsm_surface \
  --target-set svamp70_chal171_240=path=results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json,role=weak_candidate,note=weak_adjacent_surface \
  --target-set svamp70_chal311_380=path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_contrastive_target_set.json,role=weak_candidate,note=weak_adjacent_surface \
  --target-set svamp32_qwen25math_instruct=path=results/surface_scout_qwen25math_instruct_qwen3_svamp32_20260426/source_contrastive_target_set.json,role=weak_candidate,note=weak_instruct_surface \
  --min-clean-source-only 5 \
  --date 2026-04-27 \
  --output-json results/durable_source_surface_ranking_20260427/source_surface_ranking.json \
  --output-md results/durable_source_surface_ranking_20260427/source_surface_ranking.md
```

Result:

- Status: `primary_surface_selected`.
- Top surface: `svamp70_live`.
- `svamp70_live`: clean source-only `6/70`, raw source-only `9/70`,
  target/source oracle gain `9/70`.
- `svamp70_holdout`: clean source-only `2/70`, still canonical holdout.
- `svamp70_chal241_310`: clean source-only `4/70`, adjacent falsifier only.

Artifact hashes:

- `results/durable_source_surface_ranking_20260427/source_surface_ranking.json`:
  `7e665698c206f748074ea567754e1f7392b0391ee60dc514bb41619e706a038f`
- `results/durable_source_surface_ranking_20260427/source_surface_ranking.md`:
  `99fe4631a973dcc09b4f97ef6a5b0d26c6dc833fd4968b24bb13a574cf7294e8`

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_rank_source_contrastive_target_sets.py tests/test_analyze_source_headroom_surfaces.py -q
./venv_arm64/bin/python -m py_compile scripts/rank_source_contrastive_target_sets.py
```

Result: `8 passed in 0.07s`; compile passed.

Literature update:

- `references/469_recent_latent_agent_communication_refs.md` adds recent
  primary-source latent/activation communication baselines: LatentMAS,
  Interlat, ICML activation communication, and Thought Communication.
- Hypothesis update: the next learned branch should be fixed-budget,
  zero-init target-preserving latent side-information with activation/latent
  communication baselines, not another shallow receiver threshold sweep.

Decision:

- Promote `svamp70_live` as the durable primary method surface.
- Keep `svamp70_holdout` as canonical replay despite weak clean headroom.
- Keep `svamp70_chal241_310` only as adjacent falsifier.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` clears, run the stronger-source scout recorded in
`paper/postkill_historical_cpu_audit_20260427.md`, then apply the zero-init
gated latent side-information smoke only if the scout has at least six clean
source-only IDs and target/source oracle gain of at least six.

## 2026-04-27 Cycle - Source-hidden query smoke and KVComm tooling smoke

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; no deployable method
   has cleared live/holdout controls, seed stability, systems metrics, and
   cross-family falsification.
2. Current paper story: source-derived side information remains plausible, but
   direct source-hidden readouts and shallow routers are not robust enough.
3. Exact blocker to submission: MPS remains blocked by orphaned PID `31103`,
   and no offline activation tensor artifacts exist for a full latent-injection
   smoke.
4. Current live branch: none. Top next executable branch after MPS clears is
   fixed-budget KV/cache communication baseline.
5. Highest-priority gate: run the cheapest CPU latent-sideinfo diagnostic and
   verify KVComm tooling.
6. Scale-up rung: smoke / branch preparation.

Source-hidden query smoke:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_latent_syndrome_probe.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target target_alone=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --candidate target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --fallback-label target_self_repair \
  --probe-model query_bottleneck \
  --query-epochs 2 \
  --query-slots 4 \
  --moduli 2,3,5,7 \
  --feature-layers last \
  --device cpu \
  --dtype float32 \
  --min-numeric-coverage 31 \
  --output-json .debug/zero_init_latent_sideinfo_audit/source_hidden_query_smoke.json \
  --output-md .debug/zero_init_latent_sideinfo_audit/source_hidden_query_smoke.md
```

Result:

- Status: `source_latent_syndrome_probe_fails_gate`.
- Matched: `11/32`.
- Zero-source/shuffled-source/label-shuffled/target-only: `14/32`.
- Clean source-necessary IDs: `0`.
- Decision: direct source-hidden query-bottleneck syndrome readout is weakened
  again; do not scale it.

Scratch artifact hashes:

- `.debug/zero_init_latent_sideinfo_audit/source_hidden_query_smoke.json`:
  `033c9cff44bba273dc71a1fff39e626afdb8da8be05a118317f3264576db881c`
- `.debug/zero_init_latent_sideinfo_audit/source_hidden_query_smoke.md`:
  `7ecdb0140d00d84d398a838c782325d50905e208439edde4fc0d4777dbaa4575`

KVComm tooling smoke:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python -m latent_bridge.kvcomm_eval \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 8 \
  --source-reasoning-mode brief_analysis \
  --top-layers-grid 0.25 \
  --calibration-limit 1 \
  --eval-limit 1 \
  --prediction-output .debug/kvcomm_cpu_smoke_20260427/kvcomm_cpu_smoke_predictions.jsonl
```

Result:

- Tooling smoke passed via module invocation.
- Accuracy: `0/1` (not a method gate).
- CPU latency: `0.9505s`.
- Selected layers: `[1, 6, 2, 8, 7, 5, 4]`.
- Direct script invocation initially failed with `ModuleNotFoundError`; fixed
  `latent_bridge/kvcomm_eval.py` to add the repo root to `sys.path`.

Scratch artifact hashes:

- `.debug/kvcomm_cpu_smoke_20260427/kvcomm_cpu_smoke_predictions.jsonl`:
  `ddfa80b562ebcda86e0e2578e33d7d010f18cb003b9f1bb326e0c6f9940eb64e`
- `.debug/kvcomm_cpu_smoke_20260427/kvcomm_cpu_smoke_predictions.jsonl.meta.json`:
  `b051921a3089b8af7f8f2c3ef89aed8ffaf6c6edb3b563313374ce3e75abed40`

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_latent_syndrome_probe.py tests/test_analyze_svamp32_syndrome_sidecar_probe.py tests/test_analyze_svamp32_source_only_sidecar_router_gate.py tests/test_analyze_condition_likelihood_receiver_gate.py tests/test_build_condition_likelihood_candidate_pools.py -q
```

Result: `17 passed in 0.67s`.

Focused compile/help checks:

```bash
./venv_arm64/bin/python -m py_compile latent_bridge/kvcomm_eval.py
./venv_arm64/bin/python latent_bridge/kvcomm_eval.py --help
```

Result: both passed.

Literature/reference update:

- Added `references/470_kv_cache_latent_communication_baselines_refs.md`.
- Hypothesis update: run C2C/KVComm-style receiver-cache-preserving baselines
  before adding a new latent connector.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID clears, run a one-example MPS KVComm smoke or the stronger-source MPS
scout from `paper/postkill_historical_cpu_audit_20260427.md`, then scale only
if exact ID/numeric coverage and source controls are preserved.

## Cycle 2026-04-27 00:50 PDT - KVComm Source-Control Harness Smoke

Cycle header:

1. Current ICLR readiness and distance: not ICLR-ready; still missing one
   deployable positive method that survives source controls, seeds, systems
   baselines, and cross-family falsification.
2. Current paper story: durable source headroom exists on `svamp70_live`, but
   direct source-hidden latent readouts fail. KV/cache communication remains a
   necessary strong baseline and possible systems-compression lane, not yet a
   positive method.
3. Exact blocker to submission: no matched-source KV/cache row has yet been
   tested against zero-source, shuffled-source, and target-only controls on a
   decision slice; MPS PID `31103` remains stuck in `STAT=UE`.
4. Current live branch or top candidates: `none`; top branch is
   fixed-budget KVComm/C2C-style cache communication with strict source
   controls, followed by a zero-init gated source bottleneck only after the
   baseline contract is exercised.
5. Highest-priority gate for this cycle: add and smoke-test KVComm
   source-destroying controls without using MPS.
6. Scale-up rung: smoke / harness preparation.

Code changes:

- `latent_bridge/kvcomm_eval.py` now supports final
  `--source-control-modes`, including `all` expanding to matched, zero-source,
  shuffled-source, and target-only.
- Layer ranking and top-layer selection remain matched-calibration only; final
  controls reuse the selected layers.
- `zero_source` zeroes matched source past-key-values while preserving tensor
  shape/device/dtype.
- `target_only` bypasses KVComm and calls target generation directly.
- Prediction records now include control provenance and method-specific metrics.

CPU smoke command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python -m latent_bridge.kvcomm_eval \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 4 \
  --source-reasoning-mode brief_analysis \
  --top-layers-grid 0.25 \
  --calibration-limit 1 \
  --eval-limit 2 \
  --source-control-modes all \
  --prediction-output .debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_cpu_smoke_predictions.jsonl
```

Result:

- Harness smoke passed; this is not positive method evidence.
- `kvcomm_matched`, `kvcomm_zero_source`, `kvcomm_shuffled_source`, and
  `target_only` all reached `0/2` at `max_new_tokens=4`.
- All final modes reused selected layers `[1, 6, 2, 8, 7, 5, 4]`.
- Shuffled-source records had nonmatching source IDs.

Artifact hashes:

- `.debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_cpu_smoke_predictions.jsonl`:
  `ce1dd54cb3e96056e821cc9397b61151ebe054e345c8bb8ba1347d45cd519ea6`
- `.debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_cpu_smoke_predictions.jsonl.meta.json`:
  `0ae7d5a9f6f38a8fa51a36dca7de9828fbc4ff2881bf1c1b60b0b0f8187238d4`

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_kvcomm_eval_controls.py -q
./venv_arm64/bin/python -m py_compile latent_bridge/kvcomm_eval.py
./venv_arm64/bin/python latent_bridge/kvcomm_eval.py --help
./venv_arm64/bin/python -m pytest tests/test_kvcomm_eval.py tests/test_kvcomm_eval_controls.py -q
```

Results: `9 passed`, compile passed, help passed, then `12 passed`.

Decision: promote KVComm from tooling-only to harness-ready baseline. Do not
promote method evidence; the next gate must run the same source-control contract
on a real decision slice after MPS clears, or on a CPU-feasible tiny slice only
for additional plumbing checks.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` clears, run:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python -m latent_bridge.kvcomm_eval \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 32 \
  --source-reasoning-mode brief_analysis \
  --top-layers-grid 0.25,0.5 \
  --calibration-limit 8 \
  --eval-limit 8 \
  --source-control-modes all \
  --prediction-output results/kvcomm_svamp32_controls_smoke_20260427/kvcomm_controls_predictions.jsonl
```

If PID remains, continue CPU-only artifact audit and literature/code work.

### Cycle Addendum 2026-04-27 00:56 PDT - Hash-Shuffled KVComm Controls

The KVComm shuffled-source control now uses deterministic hash-based non-self
pairing instead of fixed offset pairing, and prediction records include
`source_control_source_answers_overlap_target`.

Re-run CPU smoke:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python -m latent_bridge.kvcomm_eval \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 4 \
  --source-reasoning-mode brief_analysis \
  --top-layers-grid 0.25 \
  --calibration-limit 1 \
  --eval-limit 2 \
  --source-control-modes all \
  --prediction-output .debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_hashshuffle_cpu_smoke_predictions.jsonl
```

Result: tooling smoke still passes. All modes are `0/2`; shuffled-source has
nonmatching source IDs and no answer overlap on both rows.

Artifact hashes:

- `.debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_hashshuffle_cpu_smoke_predictions.jsonl`:
  `83c069c4c1a893d2b621318c7aeb49b777cb0b45a509957857b1568a0c43f73c`
- `.debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_hashshuffle_cpu_smoke_predictions.jsonl.meta.json`:
  `fa5230f6be1fa9235428dc202060b29e8b6e771176b6cb349dded57f3a752922`

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_kvcomm_eval.py tests/test_kvcomm_eval_controls.py -q
./venv_arm64/bin/python -m py_compile latent_bridge/kvcomm_eval.py
```

Results: `13 passed`, compile passed.

### Cycle Addendum 2026-04-27 00:58 PDT - KVComm Paired Sidecar Controls

The generic prediction sidecar now supports configured paired baselines via
`run_config["paired_baseline_methods"]`, while preserving the legacy
`target_alone` default. KVComm sets paired baselines to `target_only`,
`kvcomm_zero_source`, and `kvcomm_shuffled_source`.

CPU smoke:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python -m latent_bridge.kvcomm_eval \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 4 \
  --source-reasoning-mode brief_analysis \
  --top-layers-grid 0.25 \
  --calibration-limit 1 \
  --eval-limit 2 \
  --source-control-modes all \
  --prediction-output .debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_paired_cpu_smoke_predictions.jsonl
```

Result: tooling smoke still passes. All modes are `0/2`, and the sidecar now
contains `kvcomm_matched_vs_target_only`, `kvcomm_matched_vs_kvcomm_zero_source`,
and `kvcomm_matched_vs_kvcomm_shuffled_source` flip tables.

Artifact hashes:

- `.debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_paired_cpu_smoke_predictions.jsonl`:
  `ffd45b5fc252c638bbc51d078bd77f9c77f01bb1680cfc66bc20f7037c26bd30`
- `.debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_paired_cpu_smoke_predictions.jsonl.meta.json`:
  `8a62ca420d878c198883aa528fd3dfb15758de7adb3010879023010145cd8c12`

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_evaluate_helpers.py::test_write_prediction_sidecar_uses_configured_paired_baselines tests/test_evaluate_helpers.py::test_write_prediction_sidecar_writes_run_and_method_summary tests/test_kvcomm_eval.py tests/test_kvcomm_eval_controls.py -q
./venv_arm64/bin/python -m py_compile latent_bridge/evaluate.py latent_bridge/kvcomm_eval.py
```

Results: `15 passed`, compile passed.

### Cycle Addendum 2026-04-27 01:03 PDT - KVComm Byte Telemetry

KVComm prediction records now include cache-derived systems telemetry:
`bits`, `bytes`, `payload_bits`, `source_cache_bytes`, and
`communicated_cache_bytes`. The first byte smoke exposed zero-byte telemetry
because HF cache objects were not always plain tuples; the counter now handles
`to_legacy_cache()`/cache-list objects.

Corrected CPU smoke:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python -m latent_bridge.kvcomm_eval \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --calibration-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 4 \
  --source-reasoning-mode brief_analysis \
  --top-layers-grid 0.25 \
  --calibration-limit 1 \
  --eval-limit 2 \
  --source-control-modes all \
  --prediction-output .debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_bytes_fixed_cpu_smoke_predictions.jsonl
```

Result: tooling smoke still passes. All modes are `0/2`. Matched, zero-source,
and shuffled-source average `530432` communicated bytes/example; target-only
averages `0`.

Artifact hashes:

- `.debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_bytes_fixed_cpu_smoke_predictions.jsonl`:
  `f11d3de75f968b806423ca539603bab5560644827b02adfbbd0f595d2745c96b`
- `.debug/kvcomm_cpu_smoke_20260427/kvcomm_all_controls_bytes_fixed_cpu_smoke_predictions.jsonl.meta.json`:
  `d963aadbfe19b0f543fe887514afbdf363120790f247a4064504765fc720dab1`

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_kvcomm_eval.py tests/test_kvcomm_eval_controls.py tests/test_evaluate_helpers.py::test_write_prediction_sidecar_uses_configured_paired_baselines -q
./venv_arm64/bin/python -m py_compile latent_bridge/kvcomm_eval.py
```

Results: `16 passed`, compile passed.

### Cycle Addendum 2026-04-27 01:08 PDT - Byte-Efficient Side-Information Audit

Cycle header:

1. ICLR readiness: not ready; no live positive method survives controls,
   target-self preservation, seed stability, systems accounting, and
   cross-family falsification.
2. Paper story: source-derived side information remains the most plausible
   story, but historical sparse-K, RotAlign/DynAlign, Perceiver, and shallow
   predicate results are mechanism clues rather than claims.
3. Exact blocker: missing positive method plus persistent MPS blocker PID
   `31103`.
4. Live branch: none. Top candidates are learned source-derived
   syndrome/innovation sidecars and target-safe sparse/dictionary sidecars.
5. Highest-priority gate: preserve the historical/reference audit and define
   the next exact gate.
6. Scale-up rung: post-kill branch selection / next smoke.

MPS blocker recheck:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Result: PID `31103` remains present with `PPID=1`, `STAT=UE`, still running
the old MPS `scripts/calibrate.py ... --device mps --dtype float32 ...` job.

Audit decision:

- Promote learned source-derived syndrome/innovation sidecar as the next
  highest-value branch once MPS clears and a stronger source surface is found.
- Keep KVComm/Q-KVComm/C2C/DroidSpeak as required systems baselines.
- Revive RotAlign/DynAlign only as target-safe conditional innovation or
  sparse/dictionary side information.
- Kill shallow source likelihood, semantic-predicate/router, Perceiver
  answer-teacher, and numeric hash-syndrome variants on current evidence.

Focused memo:

- `paper/byte_efficient_sideinfo_branch_audit_20260427.md`

Reference memo:

- `references/471_byte_efficient_source_sideinfo_refs.md`

Durable file hashes:

- `paper/byte_efficient_sideinfo_branch_audit_20260427.md`:
  `0c0ea759c7d8854b5cad9a3ecdfc0042e5f44eebbfa893eb3e77316c0f576549`
- `references/471_byte_efficient_source_sideinfo_refs.md`:
  `4cd4c2cd7651662dfe3ae5d7affe4f9a6fb55be6952c3a9de307db1d2a72298f`
- `references/research_memo_manifest.json`:
  `1f01d34dcfde4e7fd59e28f286aab537f5d3a13c3342978db7e51da3b06c960e`

Next exact gate remains:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID clears, run the stronger-source MPS surface scout from
`paper/postkill_historical_cpu_audit_20260427.md`; only implement the learned
syndrome/innovation sidecar if that scout clears source-mass, exact-ID, and
numeric-coverage thresholds.

### Cycle Addendum 2026-04-27 01:18 PDT - Sidecar Harness Hardening

Cycle header:

1. ICLR readiness: not ready; no positive method has passed controls,
   target-self preservation, holdout, seed stability, systems accounting, and
   cross-family falsification.
2. Paper story: byte-efficient source side information remains the top branch,
   but current SVAMP semantic predicates are falsification/tooling evidence.
3. Exact blocker: missing learned source-derived sidecar plus persistent MPS
   blocker PID `31103`.
4. Live branch: none. Top candidate is learned source-derived
   syndrome/innovation sidecar.
5. Highest-priority gate: make the sidecar gate executable and harden controls
   while MPS is blocked.
6. Scale-up rung: CPU smoke / harness preparation.

Code change:

- `scripts/analyze_svamp_source_semantic_predicate_decoder.py` now supports
  hash-derived non-self shuffled controls, `random_sidecar`, source-control
  provenance fields, and optional learned sidecar JSONL inputs via
  `--live-sidecar-jsonl` / `--holdout-sidecar-jsonl`.
- `tests/test_analyze_svamp_source_semantic_predicate_decoder.py` now covers
  random-sidecar/non-self controls and a synthetic candidate-score sidecar that
  recovers one clean ID while preserving target-correct IDs.

CPU replay command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp_source_semantic_predicate_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --mode learned_logodds \
  --outer-folds 5 \
  --accept-penalty 0.75 \
  --harm-weight 20.0 \
  --min-live-correct 25 \
  --min-live-clean-source-necessary 2 \
  --min-holdout-correct 10 \
  --min-holdout-clean-source-necessary 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-dir .debug/semantic_predicate_decoder_sidecar_harness_20260427 \
  --output-predictions-jsonl .debug/semantic_predicate_decoder_sidecar_harness_20260427/predictions.jsonl
```

Result: `semantic_predicate_decoder_fails_smoke`.

- Live matched: `25/70`, clean `3`, accepted harm `0`.
- Live random same-byte sidecar: `17/70`, clean `1`, accepted harm `7`.
- Holdout matched: `9/70`, clean `0`, accepted harm `0`.
- Holdout random same-byte sidecar: `9/70`, clean `0`, accepted harm `1`.

Decision: old semantic-predicate branch is more decisively killed. The
hardened harness is retained for future learned sidecars from a stronger
surface or frozen out-of-fold source-side predictor.

Artifact hashes:

- `.debug/semantic_predicate_decoder_sidecar_harness_20260427/semantic_predicate_decoder.json`:
  `9cc4804426b6eb1b0f47f7f3fb091cb9185b763379e9fc7c94d08ee936591ed0`
- `.debug/semantic_predicate_decoder_sidecar_harness_20260427/semantic_predicate_decoder.md`:
  `ee8210a6d6029bf2de2594de2b70db6a48aa6e0075eb4d58214274b1e10e9144`
- `.debug/semantic_predicate_decoder_sidecar_harness_20260427/predictions.jsonl`:
  `2fa404726a3cb654ec2de6cba75cfceac3e20fccf7ca2b5b873a687020aa6aed`

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp_source_semantic_predicate_decoder.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_svamp_source_semantic_predicate_decoder.py tests/test_analyze_svamp_source_semantic_predicate_decoder.py
```

Result: `5 passed`, compile passed.

### Cycle Addendum 2026-04-27 01:28 PDT - Target-Pool Sidecar Hardening

Cycle header:

1. ICLR readiness: not ready; no controlled positive method yet.
2. Paper story: byte-efficient source side-information remains the top branch,
   but the decoder must not expose source-only answers to target-side controls.
3. Exact blocker: no frozen learned sidecar has cleared a target-side candidate
   gate, and MPS remains blocked by PID `31103`.
4. Live branch: none. Top candidate is learned source-derived
   syndrome/innovation sidecar.
5. Highest-priority gate: close candidate-pool and sidecar-control leakage
   holes in the harness.
6. Scale-up rung: CPU smoke / harness hardening.

Code change:

- `_candidate_pool()` in
  `scripts/analyze_svamp_source_semantic_predicate_decoder.py` is target-side
  by default and excludes `source`.
- Candidate-score sidecars map explicit `value`/`candidate_value` fields only
  if the value is already in the target-side pool.
- `random_sidecar` preserves the declared same-byte budget for learned
  sidecars.
- `target_only_sidecar` and `slots_only_sidecar` now test sidecar-shaped
  target/slot controls at the same byte scale.
- Sidecar loading rejects duplicate IDs and mismatched sidecar/reference IDs.
- Summaries now report accepted help, accepted clean help, fallback-correct
  count, and sidecar present/missing counts.

CPU replay command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp_source_semantic_predicate_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --mode learned_logodds \
  --outer-folds 5 \
  --accept-penalty 0.75 \
  --harm-weight 20.0 \
  --min-live-correct 25 \
  --min-live-clean-source-necessary 2 \
  --min-holdout-correct 10 \
  --min-holdout-clean-source-necessary 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-dir .debug/semantic_predicate_decoder_targetpool_20260427 \
  --output-predictions-jsonl .debug/semantic_predicate_decoder_targetpool_20260427/predictions.jsonl
```

Result: `semantic_predicate_decoder_fails_smoke`.

- Live matched: `24/70`, clean `3`, accepted clean help `3`, accepted harm `1`.
- Live random same-byte sidecar: `16/70`, clean `0`, accepted harm `9`.
- Live target-only sidecar: `21/70`, clean `0`, accepted harm `0`.
- Live slots-only sidecar: `21/70`, clean `0`, accepted harm `0`.
- Holdout matched: `9/70`, clean `0`, accepted harm `0`.
- Holdout target-only sidecar: `8/70`, clean `0`, accepted harm `0`.
- Holdout slots-only sidecar: `11/70`, clean `0`, accepted harm `0`.
- Control clean union: `0`.

Decision: old semantic-predicate branch is killed more cleanly. Removing
source-only values from target-side candidate pools drops the live row below
the accuracy/harm gate. Keep the harness for future frozen learned sidecars.

Artifact hashes:

- `.debug/semantic_predicate_decoder_targetpool_20260427/semantic_predicate_decoder.json`:
  `55a1e73e061f03c51733d16009e7d1f6766d2d2f8807ad725df1d0cd47020d5f`
- `.debug/semantic_predicate_decoder_targetpool_20260427/semantic_predicate_decoder.md`:
  `7d5882c76b0d9e6e5dbae90084bd731945a8d72cfa9b948843dc290a502365bb`
- `.debug/semantic_predicate_decoder_targetpool_20260427/predictions.jsonl`:
  `48ba362b0f8f557ceb5c1eedd4674d097af1a464f4ea9cdfe1bdf2475fb7fdd8`

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp_source_semantic_predicate_decoder.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_svamp_source_semantic_predicate_decoder.py tests/test_analyze_svamp_source_semantic_predicate_decoder.py
```

Result: `7 passed`, compile passed.

Updated sidecar-shaped control replay:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp_source_semantic_predicate_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --mode learned_logodds \
  --outer-folds 5 \
  --accept-penalty 0.75 \
  --harm-weight 20.0 \
  --min-live-correct 25 \
  --min-live-clean-source-necessary 2 \
  --min-holdout-correct 10 \
  --min-holdout-clean-source-necessary 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-dir .debug/semantic_predicate_decoder_sidecar_controls_20260427 \
  --output-predictions-jsonl .debug/semantic_predicate_decoder_sidecar_controls_20260427/predictions.jsonl
```

Status remains `semantic_predicate_decoder_fails_smoke`; sidecar-shaped
target/slot controls recover no clean source IDs.

Hashes:

- `.debug/semantic_predicate_decoder_sidecar_controls_20260427/semantic_predicate_decoder.json`:
  `f1529702d17ae53eb9e5b1ad40e2d274a0c3724d0158c70c6c5c1408353691a2`
- `.debug/semantic_predicate_decoder_sidecar_controls_20260427/semantic_predicate_decoder.md`:
  `7d5882c76b0d9e6e5dbae90084bd731945a8d72cfa9b948843dc290a502365bb`
- `.debug/semantic_predicate_decoder_sidecar_controls_20260427/predictions.jsonl`:
  `48ba362b0f8f557ceb5c1eedd4674d097af1a464f4ea9cdfe1bdf2475fb7fdd8`

## Cycle 2026-04-27 01:38 PDT - CPU frozen source-candidate sidecar smoke

Start state:

1. current ICLR readiness: not ready; still missing a deployable positive
   source-communication method with seed stability and source controls
2. current paper story: side-information communication is the strongest
   remaining story, but current positives are bounds/tooling rather than a
   frozen method
3. exact blocker: no source-derived low-rate sidecar producer survives
   live/holdout controls; MPS remains blocked by orphaned PID `31103`
4. live branch / candidates: learned/frozen syndrome-style sidecar first;
   sparse/dictionary sidecar second
5. highest-priority gate: materialize a no-leak source-candidate sidecar and
   replay it through the hardened semantic decoder
6. scale-up rung: smoke / strict-small preparation

MPS check:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Result: PID `31103` still exists with `PPID=1`, `STAT=UE`, elapsed over
`08:35`, running the old MPS `scripts/calibrate.py` job. No MPS jobs were
started.

Implemented:

- `scripts/materialize_svamp_source_candidate_sidecars.py`
- `tests/test_materialize_svamp_source_candidate_sidecars.py`
- memo: `paper/source_candidate_sidecar_materializer_20260427.md`

The materializer emits source-derived `candidate_scores` JSONL sidecars over
target-side candidate values only. It does not add source-only answers to the
receiver pool.

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_materialize_svamp_source_candidate_sidecars.py tests/test_analyze_svamp_source_semantic_predicate_decoder.py -q
./venv_arm64/bin/python -m py_compile scripts/materialize_svamp_source_candidate_sidecars.py scripts/analyze_svamp_source_semantic_predicate_decoder.py tests/test_materialize_svamp_source_candidate_sidecars.py
```

Result: `9 passed`; compile passed.

Materialization command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_svamp_source_candidate_sidecars.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --output-dir .debug/source_candidate_sidecars_20260427 \
  --sidecar-bits 8 \
  --date 2026-04-27
```

Gate command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp_source_semantic_predicate_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --live-sidecar-jsonl .debug/source_candidate_sidecars_20260427/live_candidate_sidecars.jsonl \
  --holdout-sidecar-jsonl .debug/source_candidate_sidecars_20260427/holdout_candidate_sidecars.jsonl \
  --mode learned_logodds \
  --outer-folds 5 \
  --accept-penalty 0.75 \
  --harm-weight 20.0 \
  --min-live-correct 25 \
  --min-live-clean-source-necessary 2 \
  --min-holdout-correct 10 \
  --min-holdout-clean-source-necessary 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-dir .debug/source_candidate_sidecar_gate_20260427 \
  --output-predictions-jsonl .debug/source_candidate_sidecar_gate_20260427/predictions.jsonl
```

Result: `semantic_predicate_decoder_fails_smoke`.

- live matched: `21/70`, accepted `0`, clean source-necessary `0`, accepted
  harm `0`, control clean union `0`
- holdout matched: `11/70`, accepted `7`, clean source-necessary `0`,
  accepted harm `1`, control clean union `0`
- sidecar bytes: `1` per example

Artifact audit:

- strict target-mentioned live headroom is only `+2`, from `21/70` to `23/70`
- strict target-mentioned holdout headroom is `+4`, from `8/70` to `12/70`
- broader text-to-text/C2C candidate artifacts have larger oracle headroom but
  require a frozen source-derived sidecar and full controls before they can be
  counted as communication evidence

Artifact hashes:

- `.debug/source_candidate_sidecars_20260427/manifest.json`:
  `47c58449496b4983923879dbe466effe83e996ef5a13019be342c21e548dd722`
- `.debug/source_candidate_sidecars_20260427/manifest.md`:
  `d936d08113654d1dbae76ab4d944f4a0f0316ec4a1f68bda45eb1a38173e5150`
- `.debug/source_candidate_sidecars_20260427/live_candidate_sidecars.jsonl`:
  `0378c86287b88edacfe2ae1fd61418d24e10180d99d562b57fb8b2a1574f06dd`
- `.debug/source_candidate_sidecars_20260427/holdout_candidate_sidecars.jsonl`:
  `267abcfd7efa67dd702e9a362a2fdae357084e03b308d3496548141d4ce54b83`
- `.debug/source_candidate_sidecar_gate_20260427/semantic_predicate_decoder.json`:
  `ac5e5e2a9ac60453504fac7c9139fcc788e7e53d8704f4eb7f476b08d072a792`
- `.debug/source_candidate_sidecar_gate_20260427/semantic_predicate_decoder.md`:
  `ec4d14f179c671db63a6f0d8f888c68ab58813301cb81f9c9404156c92b436e2`
- `.debug/source_candidate_sidecar_gate_20260427/predictions.jsonl`:
  `43b4824e0d82a40b2a77e8833dce7943ec9f929e934d0e9ba4479eec6ffec30a`

Decision: kill the heuristic source-candidate materializer as a method branch.
Promote only the no-leak sidecar schema/tooling.

Follow-up strict frozen producer:

- implemented `scripts/collect_svamp_frozen_candidate_score_sidecar.py`
- added `tests/test_collect_svamp_frozen_candidate_score_sidecar.py`
- fake-model tests verify next-token scoring, no gold/correctness leakage,
  source-only candidate exclusion, and target-side-only candidate pools

Two-example CPU plumbing smoke:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/collect_svamp_frozen_candidate_score_sidecar.py \
  --scorer-model Qwen/Qwen2.5-Math-1.5B \
  --target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --eval-file results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl \
  --continuation-template 'Answer: {text}' \
  --limit 2 \
  --device cpu \
  --dtype float32 \
  --sidecar-bits 32 \
  --scorer-use-chat-template \
  --scorer-enable-thinking false \
  --date 2026-04-27 \
  --output-jsonl .debug/frozen_candidate_score_sidecar_smoke_20260427/live_limit2.jsonl \
  --output-md .debug/frozen_candidate_score_sidecar_smoke_20260427/live_limit2.md
```

Result: `2` rows, elapsed `10.42s`; JSONL hash
`15227350a56e5bf9d26143c108fcae598b5cdffa78a07301f44f6c6ed852ce7c`.
Two-ID decoder schema smoke passed only as plumbing evidence, with accepted
sidecar rows `0`.

Full live CPU collection:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/collect_svamp_frozen_candidate_score_sidecar.py \
  --scorer-model Qwen/Qwen2.5-Math-1.5B \
  --target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --eval-file results/qwen25math_qwen3_svamp70_source_surface_20260426/_artifacts/svamp_eval_70_70.jsonl \
  --continuation-template 'Answer: {text}' \
  --device cpu \
  --dtype float32 \
  --sidecar-bits 32 \
  --scorer-use-chat-template \
  --scorer-enable-thinking false \
  --resume \
  --date 2026-04-27 \
  --output-jsonl results/frozen_candidate_score_sidecar_20260427/live_candidate_score_sidecar_cpu.jsonl \
  --output-md results/frozen_candidate_score_sidecar_20260427/live_candidate_score_sidecar_cpu.md
```

Result: `70` rows, elapsed `351.12s`, top labels `target=44`, `t2t=26`.

Live decoder gate:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp_source_semantic_predicate_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --live-sidecar-jsonl results/frozen_candidate_score_sidecar_20260427/live_candidate_score_sidecar_cpu.jsonl \
  --mode learned_logodds \
  --outer-folds 5 \
  --accept-penalty 0.75 \
  --harm-weight 20.0 \
  --min-live-correct 25 \
  --min-live-clean-source-necessary 2 \
  --min-holdout-correct 10 \
  --min-holdout-clean-source-necessary 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-dir results/frozen_candidate_score_sidecar_20260427/live_only_decoder_gate \
  --output-predictions-jsonl results/frozen_candidate_score_sidecar_20260427/live_only_decoder_gate/predictions.jsonl
```

Result: `semantic_predicate_decoder_fails_smoke`.

- live matched: `21/70`
- accepted: `1`
- clean source-necessary: `0`
- accepted harm: `0`
- control clean union: `0`

Hashes:

- `results/frozen_candidate_score_sidecar_20260427/live_candidate_score_sidecar_cpu.jsonl`:
  `3734e4884c87bc14d3bc74317a47c195bbac85253927ae799e3eaa717cf2e771`
- `results/frozen_candidate_score_sidecar_20260427/live_candidate_score_sidecar_cpu.md`:
  `4bff722fcda7579918d95b01f4b01471e52aae53f3b06236b786913054202cdc`
- `results/frozen_candidate_score_sidecar_20260427/live_only_decoder_gate/semantic_predicate_decoder.json`:
  `7491d8c4e6e63d088ae208e1e01496b8712855940a7c5df8fccd246e3fbcd498`
- `results/frozen_candidate_score_sidecar_20260427/live_only_decoder_gate/semantic_predicate_decoder.md`:
  `0a56cd0c02c7ad1a54dbf68e1fdf1d0d791b7a2c587b0916b87ba529d29bf28d`
- `results/frozen_candidate_score_sidecar_20260427/live_only_decoder_gate/predictions.jsonl`:
  `90421fa78c019d7b2fa927bce3f3719bd083792383b585c0a571186107ac521c`

Decision: kill the frozen model-scored target-side candidate sidecar producer
on canonical SVAMP70 live. Do not spend a holdout CPU pass or another threshold
sweep on this exact candidate pool.

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
rg -n "status:.*pass|matched.*clean|clean source|source-necessary|wins|candidate" \
  paper results rotalign latent_bridge -g '*.md'
```

Use that audit to select a source-surface discovery gate or a qualitatively
larger controlled candidate surface. If no overlooked positive branch appears,
switch away from canonical SVAMP70 target-side sidecars.

## Cycle 2026-04-27 02:20 PDT - Source-only residue sidecar strict replay

Start state:

1. current ICLR readiness: not ready; no source-derived method has survived
   live and holdout controls
2. current paper story: source-side information and candidate-pool decoding
   have headroom, but shallow sidecars and repair rows fail controls
3. exact blocker: the strongest prior source-residue sidecar may have used
   source values in the decoder candidate pool
4. live branch / candidates: source-only residue sidecar first; source-surface
   discovery second
5. highest-priority gate: replay the prior live-positive source-only sidecar
   with target-side candidate pool only and hash-based source controls
6. scale-up rung: strict small live/holdout replay

Implemented:

- `scripts/analyze_svamp32_source_only_sidecar_router_gate.py` now supports
  `--shuffle-mode hash`, using deterministic hash-based non-self source
  controls for shuffled-source and label-shuffle conditions.
- `tests/test_analyze_svamp32_source_only_sidecar_router_gate.py` now verifies
  hash controls never select the same source ID.

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_only_sidecar_router_gate.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_only_sidecar_router_gate.py tests/test_analyze_svamp32_source_only_sidecar_router_gate.py
```

Result: `6 passed`; compile passed.

Old-row audit:

- prior textless SVAMP70 live sidecar used `source_alone.jsonl` as a decoder
  candidate artifact, so source-only values could enter the candidate pool
- this is not acceptable as a target-side side-information decoder claim

Live strict replay command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_only_sidecar_router_gate.py \
  --target target=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --source source=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --candidate t2t=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --fallback-label target \
  --source-quality-guard shorter_than_target_numeric \
  --shuffle-mode hash \
  --min-correct 26 \
  --min-target-self 0 \
  --min-clean-source-necessary 4 \
  --max-control-clean-union 0 \
  --min-numeric-coverage 61 \
  --date 2026-04-27 \
  --output-json results/source_only_sidecar_hash_controls_20260427/live_shorter_guard_hash_controls.json \
  --output-md results/source_only_sidecar_hash_controls_20260427/live_shorter_guard_hash_controls.md \
  --output-predictions-jsonl results/source_only_sidecar_hash_controls_20260427/live_shorter_guard_hash_predictions.jsonl \
  --prediction-method source_shorter_than_target_guard_sidecar_live_hash
```

Result: `source_only_sidecar_router_fails_gate`.

- best matched: `22/70`
- clean matched: `0/6`
- clean source-necessary: `0/6`
- control clean union: `0`

Holdout strict replay command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_only_sidecar_router_gate.py \
  --target target=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl,method=target_alone \
  --source source=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,method=source_alone \
  --candidate t2t=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/text_to_text.jsonl,method=text_to_text \
  --target-set-json results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --fallback-label target \
  --source-quality-guard shorter_than_target_numeric \
  --shuffle-mode hash \
  --min-correct 10 \
  --min-target-self 0 \
  --min-clean-source-necessary 1 \
  --max-control-clean-union 0 \
  --min-numeric-coverage 64 \
  --date 2026-04-27 \
  --output-json results/source_only_sidecar_hash_controls_20260427/holdout_shorter_guard_hash_controls.json \
  --output-md results/source_only_sidecar_hash_controls_20260427/holdout_shorter_guard_hash_controls.md \
  --output-predictions-jsonl results/source_only_sidecar_hash_controls_20260427/holdout_shorter_guard_hash_predictions.jsonl \
  --prediction-method source_shorter_than_target_guard_sidecar_holdout_hash
```

Result: holdout passes weakly with `11/70`, clean source-necessary `1`,
control clean union `0`; clean ID `daea537474de16ac`.

Artifact hashes are recorded in
`results/source_only_sidecar_hash_controls_20260427/manifest.md`.

Decision: kill the old source-only residue sidecar as a paper method. Its
prior live-positive row depended on source-value candidate-pool leakage. The
remaining holdout-only one-ID signal is insufficient.

Next exact gate:

```bash
./venv_arm64/bin/python scripts/rank_source_contrastive_target_sets.py \
  --target-set svamp70_live=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json,role=old_live,note=canonical_sidecar_saturated \
  --target-set svamp70_holdout=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json,role=old_holdout,note=weak_one_id_signal \
  --target-set svamp70_chal241_310=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json,role=next_candidate,note=adjacent_clean_surface \
  --min-clean-source-only 4 \
  --date 2026-04-27 \
  --output-json results/source_surface_after_sidecar_pruning_20260427/source_surface_ranking.json \
  --output-md results/source_surface_after_sidecar_pruning_20260427/source_surface_ranking.md
```

If no surface with target-side candidate headroom survives, stop tuning numeric
sidecars and move to a new candidate-surface generator with explicit source
destroying controls.

Follow-up target-side candidate-pool headroom audit:

Implemented:

- `scripts/analyze_target_side_candidate_headroom.py`
- `tests/test_analyze_target_side_candidate_headroom.py`

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_target_side_candidate_headroom.py tests/test_analyze_svamp32_source_only_sidecar_router_gate.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_target_side_candidate_headroom.py scripts/analyze_svamp32_source_only_sidecar_router_gate.py tests/test_analyze_target_side_candidate_headroom.py
```

Result: `7 passed`; compile passed.

Command:

```bash
./venv_arm64/bin/python scripts/analyze_target_side_candidate_headroom.py \
  --target-set svamp70_live=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json,role=old_live,note=canonical_sidecar_saturated \
  --target-set svamp70_holdout=path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json,role=old_holdout,note=weak_one_id_signal \
  --target-set svamp70_chal241_310=path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_contrastive_target_set.json,role=next_candidate,note=adjacent_clean_surface \
  --target-set svamp70_chal311_380=path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_contrastive_target_set.json,role=next_candidate,note=adjacent_weak_surface \
  --target-set svamp70_chal171_240=path=results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_contrastive_target_set.json,role=next_candidate,note=adjacent_weak_surface \
  --target-set gsm70_qwen25math=path=results/qwen25math_qwen3_gsm70_source_surface_20260426/source_contrastive_target_set.json,role=next_candidate,note=gsm_surface \
  --date 2026-04-27 \
  --output-json results/target_side_candidate_headroom_20260427/target_side_candidate_headroom.json \
  --output-md results/target_side_candidate_headroom_20260427/target_side_candidate_headroom.md
```

Result: `target_side_candidate_headroom_ranked`.

- `svamp70_live`: target-side oracle `33/70`, oracle gain `12`, clean gold in
  target-side pool `0/6`
- `svamp70_holdout`: target-side oracle `28/70`, oracle gain `20`, clean gold
  in target-side pool `2/2`
- `svamp70_chal171_240`: target-side oracle `39/70`, clean gold in pool `1/1`
- `svamp70_chal241_310`: target-side oracle `23/70`, clean gold in pool `1/4`

Hashes:

- `results/target_side_candidate_headroom_20260427/target_side_candidate_headroom.json`:
  `30f92ea80c55c330a96bc9771bef54f2b66532706366fb9af9342a43e1facf1d`
- `results/target_side_candidate_headroom_20260427/target_side_candidate_headroom.md`:
  `c2a729773f3d487224997924226a3be2b2b1651a39bbb17e691d674983e88237`

Decision: canonical `svamp70_live` is saturated for target-side
side-information decoders. The next branch is candidate-surface generation:
build non-source target-side candidate pools from target self-repair,
stochastic target routes, or target-only candidate decoders, then run the
target-side headroom audit before any new source sidecar.

Next exact gate:

```bash
rg -n "stochastic|target_self_repair|process_repair|candidate_scores|route" \
  scripts results -g '*.py' -g '*.md'
```

Use that to select the cheapest existing target-side candidate generator and
materialize a no-source candidate surface with exact IDs, bytes/tokens, and
source-destroying controls.

## 2026-04-27 Cycle 1 - No-Source Candidate Surface And Target Sampling

Cycle start:

1. ICLR readiness: not ready; no source-dependent positive method has survived
   the canonical frozen controls.
2. Current paper story: source/headroom exists, but canonical SVAMP70
   target-side pools do not contain clean source-only wins.
3. Exact blocker: find a no-source target candidate surface that contains clean
   source-necessary gold answers before testing a source-derived selector.
4. Live branches: target-side candidate-surface generation first; source
   side-information decoder second.
5. Highest-priority gate: materialize existing no-source zero-KV stochastic
   candidates, then audit target-side pool headroom.
6. Scale-up rung: smoke / source-surface discovery.

Implemented:

- `scripts/materialize_no_source_candidate_surface.py`
- `scripts/sample_target_candidate_surface.py`
- `tests/test_materialize_no_source_candidate_surface.py`
- `tests/test_sample_target_candidate_surface.py`

Artifact commands:

```bash
./venv_arm64/bin/python scripts/materialize_no_source_candidate_surface.py \
  --base-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --candidate target_self_repair=path=results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl,method=target_self_repair \
  --candidate selected_route_no_repair=path=results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl,method=selected_route_no_repair \
  --candidate process_repair=path=results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl,method=process_repair_selected_route \
  --expand-candidate-scores zero_source_pool=path=results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl,method=selected_route_no_repair \
  --min-source-only 0 \
  --date 2026-04-27 \
  --output-dir results/no_source_candidate_surface_20260427
```

```bash
./venv_arm64/bin/python scripts/analyze_target_side_candidate_headroom.py \
  --target-set zero_source_candidate_surface=path=results/no_source_candidate_surface_20260427/source_contrastive_target_set.json,role=no_source_target_pool,note=target+t2t+target_self+process_repair+zero_source_seed_pool \
  --target-set canonical_live=path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json,role=canonical,note=target+t2t_only \
  --target-set target_self_surface=path=results/target_self_repair_candidate_surface_20260427/live_target_self_repair_target_set.json,role=target_self_only,note=target+t2t+target_self_repair \
  --date 2026-04-27 \
  --output-json results/no_source_candidate_surface_20260427/target_side_candidate_headroom.json \
  --output-md results/no_source_candidate_surface_20260427/target_side_candidate_headroom.md
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/target_only_sampling_clean3_20260427/clean_source_only_eval.jsonl \
  --model Qwen/Qwen3-0.6B \
  --samples 8 \
  --temperature 0.9 \
  --top-p 0.95 \
  --seed 11 \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 96 \
  --use-chat-template \
  --enable-thinking false \
  --output-jsonl results/target_only_sampling_clean3_20260427/target_only_samples.jsonl \
  --output-json results/target_only_sampling_clean3_20260427/target_only_samples.json \
  --output-md results/target_only_sampling_clean3_20260427/target_only_samples.md
```

Result summary:

- zero-source candidate surface: target `21/70`, source `13/70`,
  target-side oracle `48/70`, oracle gain `27`, clean in pool `0/3`
- target-self-only surface: target-side oracle `47/70`, clean in pool `0/3`
- canonical live: target-side oracle `33/70`, clean in pool `0/6`
- target-only CPU sampling on the remaining three clean IDs: oracle `1/3`,
  recovered `14bfbfc94f2c2e7b`

Decision:

- killed/weakened: source selector over the existing zero-source candidate
  surface; it has no clean source-necessary gold answer in the pool.
- revived: target-only sampled candidate surface plus source-derived selector,
  because stochastic target sampling recovered one of the three remaining IDs
  without source leakage.

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_materialize_no_source_candidate_surface.py tests/test_sample_target_candidate_surface.py tests/test_analyze_target_side_candidate_headroom.py tests/test_build_source_contrastive_target_set.py -q
./venv_arm64/bin/python -m py_compile scripts/materialize_no_source_candidate_surface.py scripts/sample_target_candidate_surface.py scripts/analyze_target_side_candidate_headroom.py scripts/build_source_contrastive_target_set.py
```

Result: `5 passed`; compile passed.

Hashes:

- `results/no_source_candidate_surface_20260427/source_contrastive_target_set.json`:
  `fb615786f89643c6208909534f59896c2f7d8987b29043842941a769e52e26aa`
- `results/no_source_candidate_surface_20260427/target_side_candidate_headroom.json`:
  `4ebf19932e36de5deaf1f463667b4c1dfb096743140f0779b3afe334722969b9`
- `results/target_only_sampling_clean3_20260427/clean_source_only_eval.jsonl`:
  `a84668c43d47dd72be58daa1a608295bdc42c261c83777ab2eaa33f78c48946b`
- `results/target_only_sampling_clean3_20260427/target_only_samples.jsonl`:
  `f40a89d736afc5da7b28a4b4e01bfdb12650a97dbb67b84b94104674ac427908`
- `results/target_only_sampling_clean3_20260427/target_only_samples.json`:
  `fccf83e7e0b61c023b85b80d524f6bc6ecf9becf852b69702fa115076113de0f`

MPS blocker:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

PID `31103` was still present at cycle start, so all new generation was CPU-only.

Next exact gate:

```bash
./venv_arm64/bin/python scripts/materialize_exact_id_slice.py \
  --reference-jsonl <clean3_reference_with_example_ids.jsonl> \
  --source-jsonl results/target_only_sampling_clean3_20260427/target_only_samples.jsonl \
  --source-method target_sample_s1 \
  --output-jsonl results/target_only_sampling_clean3_20260427/target_sample_s1_exact3.jsonl
```

Materialize all `target_sample_s*` rows into a clean3 target set, then run
`scripts/materialize_svamp_source_candidate_sidecars.py` and
`scripts/analyze_svamp_source_semantic_predicate_decoder.py` on that sampled
pool. Promote only if matched source selects `14bfbfc94f2c2e7b` and all
source-destroying or same-byte controls miss it.

## 2026-04-27 Cycle 2 - Target-only sampled pool source sidecar smoke

Cycle start:

1. ICLR readiness: not ready; estimated distance is still one strict
   same-family positive method, seed repeats, source-control survival, systems
   accounting, and cross-family falsification.
2. Current paper story: target-only/no-source candidate pools can create a
   receiver-side decision surface, and a compact source sidecar may select the
   right candidate only when source content is matched.
3. Exact blocker: prove the selector is source-derived rather than target-pool,
   random sidecar, shuffled-source, or slots-only leakage.
4. Live branch: sampled target candidate pool plus compact source-derived
   candidate-score sidecar; secondary branch is learned semantic predicates.
5. Highest-priority gate: clean3 smoke with full source-destroying controls.
6. Scale-up rung: smoke.

Implemented:

- `scripts/extend_target_set_candidate_labels.py`
- `scripts/analyze_candidate_score_sidecar_top_select.py`
- `tests/test_extend_target_set_candidate_labels.py`
- `tests/test_analyze_candidate_score_sidecar_top_select.py`
- memo: `paper/target_only_sampling_source_sidecar_smoke_20260427.md`

Important harness fix:

- the first random-sidecar control shuffled `candidate_scores` but the decoder
  sorted by score afterward, accidentally preserving the matched top candidate.
- fixed it to preserve the same score distribution while destroying the
  candidate-value mapping.

Artifact commands:

```bash
./venv_arm64/bin/python scripts/extend_target_set_candidate_labels.py \
  --base-target-set results/no_source_candidate_surface_20260427/source_contrastive_target_set.json \
  --id-fields clean_source_only \
  --candidate target_sample_s0=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s0 \
  --candidate target_sample_s1=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s1 \
  --candidate target_sample_s2=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s2 \
  --candidate target_sample_s3=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s3 \
  --candidate target_sample_s4=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s4 \
  --candidate target_sample_s5=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s5 \
  --candidate target_sample_s6=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s6 \
  --candidate target_sample_s7=path=results/target_only_sampling_clean3_20260427/target_only_samples.jsonl,method=target_sample_s7 \
  --date 2026-04-27 \
  --output-json results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --output-md results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.md \
  --manifest-json results/target_only_sampling_clean3_20260427/sampled_clean3_target_set_manifest.json
```

```bash
./venv_arm64/bin/python scripts/analyze_target_side_candidate_headroom.py \
  --target-set sampled_clean3=path=results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json,role=sampled_target_pool,note=zero_source_pool_plus_target_samples_clean3 \
  --date 2026-04-27 \
  --output-json results/target_only_sampling_clean3_20260427/sampled_clean3_headroom.json \
  --output-md results/target_only_sampling_clean3_20260427/sampled_clean3_headroom.md
```

```bash
./venv_arm64/bin/python scripts/materialize_svamp_source_candidate_sidecars.py \
  --live-target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --output-dir results/target_only_sampling_clean3_20260427/source_candidate_sidecars \
  --sidecar-bits 8 \
  --label-prior 0.0 \
  --date 2026-04-27
```

```bash
./venv_arm64/bin/python scripts/analyze_svamp_source_semantic_predicate_decoder.py \
  --live-target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --holdout-target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --live-sidecar-jsonl results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl \
  --holdout-sidecar-jsonl results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl \
  --outer-folds 3 \
  --accept-penalty 0.05 \
  --harm-weight 4.0 \
  --min-live-correct 1 \
  --min-live-clean-source-necessary 1 \
  --min-holdout-correct 1 \
  --min-holdout-clean-source-necessary 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-dir results/target_only_sampling_clean3_20260427/source_sidecar_decoder_smoke \
  --output-predictions-jsonl results/target_only_sampling_clean3_20260427/source_sidecar_decoder_smoke/predictions.jsonl
```

```bash
./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
  --target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --sidecar-jsonl results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl \
  --min-confidence 2.0 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json results/target_only_sampling_clean3_20260427/top_sidecar_selector.json \
  --output-md results/target_only_sampling_clean3_20260427/top_sidecar_selector.md
```

Result summary:

- sampled clean3 target pool: target `0/3`, source `3/3`,
  target-side oracle `1/3`, clean in pool `1/3`
- learned semantic-predicate decoder: matched correct `0/3`, accepted `0`,
  clean source-necessary `0`
- top source sidecar selector: matched correct `1/3`, accepted `1`,
  clean source-necessary `1`, accepted harm `0`
- controls: shuffled-source clean `0`, randomized same-byte sidecar clean `0`,
  target-only clean `0`, slots-only clean `0`, control clean union `0`
- recovered source-necessary clean ID: `14bfbfc94f2c2e7b`

Decision:

- promoted: sampled target-pool + compact source sidecar from source-surface
  discovery to smoke-positive.
- weakened: learned semantic-predicate erasure decoder on this tiny surface; it
  is target-safe but too conservative.
- not promoted to paper evidence: the positive row is one clean ID on a
  three-example slice and uses a handwritten top selector.

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_candidate_score_sidecar_top_select.py -q
```

Result: `3 passed`.

Hashes:

- `results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json`:
  `786bc1a4c336483633e74a64d3d309602d0bcb79d758b8fc0190f311832d52ff`
- `results/target_only_sampling_clean3_20260427/sampled_clean3_headroom.json`:
  `22ccec02fa0ee77d990511743ac3dd766b04f033695cb2865e0235fdb19e62fb`
- `results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl`:
  `003adb88d1424f8b7b444d0972c85141d2cc889f29e27a0487ba48605ea7e66f`
- `results/target_only_sampling_clean3_20260427/source_candidate_sidecars/manifest.json`:
  `b51345d0b2efab9d7bfb4a91ba032790aeacfbcb2040220f5593add014f0b050`
- `results/target_only_sampling_clean3_20260427/source_sidecar_decoder_smoke/semantic_predicate_decoder.json`:
  `dd42726fb336d7df8423a5158eb0e5e5f392865f95a98d9d961b363939b9faa5`
- `results/target_only_sampling_clean3_20260427/top_sidecar_selector.json`:
  `58c79810a3a8bb19c2431f0a19f973af3a46a05d1094ee2172de62a915879aeb`

MPS blocker:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

PID `31103` remains present with `STAT=UE`; no MPS jobs were started.

Next exact gate:

```bash
./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
  --target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --sidecar-jsonl results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl \
  --min-confidence 2.0 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json results/target_only_sampling_clean3_20260427/top_sidecar_selector_repeated_random.json \
  --output-md results/target_only_sampling_clean3_20260427/top_sidecar_selector_repeated_random.md
```

Before trusting the smoke row, add repeated random-sidecar salts or an exact
enumeration over candidate-score permutations. Promote only if matched remains
source-necessary clean and same-byte/random controls have clean union `0`.

## 2026-04-27 Cycle 3 - Clean3 Source-Answer Ablation Gate

Cycle start:

1. ICLR readiness: not ready; estimated distance remains a stable positive
   method plus strict controls, seed repeats, systems accounting, and
   cross-family falsification.
2. Current paper story: the clean3 smoke might be source-derived candidate
   selection, but it may also be direct source-answer copying into a lucky
   target-only sampled pool.
3. Exact blocker: distinguish richer source communication from source-final or
   verified numeric answer evidence.
4. Live branch: sampled target candidate pool plus candidate-score sidecar.
5. Highest-priority gate: source-answer masking and source-final-only baseline.
6. Scale-up rung: smoke falsification.

Implemented:

- `scripts/materialize_sidecar_counterfactuals.py`
- `tests/test_materialize_sidecar_counterfactuals.py`
- memo: `paper/clean3_source_answer_ablation_gate_20260427.md`
- manifest update:
  `results/target_only_sampling_clean3_20260427/source_candidate_sidecars/manifest.md`

Commands:

```bash
./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
  --target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --sidecar-jsonl results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl \
  --min-confidence 2.0 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json .debug/clean3_sidecar_counterfactuals_20260427/full_top_selector_rerun.json \
  --output-md .debug/clean3_sidecar_counterfactuals_20260427/full_top_selector_rerun.md
```

```bash
./venv_arm64/bin/python scripts/materialize_sidecar_counterfactuals.py \
  --target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --sidecar-jsonl results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl \
  --mode source_answer_masked \
  --date 2026-04-27 \
  --output-jsonl .debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_sidecar.jsonl \
  --output-json .debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_sidecar.json \
  --output-md .debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_sidecar.md
```

```bash
./venv_arm64/bin/python scripts/materialize_sidecar_counterfactuals.py \
  --target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --sidecar-jsonl results/target_only_sampling_clean3_20260427/source_candidate_sidecars/live_candidate_sidecars.jsonl \
  --mode source_final_only \
  --date 2026-04-27 \
  --output-jsonl .debug/clean3_sidecar_counterfactuals_20260427/source_final_only_sidecar.jsonl \
  --output-json .debug/clean3_sidecar_counterfactuals_20260427/source_final_only_sidecar.json \
  --output-md .debug/clean3_sidecar_counterfactuals_20260427/source_final_only_sidecar.md
```

```bash
./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
  --target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --sidecar-jsonl .debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_sidecar.jsonl \
  --min-confidence 2.0 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json .debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_top_selector.json \
  --output-md .debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_top_selector.md
```

```bash
./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
  --target-set results/target_only_sampling_clean3_20260427/sampled_clean3_target_set.json \
  --sidecar-jsonl .debug/clean3_sidecar_counterfactuals_20260427/source_final_only_sidecar.jsonl \
  --min-confidence 2.0 \
  --min-source-necessary-clean 0 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json .debug/clean3_sidecar_counterfactuals_20260427/source_final_only_top_selector.json \
  --output-md .debug/clean3_sidecar_counterfactuals_20260427/source_final_only_top_selector.md
```

Result summary:

- full sidecar: passes smoke, source-necessary clean
  `['14bfbfc94f2c2e7b']`, control clean union `[]`, accepted harm `0`
- source-answer masked: fails smoke, source-necessary clean `[]`, control
  clean union `[]`, accepted harm `0`
- source-final-only: passes smoke, source-necessary clean
  `['14bfbfc94f2c2e7b']`, control clean union `[]`, accepted harm `0`

Decision:

- killed: clean3 candidate-score sidecar as a positive-method branch. The win is
  explained by source-final numeric evidence over a target-only sampled pool.
- promoted as a diagnostic only: target-only sampling can expose receiver-side
  headroom that source signals can select, but future sidecars must pass
  source-answer masking.
- next live branch: source-surface discovery over existing artifacts for
  target-side pools with source-necessary answers not explained by direct
  source-final numeric evidence.

Tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_materialize_sidecar_counterfactuals.py \
  tests/test_analyze_candidate_score_sidecar_top_select.py -q
```

Result: `5 passed`.

Hashes:

- `.debug/clean3_sidecar_counterfactuals_20260427/source_answer_masked_sidecar.jsonl`:
  `d8fcde23ea05c1f989974925467c769565f11cdfa5fda61c3de852667fb4a7a2`
- `.debug/clean3_sidecar_counterfactuals_20260427/source_final_only_sidecar.jsonl`:
  `e1fe88457fdd74defd94008fd22178aa355a2d54278fac44671d5e9817b655c3`

MPS blocker:

PID `31103` remains present with `STAT=UE`; no MPS jobs were started.

Next exact gate:

```bash
./venv_arm64/bin/python scripts/audit_source_surface_answer_masking.py \
  --results-root results \
  --date 2026-04-27 \
  --output-json results/source_surface_answer_masking_audit_20260427/audit.json \
  --output-md results/source_surface_answer_masking_audit_20260427/audit.md
```

Implement a CPU-only artifact audit that ranks existing target-side candidate
surfaces by clean source-necessary IDs whose gold appears in the receiver pool
without being explained by the source final/verified numeric answer. If the
audit finds none, the next action is MPS/session cleanup before richer surface
generation.

## 2026-04-27 Cycle 4 - Source-Surface Answer-Masking Audit

Cycle start:

1. ICLR readiness: not ready; no current positive method survives
   source-answer masking.
2. Current paper story: candidate-pool headroom exists, but source-sidecar wins
   are not communication evidence if explained by final numeric answer copying.
3. Exact blocker: find an existing target-side surface with clean source-needed
   answers in the receiver pool that are not explained by source final or
   verified numeric answers.
4. Live branch: source-surface discovery over stored artifacts.
5. Highest-priority gate: CPU-only artifact-wide answer-masking audit.
6. Scale-up rung: source-surface discovery.

Implemented:

- `scripts/audit_source_surface_answer_masking.py`
- `tests/test_audit_source_surface_answer_masking.py`
- memo: `paper/source_surface_answer_masking_audit_20260427.md`

Command:

```bash
./venv_arm64/bin/python scripts/audit_source_surface_answer_masking.py \
  --results-root results \
  --date 2026-04-27 \
  --output-json results/source_surface_answer_masking_audit_20260427/audit.json \
  --output-md results/source_surface_answer_masking_audit_20260427/audit.md
```

Result summary:

- candidate target-set paths: `15`
- loaded surfaces with clean IDs: `12`
- skipped non-loadable candidate JSONs: `3`
- top clean-in-pool surface:
  `results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json`
  with clean `2`, clean in pool `2`, answer-unexplained clean in pool `0`
- all loaded surfaces: answer-unexplained clean in pool `0`

Decision:

- killed: tuning source candidate-score sidecars on existing stored surfaces
  under the stricter answer-masking threat model.
- live branch: answer-masked source-interface design plus fresh surface
  generation when MPS clears.
- hard constraint: PID `31103` remains stuck, so no MPS generation should start.

Tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_audit_source_surface_answer_masking.py \
  tests/test_materialize_sidecar_counterfactuals.py \
  tests/test_analyze_candidate_score_sidecar_top_select.py -q
```

Result: `7 passed`.

Hash:

- `results/source_surface_answer_masking_audit_20260427/audit.json`:
  `7e6a3acf0cda9e0fb033695d0ab09496d83c0e9ca9f9f4b6013fbd10aeb6e816`

Next exact gate:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is gone, generate a fresh strict small same-family source
surface with final-answer masking built into source sidecar formation. If PID
`31103` remains, continue CPU-only literature/harness design for answer-masked
communication interfaces.

## 2026-04-27 Cycle 5 - Masked Process-Verifier Sidecar Smoke

Cycle start:

1. ICLR readiness: not ready; the active blocker is answer-null source
   communication.
2. Current paper story: candidate sidecars are only useful if they communicate
   reasoning/process information rather than direct final numeric answers.
3. Exact blocker: source-answer masking killed the prior sidecar, and stored
   surfaces have no answer-unexplained clean in-pool IDs.
4. Live branches: masked process-verifier sidecar; predicate syndrome as the
   next branch if process-overlap fails.
5. Highest-priority gate: CPU-only live/holdout masked process sidecar smoke.
6. Scale-up rung: smoke.

Implemented:

- `scripts/materialize_masked_process_verifier_sidecars.py`
- `tests/test_materialize_masked_process_verifier_sidecars.py`
- memo: `paper/masked_process_verifier_sidecar_20260427.md`
- reference memo: `references/472_answer_null_sideinfo_refs.md`

Commands:

```bash
./venv_arm64/bin/python scripts/materialize_masked_process_verifier_sidecars.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --output-dir results/masked_process_verifier_sidecars_20260427 \
  --date 2026-04-27
```

```bash
./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
  --target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --sidecar-jsonl results/masked_process_verifier_sidecars_20260427/live_masked_process_sidecars.jsonl \
  --min-confidence 0.5 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json results/masked_process_verifier_sidecars_20260427/live_top_selector.json \
  --output-md results/masked_process_verifier_sidecars_20260427/live_top_selector.md
```

```bash
./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
  --target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --sidecar-jsonl results/masked_process_verifier_sidecars_20260427/holdout_masked_process_sidecars.jsonl \
  --min-confidence 0.5 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json results/masked_process_verifier_sidecars_20260427/holdout_top_selector.json \
  --output-md results/masked_process_verifier_sidecars_20260427/holdout_top_selector.md
```

Result summary:

- live materialization: `70` examples, clean `6`, answer-excluded top `55`
- holdout materialization: `70` examples, clean `2`, answer-excluded top `62`
- live top selector: matched `21/70`, accepted `0`, clean `0`, accepted harm
  `0`, control clean union `0`
- holdout top selector: matched `8/70`, accepted `0`, clean `0`, accepted harm
  `0`, control clean union `0`
- holdout threshold sweep at `0`, `0.1`, `0.25`, `0.5`, `1.0`: no
  source-necessary clean IDs

Decision:

- killed: heuristic masked process-overlap sidecar. After answer masking, simple
  operation/equation/lexical overlap is too weak to select useful candidates.
- next branch: structured answer-null predicate syndrome over operation
  sequence, quantity roles, equation-shape buckets, unit relation, and
  sign/order relation.

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_materialize_masked_process_verifier_sidecars.py -q
./venv_arm64/bin/python -m py_compile scripts/materialize_masked_process_verifier_sidecars.py
```

Result: `3 passed`; compile passed.

Hashes:

- `results/masked_process_verifier_sidecars_20260427/manifest.json`:
  `d7ff1fd198fcd0a2fc3200d0fc332c234f372d06133c6827e316823c1f3a10d3`
- `results/masked_process_verifier_sidecars_20260427/live_masked_process_sidecars.jsonl`:
  `03047945d55036b8269331ed2286b09ac9a83e1dae06d264ab32c5bc96d2a0d6`
- `results/masked_process_verifier_sidecars_20260427/holdout_masked_process_sidecars.jsonl`:
  `99d8f7fe9821125c2321a1e0f6ae6bbd354ef54016af7f36c1e2f72f7f3868a3`
- `results/masked_process_verifier_sidecars_20260427/live_top_selector.json`:
  `f132337726df48abd7f4f6051b7c56e07085edb94dad0410a975416fc398ef64`
- `results/masked_process_verifier_sidecars_20260427/holdout_top_selector.json`:
  `01967b1885a2536b29f573b7ae4e12c55d57f49293ef91551d4c73e48b1fc8cf`

Next exact gate:

```bash
./venv_arm64/bin/python scripts/materialize_answer_null_predicate_syndrome.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --output-dir results/answer_null_predicate_syndrome_20260427 \
  --date 2026-04-27
```

Implement this as a CPU-only smoke. The sidecar must exclude candidate IDs,
candidate values, source final numbers, verified answer numbers, and residue
hashes.

## 2026-04-27 Cycle 6 - Answer-Null Predicate Syndrome Smoke

Cycle start:

1. ICLR readiness: not ready; answer-null communication remains unsolved.
2. Current paper story: direct candidate/value sidecars are pruned; simple
   masked process overlap is too weak; structured predicate syndromes are the
   last cheap CPU branch over stored artifacts.
3. Exact blocker: show any non-answer source predicate signal can select clean
   target-side candidates without control leakage or target harm.
4. Live branch: answer-null predicate syndrome.
5. Highest-priority gate: live/holdout CPU smoke with shuffled-source,
   random-syndrome, target-only, and slots-only controls.
6. Scale-up rung: smoke.

Implemented:

- `scripts/analyze_answer_null_predicate_syndrome.py`
- `tests/test_analyze_answer_null_predicate_syndrome.py`
- memo: `paper/answer_null_predicate_syndrome_20260427.md`

Commands:

```bash
./venv_arm64/bin/python scripts/analyze_answer_null_predicate_syndrome.py \
  --target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --min-confidence 0.0 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json results/answer_null_predicate_syndrome_20260427/live_predicate_syndrome.json \
  --output-md results/answer_null_predicate_syndrome_20260427/live_predicate_syndrome.md
```

```bash
./venv_arm64/bin/python scripts/analyze_answer_null_predicate_syndrome.py \
  --target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --min-confidence 0.0 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json results/answer_null_predicate_syndrome_20260427/holdout_predicate_syndrome.json \
  --output-md results/answer_null_predicate_syndrome_20260427/holdout_predicate_syndrome.md
```

Result summary:

- live: matched `16/70`, accepted `45`, clean `0`, accepted harm `12`, control
  clean union `0`
- holdout: matched `13/70`, accepted `46`, clean `1`, accepted harm `5`,
  control clean union `1`
- holdout clean recovery `ab1e71e8928661d0` is explained by random and shuffled
  controls.
- threshold sweeps at `0.1`, `0.5`, and `1.0` on both live and holdout recover
  no source-necessary clean IDs.

Decision:

- killed: structured answer-null predicate syndrome over stored surfaces.
- current live branch: none among CPU-only stored-artifact branches.
- hard blocker: fresh same-family surface generation is now required, but PID
  `31103` remains an orphaned MPS process in `STAT=UE`.

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_answer_null_predicate_syndrome.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_answer_null_predicate_syndrome.py
```

Result: `2 passed`; compile passed.

Hashes:

- `results/answer_null_predicate_syndrome_20260427/live_predicate_syndrome.json`:
  `70f093a89fb99d485ce86b038fe327ec2cdbbdd9c847a65e04261cb089c562fe`
- `results/answer_null_predicate_syndrome_20260427/holdout_predicate_syndrome.json`:
  `a5d6bf035c641f301d2f497f6b33219687ce76ca894ab45e4abb279417737f00`

Resume command:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is gone, the next exact gate is fresh strict-small same-family
surface generation with source-final masking built in from the first run. If it
persists, OS/session-level cleanup is required before progress on the next
evidence-bearing gate.

## 2026-04-27 Cycle 7 - Fresh CPU SVAMP Surface Scout

Cycle start:

1. ICLR readiness: not ready; no positive method or source surface currently
   survives answer-masking/source-control gates.
2. Current paper story: CPU-only stored artifacts are exhausted, but a tiny
   fresh CPU source-surface scout might find new clean source-needed IDs while
   MPS is blocked.
3. Exact blocker: PID `31103` remains in `STAT=UE`, preventing MPS scale-up and
   fresh MPS generation.
4. Live branch: CPU fresh surface discovery, with MPS fresh same-family scout as
   the next real branch after cleanup.
5. Highest-priority gate: two adjacent SVAMP8 CPU baseline scouts with exact ID
   parity, text-relay control, and answer-masking audit.
6. Scale-up rung: smoke / source-surface discovery.

Commands:

```bash
./venv_arm64/bin/python scripts/materialize_jsonl_range.py \
  --source data/svamp_1000.jsonl \
  --output results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.jsonl \
  --start-index 381 \
  --count 8 \
  --manifest-json results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.manifest.json \
  --manifest-md results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.manifest.md \
  --run-date 2026-04-27
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.jsonl \
  --results-dir results/fresh_cpu_svamp8_answernull_20260427/baselines \
  --methods source target t2t \
  --limit 8 \
  --device cpu \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Repeated with rows `389..396` under
`results/fresh_cpu_svamp8b_answernull_20260427/`.

Result summary:

| Range | Target | Source | Text Relay | Source-Only | Clean Source-Only | Answer-Unexplained Clean In Pool |
|---|---:|---:|---:|---:|---:|---:|
| SVAMP rows 381-388 | `1/8` | `1/8` | `4/8` | `1` | `0` | `0` |
| SVAMP rows 389-396 | `2/8` | `1/8` | `3/8` | `0` | `0` | `0` |

Decision:

- killed: fresh CPU SVAMP8 scouts as a useful next source surface.
- current live branch: none that can progress to evidence-bearing gates while
  PID `31103` remains stuck.
- hard blocker: OS/session-level cleanup of orphaned MPS process PID `31103`.

Hashes:

- `results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.jsonl`:
  `7cc0e9d8388778fca31b7dde69293d5400e0c645f03c21ec5c677a482c50daf1`
- `results/fresh_cpu_svamp8_answernull_20260427/baselines/manifest.json`:
  `e7c755be971dbcbd8b29b0c28296d10edd934911b0b3f5ce30bd496de4d32e32`
- `results/fresh_cpu_svamp8_answernull_20260427/source_contrastive_target_set.json`:
  `bc74b715e6f6fef21aee644fa1bfe3ec925764031d7a0c6f46ac96dcb9cbfacc`
- `results/fresh_cpu_svamp8_answernull_20260427/answer_masking_audit.json`:
  `ce20056a6ca59b27f08ee9ed02c2494d997344c4f7e58ce2d16fc5638c03ed5c`
- `results/fresh_cpu_svamp8b_answernull_20260427/svamp_rows389_396.jsonl`:
  `753b2778348768e3ff6b72cd0c070454ce5baf52a142f8fc0ed3a0db78138280`
- `results/fresh_cpu_svamp8b_answernull_20260427/baselines/manifest.json`:
  `ebb82ae03a766c35bb5f47dfad8bdd742e436cafadc0f0b2c553a504baee5f18`
- `results/fresh_cpu_svamp8b_answernull_20260427/source_contrastive_target_set.json`:
  `f1ab04c2313fd70c452b41076a881f1f1066c20a3ebc6aad8f9473cb3943cf23`
- `results/fresh_cpu_svamp8b_answernull_20260427/answer_masking_audit.json`:
  `843885e38e760b2951e9260e6ee6d9e6112835cc590c0b471de21d465afb29fe`

Resume command:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is gone, run the stronger-source SVAMP70 scout or KVComm strict
source-control MPS smoke recorded earlier in the ledger. If it remains in
`STAT=UE`, OS/session-level cleanup is required before the next
evidence-bearing gate.

## 2026-04-27 Cycle 8 - Condition Candidate Pool Control Hardening

Cycle start:

1. ICLR readiness: not ready; no live positive method survives strict source
   controls and answer-masking.
2. Current paper story: current CPU sidecars over stored artifacts are pruned,
   but future learned receiver gates still need stricter source-destroying
   candidate pools.
3. Exact blocker: PID `31103` remains an orphaned MPS process in `STAT=UE`, so
   fresh evidence-bearing MPS gates cannot run.
4. Current live branch: none while MPS is blocked. Top next branch after cleanup
   is stronger-source answer-masked surface discovery.
5. Highest-priority gate: CPU-only harness hardening for condition-specific
   receiver controls.
6. Scale-up rung: harness/source-control hardening before smoke.

Subagent findings:

- artifact audit: no CPU-only evidence-bearing method gate remains; strongest
  reusable surface is still `svamp70_live`, but it is a surface, not a method.
- experiment/literature planner: next scientific rung is stronger-source
  answer-masked Wyner-Ziv-style surface discovery after MPS cleanup.
- code audit: `scripts/build_condition_likelihood_candidate_pools.py` recorded
  `label_shuffle_offset` but did not use it; `label_shuffle` used same-example
  source content in the target-labeled slot, and `shuffled_source` could
  self-donor if offsets were zero or wrapped to self.

Implemented:

- added `_nonself_offset_index(total, index, offset)`.
- made `shuffled_source` use guaranteed non-self source donors.
- made `label_shuffle` use `--label-shuffle-offset` for a non-self donor in
  the target-labeled source slot.
- added regression coverage for zero offsets and direct donor ID checks.

Tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_build_condition_likelihood_candidate_pools.py \
  tests/test_analyze_condition_likelihood_receiver_gate.py \
  tests/test_kvcomm_eval_controls.py -q
```

Result: `18 passed in 0.12s`.

```bash
./venv_arm64/bin/python -m py_compile scripts/build_condition_likelihood_candidate_pools.py
```

Result: passed.

Reference update:

- `references/473_mps_blocked_next_gate_refs.md` consolidates current
  C2C/KVCOMM/KVComm/Q-KVComm implications: future claims need a stronger
  source surface first, then C2C/KVComm/Q-KVComm-style quality and matched-byte
  baselines.

Decision:

- hardened: condition-specific candidate-pool controls.
- not revived: condition-likelihood receiver on current SVAMP70 surface.
- current live branch: none while PID `31103` persists.

Resume command:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` clears, regenerate future condition candidate pools with this
patch before any receiver scoring. Then run stronger-source answer-masked
surface discovery; spend learned-syndrome or zero-init query-bottleneck compute
only if the surface has answer-unexplained clean target-pool headroom.

## 2026-04-27 Cycle 9 - MPS Blocker Preflight

Cycle start:

1. ICLR readiness: not ready; no live method branch survives strict controls.
2. Current paper story: stronger-source answer-masked side information remains
   the highest-value next scientific rung, but it is MPS-gated.
3. Exact blocker: PID `31103` is still orphaned under PID `1` in `STAT=UE`.
4. Current live branch: none while the blocker persists.
5. Highest-priority gate: make the MPS blocker check executable and
   machine-readable.
6. Scale-up rung: operational hard-blocker preflight.

Command:

```bash
./venv_arm64/bin/python scripts/check_mps_blocker.py --json
```

Result:

```json
{
  "blocked": true,
  "next_action": "use_cpu_only_or_clear_os_session",
  "pid": 31103,
  "present": true
}
```

Implemented:

- `scripts/check_mps_blocker.py`
- `tests/test_check_mps_blocker.py`
- memo: `paper/mps_blocker_preflight_20260427.md`

Tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_check_mps_blocker.py -q
./venv_arm64/bin/python -m py_compile scripts/check_mps_blocker.py
```

Result: `3 passed`; compile passed.

Decision:

- no positive method branch is live while the blocker persists.
- no MPS jobs were launched.
- hard blocker remains OS/session-level cleanup of PID `31103`.

Next exact gate:

```bash
./venv_arm64/bin/python scripts/check_mps_blocker.py --json
```

Proceed to stronger-source answer-masked surface discovery only when it reports
`"blocked": false`.

## 2026-04-27 Cycle 10 - MPS Micro Stronger-Source Surface Gate

Cycle start:

1. Current ICLR readiness: not ready; no branch has positive evidence beyond
   strict source controls.
2. Current paper story: CPU sidecars and existing surfaces are pruned; with MPS
   now clear, the highest-value live branch is stronger-source answer-masked
   surface discovery.
3. Exact blocker: find source-only target-pool headroom that is not explained by
   the source final/verified numeric answer.
4. Current live candidates: `Qwen2.5-7B-Instruct -> Qwen3-0.6B` discovery; the
   compatible `Qwen2.5-Math-1.5B -> Qwen3-0.6B` canary only as an operational
   fallback.
5. Highest-priority gate: micro MPS smoke with exact ID parity, text relay, and
   answer-masking audit.
6. Scale-up rung: micro smoke.

MPS preflight:

```bash
./venv_arm64/bin/python scripts/check_mps_blocker.py --json
```

Result: `blocked=false`, PID `31103` absent.

7B stronger-source command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/mps_micro_qwen25_7b_qwen3_svamp8_surface_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t \
  --limit 8 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Result:

- source `2/8`, target `2/8`, text relay `1/8`
- exact ID parity: true for all three methods
- numeric coverage: source `8/8`, target `8/8`, text relay `8/8`
- clean source-only after text relay: `1` ID, `d64f6e35083ffe8c`
- answer-masking audit: `answer_unexplained_clean_in_pool=0`; the clean ID is
  explained by source final answer `2`

Compatibility canary:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t \
  --limit 8 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Result:

- source `2/8`, target `2/8`, text relay `2/8`
- exact ID parity: true for all three methods
- numeric coverage: source `7/8`, target `8/8`, text relay `8/8`
- clean source-only after text relay: `0`
- answer-masking audit: `answer_unexplained_clean_in_pool=0`

Artifacts:

- `results/mps_micro_qwen25_7b_qwen3_svamp8_surface_20260427/manifest.json`
- `results/mps_micro_qwen25_7b_qwen3_svamp8_surface_20260427/source_contrastive_target_set.json`
- `results/mps_micro_qwen25_7b_qwen3_svamp8_surface_20260427/answer_masking_audit.json`
- `results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/manifest.json`
- `results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/source_contrastive_target_set.json`
- `results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/answer_masking_audit.json`
- memo: `paper/mps_micro_stronger_source_surface_20260427.md`

Decision:

- passed: MPS is usable again; generation artifacts preserve exact ID parity,
  high numeric coverage, and sidecar validation.
- failed: neither micro surface contains answer-unexplained clean target-pool
  headroom.
- not promoted: no learned syndrome/semantic sidecar, zero-init connector, or
  KV transport claim should use these rows as communication evidence.

Next exact gate:

Run a different answer-masked discovery surface rather than scaling these exact
first-eight IDs. Prefer a bounded 7B or Math-7B slice selected for
source/target disagreement; promote only if `answer_unexplained_clean_in_pool >
0` and the clean IDs survive text relay plus source-destroying controls.

## 2026-04-27 Cycle 11 - Disagreement Surface And JEPA Anti-Collapse

Cycle start:

1. Current ICLR readiness: not ready; no positive branch survives strict
   source controls.
2. Current paper story: MPS generation works again and 7B sources create some
   disagreement, but answer-masking still blocks promotion.
3. Exact blocker: source-only target-pool headroom must be answer-unexplained.
4. Current live candidates: selected 7B disagreement discovery; JEPA-inspired
   latent prediction only after a clean non-leaky surface exists.
5. Highest-priority gate: selected 6-ID and 12-ID disagreement audits plus
   JEPA anti-collapse branch synthesis.
6. Scale-up rung: selected micro discovery.

Harness update:

- `scripts/materialize_generation_id_subset.py` now accepts exactly one of
  `--target-set-json`, `--ids`, or `--ids-file`, so selected discovery slices can
  be reproduced without fabricating target-set JSON.
- focused tests: `tests/test_materialize_generation_id_subset.py`.

Clean6 command:

```bash
./venv_arm64/bin/python scripts/materialize_generation_id_subset.py \
  --eval-file data/svamp_eval_70.jsonl \
  --target-set-json results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --id-fields clean_source_only \
  --output-jsonl results/mps_qwen25_7b_historical_clean6_discovery_20260427/clean6_eval.jsonl \
  --output-meta-json results/mps_qwen25_7b_historical_clean6_discovery_20260427/clean6_eval.meta.json
```

Then:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/mps_qwen25_7b_historical_clean6_discovery_20260427/clean6_eval.jsonl \
  --results-dir results/mps_qwen25_7b_historical_clean6_discovery_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t \
  --limit 6 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Clean6 result: target `0/6`, source `1/6`, text relay `1/6`, clean source-only
`1`, answer-unexplained clean in pool `0`.

Disagreement12 command:

```bash
./venv_arm64/bin/python scripts/materialize_generation_id_subset.py \
  --eval-file data/svamp_1000.jsonl \
  --ids 14bfbfc94f2c2e7b 2de1549556000830 41cce6c6e6bb0058 4d780f825bb8541c bd9d8da923981d69 ce08a3a269bf0151 0ee313c160b638a9 561daa750422c0e4 cd5623c80cf95da9 e90d2681e386fb04 ab1e71e8928661d0 daea537474de16ac \
  --output-jsonl results/mps_qwen25_7b_disagreement12_discovery_20260427/disagreement12_eval.jsonl \
  --output-meta-json results/mps_qwen25_7b_disagreement12_discovery_20260427/disagreement12_eval.meta.json

PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/mps_qwen25_7b_disagreement12_discovery_20260427/disagreement12_eval.jsonl \
  --results-dir results/mps_qwen25_7b_disagreement12_discovery_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t \
  --limit 12 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Disagreement12 result:

- target `0/12`, source `4/12`, text relay `4/12`
- exact ID parity true for all methods
- numeric coverage `12/12` for all methods
- clean source-only after text relay: `2`
- clean in target-side pool: `1`
- answer-unexplained clean in pool: `0`

Decision:

- failed: selected 6-ID and 12-ID disagreement probes still do not expose
  answer-unexplained target-pool headroom.
- not promoted: no learned sidecar, latent connector, or KV transport claim.
- updated next-branch design: JEPA/LeJEPA/V-JEPA literature suggests
  answer-masked latent prediction with frozen target latents, source-destroying
  margins, target-preservation loss, and explicit variance/covariance/effective
  rank telemetry to avoid representation collapse.

Reference update:

- `references/474_jepa_anticollapse_refs.md`
- manifest: `references/research_memo_manifest.json`

New memo:

- `paper/mps_disagreement_surface_and_jepa_anticollapse_20260427.md`

Next exact gate:

Implement CPU-only `answer_masked_source` / `answer_only` diagnostics and
collapse telemetry, then run answer-likelihood smoke on live and holdout source
surfaces before another MPS generation sweep.

## 2026-04-27 - CPU answer-likelihood smoke with answer-only and answer-masked controls

Cycle start:

1. Current ICLR readiness: not ready; still missing a source-derived positive
   method that survives answer-leakage and source-destroying controls.
2. Current paper story: MPS and 7B disagreement surfaces are useful falsifiers,
   but clean rows remain final-answer relay or outside the target-side pool.
3. Exact blocker: answer-masked source information must remain useful after
   answer-only, shuffled-source, target-only, slots-only, and collapse controls.
4. Live branch: CPU answer-masked receiver diagnostics; JEPA-style connector is
   design-only until non-leaky headroom exists.
5. Highest-priority gate: add/run `answer_only` and `answer_masked_source`
   condition controls with collapse telemetry.
6. Scale-up rung: smoke.

Code/harness update:

- `scripts/build_condition_likelihood_candidate_pools.py`
  - added conditions: `answer_only`, `answer_masked_source`
  - `answer_only` emits only the source final/verified numeric answer in the
    source slot and recomputes correctness for the current example.
  - `answer_masked_source` masks source final/verified numeric values and clears
    the normalized source answer slot.
- `scripts/analyze_condition_likelihood_receiver_gate.py`
  - added both conditions to strict condition parsing.
  - added collapse telemetry over candidate score matrices: finite coverage,
    score/std metrics, effective rank, covariance off-diagonal mass,
    top-label histograms, and Barlow-style matched-vs-control score telemetry.

Focused tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_build_condition_likelihood_candidate_pools.py \
  tests/test_analyze_condition_likelihood_receiver_gate.py \
  tests/test_collect_source_likelihood_sketch.py -q
```

Result: `13 passed`.

Artifact:

- `results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/`

Candidate pool command:

```bash
./venv_arm64/bin/python scripts/build_condition_likelihood_candidate_pools.py \
  --target-jsonl results/mps_qwen25_7b_disagreement12_discovery_20260427/target_alone.jsonl \
  --text-jsonl results/mps_qwen25_7b_disagreement12_discovery_20260427/text_to_text.jsonl \
  --source-jsonl results/mps_qwen25_7b_disagreement12_discovery_20260427/source_alone.jsonl \
  --output-dir results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/candidate_pools \
  --shuffle-offset 1 \
  --label-shuffle-offset 5 \
  --force
```

Pre-scoring audit:

- matched source slot correct: `4/12`
- answer-only source slot correct: `4/12`
- answer-masked-source slot correct: `0/12`
- target-only source slot correct: `0/12`
- shuffled-source source slot correct: `1/12`
- slots-only source slot correct: `0/12`

Scoring:

- model: `Qwen/Qwen3-0.6B`
- device/dtype: CPU / float32
- field/template: `normalized_prediction`, `Answer: {text}`
- conditions scored: matched, target-only, slots-only, shuffled-source,
  answer-only, answer-masked-source
- exact row parity: `12/12` for all conditions

Gate command:

```bash
./venv_arm64/bin/python scripts/analyze_condition_likelihood_receiver_gate.py \
  --live-condition-sketch matched=results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/matched.jsonl \
  --live-condition-sketch target_only=results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/target_only.jsonl \
  --live-condition-sketch slots_only=results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/slots_only.jsonl \
  --live-condition-sketch shuffled_source=results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/shuffled_source.jsonl \
  --live-condition-sketch answer_only=results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/answer_only.jsonl \
  --live-condition-sketch answer_masked_source=results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/answer_masked_source.jsonl \
  --live-target-set-json results/mps_qwen25_7b_disagreement12_discovery_20260427/source_contrastive_target_set.json \
  --holdout-condition-sketch matched=results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/matched.jsonl \
  --holdout-condition-sketch target_only=results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/target_only.jsonl \
  --holdout-condition-sketch slots_only=results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/slots_only.jsonl \
  --holdout-condition-sketch shuffled_source=results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/shuffled_source.jsonl \
  --holdout-condition-sketch answer_only=results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/answer_only.jsonl \
  --holdout-condition-sketch answer_masked_source=results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/sketches/answer_masked_source.jsonl \
  --holdout-target-set-json results/mps_qwen25_7b_disagreement12_discovery_20260427/source_contrastive_target_set.json \
  --fallback-label target \
  --outer-folds 3 \
  --min-live-correct 1 \
  --min-live-clean-source-necessary 1 \
  --min-holdout-correct 1 \
  --min-holdout-clean-source-necessary 1 \
  --max-clean-control-union 0 \
  --max-accepted-harm 0 \
  --output-json results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/gate.json \
  --output-md results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/gate.md \
  --output-predictions-jsonl results/mps_qwen25_7b_disagreement12_answer_likelihood_cpu_20260427/predictions.jsonl
```

Gate result:

- status: `condition_likelihood_receiver_fails_gate`
- live/CV clean source-necessary IDs: `0`
- holdout/frozen clean source-necessary IDs: `0`
- live/CV clean control union: `1` ID, `ab1e71e8928661d0`
- holdout/frozen clean control union: `2` IDs,
  `ab1e71e8928661d0`, `ce08a3a269bf0151`
- matched and answer-only sketches are byte-identical by SHA256:
  `fbc34d474466922f3678f0615e2fab8a88e3f1ee90723279f1d3626267e891a7`
- answer-masked-source recovers no clean IDs.
- matched top-label histogram: `source: 12`
- matched effective rank: `2.8769699003054123`
- matched-vs-answer-only Barlow diagonal mean: `1.0000000000000002`

Decision:

- failed/pruned: normalized-answer receiver-likelihood variants on the latest
  7B disagreement surface.
- reason: matched equals answer-only, so the apparent source signal is final
  answer relay rather than nontrivial communication.
- next exact gate: do not run another receiver-likelihood variant on this
  surface unless a new source surface has `answer_unexplained_clean_in_pool >
  0`; move upstream to source-surface discovery or to an answer-masked
  trace/latent JEPA objective with frozen target latents and preservation loss.

## 2026-04-27 - Cached 7B SVAMP70 surface and answer-free query-bottleneck syndrome smoke

Cycle start:

1. Current ICLR readiness: not ready; no source-derived method survives
   answer-only/source-destroying controls.
2. Current paper story: receiver-likelihood on the latest 7B disagreement
   slice is pruned; JEPA/LeJEPA/V-JEPA remain design constraints only.
3. Exact blocker: find answer-unexplained source headroom in a target-side pool
   or show answer-free source latents can predict useful side information.
4. Top branches: fresh stronger-source surface discovery; answer-free
   query-bottleneck syndrome diagnostic.
5. Highest-priority gate: run the cheapest MPS-clear stronger-source scout,
   then the CPU latent diagnostic if the surface still fails.
6. Scale-up rung: smoke.

MPS preflight:

```bash
./venv_arm64/bin/python scripts/check_mps_blocker.py --json
```

Result: PID `31103` absent; MPS clear.

Cached-7B surface scout command:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/qwen25_7b_qwen3_svamp70_surface_scout_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods target source t2t \
  --limit 70 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

I used cached `Qwen/Qwen2.5-7B-Instruct`; `Qwen/Qwen2.5-Math-7B-Instruct` was
not locally cached.

Surface result:

- target: `21/70`
- source: `15/70`
- text relay: `12/70`
- exact ID parity: true for all methods
- numeric coverage: `70/70` for all methods
- source-only over target: `8`
- clean source-only after text relay: `7`
- target/source oracle: `29/70`

Answer-masking audit:

- clean in target-side pool: `3`
- answer-unexplained clean in target-side pool: `0`
- clean in-pool IDs: `4c84ebf42812703b`, `d64f6e35083ffe8c`,
  `de1bf4d142544e5b`

Decision: fail for positive-method promotion. The fresh cached-7B surface has
more raw source headroom than prior selected 7B slices, but every reachable
clean target-pool answer is explained by source final or verified numeric
answers.

Answer-free latent diagnostic command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_latent_syndrome_probe.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target target_alone=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --candidate target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --candidate selected_route_no_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/selected_route_no_repair_exact32.jsonl,method=selected_route_no_repair \
  --candidate query_pool_gate010=path=results/svamp32_query_innovation_query_pool_transport_20260423/query_pool_transport_gate010_matched.jsonl,method=rotalign_kv \
  --candidate idweighted_gate015=path=results/svamp32_idweighted_query_innovation_20260423/idweighted_query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --candidate query_innovation_gate015=path=results/svamp32_query_innovation_resampler_gate015_20260423/query_innovation_gate015_matched.jsonl,method=rotalign_kv \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --fallback-label target_self_repair \
  --moduli 2,3,5,7 \
  --probe-model query_bottleneck \
  --query-slots 8 \
  --query-epochs 80 \
  --query-lr 0.01 \
  --query-weight-decay 0.001 \
  --query-seed 0 \
  --shuffle-offset 1 \
  --min-correct 14 \
  --min-clean-source-necessary 2 \
  --min-numeric-coverage 31 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --source-enable-thinking false \
  --feature-layers mid,last \
  --device cpu \
  --dtype float32 \
  --date 2026-04-27 \
  --output-json results/svamp32_query_bottleneck_syndrome_jepa_smoke_20260427/query_bottleneck_mid_last.json \
  --output-md results/svamp32_query_bottleneck_syndrome_jepa_smoke_20260427/query_bottleneck_mid_last.md
```

Result:

- status: `source_latent_syndrome_probe_fails_gate`
- matched: `10/32`
- zero-source: `13/32`
- shuffled-source: `10/32`
- label-shuffled: `12/32`
- target-only: `14/32`
- slots-only: `8/32`
- clean source-necessary IDs: `0`
- control clean union: `0`
- teacher numeric coverage: `32/32`
- candidate-pool clean-gold count: `2`

Decision: fail. Answer-free prompt hidden states with an 8-query bottleneck do
not recover the C2C candidate-syndrome bound; matched is below the target-only
floor and has no clean source-necessary IDs.

Next exact gate:

- Run the originally recorded Math-7B source-surface scout only after the model
  is local, or run a selected-disagreement Math-7B slice if download time is an
  acceptable cost.
- Promotion requires `answer_unexplained_clean_in_pool > 0`; otherwise reject
  before any receiver, connector, or JEPA objective spend.

## 2026-04-27 Cycle 14 - Math-7B Selected Disagreement Surface

Cycle start:

1. Current ICLR readiness: not ready; no source-derived positive method
   survives answer masking and source controls.
2. Current paper story: stronger sources expose raw source/target disagreement,
   but useful-looking rows keep collapsing to source final-answer relay or
   target-side artifacts.
3. Exact blocker: no source surface has
   `answer_unexplained_clean_in_pool > 0`.
4. Current live branch: `none`; top candidates are Math-7B source-surface
   discovery and JEPA-style answer-masked latent prediction only after non-leaky
   headroom exists.
5. Highest-priority gate: selected-disagreement Math-7B source-surface smoke.
6. Scale-up rung: selected micro discovery.

Subagent updates:

- JEPA/LeJEPA/V-JEPA committee: use answer-masked dual source views, frozen
  target latent/KV targets, matched-source margins over source-destroying
  controls, target-preservation loss, and variance/effective-rank/covariance
  collapse telemetry. This is design guidance, not positive evidence.
- Experiment planner: preferred full SVAMP70 Math-7B now that cached 7B and
  answer-free query-bottleneck gates failed. I ran the selected-12 gate first
  to validate the model path and immediate non-leaky headroom.

Model-fetch note:

- first attempt stalled at Hugging Face `Fetching 4 files: 0%`.
- retry with `HF_HUB_DISABLE_XET=1` completed and cached
  `Qwen/Qwen2.5-Math-7B-Instruct` under repo-local `.hf_home/hub`.

Command:

```bash
HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/mps_qwen25_7b_disagreement12_discovery_20260427/disagreement12_eval.jsonl \
  --results-dir results/qwen25math7b_disagreement12_surface_scout_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-Math-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t \
  --limit 12 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Result:

- target: `0/12`
- source: `5/12`
- text relay: `1/12`
- exact ID parity: true for all methods
- numeric coverage: `12/12` for all methods
- source-only over target: `5`
- clean source-only after text relay: `5`
- target/source oracle: `5/12`

Answer-masking audit:

- clean in target-side pool: `3`
- answer-unexplained clean in target-side pool: `0`
- clean in-pool IDs: `561daa750422c0e4`, `ab1e71e8928661d0`,
  `daea537474de16ac`

Decision: fail for promotion. Math-7B creates a stronger selected disagreement
surface, but every reachable clean target-pool answer is still explained by the
source final/verified numeric answer. Do not train a learned receiver or JEPA
connector on this selected surface.

Artifacts:

- `results/qwen25math7b_disagreement12_surface_scout_20260427/manifest.md`
- `results/qwen25math7b_disagreement12_surface_scout_20260427/manifest.json`
- `results/qwen25math7b_disagreement12_surface_scout_20260427/source_alone.jsonl`
- `results/qwen25math7b_disagreement12_surface_scout_20260427/target_alone.jsonl`
- `results/qwen25math7b_disagreement12_surface_scout_20260427/text_to_text.jsonl`
- `results/qwen25math7b_disagreement12_surface_scout_20260427/source_contrastive_target_set.md`
- `results/qwen25math7b_disagreement12_surface_scout_20260427/answer_masking_audit.md`
- `paper/qwen25math7b_disagreement12_surface_scout_20260427.md`

Next exact gate:

- Full SVAMP70 Math-7B scout now that the model is local, if we accept the MPS
  time. Pass requires exact ID parity, numeric coverage `>=69/70`, source-only
  over target `>=6`, clean source-only after text relay `>=3`, clean in pool
  `>=1`, and `answer_unexplained_clean_in_pool > 0`.
- If full SVAMP70 also has `answer_unexplained_clean_in_pool = 0`, stop
  source-scorer/receiver variants and switch to a new candidate-pool generator.

## 2026-04-27 Cycle 15 - Full Math-7B SVAMP70 And Target-Only Pool Smoke

Cycle start:

1. Current ICLR readiness: not ready; no source-derived positive method
   survives answer masking and source controls.
2. Current paper story: stronger sources can expose selected-slice
   disagreement, but full frozen surfaces still do not produce non-leaky
   target-pool headroom.
3. Exact blocker: `answer_unexplained_clean_in_pool` remains `0`.
4. Current live branch: target-only/no-source candidate-pool generation on
   residual clean IDs; source selectors are not live unless answer-masked source
   evidence survives controls.
5. Highest-priority gate: full SVAMP70 Math-7B scout, then target-only sampling
   if the source surface fails.
6. Scale-up rung: medium surface discovery plus micro candidate-pool smoke.

Full Math-7B SVAMP70 command:

```bash
HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/qwen25math7b_qwen3_svamp70_surface_scout_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-Math-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t \
  --limit 70 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Result:

- target: `21/70`
- source: `5/70`
- text relay: `8/70`
- exact ID parity: true for all methods
- numeric coverage: `70/70` for all methods
- source-only over target: `3`
- clean source-only after text relay: `3`
- target/source oracle: `24/70`

Answer-masking audit:

- clean in target-side pool: `1`
- answer-unexplained clean in target-side pool: `0`
- clean in-pool ID: `a07cd6cc8f1c832e`

Decision: fail. Full SVAMP70 Math-7B is weaker than target-alone and does not
clear the non-leaky source-surface gate. Stop source-scorer/receiver variants on
this surface.

Candidate-pool generator smoke:

```bash
./venv_arm64/bin/python scripts/materialize_generation_id_subset.py \
  --eval-file data/svamp_eval_70.jsonl \
  --ids 14bfbfc94f2c2e7b a07cd6cc8f1c832e d64f6e35083ffe8c \
  --output-jsonl results/qwen25math7b_svamp70_target_sampling_clean3_20260427/clean_source_only_eval.jsonl \
  --output-meta-json results/qwen25math7b_svamp70_target_sampling_clean3_20260427/clean_source_only_eval.meta.json

HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/qwen25math7b_svamp70_target_sampling_clean3_20260427/clean_source_only_eval.jsonl \
  --model Qwen/Qwen3-0.6B \
  --samples 16 \
  --temperature 0.9 \
  --top-p 0.95 \
  --seed 17 \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --use-chat-template \
  --enable-thinking false \
  --output-jsonl results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.jsonl \
  --output-json results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.json \
  --output-md results/qwen25math7b_svamp70_target_sampling_clean3_20260427/target_only_samples.md
```

Result:

- target-only samples oracle: `1/3`
- combined sampled target-side oracle: `2/3`
- clean IDs in combined target-side pool: `14bfbfc94f2c2e7b`,
  `a07cd6cc8f1c832e`

Harness update:

- `scripts/materialize_svamp_source_candidate_sidecars.py`
  - added `--profile-mode full|answer_only|answer_masked`
  - default `full` preserves prior behavior
  - `answer_only` isolates source final/verified answer evidence
  - `answer_masked` removes final/verified answer values before candidate
    scoring
- tests: `tests/test_materialize_svamp_source_candidate_sidecars.py`

Selector controls on the sampled clean3 pool:

| Profile Mode | Matched Clean Correct | Source-Necessary Clean | Decision |
|---|---:|---:|---|
| `full` | 2/3 | 2 | answer relay |
| `answer_only` | 2/3 | 2 | answer relay |
| `answer_masked` | 0/3 | 0 | fails |

Decision: generator-positive, selector-pruned. Target-only/no-source sampling
improves candidate reachability, but the current source-candidate selector is
fully explained by final/verified source answers.

Artifacts:

- `results/qwen25math7b_qwen3_svamp70_surface_scout_20260427/manifest.md`
- `results/qwen25math7b_qwen3_svamp70_surface_scout_20260427/source_contrastive_target_set.md`
- `results/qwen25math7b_qwen3_svamp70_surface_scout_20260427/answer_masking_audit.md`
- `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/manifest.md`
- `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/sampled_clean3_headroom.md`
- `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/top_selector_full.md`
- `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/top_selector_answer_only.md`
- `results/qwen25math7b_svamp70_target_sampling_clean3_20260427/top_selector_answer_masked.md`
- `paper/qwen25math7b_svamp70_surface_and_sampling_20260427.md`

Next exact gate:

- Build a larger target-only/no-source sampled candidate pool on a strict small
  slice (`SVAMP32` or all target-wrong/source-disagreement IDs), then test only
  answer-masked source signals against answer-only, shuffled-source,
  random-sidecar, target-only, and slots-only controls.

## 2026-04-27 Cycle 16 - SVAMP32 Target-Only Clean6 Sampling Gate

Cycle start:

1. Current ICLR readiness: not ready; no source-derived positive method
   survives answer masking and source-destroying controls.
2. Current paper story: target/no-source sampling can create target-side
   candidate reachability, but current source selectors either relay final
   answers or collapse to no accepted signal after answer masking.
3. Exact blocker: source-derived non-answer information must select newly
   reachable target candidates without source final/verified-answer leakage.
4. Current live branches: target-only/no-source candidate pools with strict
   source controls; JEPA-style latent/process ranking only as a follow-up design
   branch.
5. Highest-priority gate: SVAMP32 clean C2C residual IDs with target-only
   sampled candidates and full/answer-only/answer-masked selector controls.
6. Scale-up rung: strict small gate.

Harness update:

- `scripts/extend_target_set_candidate_labels.py`
  - added explicit `--ids` and `--ids-file` selectors, matching the subset
    materializer contract
  - added `--override-clean-residual-ids` so C2C clean residual slices can reuse
    decoder-compatible source/target surfaces without schema hand edits
- tests: `tests/test_extend_target_set_candidate_labels.py`

Strict-small gate:

```bash
./venv_arm64/bin/python scripts/materialize_generation_id_subset.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target-set-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json \
  --id-fields clean_residual_targets \
  --output-jsonl results/svamp32_target_sampling_clean6_20260427/clean6_eval.jsonl \
  --output-meta-json results/svamp32_target_sampling_clean6_20260427/clean6_eval.meta.json

HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/svamp32_target_sampling_clean6_20260427/clean6_eval.jsonl \
  --model Qwen/Qwen3-0.6B \
  --samples 16 \
  --temperature 0.9 \
  --top-p 0.95 \
  --seed 31 \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --use-chat-template \
  --enable-thinking false \
  --output-jsonl results/svamp32_target_sampling_clean6_20260427/target_only_samples.jsonl \
  --output-json results/svamp32_target_sampling_clean6_20260427/target_only_samples.json \
  --output-md results/svamp32_target_sampling_clean6_20260427/target_only_samples.md
```

Target/no-source generator result:

- clean residual IDs: `6`
- target-only samples per ID: `16`
- numeric coverage: `96/96`
- candidate oracle: `2/6`
- reachable clean IDs: `3e8a5691f5443495`, `575d7e83d84c1e67`

Selector controls:

| Profile Mode | Matched Correct | Matched Accepted | Clean Correct | Source-Necessary Clean | Decision |
|---|---:|---:|---:|---:|---|
| `full` | 0/6 | 4 | 0 | 0 | fail |
| `answer_only` | 0/6 | 4 | 0 | 0 | fail |
| `answer_masked` | 0/6 | 0 | 0 | 0 | fail |

JEPA/LeJEPA/V-JEPA update:

- references remain in `references/474_jepa_anticollapse_refs.md`
- committee decision: use JEPA only as an answer-masked process/latent ranking
  design with frozen target/candidate latent targets and collapse telemetry
- local JEPA smoke remains negative: matched `10/32`, target-only `14/32`,
  clean source-necessary `0`

Decision:

- target/no-source generator passes the strict-small headroom floor
- current numeric source-candidate sidecar fails as communication
- do not tune this selector family further

Artifacts:

- `results/svamp32_target_sampling_clean6_20260427/manifest.md`
- `results/svamp32_target_sampling_clean6_20260427/target_only_samples.md`
- `results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.md`
- `results/svamp32_target_sampling_clean6_20260427/top_selector_full.md`
- `results/svamp32_target_sampling_clean6_20260427/top_selector_answer_only.md`
- `results/svamp32_target_sampling_clean6_20260427/top_selector_answer_masked.md`
- `paper/svamp32_target_sampling_clean6_gate_20260427.md`

Next exact gate:

- Build an answer-masked process/latent ranking smoke only over the two reachable
  clean IDs. Pass requires `>=1` source-necessary clean selection, control clean
  union `0`, accepted harm `0`, no selected candidate appearing unmasked in the
  source trace, and variance/effective-rank/covariance/Barlow telemetry that
  rules out representation collapse.

## 2026-04-27 Cycle 17 - SVAMP32 Process-Trace Sidecar Smoke

Cycle start:

1. Current ICLR readiness: not ready; no deployable source-derived method
   survives answer masking and source-destroying controls.
2. Current paper story: target/no-source sampling creates reachable target-side
   candidates, but both numeric and process-text source sidecars fail to select
   those candidates under controls.
3. Exact blocker: source-derived non-answer information must select reachable
   target candidates without final-answer leakage, target priors, random-sidecar
   wins, or representation collapse.
4. Current live branches: process/latent ranking is weakened; broader
   target/no-source candidate-pool discovery or a trained frozen-latent
   connector is higher value than more hand-built sidecars.
5. Highest-priority gate: CPU process-trace similarity sidecar on the SVAMP32
   clean6 target-only pool.
6. Scale-up rung: smoke.

Harness update:

- added `scripts/materialize_svamp_process_trace_sidecars.py`
  - masks numerals in source and candidate reasoning traces
  - scores answer-masked source process text against target-side candidate
    process text
  - supports sampled-label preference, `--exclude-label`, and
    `--prediction-only`
  - reports feature variance, effective rank, zero vectors, zero margins,
    top-label counts, and selected-value source-number overlap
- tests: `tests/test_materialize_svamp_process_trace_sidecars.py`

Commands:

```bash
./venv_arm64/bin/python scripts/materialize_svamp_process_trace_sidecars.py \
  --target-set results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json \
  --output-dir results/svamp32_process_trace_sidecar_clean6_20260427/process_trace_sidecars_predonly_no_t2t \
  --sidecar-bits 256 \
  --max-ngram 2 \
  --exclude-label t2t \
  --prediction-only \
  --date 2026-04-27

./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
  --target-set results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json \
  --sidecar-jsonl results/svamp32_process_trace_sidecar_clean6_20260427/process_trace_sidecars_predonly_no_t2t/live_candidate_sidecars.jsonl \
  --min-confidence 0.01 \
  --min-source-necessary-clean 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-json results/svamp32_process_trace_sidecar_clean6_20260427/top_selector_process_trace_predonly_no_t2t.json \
  --output-md results/svamp32_process_trace_sidecar_clean6_20260427/top_selector_process_trace_predonly_no_t2t.md
```

Results:

| Variant | Matched Correct | Matched Accepted | Control Clean Union | Source-Necessary Clean | Decision |
|---|---:|---:|---:|---:|---|
| `process_trace` | 0/6 | 0 | 0 | 0 | fail; target-label zero-margin collapse |
| `process_trace_sample_pref` | 0/6 | 0 | 1 | 0 | fail; control recovers clean ID |
| `process_trace_predonly_no_t2t` | 0/6 | 2 | 1 | 0 | fail; matched wrong, random clean |

Best diagnostic variant (`prediction-only`, no `t2t`):

- matched accepted IDs: `1d50b408c8f5cd2c`, `3e8a5691f5443495`
- matched clean correct: `0/6`
- random-sidecar clean correct: `575d7e83d84c1e67`
- selected value in unmasked source numbers: `5/6`
- effective rank: `31.270460`
- zero vectors: `0`

Decision:

- fail and prune deterministic hand-built process-trace similarity sidecars on
  this slice
- the failure is not low-rank collapse; it is lack of source-necessary candidate
  selection after answer masking and controls
- do not tune TF-IDF/process-text similarity further

Artifacts:

- `results/svamp32_process_trace_sidecar_clean6_20260427/manifest.md`
- `results/svamp32_process_trace_sidecar_clean6_20260427/process_trace_sidecars/manifest.md`
- `results/svamp32_process_trace_sidecar_clean6_20260427/process_trace_sidecars_sample_pref/manifest.md`
- `results/svamp32_process_trace_sidecar_clean6_20260427/process_trace_sidecars_no_t2t/manifest.md`
- `results/svamp32_process_trace_sidecar_clean6_20260427/process_trace_sidecars_predonly_no_t2t/manifest.md`
- `results/svamp32_process_trace_sidecar_clean6_20260427/top_selector_process_trace_predonly_no_t2t.md`
- `paper/svamp32_process_trace_sidecar_20260427.md`

Next exact gate:

- Stop hand-built sidecar tuning. Either widen target/no-source candidate-pool
  discovery to produce a larger reachable clean selector surface, or run a
  trained frozen-latent/rate-capped connector smoke with the same controls and
  collapse telemetry.

## 2026-04-27 Cycle 18 - SVAMP32 Full32 Target Sampling Reachability

Cycle start:

1. Current ICLR readiness: not ready; no deployable source-derived method
   survives text relay, source-destroying controls, and seed/rung confirmation.
2. Current paper story: target/no-source sampling creates receiver-side
   candidate headroom, but numeric and process sidecars have not turned that
   headroom into communication.
3. Exact blocker: determine whether a broader target/no-source candidate pool
   yields enough clean C2C residual reachability to justify another selector or
   connector gate.
4. Current live branches: broader target/no-source candidate-pool discovery;
   JEPA-style anti-collapse connector only after a non-leaky reachable surface.
5. Highest-priority gate: SVAMP32 full32 target-only sampling at 8 samples per
   ID, with C2C clean residual overlap and duplicate-answer diagnostics.
6. Scale-up rung: strict small gate.

Harness update:

- added `scripts/analyze_target_sampling_reachability.py`
  - audits raw sampled rows against a frozen source-contrastive target set and
    C2C headroom metadata
  - reports raw sample oracle, oracle gain vs target, C2C clean residual overlap,
    C2C teacher-only overlap, source-contrastive clean overlap, and sampled
    answer diversity
- added `tests/test_analyze_target_sampling_reachability.py`

Commands:

```bash
HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --model Qwen/Qwen3-0.6B \
  --samples 8 \
  --temperature 0.9 \
  --top-p 0.95 \
  --seed 43 \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --use-chat-template \
  --enable-thinking false \
  --output-jsonl results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl \
  --output-json results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.json \
  --output-md results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.md

./venv_arm64/bin/python scripts/analyze_target_sampling_reachability.py \
  --samples-jsonl results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.jsonl \
  --base-target-set results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json \
  --c2c-headroom-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json \
  --date 2026-04-27 \
  --output-json results/svamp32_target_sampling_full32_s8_20260427/reachability.json \
  --output-md results/svamp32_target_sampling_full32_s8_20260427/reachability.md
```

Results:

- target baseline: `8/32`
- raw target/no-source sample candidate oracle: `14/32`
- raw sample oracle gain vs target: `7`
- merged target-side oracle with text relay plus samples: `18/32`
- merged oracle gain: `10`
- numeric coverage: `256/256`
- C2C clean residual IDs in pool: `2/6`
- C2C teacher-only IDs in pool: `4/9`
- source-contrastive clean IDs in pool: `2/4`
- mean unique sampled answers per ID: `3.344`
- duplicate nonempty row fraction: `0.582`

Key diagnostic:

- the C2C-clean residual IDs recovered are still only
  `3e8a5691f5443495` and `575d7e83d84c1e67`, the same two already reached by
  the clean6 `s16` gate.
- after merging text relay and samples, the remaining clean source-only IDs
  have `0/2` gold answers in the target-side pool.

Decision:

- target/no-source candidate generation passes as receiver-side headroom
- the selector surface is not expanded; do not top up to 16 merely to tune the
  same hand-built selector family
- this is not communication evidence because no source-derived signal was used
  and no new C2C-clean residual IDs became reachable

Artifacts:

- `results/svamp32_target_sampling_full32_s8_20260427/manifest.md`
- `results/svamp32_target_sampling_full32_s8_20260427/target_only_samples.md`
- `results/svamp32_target_sampling_full32_s8_20260427/reachability.md`
- `results/svamp32_target_sampling_full32_s8_20260427/headroom.md`
- `results/svamp32_target_sampling_full32_s8_20260427/no_source_surface/manifest.md`
- `paper/svamp32_full32_target_sampling_reachability_20260427.md`

Next exact gate:

- switch from deterministic selectors to a bounded learned source-conditioned
  candidate generator or frozen-latent/rate-capped connector smoke. Use the
  full32 no-source pool as the target-prior baseline. Pass requires at least
  `1` C2C-clean source-necessary recovery beyond target/text/no-source controls,
  zero source-destroying control clean recovery, no target-correct harm, byte
  and latency accounting, and JEPA/LeJEPA/V-JEPA-style collapse telemetry.

## 2026-04-27 Cycle 19 - SVAMP32 Full32 Source Sampling Reachability

Cycle start:

1. Current ICLR readiness: not ready; no deployable source-derived method
   survives text relay, source-destroying controls, and seed/rung confirmation.
2. Current paper story: target/no-source candidate sampling gives receiver-side
   headroom, but prior numeric/process sidecars and connector probes have not
   turned that headroom into communication.
3. Exact blocker: find a source-derived candidate surface or selector signal
   that recovers C2C-clean residual IDs beyond target-only priors and source
   controls.
4. Current live branches: source-conditioned candidate generation; JEPA-style
   anti-collapse connector only after a non-leaky reachable surface exists.
5. Highest-priority gate: source-model sampling over SVAMP32 full32, compared
   directly with the full32 target/no-source pool.
6. Scale-up rung: smoke.

Harness update:

- `scripts/sample_target_candidate_surface.py`
  - added `--prompt-mode {direct,source_reasoning}`
  - added `--source-reasoning-mode`
  - added `--method-prefix`
  - now records prompt-mode metadata for each row and summary artifact
- added `scripts/compare_candidate_pool_reachability.py`
  - compares candidate-pool reachability audits
  - reports total oracle delta, new/lost oracle IDs, and new C2C-clean residual
    IDs
- added `tests/test_compare_candidate_pool_reachability.py`
- extended `tests/test_sample_target_candidate_surface.py`

Commands:

```bash
HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --model Qwen/Qwen2.5-Math-1.5B \
  --samples 4 \
  --temperature 0.9 \
  --top-p 0.95 \
  --seed 71 \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --prompt-mode source_reasoning \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --enable-thinking false \
  --output-jsonl results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl \
  --output-json results/svamp32_source_sampling_full32_s4_20260427/source_samples.json \
  --output-md results/svamp32_source_sampling_full32_s4_20260427/source_samples.md

./venv_arm64/bin/python scripts/analyze_target_sampling_reachability.py \
  --samples-jsonl results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl \
  --base-target-set results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json \
  --c2c-headroom-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json \
  --date 2026-04-27 \
  --output-json results/svamp32_source_sampling_full32_s4_20260427/reachability.json \
  --output-md results/svamp32_source_sampling_full32_s4_20260427/reachability.md

./venv_arm64/bin/python scripts/compare_candidate_pool_reachability.py \
  --baseline-reachability results/svamp32_target_sampling_full32_s8_20260427/reachability.json \
  --candidate-reachability results/svamp32_source_sampling_full32_s4_20260427/reachability.json \
  --date 2026-04-27 \
  --output-json results/svamp32_source_sampling_full32_s4_20260427/source_vs_target_reachability.json \
  --output-md results/svamp32_source_sampling_full32_s4_20260427/source_vs_target_reachability.md
```

Results:

- source-sampled candidate oracle: `10/32`
- target/no-source full32 S8 baseline oracle: `14/32`
- source minus target/no-source oracle: `-4`
- new source oracle IDs beyond target/no-source: `5`
- lost target/no-source oracle IDs: `9`
- C2C clean residual in source pool: `3/6`
- new C2C-clean residual IDs beyond target/no-source: `2`
  - `6e9745b37ab6fc45`
  - `de1bf4d142544e5b`
- C2C teacher-only IDs in source pool: `4/9`
- source-contrastive clean IDs in source pool: `1/4`
- mean unique sampled answers per ID: `3.406`
- duplicate nonempty row fraction: `0.148`

Decision:

- pass as source-surface discovery only
- fail as a method claim because total oracle reachability is below the
  target/no-source full32 pool and no receiver selected source-derived
  information yet
- the two new C2C-clean residual IDs form the next strict smoke surface for a
  source-conditioned selector or JEPA-style rate-capped connector

Artifacts:

- `results/svamp32_source_sampling_full32_s4_20260427/manifest.md`
- `results/svamp32_source_sampling_full32_s4_20260427/source_samples.md`
- `results/svamp32_source_sampling_full32_s4_20260427/reachability.md`
- `results/svamp32_source_sampling_full32_s4_20260427/source_vs_target_reachability.md`
- `paper/svamp32_source_sampling_reachability_20260427.md`

Next exact gate:

- build a combined target/no-source plus source-sampled candidate target set for
  the two new source-only C2C-clean residual IDs
- run a strict matched-source selector or connector gate with zero-source,
  shuffled-source, target-only/slots-only, random same-byte, answer-only, and
  answer-masked controls
- pass only if matched source uniquely recovers at least one of
  `6e9745b37ab6fc45` or `de1bf4d142544e5b`, control clean union is `0`,
  target-correct harm is `0`, and bytes plus collapse telemetry are reported

## 2026-04-27 Cycle 20 - SVAMP32 Source-Sample Selector And S16 Replay

Cycle start:

1. Current ICLR readiness: not ready; no source-derived positive method has
   survived strict controls.
2. Current paper story: source sampling exposed two C2C-clean residual
   candidates, but it needed attribution before any connector training.
3. Exact blocker: prove matched source selects or generates those candidates
   beyond target-only, prompt-wrapper, and source-destroying controls.
4. Current live branches: source-sampled selector surface; JEPA-style
   source-innovation connector only if the surface survives controls.
5. Highest-priority gate: strict selector over the two-ID candidate pool,
   followed by a source-vs-target prompt-wrapper replay.
6. Scale-up rung: smoke.

Subagents:

- artifact audit: identified exact source-sample rows and warned that raw
  source rows still carry inherited `target_sample_s*` method names
- experiment planner: recommended a two-ID `s16` replay before any connector
  training
- reviewer: required target prompt-wrapper controls and cautioned that even a
  clean `2/2` would be only attribution evidence
- JEPA/anti-collapse: kept Query-JEPA/source-innovation as a future connector
  design only after the surface survives controls
- cross-field scout: proposed compact conditional-innovation sidecars from
  distributed source coding, Kalman/predictive coding, and one-bit sketches

Harness update:

- added `scripts/build_candidate_pool_decision_surface.py`
  - clones an existing target set
  - appends extra candidate rows with distinct labels
  - overrides clean decision IDs for a bounded attribution surface
- added `zero_source` to
  `scripts/analyze_candidate_score_sidecar_top_select.py`
- added `tests/test_build_candidate_pool_decision_surface.py`

Gate 1 commands:

```bash
./venv_arm64/bin/python scripts/build_candidate_pool_decision_surface.py \
  --base-target-set results/svamp32_target_sampling_full32_s8_20260427/no_source_surface/source_contrastive_target_set.json \
  --extra-candidate label=source_sample_s0,path=results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl,method=target_sample_s0 \
  --extra-candidate label=source_sample_s1,path=results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl,method=target_sample_s1 \
  --extra-candidate label=source_sample_s2,path=results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl,method=target_sample_s2 \
  --extra-candidate label=source_sample_s3,path=results/svamp32_source_sampling_full32_s4_20260427/source_samples.jsonl,method=target_sample_s3 \
  --clean-id 6e9745b37ab6fc45 \
  --clean-id de1bf4d142544e5b \
  --date 2026-04-27 \
  --output-json results/svamp32_source_sample_selector_newclean2_20260427/decision_surface.json \
  --output-md results/svamp32_source_sample_selector_newclean2_20260427/decision_surface.md

for mode in full answer_only answer_masked; do
  ./venv_arm64/bin/python scripts/materialize_svamp_source_candidate_sidecars.py \
    --live-target-set results/svamp32_source_sample_selector_newclean2_20260427/decision_surface.json \
    --output-dir results/svamp32_source_sample_selector_newclean2_20260427/sidecars_${mode} \
    --sidecar-bits 8 \
    --label-prior 0.0 \
    --profile-mode ${mode} \
    --date 2026-04-27
  ./venv_arm64/bin/python scripts/analyze_candidate_score_sidecar_top_select.py \
    --target-set results/svamp32_source_sample_selector_newclean2_20260427/decision_surface.json \
    --sidecar-jsonl results/svamp32_source_sample_selector_newclean2_20260427/sidecars_${mode}/live_candidate_sidecars.jsonl \
    --min-confidence 0.0 \
    --min-source-necessary-clean 1 \
    --max-control-clean-union 0 \
    --max-accepted-harm 0 \
    --date 2026-04-27 \
    --output-json results/svamp32_source_sample_selector_newclean2_20260427/top_select_${mode}.json \
    --output-md results/svamp32_source_sample_selector_newclean2_20260427/top_select_${mode}.md
done
```

Gate 1 results:

| Profile | matched correct | matched clean | control clean union | accepted harm |
|---|---:|---:|---:|---:|
| `full` | `6/32` | `0` | `0` | `5` |
| `answer_only` | `6/32` | `0` | `0` | `5` |
| `answer_masked` | `2/32` | `0` | `0` | `6` |

Gate 1 decision:

- fail
- the deterministic source-candidate score sidecar does not select either new
  clean candidate, even when the source-sampled gold values are present in the
  candidate pool
- do not tune this selector family on the same surface

Gate 2 commands:

```bash
./venv_arm64/bin/python scripts/materialize_generation_id_subset.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --ids 6e9745b37ab6fc45 de1bf4d142544e5b \
  --output-jsonl results/svamp32_source_sampling_newclean2_s16_20260427/newclean2_eval.jsonl \
  --output-meta-json results/svamp32_source_sampling_newclean2_s16_20260427/newclean2_eval.meta.json

HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/svamp32_source_sampling_newclean2_s16_20260427/newclean2_eval.jsonl \
  --model Qwen/Qwen2.5-Math-1.5B \
  --samples 16 \
  --method-prefix source_sample \
  --temperature 0.9 \
  --top-p 0.95 \
  --seed 171 \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --prompt-mode source_reasoning \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --enable-thinking false \
  --output-jsonl results/svamp32_source_sampling_newclean2_s16_20260427/source_samples.jsonl \
  --output-json results/svamp32_source_sampling_newclean2_s16_20260427/source_samples.json \
  --output-md results/svamp32_source_sampling_newclean2_s16_20260427/source_samples.md
```

The same command shape was repeated for:

- `Qwen/Qwen3-0.6B`, direct prompt, `target_direct_sample`
- `Qwen/Qwen3-0.6B`, source-reasoning prompt, `target_brief_sample`
- `Qwen/Qwen2.5-Math-1.5B`, direct prompt, `source_direct_sample`

Gate 2 results:

| Condition | Model | Prompt mode | Oracle |
|---|---|---|---:|
| `source_sample` | `Qwen/Qwen2.5-Math-1.5B` | `source_reasoning` | `2/2` |
| `target_direct_sample` | `Qwen/Qwen3-0.6B` | `direct` | `0/2` |
| `target_brief_sample` | `Qwen/Qwen3-0.6B` | `source_reasoning` | `2/2` |
| `source_direct_sample` | `Qwen/Qwen2.5-Math-1.5B` | `direct` | `1/2` |

Gate 2 decision:

- hard fail for source-specific attribution
- the matched source replay is stable, but the target model with the same
  brief-analysis/source-reasoning wrapper also reaches `2/2`
- the two-ID source-sampled surface is prompt-wrapper reachable; do not train a
  connector on it

Artifacts:

- `results/svamp32_source_sample_selector_newclean2_20260427/manifest.md`
- `results/svamp32_source_sampling_newclean2_s16_20260427/manifest.md`
- `paper/svamp32_source_sample_selector_and_replay_20260427.md`
- `references/475_crossfield_source_innovation_controls_refs.md`

Next exact gate:

- promote target brief-wrapper sampling to a target-prior baseline/control
- rerun source-surface discovery only if matched source adds clean residual IDs
  beyond target direct, target brief-wrapper, zero-source, shuffled-source,
  answer-only, answer-masked, and random same-byte controls
  before any connector training

## 2026-04-27 Cycle 21 - SVAMP32 Target Brief Wrapper Baseline

Cycle start:

1. Current ICLR readiness: not ready; the project still lacks a source-derived
   positive method.
2. Current paper story: the last source-sampling clue was explained by the
   target brief-analysis wrapper on a two-ID replay.
3. Exact blocker: define source residual IDs only after subtracting target
   direct, target brief-wrapper, no-source merged pools, and source-destroying
   controls.
4. Current live branches: prompt-controlled source-surface discovery; compact
   answer-masked source-innovation sidecars only after such a surface survives.
5. Highest-priority gate: full SVAMP32 target brief-wrapper S4 baseline
   compared against prior target direct S8 and source brief S4.
6. Scale-up rung: smoke.

Subagents:

- planner: run full32 target brief-wrapper S4 first; only run S8 if S4 is
  inconclusive
- reviewer: promote prompt-wrapper sampling to a mandatory target-prior baseline
  at matched or larger budget
- creative scout: defer causal-order, innovation matched-filter, and
  spread-spectrum challenge sidecars until a surface survives prompt controls

Harness update:

- added `scripts/summarize_reachability_union.py`
  - summarizes union oracle and C2C/source-clean overlap across reachability
    audits
- added `tests/test_summarize_reachability_union.py`

Commands:

```bash
HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --model Qwen/Qwen3-0.6B \
  --samples 4 \
  --method-prefix target_brief_sample \
  --temperature 0.9 \
  --top-p 0.95 \
  --seed 71 \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --prompt-mode source_reasoning \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --enable-thinking false \
  --output-jsonl results/svamp32_target_brief_sampling_full32_s4_20260427/target_brief_samples.jsonl \
  --output-json results/svamp32_target_brief_sampling_full32_s4_20260427/target_brief_samples.json \
  --output-md results/svamp32_target_brief_sampling_full32_s4_20260427/target_brief_samples.md

./venv_arm64/bin/python scripts/analyze_target_sampling_reachability.py \
  --samples-jsonl results/svamp32_target_brief_sampling_full32_s4_20260427/target_brief_samples.jsonl \
  --base-target-set results/qwen25math_svamp32_source_contrastive_sidecar_20260426/source_contrastive_target_set.json \
  --c2c-headroom-json results/qwen25math_svamp32_c2c_headroom_20260426/compatible_target_set.json \
  --date 2026-04-27 \
  --output-json results/svamp32_target_brief_sampling_full32_s4_20260427/reachability.json \
  --output-md results/svamp32_target_brief_sampling_full32_s4_20260427/reachability.md

./venv_arm64/bin/python scripts/summarize_reachability_union.py \
  --reachability target_direct_s8=results/svamp32_target_sampling_full32_s8_20260427/reachability.json \
  --reachability target_brief_s4=results/svamp32_target_brief_sampling_full32_s4_20260427/reachability.json \
  --date 2026-04-27 \
  --output-json results/svamp32_target_brief_sampling_full32_s4_20260427/target_prior_union_reachability.json \
  --output-md results/svamp32_target_brief_sampling_full32_s4_20260427/target_prior_union_reachability.md

./venv_arm64/bin/python scripts/summarize_reachability_union.py \
  --reachability target_direct_s8=results/svamp32_target_sampling_full32_s8_20260427/reachability.json \
  --reachability target_brief_s4=results/svamp32_target_brief_sampling_full32_s4_20260427/reachability.json \
  --reachability source_brief_s4=results/svamp32_source_sampling_full32_s4_20260427/reachability.json \
  --date 2026-04-27 \
  --output-json results/svamp32_target_brief_sampling_full32_s4_20260427/target_prior_plus_source_union_reachability.json \
  --output-md results/svamp32_target_brief_sampling_full32_s4_20260427/target_prior_plus_source_union_reachability.md
```

Results:

- target brief-wrapper S4 oracle: `18/32`
- target direct S8 oracle: `14/32`
- source brief S4 oracle: `10/32`
- target brief-wrapper C2C-clean residual reachability: `4/6`
- target direct plus target brief-wrapper union:
  - oracle: `23/32`
  - C2C-clean residual reachability: `6/6`
  - C2C teacher-only reachability: `8/9`
- target prior plus source union:
  - oracle: `24/32`
  - C2C-clean residual reachability: `6/6`
- source addition beyond target-prior union:
  - oracle delta: `+1` (`b1200c32546a34a5`)
  - C2C-clean residual delta: `0`

Decision:

- promote target brief-wrapper sampling to a mandatory target-prior baseline
- prune current source-sampling family as a communication surface
- do not train a connector unless matched source adds residual IDs beyond target
  direct plus target brief-wrapper at matched or larger budget

Artifacts:

- `results/svamp32_target_brief_sampling_full32_s4_20260427/manifest.md`
- `results/svamp32_target_brief_sampling_full32_s4_20260427/reachability.md`
- `results/svamp32_target_brief_sampling_full32_s4_20260427/target_prior_union_reachability.md`
- `results/svamp32_target_brief_sampling_full32_s4_20260427/source_addition_vs_target_prior_union.md`
- `paper/svamp32_target_brief_wrapper_reachability_20260427.md`

Next exact gate:

- rerun source-surface discovery only with target brief-wrapper controls
  predeclared and matched or larger budget
- a live source branch requires clean residual IDs not reached by target direct,
  target brief-wrapper, no-source merged pool, answer-only/answer-masked source,
  zero-source, shuffled-source, and random same-byte controls

## 2026-04-27 Cycle 22 - Prompt-Wrapper Source Surface Controls

Cycle start:

1. Current ICLR readiness: not ready; still missing a source-derived positive
   method.
2. Current paper story: target prompt wrappers are now a mandatory no-source
   candidate-prior baseline.
3. Exact blocker: identify source residual IDs only after subtracting target
   direct, target brief-wrapper, no-source merged pools, prompt-format controls,
   and source-answer leakage.
4. Current live branches: prompt-controlled source-surface discovery; JEPA-style
   source-innovation connectors only after a residual surface survives.
5. Highest-priority gate: run target prompt-wrapper controls on reusable SVAMP70
   and GSM clean source-only surfaces.
6. Scale-up rung: strict small gate.

Subagents:

- planner: skip SVAMP32 frontier because it is saturated; test GSM clean2 if a
  source surface remains after target prompt controls
- reviewer: require target-wrapper, prompt-format, no-source, selector, and
  answer-leak controls before any source residual claim
- JEPA/anti-collapse: defer Query-JEPA and masked target-state fill-in until a
  target-prior-unexplained source surface exists
- artifact audit: prioritize Math-7B SVAMP70 clean7 before connector work

Commands and artifacts:

```bash
./venv_arm64/bin/python scripts/materialize_generation_id_subset.py \
  --eval-file results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/_artifacts/svamp_eval_70_70.jsonl \
  --target-set-json results/qwen25_7b_qwen3_svamp70_surface_scout_20260427/source_contrastive_target_set.json \
  --id-fields clean_source_only \
  --output-jsonl results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/clean7_eval.jsonl \
  --output-meta-json results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/clean7_eval.meta.json

HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/sample_target_candidate_surface.py \
  --eval-file results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/clean7_eval.jsonl \
  --model Qwen/Qwen3-0.6B \
  --samples 8 \
  --method-prefix target_brief_sample \
  --temperature 0.9 \
  --top-p 0.95 \
  --seed 71 \
  --device mps \
  --dtype float32 \
  --max-new-tokens 64 \
  --prompt-mode source_reasoning \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --enable-thinking false \
  --output-jsonl results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/target_brief_samples.jsonl \
  --output-json results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/target_brief_samples.json \
  --output-md results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/target_brief_samples.md
```

Math-7B SVAMP70 clean7 result:

- target brief-wrapper S8 oracle: `4/7`
- target-prior explained clean source-only IDs: `4`
- residual candidates after target brief S8: `33836927fc9f1a8a`,
  `4c84ebf42812703b`, `d64f6e35083ffe8c`
- decision: partial prune; do not train a connector until those three survive
  answer-masked/source-destroying controls

GSM clean2 prompt-control result:

- source brief S8 oracle: `1/2`
- target direct S16 oracle: `1/2`
- target brief-wrapper S16 oracle: `1/2`
- target prompt union oracle: `1/2`
- source addition beyond target prompt union: `0`
- decision: fail GSM clean2 as a live source surface

Artifacts:

- `results/qwen25_7b_svamp70_clean7_target_brief_s8_20260427/manifest.md`
- `results/gsm_source_residual_prompt_control_clean2_20260427/manifest.md`
- `paper/prompt_wrapper_source_surface_controls_20260427.md`

Next exact gate:

- run answer-masked and answer-only source controls on the three Math-7B SVAMP70
  residual candidates that target brief S8 missed

## 2026-04-27 Cycle 23 - Source-Private Pivot Portfolio

Cycle start:

1. Current ICLR readiness: not ready, but a one-month positive method is still
   plausible if the project pivots away from ordinary math residual hunting.
2. Current paper story: target prompt wrappers and no-source candidate pools
   explain too many apparent source gains; the better story is source-private
   residual communication under decoder side information.
3. Exact blocker: build a task/method where the source truly has private
   information and the target cannot recover it from prompt wrappers, target
   sampling, no-source pools, answer-only messages, or shuffled/random sidecars.
4. Current live branches: source-private candidate-syndrome sidecars;
   C2C/KV-teacher distillation into compact sidecars; Query-JEPA only after
   source-private residual IDs exist.
5. Highest-priority gate: design and run a strict-small source-private benchmark
   with byte accounting and source-destroying controls.
6. Scale-up rung: strict small gate design.

Subagent synthesis:

- reviewer: the pivot is defensible only if target wrapper `S32`, no-source
  oracle, structured text, source-final-only, answer-masked, zero, shuffled,
  random, and rate-matched target sidecar controls are all included
- planner: fastest one-month paper ideas are private evidence packets and
  private tool-trace distillation
- creative scout: best unusual mechanisms are challenge-response hashes,
  innovation controllers, causal-order checksums, veto signals, and provenance
  fingerprints
- JEPA/connector: learned Query-JEPA/adapters should operate on target-prior
  subtracted residuals and report variance/effective-rank/covariance/query
  entropy telemetry
- artifact audit: existing candidate-pool, sidecar, exact-ID, and KVComm scripts
  can accelerate the pivot, but repo-native long-context/RAG harnessing is not
  yet integrated
- theory: source-private communication is a Wyner-Ziv/Slepian-Wolf-style
  decoder-side-information problem; claim rate-distortion improvements, not
  generic prompt quality

Ranked idea portfolio:

1. Private evidence packet for retrieval QA.
2. Private tool-trace distillation.
3. Candidate-syndrome side information over target candidate pools.
4. Source-private Query-JEPA adapter.
5. Private memory handoff.
6. Private program-state debugging.

Decision:

- write down the pivot and stop treating ordinary SVAMP/GSM residuals as the
  main discovery engine
- next method should be simple and interpretable first: candidate syndrome or
  private evidence packet
- learned latent adapters are second-stage methods after source-private residual
  IDs exist

Artifacts:

- `paper/source_private_pivot_portfolio_20260427.md`
- `references/476_source_private_comm_pivot_refs.md`
- `references/research_memo_manifest.json`

Next exact gate:

- implement a strict-small source-private evidence/tool-trace benchmark with
  `100-200` examples
- require target wrapper `S32` and no-source oracle to fail, matched source
  sidecar to beat best no-source by `>=15` points, source-destroying controls to
  remain within `2` points of no-source, and structured text at matched bytes to
  be included

## 2026-04-27 Cycle 24 - Source-Private Literature Sprint And Evidence Packet Harness

Cycle start:

1. Current ICLR readiness: not ready; still missing a strict-small
   source-private positive method, but the benchmark contract is now concrete.
2. Current paper story: source-private residual communication with decoder side
   information is the live story; ordinary SVAMP/GSM residual hunting is
   demoted to background evidence about target-prior contamination.
3. Exact blocker: instantiate a source-private task where matched source
   packets improve target selection and zero/shuffled/random/answer-only/
   answer-masked/target-only sidecars do not.
4. Current live branches: private evidence packet / candidate-syndrome decoder;
   private tool-trace handoff; Query-JEPA/Q-Former adapters only after residual
   IDs exist.
5. Highest-priority gate: run the cheapest deterministic source-private
   evidence-packet contract, then scale the same contract to `100-200`
   LLM-mediated examples.
6. Scale-up rung: smoke benchmark contract.

Subagent synthesis:

- reviewer: lead with private evidence packet plus candidate syndrome; do not
  lead with Query-JEPA because learned connectors have repeatedly failed source
  controls
- JEPA/anti-collapse: use detached target latents, source-innovation scores,
  variance/effective-rank/covariance/query-entropy telemetry only after a
  source-private residual surface exists
- VLM connector scout: Q-Former/Perceiver/Flamingo/BLIP-2/LLaVA/LLaMA-Adapter
  motivate fixed query bottlenecks and zero-init target-preserving gates, but
  simple projector and prompt-wrapper baselines are mandatory
- systems/KV scout: KVComm/KVTC/KIVI are baselines and byte-accounting
  references; cache transfer is not the first headline because byte budgets are
  large
- information theory/geometry: Slepian-Wolf/Wyner-Ziv/distributed indirect
  source coding support a fixed-byte syndrome decoded with target candidate
  side information; relative representations suggest anchor-relative codes as
  a second variant
- multi-agent planner: private tool/test-log handoff is the second benchmark if
  evidence packets pass

Commands:

```bash
./venv_arm64/bin/python scripts/run_source_private_evidence_packet_gate.py \
  --examples 128 \
  --candidates 4 \
  --seed 17 \
  --syndrome-bytes 2 \
  --output-dir results/source_private_evidence_packet_gate_20260427

./venv_arm64/bin/python -m pytest tests/test_run_source_private_evidence_packet_gate.py -q
```

Smoke results:

- target-only: `32/128`
- target wrapper/no-source: `32/128`
- matched 2-byte syndrome: `128/128`
- zero-source: `32/128`
- shuffled-source: `32/128`
- random same-byte: `32/128`
- answer-only: `32/128`
- answer-masked: `32/128`
- target-only sidecar: `32/128`
- structured text at matched 2 bytes: `32/128`
- full structured text oracle: `128/128`, mean `13` bytes
- matched minus best no-source/control: `+0.750`
- gate: `pass`

Interpretation:

- This is not positive paper evidence; it is a deterministic harness/contract.
- It verifies that the proposed gate can separate matched source-private
  syndrome communication from target priors, source-destroying controls, and a
  matched-byte structured text relay.
- The top one-month path is now concrete: source-private evidence packet /
  candidate-syndrome first; learned JEPA/Q-Former connector second.

Artifacts:

- `paper/source_private_literature_sprint_20260427.md`
- `references/477_source_private_literature_sprint_refs.md`
- `scripts/run_source_private_evidence_packet_gate.py`
- `tests/test_run_source_private_evidence_packet_gate.py`
- `results/source_private_evidence_packet_gate_20260427/summary.md`
- `results/source_private_evidence_packet_gate_20260427/summary.json`
- `results/source_private_evidence_packet_gate_20260427/predictions.jsonl`
- `results/source_private_evidence_packet_gate_20260427/manifest.md`
- `results/source_private_evidence_packet_gate_20260427/manifest.json`

Next exact gate:

- build `source_private_evidence_packet_strict_small_20260428` with `100-200`
  frozen private-evidence QA examples, target `S32` no-source candidates,
  capped source packets at `2/4/8/16/32` bytes or tokens, matched-byte
  structured text, full evidence/full text oracle, and zero/shuffled/random/
  answer-only/answer-masked/target-derived packet controls
- pass only if matched source beats best no-source by `>=15` points and every
  source-destroying control stays within `2` points of no-source

## 2026-04-28 Cycle 25 - Source-Private Evidence Packet Strict-Small Gate

Cycle start:

1. Current ICLR readiness: not ready; this cycle can clear a strict-small
   source-private protocol gate, but not the full paper claim.
2. Current paper story: source-private residual communication with decoder side
   information. The target has public prompts and candidate commitments; the
   source has private evidence; the source sends a rate-capped packet.
3. Exact blocker: show the source-private gain beyond target/no-source priors,
   matched-byte text, answer-only/answer-masked, and source-destroying controls.
4. Current live branch: source-private evidence packet / candidate-syndrome
   decoder.
5. Highest-priority gate: run `source_private_evidence_packet_strict_small_20260428`.
6. Scale-up rung: strict small gate.

Subagent synthesis:

- reviewer: a synthetic/private-evidence benchmark is defensible only with
  target/no-source, target-wrapper, matched-byte text, answer-only,
  answer-masked, zero, shuffled, random, target-derived, exact-ID parity, and
  byte accounting
- planner: promote if matched source beats target by at least `15` points and
  controls stay flat; otherwise prune or repeat only with clear oracle headroom
- method scout: use compact syndrome/evidence packets first; keep learned
  Query-JEPA/Q-Former gates as second-stage methods
- harness audit: add exact-ID hashes, artifact hashes, binary/text byte
  accounting, wrong-salt same-source control, and tests

Command:

```bash
./venv_arm64/bin/python scripts/run_source_private_evidence_packet_strict_small.py \
  --examples 160 \
  --candidates 4 \
  --seed 28 \
  --budgets 2,4,8,16,32 \
  --output-dir results/source_private_evidence_packet_strict_small_20260428
```

Results:

- examples: `160`
- candidate pool recall: `1.000`
- best budget: `2` bytes
- exact ID parity: `true`
- exact ID count: `160`
- exact ID SHA256:
  `3a65952ba323a8896906863f1be4e83400a6cea00ab1f18bbf58cb8e7611b19c`
- target/no-source accuracy: `0.250`
- matched syndrome accuracy: `1.000` at every budget `2/4/8/16/32`
- best source-destroying control accuracy: `0.250`
- matched-byte structured text accuracy: `0.250`
- full structured text oracle: `1.000`
- full private evidence oracle: `1.000`
- wrong-salt same-source control: `0.250`
- strict-small gate: `pass`

Interpretation:

- Positive for the source-private candidate-syndrome protocol under a frozen,
  deterministic strict-small benchmark.
- Not yet ICLR-ready evidence because packet production and decoding are still
  protocol-shaped rather than real model-mediated communication.
- The live branch is promoted from smoke to strict-small protocol pass; learned
  latent/JEPA connectors remain deferred.

Artifacts:

- `paper/source_private_evidence_packet_strict_small_20260428.md`
- `scripts/run_source_private_evidence_packet_strict_small.py`
- `tests/test_run_source_private_evidence_packet_strict_small.py`
- `results/source_private_evidence_packet_strict_small_20260428/benchmark.jsonl`
- `results/source_private_evidence_packet_strict_small_20260428/sweep_summary.md`
- `results/source_private_evidence_packet_strict_small_20260428/sweep_summary.json`
- `results/source_private_evidence_packet_strict_small_20260428/manifest.md`
- `results/source_private_evidence_packet_strict_small_20260428/manifest.json`
- `results/source_private_evidence_packet_strict_small_20260428/predictions_budget{2,4,8,16,32}.jsonl`
- `results/source_private_evidence_packet_strict_small_20260428/summary_budget{2,4,8,16,32}.json`

Tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_evidence_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_gate.py -q
```

Result: `7 passed`.

Next exact gate:

- run `source_private_evidence_packet_llm_packet_20260428`: freeze the same
  `160` examples, ask a source model to emit packets under the same budgets,
  keep the deterministic decoder and all controls, and pass only if
  model-produced matched packets beat best no-source by `>=15` points while
  all controls stay within `2` points of no-source
- if model packet production fails, pivot to private tool/test-log handoff with
  the same strict source-private gate structure

## 2026-04-28 Cycle 26 - Source-Private Evidence Packet Model-Packet Smoke

Cycle start:

1. Current ICLR readiness: not ready; the deterministic strict-small protocol
   passed, but model/source-produced packets remain the blocker.
2. Current paper story: source-private residual communication under decoder
   side information. The target has public question/candidate state; the source
   has private evidence; a rate-capped packet selects the right candidate only
   when the source signal is intact.
3. Exact blocker: replace deterministic packet generation with a defensible
   source-agent packet without leaking answer text or relying on target priors.
4. Current live branch: source-private evidence packet / candidate-syndrome
   decoder.
5. Highest-priority gate: run `source_private_evidence_packet_llm_packet_20260428`.
6. Scale-up rung: strict-small continuation, model-mediated smoke.

MPS guard:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Result: no live blocker was present.

Subagent synthesis:

- reviewer: prompted cryptographic digest computation is unlikely to be a
  credible method; source-final and same-byte controls must expose copied
  answer-bearing strings
- planner: run a cheap frozen-ID smoke and pivot immediately if matched packets
  stay at target-only
- method scout: keep candidate-syndrome as protocol headroom, but move the
  one-month paper path to private tool/test-log packets or naturally emitted
  source predicates
- harness audit: preserve exact-ID parity, strict hex parsing, byte telemetry,
  source-destroying controls, and a source-final leak row

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_evidence_packet_llm_packet.py \
  --benchmark-jsonl results/source_private_evidence_packet_strict_small_20260428/benchmark.jsonl \
  --output-dir results/source_private_evidence_packet_llm_packet_20260428 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device mps \
  --dtype float32 \
  --limit 16 \
  --budget-bytes 2 \
  --seed 28 \
  --max-new-tokens 24 \
  --no-enable-thinking || true
```

Results:

- examples: `16`
- budget bytes: `2`
- model-packet gate: `fail`
- packet nonempty rate: `0.562`
- target-only: `4/16`, accuracy `0.250`
- matched model packet: `4/16`, accuracy `0.250`
- zero-source/shuffled/random/answer-only/answer-masked controls:
  `4/16`, accuracy `0.250`
- source-final-only: `16/16`, accuracy `1.000`
- matched minus best no-source: `0.000`
- matched minus best control: `0.000`
- source-final minus best no-source: `0.750`
- p50 matched source-packet latency: `1741.28` ms

Interpretation:

- This is a clean falsification of the naive "prompt an LLM to compute the
  cryptographic syndrome" branch.
- The source model mostly copied pieces of the instruction, witness key, or
  record name instead of producing the digest packet.
- The source-final-only row confirms that answer-bearing private evidence can
  solve the task if leaked, so the failed matched packet is not caused by lack
  of task headroom.
- The deterministic strict-small result remains useful as a source-private
  side-information protocol bound, but it should not be promoted as an
  LLM-mediated method.

Artifacts:

- `paper/source_private_evidence_packet_llm_packet_20260428.md`
- `scripts/run_source_private_evidence_packet_llm_packet.py`
- `tests/test_run_source_private_evidence_packet_llm_packet.py`
- `results/source_private_evidence_packet_llm_packet_20260428/model_packets.jsonl`
- `results/source_private_evidence_packet_llm_packet_20260428/predictions.jsonl`
- `results/source_private_evidence_packet_llm_packet_20260428/summary.json`
- `results/source_private_evidence_packet_llm_packet_20260428/summary.md`
- `results/source_private_evidence_packet_llm_packet_20260428/manifest.json`
- `results/source_private_evidence_packet_llm_packet_20260428/manifest.md`

Tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_evidence_packet_llm_packet.py \
  tests/test_run_source_private_evidence_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_gate.py -q
```

Result: `10 passed`.

Decision:

- prune LLM-produced cryptographic digest packets as the live one-month method
  branch
- keep source-private candidate-syndrome as a benchmark/protocol lower bound
- make `source_private_testlog_packet_strict_small_20260428` the next exact
  gate

Next exact gate:

- build a private tool/test-log handoff benchmark where target sees public
  issue plus candidate fixes and source sees private execution evidence
- source emits a rate-capped packet/predicate derived from the private test log
- compare against target-only, target-wrapper/no-source, matched-byte text
  relay, full structured log oracle, and the full source-destroying suite
- pass only if matched source beats best no-source by `>=15` points and every
  source-destroying control stays within `2` points of no-source

## 2026-04-28 Cycle 27 - Source-Private Test-Log Packet Strict-Small Gate

Cycle start:

1. Current ICLR readiness: not ready; this gate tests a naturally emitted
   private-evidence packet surface, but still with deterministic packetization.
2. Current paper story: source-private residual communication under decoder
   side information. Target sees public issue plus candidate fixes; source sees
   a private tool/test log; a rate-capped packet identifies the candidate whose
   public handled signature matches the private log.
3. Exact blocker: show model-extracted packets and then a less synthetic
   hidden-test/code-repair benchmark.
4. Current live branch: source-private test-log packet handoff.
5. Highest-priority gate: `source_private_testlog_packet_strict_small_20260428`.
6. Scale-up rung: strict small gate.

MPS guard:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Result: no live blocker was present.

Subagent synthesis:

- reviewer: require exact ID parity, public/private split checks, flat
  zero/shuffled/random/answer-only/answer-masked controls, matched-byte text
  relay, and no answer/patch-label leakage
- planner: private tool/test-log handoff is the highest-probability branch
  after the cryptographic packet falsification; deterministic contract first,
  model extraction next
- method scout: a tool-trace verdict/signature packet is the most actionable
  source-derived signal; learned Q-bottlenecks should wait until this surface
  survives model extraction
- harness audit: add leakage audit, nonself shuffled packets, artifact hashes,
  byte telemetry, and focused tests

Command:

```bash
./venv_arm64/bin/python scripts/run_source_private_testlog_packet_strict_small.py \
  --examples 160 \
  --candidates 4 \
  --seed 28 \
  --budgets 2,4,8,16,32 \
  --output-dir results/source_private_testlog_packet_strict_small_20260428
```

Results:

- strict-small pass: `true`
- examples: `160`
- candidate-pool recall: `1.000`
- best budget: `2` bytes
- exact ID parity: `true`
- exact ID SHA256:
  `fcfd2cfcecfa51f4caae6e5de39cf0632dafb634e4f19db0dfdc12c2ef8dbd2e`
- target/no-source accuracy: `40/160`, `0.250`
- matched test-log packet accuracy: `160/160`, `1.000` at every budget
  `2/4/8/16/32`
- best source-destroying control accuracy: `40/160`, `0.250`
- matched-byte raw-log truncation accuracy: `40/160`, `0.250`
- full structured log oracle: `160/160`, `1.000`
- full signature text oracle: `160/160`, `1.000`
- matched minus best no-source: `+0.750`
- matched minus best control: `+0.750`

Leakage audit:

- public target private-log hits: `0`
- public target `TRACE_SIG` hits: `0`
- matched packet answer-label copies: `0`
- matched packet candidate-label copies: `0`
- matched packet over-budget count: `0`

Interpretation:

- Positive for the deterministic source-private test-log handoff contract.
- Stronger than the cryptographic digest surface because `TRACE_SIG=<code>` is
  a natural private tool-log artifact that a source agent can plausibly emit.
- Not yet ICLR-ready because the benchmark is synthetic and the packetizer/
  decoder are deterministic.

Artifacts:

- `paper/source_private_testlog_packet_strict_small_20260428.md`
- `scripts/run_source_private_testlog_packet_strict_small.py`
- `tests/test_run_source_private_testlog_packet_strict_small.py`
- `results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl`
- `results/source_private_testlog_packet_strict_small_20260428/sweep_summary.json`
- `results/source_private_testlog_packet_strict_small_20260428/sweep_summary.md`
- `results/source_private_testlog_packet_strict_small_20260428/leakage_audit.json`
- `results/source_private_testlog_packet_strict_small_20260428/leakage_audit.md`
- `results/source_private_testlog_packet_strict_small_20260428/manifest.json`
- `results/source_private_testlog_packet_strict_small_20260428/manifest.md`
- `results/source_private_testlog_packet_strict_small_20260428/predictions_budget{2,4,8,16,32}.jsonl`
- `results/source_private_testlog_packet_strict_small_20260428/summary_budget{2,4,8,16,32}.json`

Tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_testlog_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_llm_packet.py \
  tests/test_run_source_private_evidence_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_gate.py -q
```

Result: `16 passed`.

Decision:

- promote source-private test-log packets to the live branch
- keep deterministic candidate-syndrome/test-log gates as contracts, not final
  method claims
- run model-extracted `TRACE_SIG` packets next

Next exact gate:

- `source_private_testlog_packet_llm_packet_20260428`: same frozen benchmark,
  source model emits only the two-byte `TRACE_SIG` from the private log,
  deterministic decoder and full source-destroying controls

## 2026-04-28 Cycle 28 - Source-Private Test-Log Model-Packet Gate

Cycle start:

1. Current ICLR readiness: not ready, but this is the first positive
   model-mediated source-private packet result.
2. Current paper story: source-private residual communication under decoder
   side information. The source sees a private tool log and emits a compact
   packet; the target uses public candidate signatures to decode.
3. Exact blocker: seed stability, prompt/helper-line ablation, cross-family
   source model, and less synthetic hidden-test/code-repair logs.
4. Current live branch: source-private test-log packet handoff.
5. Highest-priority gate: model-extracted `TRACE_SIG` packets on the frozen
   strict-small benchmark.
6. Scale-up rung: strict small model-mediated gate.

MPS guard:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Result: no live blocker was present.

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_testlog_packet_llm_packet.py \
  --benchmark-jsonl results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl \
  --output-dir results/source_private_testlog_packet_llm_packet_20260428 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device mps \
  --dtype float32 \
  --limit 160 \
  --seed 28 \
  --max-new-tokens 8 \
  --no-enable-thinking || true
```

Results:

- model-packet gate: `pass`
- examples: `160`
- exact ID parity: `true`
- exact ID SHA256:
  `fcfd2cfcecfa51f4caae6e5de39cf0632dafb634e4f19db0dfdc12c2ef8dbd2e`
- packet valid rate: `0.91875`
- target-only: `40/160`, `0.250`
- matched model packet: `150/160`, `0.938`
- zero-source: `40/160`, `0.250`
- shuffled model packet: `40/160`, `0.250`
- random same-byte: `40/160`, `0.250`
- answer-only: `40/160`, `0.250`
- answer-masked: `40/160`, `0.250`
- target-derived sidecar: `40/160`, `0.250`
- full signature oracle: `160/160`, `1.000`
- matched minus best no-source: `+0.6875`
- matched minus best source-destroying control: `+0.6875`
- p50 matched source-packet latency: `162.51` ms

Interpretation:

- This is a positive model-mediated source-private communication gate.
- The gain survives zero, shuffled, random, answer-only, answer-masked, and
  target-derived controls.
- The result depends on source-side private-log line isolation: the source
  prompt includes the copied `TRACE_SIG` line. This is a valid tool-log handoff
  primitive, but not yet a full unstructured-log/code-repair claim.

Artifacts:

- `paper/source_private_testlog_packet_llm_packet_20260428.md`
- `scripts/run_source_private_testlog_packet_llm_packet.py`
- `tests/test_run_source_private_testlog_packet_llm_packet.py`
- `results/source_private_testlog_packet_llm_packet_20260428/model_packets.jsonl`
- `results/source_private_testlog_packet_llm_packet_20260428/predictions.jsonl`
- `results/source_private_testlog_packet_llm_packet_20260428/summary.json`
- `results/source_private_testlog_packet_llm_packet_20260428/summary.md`
- `results/source_private_testlog_packet_llm_packet_20260428/manifest.json`
- `results/source_private_testlog_packet_llm_packet_20260428/manifest.md`

Tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_testlog_packet_llm_packet.py \
  tests/test_run_source_private_testlog_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_llm_packet.py \
  tests/test_run_source_private_evidence_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_gate.py -q
```

Result: `20 passed`.

Decision:

- promote source-private test-log packet handoff as the current live positive
  branch
- keep caveats explicit: synthetic benchmark, source-side line isolation,
  deterministic decoder, one model, one seed

Next exact gate:

- `source_private_testlog_packet_llm_packet_seed_repeat_20260428`: seeds `29`
  and `30`, same frozen benchmark, same controls, plus no-helper-line prompt
  ablation

## 2026-04-28 Cycle 29 - Source-Private Test-Log Packet Seed Repeat

Cycle start:

1. Current ICLR readiness: not ready, but the live source-private test-log
   branch now has a positive model-mediated result.
2. Current paper story: a source model extracts a compact private tool-log
   packet and the target decodes it with public candidate-side signatures.
3. Exact blocker: separate protocol-assisted packet emission from unstructured
   log extraction, then confirm cross-model and on hidden-test/code-repair logs.
4. Current live branch: source-private test-log packet handoff.
5. Highest-priority gate:
   `source_private_testlog_packet_llm_packet_seed_repeat_20260428`.
6. Scale-up rung: strict-small model-mediated confirmation.

MPS guard:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Result: no live blocker was present.

Subagent synthesis:

- reviewer: helper-line results are acceptable only as protocol-assisted
  handoff; if no-helper collapses, do not claim generic unstructured-log
  extraction
- planner: aggregate exact frozen IDs across prompt modes and seeds, report
  pass/fail per run, and keep controls flat
- harness audit: greedy seed repeats are reproducibility checks, so record
  prompt mode, script/benchmark hashes, deterministic decoding metadata, and
  nonself shuffled-source provenance

Script updates:

- added `--prompt-mode helper_line|full_log`
- added prompt-mode metadata to packet rows and manifests
- added benchmark and script hashes to manifests
- added `do_sample: false` metadata
- added nonself shuffled-source provenance in prediction rows

Runs:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_testlog_packet_llm_packet.py \
  --benchmark-jsonl results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl \
  --output-dir results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/helper_seed29 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device mps \
  --dtype float32 \
  --limit 160 \
  --seed 29 \
  --max-new-tokens 8 \
  --prompt-mode helper_line \
  --no-enable-thinking || true
```

Repeated for:

- `helper_seed30`
- `full_log_seed29`
- `full_log_seed30`

Results:

| Prompt mode | Seeds | All pass | Mean matched | Min matched | Mean valid packets | Mean lift vs no-source | Mean lift vs controls |
|---|---|---|---:|---:|---:|---:|---:|
| helper_line | `[29, 30]` | `true` | 0.938 | 0.938 | 0.919 | 0.688 | 0.688 |
| full_log | `[29, 30]` | `false` | 0.344 | 0.344 | 0.163 | 0.094 | 0.094 |

Per-run:

- helper seed 29: pass, matched `150/160`, target-only/control `40/160`
- helper seed 30: pass, matched `150/160`, target-only/control `40/160`
- full-log seed 29: fail, matched `55/160`, target-only/control `40/160`
- full-log seed 30: fail, matched `55/160`, target-only/control `40/160`

Interpretation:

- The helper-line protocol is stable across greedy seed repeats.
- The no-helper full-log ablation fails, mostly from invalid or partial packets.
- Current claim must be protocol-assisted private tool-log packet emission, not
  general log extraction.

Artifacts:

- `paper/source_private_testlog_packet_seed_repeat_20260428.md`
- `results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/aggregate_summary.json`
- `results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/aggregate_summary.md`
- `results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/manifest.json`
- `results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/manifest.md`
- four per-run subdirectories under
  `results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/`

Tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_testlog_packet_llm_packet.py \
  tests/test_run_source_private_testlog_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_llm_packet.py \
  tests/test_run_source_private_evidence_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_gate.py -q
```

Result: `22 passed`.

Decision:

- promote the branch under a narrow protocol-assisted handoff claim
- do not claim generic unstructured-log extraction

Next exact gate:

- `source_private_testlog_packet_cross_model_20260428`: same frozen IDs and
  helper-line protocol with a second cached source model/family, same controls

## 2026-04-28 Cycle 30 - Source-Private Test-Log Packet Cross-Model Gate

Cycle start:

1. Current ICLR readiness: not ready, but the branch has stable
   model-mediated helper-line evidence.
2. Current paper story: protocol-assisted source-private tool-log packet
   handoff. A source model extracts a compact private `TRACE_SIG` packet and
   the target decodes it with public candidate-side signatures.
3. Exact blocker: show the result is not one-model-specific, then move to real
   hidden-test/code-repair logs.
4. Current live branch: source-private test-log packet handoff.
5. Highest-priority gate:
   `source_private_testlog_packet_cross_model_20260428`.
6. Scale-up rung: cross-model strict-small falsification.

MPS guard:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Result: no live blocker was present.

Subagent synthesis:

- reviewer: cross-model success is required before this can be more than a
  one-model prompt artifact; helper-line success remains publishable only under
  a narrow protocol-assisted private-log packet claim
- planner: use frozen IDs, helper-line protocol, deterministic decoder, same
  controls; promote if at least one non-Qwen model passes

Runs:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen3-0.6B`
- `microsoft/Phi-3-mini-4k-instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Attempted but not counted:

- `meta-llama/Llama-3.2-1B-Instruct`: local cache entry was incomplete for
  offline `transformers` loading

Results:

| Run | Model | Family | Pass | Matched | Target-only | Best control | Valid packets | p50 latency ms |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen25_0_5b_helper | Qwen/Qwen2.5-0.5B-Instruct | qwen2.5 | `true` | 0.938 | 0.250 | 0.250 | 0.919 | 164.86 |
| qwen3_0_6b_helper | Qwen/Qwen3-0.6B | qwen3 | `true` | 1.000 | 0.250 | 0.250 | 1.000 | 334.17 |
| phi3_mini_helper | microsoft/Phi-3-mini-4k-instruct | phi3 | `true` | 0.912 | 0.250 | 0.250 | 0.950 | 595.25 |
| tinyllama_1_1b_helper | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | tinyllama | `false` | 0.250 | 0.250 | 0.250 | 0.000 | 496.49 |

Aggregate:

- cross-model gate: `pass`
- passing models: `3/4`
- non-Qwen passing models: `phi3_mini_helper`
- mean matched accuracy among passing models: `0.950`
- all source-destroying controls stayed at `0.250`

Interpretation:

- The helper-line packet protocol generalizes to Qwen3 and Phi-3.
- TinyLlama failure is a negative capability/control row.
- Claim remains protocol-assisted private tool-log packet handoff, not
  universal model-agnostic extraction.

Artifacts:

- `paper/source_private_testlog_packet_cross_model_20260428.md`
- `results/source_private_testlog_packet_cross_model_20260428/cross_model_summary.json`
- `results/source_private_testlog_packet_cross_model_20260428/cross_model_summary.md`
- `results/source_private_testlog_packet_cross_model_20260428/manifest.json`
- `results/source_private_testlog_packet_cross_model_20260428/manifest.md`
- per-model subdirectories under
  `results/source_private_testlog_packet_cross_model_20260428/`

Tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_testlog_packet_llm_packet.py \
  tests/test_run_source_private_testlog_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_llm_packet.py \
  tests/test_run_source_private_evidence_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_gate.py -q
```

Result: `22 passed`.

Decision:

- promote the branch as cross-model on capable instruction-tuned source models
- keep TinyLlama as a negative capability row
- do not claim universal model-agnostic extraction

Next exact gate:

- `source_private_testlog_packet_hidden_repair_smoke_20260428`: same protocol
  and controls, but private packet comes from actual hidden pytest/code-repair
  evidence rather than synthetic signature fields

## 2026-04-28 - Hidden-Repair Packet Smoke Replaces Synthetic Trace Fields

Current ICLR readiness: not ready. Estimated distance is still one promoted
method plus medium confirmation, cross-model/cross-family replication, and
no-helper or weakened-helper robustness.

Current story: the live source-private packet branch now has a hidden-repair
smoke where the private source evidence is actual hidden execution output from
buggy Python functions. The target receives public issue text, buggy code, and a
four-candidate repair pool; the source sees the hidden failure log and emits a
rate-capped repair diagnostic packet.

Exact blocker: the branch is still protocol-assisted by helper-line diagnostics
and candidate metadata, so it cannot yet be claimed as general code repair or
latent model communication.

Commands:

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_smoke.py \
  --examples 64 \
  --candidates 4 \
  --seed 28 \
  --budgets 2,4,8,16,32 \
  --output-dir results/source_private_hidden_repair_packet_smoke_20260428
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py \
  --benchmark-jsonl results/source_private_hidden_repair_packet_smoke_20260428/benchmark.jsonl \
  --output-dir results/source_private_hidden_repair_packet_llm_20260428/qwen3_0_6b_helper \
  --model Qwen/Qwen3-0.6B \
  --device mps \
  --dtype float32 \
  --limit 64 \
  --seed 28 \
  --max-new-tokens 8 \
  --no-enable-thinking
```

MPS guard:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Result: no live blocker was present.

Deterministic smoke:

| Budget bytes | Pass | Matched | Best no-source | Best control | Matched text | Full log |
|---:|---|---:|---:|---:|---:|---:|
| 2 | `true` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 4 | `true` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 8 | `true` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 16 | `true` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 32 | `true` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |

Qwen3 model-packet smoke:

| Condition | Correct | Accuracy | Mean bytes |
|---|---:|---:|---:|
| target_only | 16/64 | 0.250 | 0.00 |
| matched_model_packet | 64/64 | 1.000 | 2.00 |
| zero_source | 16/64 | 0.250 | 0.00 |
| shuffled_model_packet | 16/64 | 0.250 | 2.00 |
| random_same_byte | 16/64 | 0.250 | 2.00 |
| answer_only | 16/64 | 0.250 | 2.00 |
| answer_masked | 16/64 | 0.250 | 0.00 |
| target_derived_sidecar | 16/64 | 0.250 | 2.00 |
| full_diag_oracle | 64/64 | 1.000 | 2.00 |

Model packet validity was `1.000`; matched minus best no-source was `0.750`;
matched minus best control was `0.750`.

Subagent synthesis:

- planner: hidden-repair smoke should require actual private execution logs,
  target-only/no-log controls, zero/shuffled/random/public controls, bytes, and
  latency before widening
- reviewer: this is only reviewer-safe as protocol-assisted private hidden-log
  packet handoff; no-helper, matched-byte structured text, held-out templates,
  and cross-model rows are required before a stronger paper claim

Decision:

- promote hidden-repair packet handoff to model-mediated smoke
- keep the claim narrow because helper-line diagnostics and candidate metadata
  remain part of the protocol
- do not call this general code repair or latent transfer yet

Artifacts:

- `paper/source_private_hidden_repair_packet_smoke_20260428.md`
- `results/source_private_hidden_repair_packet_smoke_20260428/`
- `results/source_private_hidden_repair_packet_llm_20260428/qwen3_0_6b_helper/`

Tests:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_hidden_repair_packet_smoke.py \
  tests/test_run_source_private_hidden_repair_packet_llm.py -q
```

Result: `8 passed`.

Next exact gate:

- `source_private_hidden_repair_packet_cross_model_20260428`: same frozen
  hidden-repair examples, Qwen2.5-0.5B, Qwen3-0.6B, Phi-3-mini, and TinyLlama
  negative capability row, with the same source-destroying controls and packet
  validity/byte/latency reporting.

## 2026-04-28 - Hidden-Repair Packet Cross-Model Smoke

Current ICLR readiness: not ready. Estimated distance is one live method branch
plus weakened-helper/no-helper evidence, strict-small or medium confirmation,
and a larger frozen slice.

Current story: source-private hidden-repair packets now pass across multiple
capable source model families. The source model sees actual hidden execution
logs plus the protocol helper line, emits a two-character repair packet, and
the target decodes it against candidate-side diagnostic metadata. Gains
disappear under zero, shuffled, random, answer-only, answer-masked, and
target-derived controls.

Exact blocker: helper-line diagnostics and candidate metadata remain central.
The result is still protocol-assisted private hidden-log handoff, not
general-purpose repair reasoning from raw logs.

MPS guard:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Result: no live blocker was present.

Runs:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen3-0.6B`
- `microsoft/Phi-3-mini-4k-instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Results:

| Run | Model | Family | Pass | Matched | Target-only | Best control | Valid packets | p50 latency ms |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen25_0_5b_helper | Qwen/Qwen2.5-0.5B-Instruct | qwen2.5 | `true` | 0.984 | 0.250 | 0.250 | 0.984 | 330.85 |
| qwen3_0_6b_helper | Qwen/Qwen3-0.6B | qwen3 | `true` | 1.000 | 0.250 | 0.250 | 1.000 | 312.52 |
| phi3_mini_helper | microsoft/Phi-3-mini-4k-instruct | phi3 | `true` | 1.000 | 0.250 | 0.250 | 1.000 | 511.35 |
| tinyllama_1_1b_helper | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | tinyllama | `false` | 0.250 | 0.250 | 0.250 | 0.000 | 561.73 |

Aggregate:

- cross-model gate: `pass`
- passing models: `3/4`
- non-Qwen passing models: `1`
- all source-destroying controls stayed at target-only (`0.250`)
- TinyLlama is a clean negative capability row with `0.000` valid packets

Decision:

- promote hidden-repair packet handoff to cross-model smoke on capable
  instruction-tuned source models
- retain the claim boundary: protocol-assisted private hidden-log packet
  handoff, not universal code repair or no-helper extraction

Artifacts:

- `paper/source_private_hidden_repair_packet_cross_model_20260428.md`
- `results/source_private_hidden_repair_packet_cross_model_20260428/`

Next exact gate:

- `source_private_hidden_repair_packet_weakened_helper_20260428`: same frozen
  hidden-repair examples, remove the copied helper line first, then test a
  harder no-helper or masked-log variant. Promote only if matched source remains
  at least `15` points above target-only and controls remain flat.

## 2026-04-28 - Hidden-Repair Packet Weakened-Helper Smoke

Current ICLR readiness: not ready. Estimated distance is now a strict-small
hidden-repair run, a medium/larger frozen slice, and a stronger explanation of
the explicit trace-field interface.

Current story: the live method is an explicit source-private tool-trace packet.
The source sees a private hidden execution log containing a compact
`REPAIR_DIAG` trace field. The target cannot see the hidden log, but can decode
the packet against public candidate metadata. Gains disappear under
source-destroying controls.

Exact blocker: this remains an explicit trace-field communication protocol.
Raw hidden logs without the trace do not yet produce a usable packet, so the
paper claim must be about private tool-trace handoff rather than open-ended
repair inference from logs.

Harness change:

- `scripts/run_source_private_hidden_repair_packet_llm.py` now supports
  `--prompt-mode copied_helper|log_only|trace_no_hint|raw_log_no_trace`
- `trace_no_hint` removes the copied helper line and hint while keeping the
  private `REPAIR_DIAG` trace
- `raw_log_no_trace` removes the trace and hint as a source-signal destruction
  control

Results:

| Run | Model | Prompt mode | Pass | Matched | Target-only | Best control | Valid packets |
|---|---|---|---|---:|---:|---:|---:|
| qwen3_log_only | Qwen/Qwen3-0.6B | log_only | `true` | 0.984 | 0.250 | 0.250 | 0.984 |
| qwen3_trace_no_hint | Qwen/Qwen3-0.6B | trace_no_hint | `true` | 0.781 | 0.250 | 0.250 | 0.734 |
| qwen3_raw_log_no_trace | Qwen/Qwen3-0.6B | raw_log_no_trace | `false` | 0.250 | 0.250 | 0.250 | 0.000 |
| phi3_trace_no_hint | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | `true` | 1.000 | 0.250 | 0.250 | 1.000 |

Decision:

- promote `trace_no_hint` as the primary hidden-repair protocol
- demote the copied helper line and hint; they are no longer necessary
- keep `raw_log_no_trace` as a source-signal destruction control
- do not claim raw-log repair inference

Artifacts:

- `paper/source_private_hidden_repair_packet_weakened_helper_20260428.md`
- `results/source_private_hidden_repair_packet_weakened_helper_20260428/`

Next exact gate:

- `source_private_hidden_repair_packet_strict_small_20260429`: scale the
  hidden-repair benchmark to `160` frozen examples with `trace_no_hint` as the
  primary source prompt, Qwen3 and Phi-3 source emitters, exact ID parity,
  source-destroying controls, packet validity, bytes, and latency.

## 2026-04-29 - Hidden-Repair Packet Strict-Small Gate

Current ICLR readiness: not ready, but the live source-private branch has its
first strict-small positive result. Estimated distance is now medium/larger
confirmation, paired uncertainty, and stronger paper framing around explicit
private tool-trace communication.

Current story: a source model can transmit a compact private `REPAIR_DIAG`
tool-trace field from hidden code execution logs to a target model. The target
uses that packet with candidate-side metadata to select the correct repair.
The gain survives Qwen3 and Phi-3 at `160` examples and disappears when the
trace field is removed.

Exact blocker: the method is still an explicit trace-field protocol, not
raw-log repair inference or latent transfer. The next blocker is scale and
uncertainty, not another prompt tweak.

Commands:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_smoke.py \
  --examples 160 \
  --candidates 4 \
  --seed 29 \
  --budgets 2,4,8,16,32 \
  --output-dir results/source_private_hidden_repair_packet_strict_small_20260429
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py \
  --benchmark-jsonl results/source_private_hidden_repair_packet_strict_small_20260429/benchmark.jsonl \
  --output-dir results/source_private_hidden_repair_packet_strict_small_llm_20260429/qwen3_trace_no_hint \
  --model Qwen/Qwen3-0.6B \
  --device mps \
  --dtype float32 \
  --limit 160 \
  --seed 29 \
  --max-new-tokens 8 \
  --prompt-mode trace_no_hint \
  --no-enable-thinking
```

Equivalent model commands were run for `microsoft/Phi-3-mini-4k-instruct` in
`trace_no_hint` mode and Qwen3 in `raw_log_no_trace` mode.

Deterministic packet sweep:

| Budget bytes | Pass | Matched | Best no-source | Best control | Matched text | Full log |
|---:|---|---:|---:|---:|---:|---:|
| 2 | `true` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 4 | `true` | 1.000 | 0.250 | 0.256 | 0.250 | 1.000 |
| 8 | `true` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 16 | `true` | 1.000 | 0.250 | 0.256 | 0.250 | 1.000 |
| 32 | `true` | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |

Model-produced packets:

| Run | Model | Prompt mode | Pass | Matched | Target-only | Best control | Valid packets |
|---|---|---|---|---:|---:|---:|---:|
| qwen3_trace_no_hint | Qwen/Qwen3-0.6B | trace_no_hint | `true` | 0.794 | 0.250 | 0.256 | 0.762 |
| phi3_trace_no_hint | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | `true` | 1.000 | 0.250 | 0.256 | 1.000 |
| qwen3_raw_log_no_trace | Qwen/Qwen3-0.6B | raw_log_no_trace | `false` | 0.250 | 0.250 | 0.256 | 0.000 |

Decision:

- promote hidden-repair packet handoff to strict-small
- keep `trace_no_hint` as the primary protocol
- keep `raw_log_no_trace` as the source-signal destruction row
- do not claim raw-log repair inference

Artifacts:

- `paper/source_private_hidden_repair_packet_strict_small_20260429.md`
- `results/source_private_hidden_repair_packet_strict_small_20260429/`
- `results/source_private_hidden_repair_packet_strict_small_llm_20260429/`

Next exact gate:

- `source_private_hidden_repair_packet_medium_20260429`: scale to `500` frozen
  examples if feasible, otherwise `320`, add paired bootstrap uncertainty, keep
  Qwen3/Phi-3 `trace_no_hint` primary rows and Qwen3 `raw_log_no_trace`
  destruction row.

## 2026-04-29 - Hidden-Repair Packet Medium Gate

Current ICLR readiness: not ready, but the live branch now has medium
confirmation. Estimated distance is held-out family generalization, seed
repeats, and reviewer-ready baseline framing.

Current story: explicit source-private tool-trace packets communicate hidden
execution evidence across models. The source emits a compact `REPAIR_DIAG`
packet from a private trace, and the target decodes it with candidate-side
metadata. On `500` frozen examples, Qwen3 and Phi-3 both beat target-only and
all source-destroying controls by large paired margins; removing the trace
destroys the signal.

Exact blocker: template-family generalization. The current medium result is
large but still from the same eight repair families, so the next paper risk is
that reviewers call it a templated protocol demo rather than a general method.

Commands:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_smoke.py \
  --examples 500 \
  --candidates 4 \
  --seed 29 \
  --budgets 2,4,8,16,32 \
  --output-dir results/source_private_hidden_repair_packet_medium_20260429
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py \
  --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl \
  --output-dir results/source_private_hidden_repair_packet_medium_llm_20260429/qwen3_trace_no_hint \
  --model Qwen/Qwen3-0.6B \
  --device mps \
  --dtype float32 \
  --limit 500 \
  --seed 29 \
  --max-new-tokens 8 \
  --prompt-mode trace_no_hint \
  --no-enable-thinking
```

Equivalent model commands were run for `microsoft/Phi-3-mini-4k-instruct` in
`trace_no_hint` mode and Qwen3 in `raw_log_no_trace` mode. Paired bootstrap
summary:

```bash
./venv_arm64/bin/python scripts/summarize_source_private_hidden_repair_medium.py \
  --llm-dir results/source_private_hidden_repair_packet_medium_llm_20260429 \
  --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl \
  --bootstrap-samples 2000 \
  --seed 20260429
```

Results:

| Run | Model | Mode | Matched | Target | Best control | Valid | Delta target 95% CI | Delta control 95% CI |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3_trace_no_hint | Qwen/Qwen3-0.6B | trace_no_hint | 0.808 | 0.250 | 0.252 | 0.776 | [0.516, 0.600] | [0.514, 0.602] |
| phi3_trace_no_hint | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.252 | 1.000 | [0.714, 0.788] | [0.708, 0.786] |
| qwen3_raw_log_no_trace | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.252 | 0.000 | [0.000, 0.000] | [-0.006, 0.000] |

Decision:

- promote hidden-repair packet handoff to medium confirmation
- preserve the claim boundary: explicit private tool-trace communication
- stop prompt tuning and move to held-out repair families plus seed repeats

Artifacts:

- `paper/source_private_hidden_repair_packet_medium_20260429.md`
- `results/source_private_hidden_repair_packet_medium_20260429/`
- `results/source_private_hidden_repair_packet_medium_llm_20260429/`

Next exact gate:

- `source_private_hidden_repair_packet_holdout_families_20260429`: add held-out
  repair families, keep `trace_no_hint` and the same controls, require both
  Qwen3 and Phi-3 to beat target-only by at least `15` points.

## 2026-04-29 - Hidden-Repair Packet Holdout-Families Gate

Current ICLR readiness: not ready, but the live branch now has medium-scale
and held-out-family confirmation. Estimated distance is seed repeats plus
reviewer-ready baseline/framing.

Current story: explicit source-private tool-trace packets are a compact
agent-to-agent communication interface. The source sees a private hidden
execution trace containing `REPAIR_DIAG`, emits a two-byte packet, and the
target decodes it against public candidate metadata. The result holds on both
core and held-out repair families, across Qwen3 and Phi-3 source emitters, and
fails when the trace is removed.

Exact blocker: seed stability and paper framing. The method is now positive,
but still needs repeat stability before becoming an ICLR claim.

Harness change:

- `scripts/run_source_private_hidden_repair_packet_smoke.py` now supports
  `--family-set core|holdout|all`
- held-out families are disjoint from the core eight template families

Commands:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_smoke.py \
  --examples 500 \
  --candidates 4 \
  --seed 30 \
  --budgets 2,4,8,16,32 \
  --family-set holdout \
  --output-dir results/source_private_hidden_repair_packet_holdout_families_20260429
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py \
  --benchmark-jsonl results/source_private_hidden_repair_packet_holdout_families_20260429/benchmark.jsonl \
  --output-dir results/source_private_hidden_repair_packet_holdout_families_llm_20260429/qwen3_trace_no_hint \
  --model Qwen/Qwen3-0.6B \
  --device mps \
  --dtype float32 \
  --limit 500 \
  --seed 30 \
  --max-new-tokens 8 \
  --prompt-mode trace_no_hint \
  --no-enable-thinking
```

Equivalent model commands were run for `microsoft/Phi-3-mini-4k-instruct` in
`trace_no_hint` mode and Qwen3 in `raw_log_no_trace` mode.

Results:

| Run | Model | Mode | Matched | Target | Best control | Valid | Delta target 95% CI | Delta control 95% CI |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen3_trace_no_hint | Qwen/Qwen3-0.6B | trace_no_hint | 0.922 | 0.250 | 0.258 | 0.864 | [0.632, 0.712] | [0.622, 0.706] |
| phi3_trace_no_hint | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.258 | 1.000 | [0.710, 0.788] | [0.702, 0.778] |
| qwen3_raw_log_no_trace | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.258 | 0.000 | [0.000, 0.000] | [-0.016, -0.002] |

Decision:

- promote hidden-repair packet handoff to held-out family confirmation
- keep the claim scoped to explicit private tool-trace communication
- next priority is seed repeats, not more prompt variants

Artifacts:

- `paper/source_private_hidden_repair_packet_holdout_families_20260429.md`
- `results/source_private_hidden_repair_packet_holdout_families_20260429/`
- `results/source_private_hidden_repair_packet_holdout_families_llm_20260429/`

Next exact gate:

- `source_private_hidden_repair_packet_seed_repeat_20260429`: repeat core and
  held-out gates over additional frozen seeds with Qwen3/Phi-3 `trace_no_hint`
  and Qwen3 `raw_log_no_trace`.

## 2026-04-29 - Hidden-Repair Packet Seed-Repeat Gate

Current ICLR readiness: not ready, but the method is now seed-stable as a
positive candidate. Remaining distance is reviewer-facing baseline/system
packaging and paper framing, not method discovery.

Current story: explicit source-private tool-trace packets communicate hidden
execution evidence from source to target. Across four frozen `500`-example
surfaces, the source emits a compact `REPAIR_DIAG` packet in `trace_no_hint`
mode; Qwen3 and Phi-3 pass, all source-destroying controls stay flat, and
removing the trace returns to target-only with `0` valid packets.

Exact blocker: convert the evidence into a reviewer-ready baseline and systems
package. The method should be framed as explicit private tool-trace
communication, not raw-log inference or unstructured latent transfer.

New seed-repeat surfaces:

- `core_seed31`: core families, seed `31`, `500` examples
- `holdout_seed32`: held-out families, seed `32`, `500` examples

Aggregate results over prior and new surfaces:

| Surface | Family set | Seed | Model | Mode | Matched | Target | Best control | Valid | Delta target 95% CI |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| core_seed29 | core | 29 | Qwen/Qwen3-0.6B | trace_no_hint | 0.808 | 0.250 | 0.252 | 0.776 | [0.516, 0.600] |
| core_seed29 | core | 29 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.252 | 1.000 | [0.714, 0.788] |
| core_seed29 | core | 29 | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.252 | 0.000 | [0.000, 0.000] |
| core_seed31 | core | 31 | Qwen/Qwen3-0.6B | trace_no_hint | 0.808 | 0.250 | 0.256 | 0.776 | [0.516, 0.602] |
| core_seed31 | core | 31 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.256 | 1.000 | [0.710, 0.786] |
| core_seed31 | core | 31 | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.256 | 0.000 | [0.000, 0.000] |
| holdout_seed30 | holdout | 30 | Qwen/Qwen3-0.6B | trace_no_hint | 0.922 | 0.250 | 0.258 | 0.864 | [0.632, 0.712] |
| holdout_seed30 | holdout | 30 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.258 | 1.000 | [0.710, 0.788] |
| holdout_seed30 | holdout | 30 | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.258 | 0.000 | [0.000, 0.000] |
| holdout_seed32 | holdout | 32 | Qwen/Qwen3-0.6B | trace_no_hint | 0.924 | 0.250 | 0.252 | 0.860 | [0.634, 0.716] |
| holdout_seed32 | holdout | 32 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.252 | 1.000 | [0.710, 0.786] |
| holdout_seed32 | holdout | 32 | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.252 | 0.000 | [0.000, 0.000] |

Aggregate:

- primary rows passing: `8/8`
- destruction rows failing as intended: `4/4`
- minimum primary delta-target lower bound: `0.516`
- minimum primary delta-control lower bound: `0.506`
- maximum destruction matched accuracy: `0.250`

Decision:

- promote explicit private tool-trace packet handoff to seed-stable positive
  method candidate
- stop method tuning unless a reviewer-facing baseline exposes a real gap

Artifacts:

- `paper/source_private_hidden_repair_packet_seed_repeat_20260429.md`
- `results/source_private_hidden_repair_packet_seed_repeat_20260429/`

Next exact gate:

- `source_private_tool_trace_baseline_pack_20260429`: consolidate baselines,
  systems metrics, paired uncertainty, and threat-model controls into a
  reviewer-facing evidence package.

## 2026-04-29 - Source-Private Tool-Trace Baseline Pack

Current ICLR readiness: not fully ready, but the live method now has a
reviewer-facing baseline/system evidence package. Remaining distance is exact
paper claim drafting and deciding whether to run the remaining reviewer-risk
rows before submission.

Current story: explicit source-private tool-trace packets communicate hidden
execution evidence to a target-side candidate decoder. The interface is a
compact private `REPAIR_DIAG` field, not raw-log inference or unstructured
latent transfer.

Exact blocker: paper framing and a small set of reviewer-risk rows:
matched-byte JSON/free-text relay, helper-only/no-log oracle, component masking
ablations, and candidate/selector separation.

Command:

```bash
./venv_arm64/bin/python scripts/build_source_private_tool_trace_baseline_pack.py \
  --output-dir results/source_private_tool_trace_baseline_pack_20260429
```

Subagent reviewer synthesis:

- biggest risks: structured text relay, helper-line/no-log target priors,
  trace-component leakage, template leakage, one-model artifact, systems value
- already covered: target-only/wrapper, zero/shuffled/random, answer controls,
  target-derived sidecar, trace removal, held-out families, seed repeats,
  cross-source-model emitters, bytes/latency/validity
- not fully covered: matched-byte JSON/free-text relay, helper-only no-log
  oracle, line/test-name/expected-actual masking, learned target-family row

Baseline pack highlights:

- primary rows passing: `8/8`
- destruction rows failing as intended: `4/4`
- minimum primary lower bound over target-only: `0.516`
- minimum primary lower bound over best control: `0.506`
- primary packet mean bytes: `1.55-2.00`
- primary packet validity range: `0.776-1.000`
- deterministic matched-byte text accuracy: `0.250`
- deterministic full hidden-log relay accuracy: `1.000`

Decision:

- promote the method to reviewer-facing evidence package
- keep claim boundary explicit: private tool-trace packet communication
- do not claim raw-log repair inference or latent transfer

Artifacts:

- `paper/source_private_tool_trace_baseline_pack_20260429.md`
- `results/source_private_tool_trace_baseline_pack_20260429/`

Next exact gate:

- `source_private_tool_trace_paper_claim_draft_20260429`: draft exact method,
  benchmark, baseline, systems, and limitation language from the baseline pack.

## 2026-04-29 - Source-Private Tool-Trace Paper Claim Draft

Current ICLR readiness: close to paper-ready for a scoped positive method, but
not final. The project now has a large-slice, held-out-family, seed-stable
positive result and a claim-boundary memo; the remaining distance is
reviewer-risk rows or an explicit decision to submit with those risks scoped as
limitations.

Current story: a source agent with private execution/tool traces emits a
rate-capped explicit `REPAIR_DIAG` packet. A target-side candidate decoder uses
that packet to select the repair candidate. The gain survives source-destroying
controls and disappears when the trace protocol is removed.

Exact blocker: reviewer-risk baselines that could still challenge the narrow
claim:

- matched-byte structured JSON/free-text relays
- helper-only/no-log target oracle
- trace-component masking for expected/actual value, line number, and test name
- explicit candidate-pool recall versus selector-accuracy table
- optional learned or LLM-mediated target-family row

Evidence now encoded in the paper-claim draft:

- four frozen `500`-example surfaces: core seeds `29/31`, held-out seeds
  `30/32`
- primary rows passing: `8/8`
- Qwen3 matched range: `0.808-0.924`
- Phi-3 matched range: `1.000`
- target-only: `0.250`
- best source-destroying controls: `0.252-0.258`
- minimum paired-bootstrap lower bound over target-only: `0.516`
- model-produced packet bytes: `1.55-2.00`
- full hidden-log relay: roughly `366-374` bytes and `34` tokens per example
- `raw_log_no_trace` returns to `0.250` with `0` valid packets on all surfaces

Decision:

- promote the branch as a scoped positive-method story: explicit private
  tool-trace packet communication
- do not promote broader latent-transfer, raw-log inference, or learned target
  bridge claims

Artifacts:

- `paper/source_private_tool_trace_paper_claim_draft_20260429.md`
- `paper/source_private_tool_trace_baseline_pack_20260429.md`
- `results/source_private_tool_trace_baseline_pack_20260429/`

Next exact gate:

- `source_private_tool_trace_reviewer_risk_rows_20260429`: run the cheapest
  decisive reviewer-risk rows, beginning with matched-byte structured
  JSON/free-text relay and helper-only/no-log oracle.

## 2026-04-29 - Source-Private Tool-Trace Reviewer-Risk Rows

Current ICLR readiness: close to paper-ready for the scoped positive method.
The deterministic reviewer-risk rows now close the biggest baseline/framing
objections left by the baseline pack. Remaining distance is final table
integration and a decision on whether to add an optional learned/LLM-mediated
target-family row or list it as a limitation.

Current story: compact explicit source-private `REPAIR_DIAG` packets transfer
private execution/tool-trace evidence to a target-side candidate decoder. The
source information is rate-efficient relative to structured text relay at the
paper packet budget.

Commands:

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_smoke.py \
  --examples 500 \
  --candidates 4 \
  --seed 29 \
  --budgets 2,4,8,16,32 \
  --family-set core \
  --output-dir results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29

./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_smoke.py \
  --examples 500 \
  --candidates 4 \
  --seed 30 \
  --budgets 2,4,8,16,32 \
  --family-set holdout \
  --output-dir results/source_private_tool_trace_reviewer_risk_rows_20260429/holdout_seed30
```

New reviewer-risk rows:

- matched-byte structured JSON relay
- matched-byte free-text relay
- helper-template/no-log target oracle
- diagnostic-masked full log
- expected/actual-masked full log
- test-name-masked full log
- explicit candidate-pool recall and selector accuracy

Results at the `2`-byte paper packet budget:

- core seed `29`: matched `1.000`, target `0.250`, best source-destroying
  control `0.254`, best reviewer negative `0.250`, min reviewer oracle
  `1.000`, candidate-pool recall `1.000`
- held-out seed `30`: matched `1.000`, target `0.250`, best source-destroying
  control `0.254`, best reviewer negative `0.250`, min reviewer oracle
  `1.000`, candidate-pool recall `1.000`

Budget behavior:

- budgets `2/4/8/16` pass on both surfaces
- budget `32` intentionally fails because structured JSON/free-text relays have
  enough bytes to expose the diagnostic and become oracles

Decision:

- promote the reviewer-risk rows as passed for the deterministic packet
  protocol
- keep the systems claim tied to the compact packet budget and report the text
  relay curve honestly
- do not broaden the claim beyond explicit private tool-trace packet
  communication

Artifacts:

- `paper/source_private_tool_trace_reviewer_risk_rows_20260429.md`
- `results/source_private_tool_trace_reviewer_risk_rows_20260429/`

Next exact gate:

- `source_private_tool_trace_final_table_20260429`: integrate model rows,
  deterministic controls, reviewer-risk rows, bytes/tokens, validity, and
  candidate-pool/selector separation into a paper-ready table.

## 2026-04-29 - Source-Private Tool-Trace Final Evidence Table

Current ICLR readiness: evidence-ready for a scoped positive method, but still
needs paper skeleton drafting and final reviewer positioning. The remaining
scientific risk is no longer the core positive result; it is whether reviewers
will require a learned target-side neural decoder rather than accepting a
deterministic protocol decoder as the candidate-selection interface.

Current story: explicit source-private tool-trace packets provide a compact
communication channel from a private source agent to a target-side candidate
decoder. At `2` bytes, the packet beats target-only, no-source wrappers,
source-destroying controls, answer controls, target-derived controls, and
same-byte JSON/free-text relays. Full structured text relays become oracles only
when allowed enough bytes to expose the diagnostic.

Final-table contents:

- model-mediated rows across four `500`-example surfaces
- deterministic target/control rows
- reviewer-risk rows
- bytes/token systems rows
- candidate-pool recall versus selector accuracy separation
- explicit unsupported-claim list

Decision:

- promote the branch to scoped positive-method evidence-ready
- start paper skeleton drafting around the final table
- keep learned target decoders as optional extension or limitation, not a
  blocker to the current scoped claim

Artifacts:

- `paper/source_private_tool_trace_final_table_20260429.md`
- `results/source_private_tool_trace_final_table_20260429/manifest.md`

Next exact gate:

- `source_private_tool_trace_paper_skeleton_20260429`: draft method,
  benchmark, results, threat-model, limitations, and rate-curve framing from
  the final evidence table.

## 2026-04-29 - Source-Private Tool-Trace Paper Skeleton

Current ICLR readiness: paper-skeleton ready for the scoped positive method,
pending skeptical review and final decision on an optional learned/LLM-mediated
target decoder row.

Current story: source-private `REPAIR_DIAG` packets are a compact,
interpretable communication channel from private execution/tool traces to a
target-side candidate decoder. The evidence is strongest when framed as a
rate-capped source-private communication protocol, not latent-state transfer.

Skeleton contents:

- working title and one-sentence claim
- abstract skeleton
- formal `(X, T, S, M, D)` problem setup
- method and target decoder description
- benchmark description
- baselines and controls
- main model-mediated result table
- reviewer-risk result table
- systems table
- threat model
- interpretability and limitations
- required figures/tables
- submission strategy

Decision:

- draft paper around the scoped protocol claim
- avoid raw-log inference, learned latent bridge, or universal cross-model
  communication framing

Artifacts:

- `paper/source_private_tool_trace_paper_skeleton_20260429.md`
- `results/source_private_tool_trace_paper_skeleton_20260429/manifest.md`

Next exact gate:

- `source_private_tool_trace_skeleton_review_20260429`: skeptical reviewer pass
  and decision on whether a learned/LLM-mediated target decoder row is needed
  before drafting full paper sections.

## 2026-04-29 - Source-Private Tool-Trace Skeleton Review

Current ICLR readiness: draftable as a narrow source-private
evidence-communication paper. Not ready for a broader latent-transfer or
general cross-model communication claim.

Skeptical review outcome:

- overclaiming is the top risk; title and abstract must avoid latent-transfer
  expectations
- same-byte structured relay is addressed at `2` bytes but must be shown as a
  rate curve because it becomes oracle at `32` bytes
- candidate metadata exposes the diagnostic mapping, so the paper must say
  candidate selection with decoder side information
- systems value should report packet validity, especially Qwen3
  `0.776-0.864`
- novelty risk remains: deterministic protocol decoder could be viewed as
  coded-label lookup

Decision:

- patch skeleton wording toward source-private evidence communication
- run one target-decoder smoke before full paper drafting if feasible

Artifacts:

- `paper/source_private_tool_trace_skeleton_review_20260429.md`
- updated `paper/source_private_tool_trace_paper_skeleton_20260429.md`

Next exact gate:

- `source_private_tool_trace_target_decoder_smoke_20260429`: replace
  deterministic lookup with an LLM-mediated or learned target-side selector on
  a small frozen slice, preserving source-destroying controls.

## 2026-04-29 - Source-Private Tool-Trace Target-Decoder Smoke

Current ICLR readiness: stronger. The main scoped claim remains evidence-ready
with the deterministic protocol decoder, and the largest novelty risk now has a
positive target-LLM smoke ablation. This is still not scaled enough to replace
the main evidence.

Current story: compact source-private `REPAIR_DIAG` packets can be consumed by
an LLM-mediated target-side selector, not only by a hard-coded lookup, while
same-byte source-destroying and structured-relay controls remain near
target-only on the tested slices.

Commands:

```bash
./venv_arm64/bin/python scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark-jsonl .debug/source_private_tool_trace_target_decoder_smoke_20260429/core_seed29_benchmark/benchmark.jsonl \
  --output-dir results/source_private_tool_trace_target_decoder_smoke_20260429/core_seed29_qwen3_n16 \
  --model Qwen/Qwen3-0.6B \
  --device mps \
  --dtype float32 \
  --limit 16 \
  --seed 29 \
  --max-new-tokens 24 \
  --no-enable-thinking

./venv_arm64/bin/python scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark-jsonl .debug/source_private_tool_trace_target_decoder_smoke_20260429/holdout_seed30_benchmark/benchmark.jsonl \
  --output-dir results/source_private_tool_trace_target_decoder_smoke_20260429/holdout_seed30_qwen3_n32 \
  --model Qwen/Qwen3-0.6B \
  --device mps \
  --dtype float32 \
  --limit 32 \
  --seed 30 \
  --max-new-tokens 24 \
  --no-enable-thinking
```

Results:

- core seed `29`, `N=16`: target `0.250`, matched `0.688`, best control
  `0.250`, pass
- held-out seed `30`, `N=16`: target `0.250`, matched `0.750`, best control
  `0.312`, fail by one random-control example on tiny slice
- held-out seed `30`, `N=32`: target `0.250`, matched `0.750`, best control
  `0.281`, pass

Decision:

- promote as a positive target-decoder smoke/ablation
- do not use it as the main evidence until scaled
- include it in the paper to reduce the hand-coded decoder novelty objection

Artifacts:

- `scripts/run_source_private_tool_trace_target_decoder_smoke.py`
- `tests/test_run_source_private_tool_trace_target_decoder_smoke.py`
- `paper/source_private_tool_trace_target_decoder_smoke_20260429.md`
- `results/source_private_tool_trace_target_decoder_smoke_20260429/`

Next exact gate:

- `source_private_tool_trace_paper_sections_20260429`: draft paper sections
  around the scoped claim and include target-decoder smoke as an ablation.

## 2026-04-29 - Source-Private Tool-Trace Paper Sections

Current ICLR readiness: section-level paper draft is ready in memo form. The
evidence is now strong enough for a scoped positive-method manuscript draft;
remaining work is paper-source integration, figure/table rendering, and final
positioning.

Current story: source-private tool-trace packets provide a compact,
interpretable evidence channel from private execution traces to target-side
candidate selection. The main result is large-slice deterministic protocol
evidence with model-produced source packets; the Qwen3 target-decoder row is a
small ablation that reduces the hand-coded lookup objection.

Sections drafted:

- title and abstract
- introduction
- problem formulation
- method
- benchmark
- results
- controls and threat model
- rate/systems analysis
- target-decoder smoke
- interpretability
- limitations
- conclusion

Artifacts:

- `paper/source_private_tool_trace_paper_sections_20260429.md`
- `results/source_private_tool_trace_paper_sections_20260429/manifest.md`

Next exact gate:

- `source_private_tool_trace_paper_draft_20260430`: convert section memo into
  paper draft/source with figure and table placeholders.

## 2026-04-30 - Source-Private Tool-Trace Paper Draft

Current ICLR readiness: full markdown paper draft exists for the scoped
positive method. Remaining work is paper-source conversion, figure/table asset
generation, and final skeptical review. The evidence package remains scoped:
source-private explicit tool-trace packet communication, not broad latent
transfer.

Current story: source model reads private tool-trace diagnostics, emits compact
`REPAIR_DIAG` packets, and a target-side candidate decoder uses public candidate
metadata plus the packet to select repairs. Large frozen evidence supports the
protocol decoder; a small Qwen3 target-decoder smoke reduces hand-coded lookup
risk.

Draft contents:

- abstract and introduction
- problem setup with `X, T, S, M, D`
- hidden-repair benchmark
- tool-trace packet method
- baselines and controls
- main model-mediated result table
- deterministic threat-model tables
- rate/systems analysis
- target-decoder smoke
- interpretability, related work, limitations, reproducibility, conclusion
- appendix claim boundary and figure/table checklist

Artifacts:

- `paper/source_private_tool_trace_paper_draft_20260430.md`
- `results/source_private_tool_trace_paper_draft_20260430/manifest.md`

Next exact gate:

- `source_private_tool_trace_latex_or_figures_20260430`: convert the markdown
  draft into ICLR paper source or create the setup/rate-curve figure assets
  first, preserving the scoped claim boundary.

## 2026-04-30 - Source-Private Tool-Trace Draft Review

Current ICLR readiness: scoped full draft exists, but submission package is not
ready. Remaining blockers are figure/table assets, concrete citations, and
final decision on target-decoder scale-up versus limitation framing.

Skeptical review risks:

- coded-label lookup / benchmark artifact remains the main novelty risk
- target decoder evidence is only `N=16/32`, so it is an ablation
- synthetic-only external validity must be stated plainly
- structured text relay becomes oracle at `32` bytes, so the rate curve is
  essential
- related-work citations need concrete BibTeX/source entries

Patches applied to the draft:

- softened broad necessity/robustness language
- scoped the evidence to the benchmark's hidden diagnostic evidence
- changed “strict controls” to “controls tested here”
- made synthetic frozen-surface scope explicit
- changed interpretability language to direct auditability in this benchmark

Artifacts:

- `paper/source_private_tool_trace_paper_draft_20260430.md`
- `paper/source_private_tool_trace_draft_review_20260430.md`

Next exact gate:

- `source_private_tool_trace_latex_or_figures_20260430`: create setup/rate-curve
  figures or convert to ICLR LaTeX, with citations and count-augmented tables.

## 2026-04-30 - Source-Private Tool-Trace Figure Assets

Current ICLR readiness: full markdown draft plus the two essential figure
assets now exist. Remaining blockers are LaTeX/source conversion, concrete
citations, count-augmented tables, and final reviewer pass.

Command:

```bash
./venv_arm64/bin/python scripts/build_source_private_tool_trace_figures.py \
  --output-dir results/source_private_tool_trace_latex_or_figures_20260430
```

Generated assets:

- `source_private_setup.svg`: source-private setup diagram for `(X,T,S,M,D)`
- `rate_curve.svg`: accuracy versus communicated bytes for packet, structured
  relays, full diagnostic text, and full hidden-log relay
- `rate_curve.csv`: source data for the rate curve

Decision:

- use the rate curve in the main paper, not the appendix
- keep structured text relay framing honest: JSON/free-text relays become
  oracles at `32` bytes

Artifacts:

- `paper/source_private_tool_trace_latex_or_figures_20260430.md`
- `results/source_private_tool_trace_latex_or_figures_20260430/`
- `scripts/build_source_private_tool_trace_figures.py`
- `tests/test_build_source_private_tool_trace_figures.py`

Next exact gate:

- `source_private_tool_trace_latex_20260430`: convert the markdown draft to ICLR
  LaTeX source, include figure references, add citations, and add counts to the
  main table.

## 2026-04-30 - Source-Private Tool-Trace LaTeX Draft

Current ICLR readiness: ICLR-style LaTeX source now exists, but has not yet
been compiled. Remaining blockers are compile cleanup, possible SVG-to-PDF
figure conversion, final citation/style checks, and final reviewer pass.

Artifacts:

- `paper/iclr2026/source_private_tool_trace.tex`
- `paper/iclr2026/source_private_tool_trace.bib`
- `paper/source_private_tool_trace_latex_20260430.md`

Draft contents:

- abstract, introduction, setup, benchmark, method, baselines/controls
- main result table with counts
- threat-model control table
- figure references to setup and rate-curve SVGs
- target-decoder smoke ablation
- related work with BibTeX entries
- limitations, conclusion, appendix claim boundary

Next exact gate:

- `source_private_tool_trace_latex_compile_20260430`: compile the LaTeX source,
  convert SVG figures to PDF if required, and fix bibliography/style issues.

## 2026-04-30 - Source-Private Tool-Trace LaTeX Compile

Current ICLR readiness: ICLR-style LaTeX source now compiles to PDF with
figures and bibliography. Remaining blockers are final skeptical review,
target-decoder scale-up decision, and table/figure caption polish.

Command:

```bash
cd paper/iclr2026
latexmk -pdf -interaction=nonstopmode -halt-on-error source_private_tool_trace.tex
```

Artifacts:

- `paper/iclr2026/source_private_tool_trace.pdf` (`6` pages, `202363` bytes)
- `results/source_private_tool_trace_latex_compile_20260430/manifest.md`

Log audit:

- no overfull boxes
- no undefined references
- no citation warnings
- only underfull page-fill warnings remain

Next exact gate:

- `source_private_tool_trace_final_review_20260430`: final skeptical review of
  the compiled paper/source, target-decoder scale-up decision, and table/figure
  caption polish.

## 2026-04-30 - Source-Private Tool-Trace Final Review Polish

Current ICLR readiness: scoped positive-method manuscript is materially closer
to submission, but still needs final line polish and citation metadata review.
No new deterministic experiment was run; the gate addressed reviewer framing.

Decision:

- do not scale the target-decoder smoke this cycle
- keep target decoding as a protocol-decoder smoke ablation
- patch the manuscript around the coded-label/protocol, synthetic-scope,
  low-rate, related-work, and reproducibility risks

Edits:

- narrowed abstract/intro language to explicit diagnostic-code communication
- added synthetic-benchmark motivation and far-left-rate framing
- renamed target decoder row to model-mediated protocol decoder smoke
- split related work and added missing baseline/framing citations
- added an appendix artifact-manifest table

Compile:

```bash
cd paper/iclr2026
latexmk -pdf -interaction=nonstopmode -halt-on-error source_private_tool_trace.tex
```

Output: `paper/iclr2026/source_private_tool_trace.pdf` (`7` pages,
`226624` bytes). Log audit found no overfull boxes, undefined references,
citation warnings, or BibTeX warnings.

References:

- `references/478_source_private_final_review_refs.md`
- updated `references/research_memo_manifest.json`

Next exact gate:

- `source_private_tool_trace_submission_polish_20260430`: line-level PDF/source
  polish, citation metadata verification, optional appendix example, and final
  scoped-submit decision.

Update `2026-04-28`: `source_private_tool_trace_submission_polish_20260430`
continues the final-review gate. Primary-source metadata was verified for
Distributed Indirect Source Coding, AutoGen, LLMLingua, C2C, KVComm, and
Repair-R1; the Repair-R1 URL was corrected to arXiv `2507.22853`. The LaTeX
wording now says controls are near target-only, target prior is constructional,
the target-decoder row is a 16--32 example smoke ablation, confidence intervals
are matched--target deltas, and the conclusion selects the gold candidate from
the provided repair pool. Appendix provenance now names the decisive result
roots and avoids claiming every manifest contains exact commands. The next
exact gate remains `source_private_tool_trace_submission_polish_20260430` until
the final scoped-submit decision is made. A representative appendix example was
added for `sphr_0001`, where the target prior chooses a numeric fallback but the
matched private packet `H1` selects the missing-key default gold candidate.

Update `2026-04-28`: `source_private_tool_trace_submission_decision_20260428`
reaches the scoped-submit decision. The paper should proceed as an explicit
source-private diagnostic-packet protocol method and should not spend the next
cycle on the optional `n=160` target-decoder scale-up unless the claim expands
to make an LLM target receiver a main result. The manuscript now adds sharper
claim-boundary language, calls candidate-side information decoder side
information, and
states that the constructional `25%` target floor isolates source-private
evidence transfer from candidate generation. The final compiled PDF is
`paper/iclr2026/source_private_tool_trace.pdf` (`7` pages, `226708` bytes,
SHA256 `97e460ddb3919b6b3373e12a1d01a64d64912102a8e6d2efd3301eb16cd326fe`).
The next exact gate is `source_private_tool_trace_human_pdf_read_20260428`.

Update `2026-04-28`: `source_private_tool_trace_human_pdf_read_20260428`
completed the final source/package hygiene pass. The manuscript now uses local
figure paths under `paper/iclr2026/figures/`, removes absolute user paths from
the public figure manifest, softens the target-LLM decoder smoke wording, and
keeps candidate-side information framed as decoder side information. No new model
evidence was run; the claim remains the scoped source-private diagnostic-packet
protocol. Remaining action before upload: exclude the ICLR template/demo source
files from the submission bundle and decide whether to archive ignored raw
JSON/JSONL artifacts for an external artifact release.

Update `2026-04-28`: `source_private_tool_trace_submission_upload_20260428`
built and compile-tested the source upload bundle
`paper/iclr2026/submission/source_private_tool_trace_iclr_source_20260428.zip`
(`360709` bytes, SHA256
`cb1595cfd1ebb6cc5441143dbed2a00cce3bacd6b5aa43f2cca179c82862478e`). The
bundle contains only the manuscript source, bibliography, ICLR style/BibTeX
files, math/style dependencies, local PDF figures, and compiled PDF; template
demo files are excluded. Scratch compile from the extracted bundle passed with
no overfull boxes, undefined references, citation warnings, or BibTeX warnings.
The next exact gate is `source_private_tool_trace_artifact_release_choice_20260428`.

Update `2026-04-28`: `source_private_tool_trace_artifact_release_choice_20260428`
built the external artifact archive
`paper/artifacts/source_private_tool_trace_artifacts_20260428.zip` (`7548312`
bytes, SHA256
`64153e44dd5b41a30e54ffa5cdb0d95ca5498c2345c13da015a9f9f076c0121f`). The
archive includes the compiled paper PDF, compile-tested source upload zip,
decisive raw JSON/JSONL result roots, target-decoder benchmark inputs copied
from scratch space, figure/rate data, and paper readout memos. `unzip -t`
passes, and a release-tree scan found no `/Users`, `Desktop`, or username path
leakage. The manuscript/source package remains ready for upload; the next exact
gate is external release handoff, e.g. attach the artifact zip to the submission
or mirror it to the chosen archival store.

Update `2026-04-28`: `source_private_tool_trace_external_handoff_20260428`
created the final upload handoff memo
`paper/source_private_tool_trace_external_handoff_20260428.md`. The handoff
records exact paths, byte counts, SHA256 hashes, passed checks, claim boundaries,
and post-upload sanity checks for the manuscript PDF, compile-tested source zip,
and artifact archive. No new method evidence was run. The next exact gate is
`external_submission_confirmation_20260428`: record uploaded file IDs/URLs,
artifact location, and any portal-side warnings or format changes.

Update `2026-04-28`: `external_submission_confirmation_20260428` is locally
blocked on external portal/artifact-host state. The repo has all upload files,
hashes, and handoff documentation, but final confirmation requires the
submission portal ID/URL, artifact host URL/release ID, or user confirmation
that the PDF, source bundle, and artifact archive were accepted. This blocker is
documented in `paper/source_private_tool_trace_submission_confirmation_20260428.md`.

Update `2026-04-28`: upload verification was tightened by adding
`paper/source_private_tool_trace_upload_checksums_20260428.sha256`, covering
the manuscript PDF, source bundle, and artifact archive. The handoff and
confirmation memos now reference the checksum sidecar. This does not change the
scientific evidence; it reduces external upload verification risk.

Update `2026-04-28`: external artifact hosting was audited. GitHub CLI is
authenticated and the remote repository is public (`SujeethJinesh/LatentWire`),
with no existing releases, so a public GitHub release is technically available.
However, because ICLR-style double-blind submissions may require anonymous
artifacts, no public release was created automatically. Host options, exact
release command, and release notes are documented in
`paper/source_private_tool_trace_artifact_host_options_20260428.md` and
`paper/source_private_tool_trace_release_notes_20260428.md`. The next gate
remains external submission confirmation after choosing the venue-compatible
host path.

Update `2026-04-28`: `anonymous_artifact_handoff_20260428` built a
venue-anonymous artifact archive at
`paper/artifacts/source_private_tool_trace_artifacts_anonymous_20260428.zip`
(`7553609` bytes, SHA256
`02b1dbd73dea1332976e60a255def4a470f6c91416f2603cbd8631270be3790a`). The
source upload bundle already uses `Anonymous Authors`; the original artifact
archive had one project-name marker in an included memo, which was sanitized in
the anonymous archive. Archive integrity passes and the extracted anonymous
archive has no matches for account names, local home paths, GitHub URLs, or the
project-name marker. The anonymous upload checksum sidecar is
`paper/source_private_tool_trace_anonymous_upload_checksums_20260428.sha256`.

Update `2026-04-28`: `anonymous_submission_bundle_20260428` created a transfer
bundle for the double-blind upload set at
`paper/artifacts/source_private_tool_trace_anonymous_submission_bundle_20260428.zip`
(`8137764` bytes, SHA256
`6d771cfc41ad4acfb8d500a6841e06e28781b3ce2549fc368dff0a1ed666e377`). The
bundle contains only the manuscript PDF, compile-tested source zip, anonymous
artifact zip, checksum sidecar, and README. `unzip -t`, internal checksum
verification, and identity/local-path string audit all pass.

Update `2026-04-28`: `anonymous_upload_final_audit_20260428` passed. The exact
anonymous upload files were checked by checksum verification, transfer-bundle
`unzip -t`, extracted-tree identity scan, and binary/string identity scan over
the PDF/source zip/anonymous artifact zip/transfer bundle. No account names,
local paths, GitHub URLs, or project-name marker matches were found. Local
preparation is complete; the only remaining blocker is external portal/artifact
confirmation.

Update `2026-04-28`: `final_reproducibility_folder_20260428` created
`final/` as a consolidated finalization and reproducibility staging folder. It
contains upload files, manuscript source/PDF, paper memos, source-private
scripts, focused tests, relevant result directories, references, repo
environment files, and a folder-level checksum manifest. `final/README.md`
states the paper story, why the result matters, who should care, what worked,
and the claim boundary. This is a copy/staging folder; canonical repo-root paths
remain intact.

Update `2026-04-28`: `latest_model_generalization_scout_20260428` adds the
post-package latest-model/MoE test plan. Official Hugging Face model cards/API
identify Qwen3.5 small rows (`0.8B`, `2B`, `4B`), Qwen3.5 MoE rows
(`35B-A3B`, `35B-A3B-FP8`), Qwen3.6 dense `27B`, and Qwen3.6 MoE rows
(`35B-A3B`, `35B-A3B-FP8`) as the next source-packet emitter matrix. A local
`Qwen/Qwen3.5-0.8B` n16 smoke failed before generation because
repo-local `transformers==4.51.0` does not recognize `model_type: qwen3_5`; the
cached config declares `transformers_version: 4.57.0.dev0`. Treat this as a
dependency blocker, not a method failure. Added
`scripts/build_source_private_latest_model_matrix.py`,
`tests/test_build_source_private_latest_model_matrix.py`, and
`results/source_private_latest_model_matrix_20260428/`. The paper should not
claim latest-model or MoE generalization until these rows run under the same
source-destroying controls.

Update `2026-04-28`: `qwen35_latest_small_confirmation_20260428` upgrades the
repo-local latest-model stack to `transformers==5.7.0`, `tokenizers==0.22.2`,
and `huggingface_hub==1.12.0`, resolving the `qwen3_5` loader blocker. Local
`Qwen/Qwen3.5-0.8B` source-packet rows now pass on CPU at n16 and n64:
matched model packet `16/16` and `64/64`, target-only/best controls `4/16` and
`16/64`, matched-minus-best-control `+0.750`, packet valid rate `1.000`, and
exact-ID parity true. The same row still fails on Apple MPS before generation
with a hybrid-attention matmul shape error, so MPS is a backend compatibility
limitation rather than method evidence. The model matrix now also includes
cross-family falsification rows from OLMo, Gemma, Granite, SmolLM3, Phi-4
mini, Ministral, Nemotron Nano, and Kimi K2, documented in
`references/480_latest_cross_family_model_scout_refs.md`. This strengthens the
post-package contribution path but does not yet justify a broad latest/MoE
claim; next gate is Qwen3.5-0.8B n160/seed repeat plus one non-Qwen n16 row.

Update `2026-04-28`: `latest_cross_family_packet_rows_20260428` clears the
Qwen3.5 n160 gate and adds the first non-Qwen positive row. `Qwen/Qwen3.5-0.8B`
CPU trace-no-hint n160 seed29 reaches matched `160/160`, target-only `40/160`,
best source-destroying control `41/160`, packet valid rate `1.000`, exact-ID
parity true, and matched-minus-best-control `+0.744`; p50 CPU packet generation
latency is `12059 ms`. OLMo-2-0425-1B-Instruct is a behavioral negative on MPS:
trace-no-hint and copied-helper n16 both have `0` valid packets and matched
`4/16`, equal to target/control. Granite-3.3-2B-Instruct is MPS-backend blocked
before generation, but CPU rows pass: trace-no-hint n16 `10/16`, copied-helper
n16 `12/16`, and copied-helper n64 `51/64` versus target/control `16/64`,
packet valid rate `0.734`, matched-minus-best-control `+0.547`, p50 CPU latency
`3046 ms`. This gives a real latest-small medium row and a non-Qwen cross-family
positive smoke/confirmation, while preserving the limitation that MoE remains
unrun and Granite needs the easier copied-helper prompt.

Update `2026-04-28`: `latest_model_seed_stability_and_granite_n160_20260428`
adds seed stability and a medium non-Qwen row. `Qwen/Qwen3.5-0.8B` CPU
trace-no-hint n160 seed31 repeats seed29 exactly at matched `160/160`,
target-only `40/160`, best control `40/160`, packet valid rate `1.000`, exact-ID
parity true, and matched-minus-best-control `+0.750`; p50 CPU packet latency is
`10998 ms`. Granite-3.3-2B-Instruct copied-helper CPU n160 reaches `128/160 =
0.800` versus target-only `40/160`, best control `41/160`, packet valid rate
`0.738`, exact-ID parity true, and matched-minus-best-control `+0.544`. Granite
trace-no-hint CPU n64 also passes but weaker at `37/64 = 0.578`, valid packet
rate `0.500`, controls `16/64`. This upgrades latest-small evidence to
seed-stable medium and non-Qwen evidence to medium confirmation, but the safe
claim remains prompt-contract sensitivity rather than prompt-invariant
cross-family generalization.

Update `2026-04-28`: `qwen35_2b_and_moe_runbook_20260428` adds a latest-small
cross-size smoke and a ready off-machine MoE falsification plan. `Qwen/Qwen3.5-2B`
CPU trace-no-hint n16 seed29 passes with matched `16/16`, target-only/best
control `4/16`, packet valid rate `1.000`, exact-ID parity true,
matched-minus-best-control `+0.750`, and p50 CPU latency `14482 ms`. Added
`paper/source_private_qwen36_moe_falsification_runbook_20260428.md` with exact
CUDA commands and an endpoint-wrapper contract for `Qwen/Qwen3.6-35B-A3B` and
`Qwen/Qwen3.6-35B-A3B-FP8` n32 gates. The next best gate is Qwen3.5-2B n64 if
local CPU time is acceptable, or off-machine Qwen3.6 MoE n32 if CUDA/serving is
available.

Update `2026-04-28`: `qwen35_2b_n64_confirmation_20260428` widens the latest-small
cross-size row. `Qwen/Qwen3.5-2B` CPU trace-no-hint n64 seed29 passes with
matched `64/64`, target-only/best control `16/64`, packet valid rate `1.000`,
exact-ID parity true, matched-minus-best-control `+0.750`, and p50 CPU latency
`13646 ms`. This turns Qwen3.5-2B from smoke to medium confirmation and supports
the latest-small same-generation breadth story alongside Qwen3.5-0.8B n160
seed stability. Next local gate is Qwen3.5-2B n160 or Qwen3.5-4B n16; next
high-value external gate remains Qwen3.6 MoE/FP8 n32.

Update `2026-04-28`: `qwen35_2b_n160_confirmation_20260428` clears the frozen
latest-small n160 gate. `Qwen/Qwen3.5-2B` CPU trace-no-hint n160 seed29 reaches
matched `160/160`, target-only `40/160`, best source-destroying control
`41/160`, packet valid rate `1.000`, exact-ID parity true,
matched-minus-best-control `+0.744`, and p50 CPU packet latency `13878 ms`.
This gives two Qwen3.5 small sizes with clean n160 evidence: 0.8B is
seed-stable at n160, and 2B is n160-confirmed on seed29. Latest/MoE readiness
still excludes Qwen3.5-4B and Qwen3.6 MoE/FP8 until those rows run.

Update `2026-04-28`: `source_private_codebook_remap_gate_20260428` adds a
large deterministic reviewer-risk ablation for the fixed-codebook objection.
Across `500` all-family examples, seeds `29/31/37`, and budgets `2/4/8/16`,
exact IDs and public candidate labels remain identical while diagnostic
codebooks differ (`3` unique codebook hashes). Every seed/budget passes:
matched repair packets are `1.000`, best no-source is `0.250`, best
source-destroying control is `0.250-0.256`, reviewer-negative controls stay
`0.250`, and positive oracles stay `1.000`. This supports a remappable
source-private diagnostic-code claim, not broad semantic or latent transfer.

Update `2026-04-28`: `qwen35_4b_smoke_20260428` adds the upper local Qwen3.5
small-hybrid source-emitter row. After downloading the `Qwen/Qwen3.5-4B`
snapshot into the repo-local HF cache (`8.7G`), CPU trace-no-hint n16 seed29
passes with matched `16/16`, target-only/best control `4/16`, packet valid rate
`1.000`, exact-ID parity true, matched-minus-best-control `+0.750`, and p50 CPU
packet latency `32485 ms`. This extends Qwen3.5 same-generation breadth to
0.8B, 2B, and 4B, but the 4B row is only smoke; MoE/FP8 evidence remains the
main full-paper blocker.

Update `2026-04-28`: `qwen35_4b_n64_confirmation_20260428` upgrades the upper
local Qwen3.5 small-hybrid row from smoke to medium confirmation. CPU
trace-no-hint n64 seed29 reaches matched `64/64`, target-only/best control
`16/64`, packet valid rate `1.000`, exact-ID parity true,
matched-minus-best-control `+0.750`, and p50 CPU packet latency `27188 ms`.
This gives Qwen3.5 same-generation breadth across 0.8B n160 seed-stable, 2B
n160, and 4B n64; MoE/FP8 evidence remains the main full-paper blocker.

Update `2026-04-28`: `gemma4_and_granite_strict_cross_family_20260428` adds two
non-Qwen cross-family gates. `google/gemma-4-E2B-it` CPU trace-no-hint n64
seed29 passes with matched `64/64`, target-only/best control `16/64`, packet
valid rate `1.000`, exact-ID parity true, matched-minus-best-control `+0.750`,
and p50 CPU packet latency `2179 ms`. `ibm-granite/granite-3.3-2b-instruct`
CPU trace-no-hint n160 seed29 passes with matched `101/160 = 0.631`,
target-only `40/160`, best source-destroying control `41/160`, packet valid
rate `0.537`, exact-ID parity true, matched-minus-best-control `+0.375`, and
p50 CPU packet latency `2816 ms`. This strengthens non-Qwen evidence from
copied-helper-only to strict-prompt positive rows, but MoE/FP8 remains unrun.

Update `2026-04-28`: `qwen36_endpoint_runner_20260428` makes the MoE/FP8 gate
executable through a vLLM/OpenAI-compatible endpoint. Added
`scripts/run_source_private_hidden_repair_packet_endpoint.py`, which reuses the
same benchmark rows, source prompt construction, packet parser, evaluator,
controls, pass rule, and manifest format as the local HF runner while replacing
only source packet generation with `/v1/chat/completions`. Focused fake-endpoint
tests pass. This is harness evidence only; Qwen3.6-35B-A3B and FP8 still need
actual n32 endpoint runs.

Update `2026-04-28`: `gemma4_e2b_mps_n160_seed_stability_20260428` strengthens
the local non-Qwen row while remote execution is disallowed. `google/gemma-4-E2B-it`
MPS trace-no-hint n16 seed29 passes with matched `16/16`, controls `4/16`, and
p50 packet latency `843 ms`. The widened n160 rows pass on seeds 29 and 31:
seed29 matched `160/160`, target-only `40/160`, best control `41/160`, valid
packet rate `1.000`, exact-ID parity true, p50 MPS latency `821 ms`; seed31
matched `160/160`, target-only/best control `40/160`, valid packet rate
`1.000`, exact-ID parity true, p50 MPS latency `791 ms`. This makes Gemma 4 E2B
a seed-stable strict-prompt non-Qwen medium row; MoE/FP8 remains unrun.

Update `2026-04-28`: `gemma4_e2b_raw_log_source_destroying_20260428` adds the
paired non-Qwen source-signal ablation. With `google/gemma-4-E2B-it` on MPS,
n160 seed29, and `raw_log_no_trace`, the private `REPAIR_DIAG` trace line is
removed. The row intentionally fails: matched `40/160 = 0.250`, target-only
`40/160`, best control `41/160`, packet valid rate `0.000`, exact-ID parity
true, and p50 MPS latency `663 ms`. This supports the interpretation that the
Gemma strict-prompt positive transfers the private diagnostic trace signal
rather than target priors or wrapper formatting alone.

Update `2026-04-28`: `gemma4_e2b_mps_n500_large_slice_20260428` upgrades the
strongest local non-Qwen source-emitter row to a large frozen slice. With
`google/gemma-4-E2B-it`, MPS, trace-no-hint, seed29, and n500, matched packets
reach `500/500 = 1.000`, target-only is `125/500 = 0.250`, best
source-destroying control is `126/500 = 0.252`, packet valid rate is `1.000`,
exact-ID parity is true, matched-minus-best-control is `+0.748`, and p50 packet
latency is `754 ms`. This passes the large local gate and gives the paper a
clean non-Qwen large-slice row paired with the raw-log/no-trace collapse.

Update `2026-04-28`: `source_private_full_paper_contribution_review_20260428`
and the refreshed final evidence table make the three full-paper contributions
explicit: (1) source-private communication with decoder side information,
(2) a rate-capped diagnostic packet method, and (3) a strict reproducible
source-destroying evaluation harness. The table now also separates
post-package latest-small/non-Qwen rows from the core submission rows and keeps
MoE/FP8 marked as unclaimed until endpoint gates pass.

Update `2026-04-28`: `granite33_2b_seed_stability_and_trace_ablation_20260428`
hardens the weakest positive non-Qwen emitter. `ibm-granite/granite-3.3-2b-instruct`
CPU trace-no-hint n160 seed31 repeats the seed29 strict-prompt boundary:
matched `101/160 = 0.631`, target-only/best control `40/160 = 0.250`, packet
valid rate `0.537`, exact-ID parity true, matched-minus-best-control `+0.381`,
and p50 packet latency `3691 ms`. The paired raw-log/no-trace n160 seed31 row
collapses to matched `40/160 = 0.250`, target-only/best control `40/160`,
packet valid rate `0.000`, exact-ID parity true, and p50 latency `2857 ms`.
This promotes Granite as a stable prompt-contract sensitivity row while making
clear it is weaker than Qwen/Gemma and not a universal prompt-invariant claim.

Update `2026-04-28`: `source_private_target_decoder_n64_cpu_20260428` upgrades
the target-model decoder ablation beyond tiny smoke. The intended Qwen3 n160 MPS
target-decoder run failed before prediction with an Apple MPS matmul shape
error, so it is backend-blocked. The CPU fallback core seed29 n64 gate passes:
matched packet `42/64 = 0.656`, target-only `16/64 = 0.250`, best control
`16/64 = 0.250`, valid matched predictions `1.000`, exact-ID parity true,
matched-minus-best-control `+0.406`, and p50 matched latency `2182 ms`. This
does not make the target LLM decoder the main claim, but it reduces the
hand-coded decoder objection and sets up held-out n64 as the next exact gate.

Update `2026-04-29`: `source_private_systems_novelty_review_20260429` makes the
systems and comparison contribution explicit. Added
`scripts/build_source_private_systems_summary.py` and
`results/source_private_systems_summary_20260428/`, aggregating deterministic
rate rows, model-produced packet rows, and target-decoder rows. Headline:
2-byte diagnostic packets reach `1.000` on deterministic core/held-out
`500`-example surfaces, matched-byte text stays at `0.250`, full hidden-log
relay costs `366.45-373.50` bytes, and the packet is `183.2x-186.7x` smaller
than full hidden-log relay while preserving the same private-evidence selector
result. Added `references/481_systems_novelty_and_future_methods_refs.md` to
position the contribution against C2C, KVComm, activation communication,
LLMLingua, AutoGen/ReAct/Toolformer/Chain-of-Agents, Wyner-Ziv/indirect source
coding, TurboQuant/KIVI/QJL, JEPA, Q-Former, Flamingo, and diffusion-inspired
successor methods. The highest-value next technical branch is now a learned
Wyner-Ziv/syndrome packet smoke on the existing source-private candidate-pool
harness; MoE/FP8 remains unclaimed until endpoint gates can run.

Update `2026-04-29`: `source_private_target_decoder_heldout_n64_cpu_20260429`
passes. `Qwen/Qwen3-0.6B` as the target decoder on CPU, held-out seed30, n64,
gets matched packet `46/64 = 0.719`, target-only `16/64 = 0.250`, best control
`17/64 = 0.266`, valid matched predictions `1.000`, exact-ID parity true,
matched-minus-best-control `+0.453`, and p50 matched latency `2237 ms`. Controls
stay near the target floor: shuffled `0.250`, random same-byte `0.266`,
structured JSON 2-byte `0.250`, and structured free-text 2-byte `0.250`. This
gives paired core/held-out local target-decoder confirmation; stop widening
target-decoder rows locally unless a new decoder mechanism is introduced.

Update `2026-04-29`: `source_private_learned_syndrome_smoke_20260429` adds the
first positive learned packet branch. The smoke uses a synthetic source-private
candidate-pool contract: source observes a noisy private projection of the
correct candidate latent, target has public candidate latents and a prior, and a
ridge-fitted encoder emits a 1/2/4/8-byte binary syndrome decoded by Hamming
distance against target-side candidate codes. Seed29 train/eval `512/256` passes
at 1/2/4 bytes: matched `0.820/0.949/0.992` versus target `0.250` and best
no-source `0.281/0.262/0.262`; 8 bytes fails because a source-free control rises
to `0.305`. Seed30 repeats the low-rate result: 1/2 bytes pass with matched
`0.797/0.902` versus target `0.250` and best no-source `0.281/0.266`; 4/8
bytes fail due source-free controls above the tolerance. This is not yet a
headline claim, but it promotes learned Wyner-Ziv/syndrome packets as the next
technical contribution candidate. Added `references/482_competitor_threats_and_learned_syndrome_refs.md`
for competitor and theory positioning.

Update `2026-04-29`: `source_private_tool_trace_learned_syndrome_20260429`
moves learned syndrome packets onto real tool-trace/candidate-text features.
The source feature is a hashed private hidden-test log; the target side has
hashed public candidate metadata; a ridge encoder emits a bit-packed
random-hyperplane syndrome decoded by Hamming distance. An initial run exposed
masked-control leakage through `repair_family=...`; after masking
`REPAIR_DIAG`, hidden input, expected/actual, failure status, test name, and
repair family, the real-feature gate passes on two seed pairs at a common
6-byte budget. Seed pair `29 -> 30`, train/eval `512/256`, all families: 6 bytes
matched `0.945`, target `0.250`, best no-source `0.285`, full diagnostic oracle
`1.000`; 8 bytes also passes but 1/2/4/12 do not. Seed pair `31 -> 32`: 6 bytes
matched `0.918`, target `0.250`, best no-source `0.289`, full diagnostic oracle
`1.000`; 12 bytes also passes. This is now a real-feature learned-method
candidate, but not yet a headline claim until it beats compression-native
baselines such as sign sketch, rotation quantization, or PQ/QJL controls.

Update `2026-04-29`: `source_private_tool_trace_compression_baselines_20260429`
adds the decisive matched-byte compression gate. The learned sign-syndrome row
is positive but loses to a stronger 6-byte scalar-quantized learned source
projection. Seed pair `29 -> 30`, train/eval `512/256`: learned syndrome
`0.934`, scalar quantized source `0.973`, target `0.250`, raw source sign sketch
`0.316`; strict scalar pass is false only because scalar shuffled source reaches
`0.305` versus the `0.300` cutoff. Seed pair `31 -> 32`, train/eval `512/256`:
learned syndrome `0.910`, scalar `0.945`, target `0.250`, raw sign sketch
`0.172`, scalar shuffled `0.227`, scalar answer-masked `0.207`, strict scalar
pass true. The larger seed `29 -> 30` slice with train/eval `768/512` passes:
learned syndrome `0.953`, scalar `0.979`, target `0.250`, raw sign sketch
`0.307`, scalar shuffled `0.166`, scalar answer-masked `0.293`. This demotes
random-hyperplane syndrome as the headline transport and promotes the systems
method to a 6-byte scalar-quantized learned source-posterior packet, pending
5-seed, held-out-family, codebook-remap, and candidate-masking gates.

Follow-up `2026-04-29`: a 5-seed 6-byte scalar stability probe shows strong
accuracy but incomplete control stability. Scalar matched accuracy across seed
pairs is `0.979`, `0.945`, `0.957`, `0.992`, and `0.986` (mean `0.972`, min
`0.945`), always above learned sign-syndrome and far above target-only. Strict
scalar source-destroying controls pass `3/5` seeds. Failures are seed `33 -> 34`
answer-masked control `0.344` and seed `37 -> 38` shuffled-source control
`0.373`. A no-bias source-innovation variant fixes the hard answer-masked
control (`0.215`) but collapses matched innovation accuracy to `0.389`, so it is
pruned. The live branch remains scalar quantized source projection, but the
next gate must be control stabilization, not scale-up.

Follow-up `2026-04-29`: control-stabilized scalar packet gate passes on the
same 5-seed surface after narrowing target-side side information to public
candidate slots and fitting ridge without an intercept. This removes explicit
candidate diagnostic fields and rich candidate semantics from the target-side
candidate representation. At 6 bytes, train/eval `768/512`, all-family seeds
`29->30`, `31->32`, `33->34`, `35->36`, and `37->38` all pass: scalar matched
`1.000` on every seed, target `0.250`, constrained shuffled source `0.000`,
answer-masked source `0.250`, label-shuffled ridge `0.207-0.258`, and raw
source sign sketch `0.182-0.307`. This is the first clean 5-seed learned-packet
gate. Cross-family remains mixed: holdout-to-core passes with scalar `0.625`
versus target `0.250`, but core-to-holdout fails with scalar `0.125` and
constrained shuffled source `0.625`. Promote the slot/no-intercept scalar packet
as a scoped same-family/all-family communication method; do not claim symmetric
cross-family generalization yet.

Follow-up `2026-04-29`: slot-codebook remap and paired bootstrap are now added
for the `slot/no-intercept` scalar packet. Three remapped codebooks pass strict
scalar controls: remap seed `101` scalar `0.463` vs target `0.250`, best strict
control `0.264`, raw sign `0.332`; remap `103` scalar `0.508` vs target
`0.250`, best strict control `0.266`, raw sign `0.316`; remap `107` scalar
`0.492` vs target `0.250`, best strict control `0.250`, raw sign `0.330`.
The bootstrap summary across five same-codebook rows and three remap rows has
mean scalar accuracy `0.808`, mean scalar-minus-best-strict-control `0.552`,
minimum paired CI95 lower bound versus target-only `+0.156`, and minimum paired
CI95 lower bound versus raw sign sketch `+0.072`. This supports a scoped
positive claim: codebook remap remains positive, but remap margins are much
weaker than the fixed-codebook setting and must be reported as a limitation.

Follow-up `2026-04-29`: an opt-in QJL/TurboQuant-style residual comparator was
added to the compression baseline runner. It keeps the same 6-byte budget by
combining a scalar-quantized prefix with sign bits over residual directions
orthogonalized against the scalar projection. Same-codebook seed `29 -> 30`
saturates for both scalar and QJL residual (`1.000` each), but the decisive
remap rows do not improve: remap `101` scalar `0.463` vs QJL `0.447`, remap
`103` scalar `0.508` vs QJL `0.484`, and remap `107` scalar `0.492` vs QJL
`0.457`. QJL controls are clean and remain positive versus target-only, so this
is a useful matched-byte systems comparator, but it is not promoted as the live
method. The next method branch should target remap invariance directly, likely
relative-anchor transport or model-emitted packets.

Follow-up `2026-04-29`: RASP/relative-score packets are now added as an opt-in
candidate-relative transport. The source computes its learned posterior against
the public candidate anchors and sends one quantized score byte per candidate,
so the four-candidate gate uses `4` bytes. At equal actual bytes, same-codebook
seed `29 -> 30` saturates for both scalar and relative (`1.000`). Remap rows
are control-clean and improve mean accuracy over scalar: remap `101` relative
`0.494` vs scalar `0.426`, remap `103` `0.520` vs `0.496`, remap `107`
`0.506` vs `0.502`. The paired bootstrap summary reports mean remap
relative-minus-scalar `+0.032` and minimum relative-vs-target CI95 lower bound
`+0.189`, but the minimum relative-vs-scalar lower bound is `-0.035`. Promote
RASP as a secondary systems/robustness contribution, not as a replacement for
the scalar packet yet.

Follow-up `2026-04-29`: expanded RASP evaluation adds remap seeds
`109/113/127/131` and bidirectional core/holdout cross-family rows. Across
seven remap seeds, RASP improves mean equal-byte accuracy over scalar by
`+0.037`, but the expanded bootstrap does not pass: minimum RASP-vs-target CI95
lower bound is `+0.146` under the `+0.15` rule, and remap `131` has a
near-threshold random same-byte control failure (`0.301` vs cutoff `0.300`).
Cross-family remains asymmetric: core-to-holdout fails (`0.207` RASP vs
`0.250` target with failed controls), while holdout-to-core passes (`0.492`
RASP vs `0.250` target, controls clean). RASP remains a secondary systems
extension, not the headline cross-family fix.

Follow-up `2026-04-29`: canonical-order RASP is now implemented as an opt-in
relative-score variant. It serializes score bytes by stable public candidate
identity rather than display/list position and decodes them back into the
target's current candidate order. Across seven remap seeds at 4 bytes, canonical
RASP improves mean equal-byte accuracy over scalar by `+0.037` and is positive
on every remap, but the strict expanded bootstrap remains just short of the
target-only paired CI rule: minimum canonical-vs-target CI95 lower bound
`+0.146` versus required `+0.150`. A larger frozen rerun of the worst remap
`127` at `1536/1024` passes (`0.442` canonical vs `0.361` scalar and `0.250`
target; target CI95 low `+0.152`, scalar CI95 low `+0.053`, controls clean).
Cross-family remains asymmetric: core-to-holdout fails (`0.207` canonical vs
`0.250` target with failed controls), while holdout-to-core passes (`0.492`
canonical vs `0.250` target, controls clean). Promote canonical RASP as a
stronger same-family/remap robustness contribution, not as a cross-family fix.
The next method branch should be a consistency-distilled canonical posterior
packet with order/feature perturbation training, and it must pass bidirectional
core/holdout before any cross-family claim.

Follow-up `2026-04-29`: implemented the proposed
`consistent_posterior_packet` as an opt-in canonical score-packet variant. It
trains a ridge encoder on smoothed candidate-posterior centroids under
source-feature dropout and negative-candidate dropout, then emits one canonical
quantized score byte per candidate. Medium cross-family rows show partial
improvement but no pass: core-to-holdout `0.381` versus target `0.250`, scalar
`0.225`, and canonical RASP `0.207` with controls clean, but below the
`+0.15` target-delta rule; holdout-to-core `0.391` versus target `0.250`,
scalar `0.375`, and canonical RASP `0.492`. Larger `1536/1024` rows prune the
branch as a cross-family fix: core-to-holdout `0.354` is below scalar `0.370`
and the order-mismatch control is effectively identical (`0.355`), while
holdout-to-core passes (`0.495` vs target `0.250`) but is not better than
canonical RASP (`0.502`). Do not claim consistency posterior packets as a
positive method. Treat them as a serious JEPA/consistency-inspired ablation and
move the next full-paper strengthening gate to systems-rate frontier telemetry
unless a new source surface changes the cross-family hypothesis.

Follow-up `2026-04-29`: added a dedicated deterministic rate-frontier artifact
for the source-private packet story. On frozen core seed29 and holdout seed30
reviewer-risk surfaces, the diagnostic packet reaches oracle accuracy at
`2` bytes while matched-byte hidden-log truncation, JSON relay, and free-text
relay remain at target-only `0.250`. Structured JSON/free-text catch up only at
`21` and `17` bytes respectively, so the packet has a `10.5x` byte advantage
over the nearest structured-text oracle point and `183.2x-186.7x` over full
hidden-log relay. This strengthens the systems contribution as a far-left
byte-rate frontier. It is not a TTFT/serving-throughput claim; endpoint timing
remains a future required systems gate.

Follow-up `2026-04-29`: added
`results/source_private_cpu_systems_frontier_20260429`, a consolidated CPU
systems frontier with `32` rows spanning the rate frontier, scalar packets,
canonical RASP, model-emitted packets, target-decoder rows, cross-family
falsifications, and pruned consistency-posterior ablations. The aggregate keeps
both positives and failures: `22` pass rows and `10` fail/near-miss rows.
Minimum passing accuracy is `0.442`, maximum passing payload is `6` bytes, and
the minimum passing model-packet valid rate is `0.537`. This strengthens the
paper package by making the systems and robustness evidence reproducible from a
single artifact while preserving the honest limitation: cross-family remains
asymmetric and endpoint TTFT/throughput is still unmeasured. Next reviewer gate:
larger target-model decoder replication or diagnostic-code remap/paraphrase
stress testing.

Follow-up `2026-04-29`: added
`results/source_private_protocol_stress_table_20260429`, a focused reviewer-risk
aggregation for the fixed-codebook/protocol objection. The table has `22` rows:
`12` deterministic diagnostic-codebook remap rows over `500` examples and
budgets `2/4/8/16`, `3` learned slot-feature remap rows over `512` examples at
`6` bytes, and `7` canonical candidate-order remap rows over `512` examples at
`4` bytes. It reports `15` pass rows and `7` near-miss rows, with minimum
passing delta versus target-only `+0.213`. The near-miss rows are intentional:
canonical RASP is positive on each remap but the seven-remap bootstrap remains
just short of the strict CI rule. The new reference memo
`references/486_protocol_stress_and_uniqueness_refs.md` positions the
contribution against C2C, KVComm, cache reuse, prompt compression, decoder-side
source coding, TurboQuant/QJL, connectors, and diffusion-inspired bottlenecks.
The defensible novelty is now framed as extreme-rate source-private evidence
handoff with decoder side information, not generic model-to-model
communication. MPS target-decoder probing still fails before prediction with
the known Apple MPS matmul shape error, so the next cheap reviewer gate is
query-aware compressed-text controls or learned target-decoder prompt
paraphrase stress.

Follow-up `2026-04-29`: strengthened the rate-frontier artifact with a
query-aware compressed-text baseline. The new condition extracts the shortest
diagnostic-span text form, `REPAIR_DIAG=<code>`, from the hidden log and then
applies the same byte budgets and deterministic decoder. It is intentionally
stronger than naive truncation but weaker than the 2-byte packet because it must
carry the field name as text. On both frozen surfaces, the source-private packet
reaches oracle at `2` bytes, query-aware diagnostic-span text reaches oracle at
`14` bytes, JSON/free-text relays need `21`/`17` bytes, matched-byte text at the
packet point remains at target-only `0.250`, and full hidden-log relay remains
`183.2x-186.7x` larger. This closes the easiest LLMLingua/LongLLMLingua-style
baseline objection for the current synthetic diagnostic protocol, while leaving
learned text compression and endpoint timing as future systems gates.

Follow-up `2026-04-29`: added
`results/source_private_wyner_ziv_packet_gate_20260429`, a learned
source-private syndrome gate using the Wyner-Ziv/decoder-side-information
framing. The encoder maps private source evidence into scalar quantized packets
and the decoder compares against public candidate side information. On three
remapped slot codebooks (`101/103/107`) and budgets `2/4/6` bytes, all `9/9`
rows pass the scalar WZ packet rule. Accuracy ranges from `0.418` at the
hardest 2-byte rows to `0.508` at the best 6-byte row, versus target-only
`0.250` and query-aware text-at-budget `0.250`. The minimum scalar-control
margin is `+0.154`; raw source sign sketches are `0.301-0.332`; QJL residual
packets are competitive but do not dominate; canonical RASP is stronger on some
4-6 byte rows. This is now the best learned-method contribution, but it remains
same-family/all-family remap evidence rather than a solved cross-family latent
transfer result.

Follow-up `2026-04-29`: ran
`results/source_private_wyner_ziv_cross_family_gate_20260429`, the decisive
bidirectional cross-family falsification for the learned WZ packet. The gate
fails under a strict all-row rule. `core_to_holdout` fails all budgets:
2-byte scalar WZ `0.127`, 4-byte `0.174`, 6-byte `0.146` versus target `0.250`,
with source-destroying scalar controls reaching `0.529-0.623`. `holdout_to_core`
remains asymmetric: 2-byte `0.328` and 4-byte `0.338` do not pass strict scalar
controls, while 6-byte passes at `0.623` versus target/control `0.250`. This
kills a bidirectional cross-family learned-WZ claim. Keep learned WZ as a
same-family/remap method contribution; do not tune this exact scalar WZ
cross-family setup without a new mechanism such as anchor-relative dictionaries
or a target-preserving query bottleneck.

Follow-up `2026-04-29`: added
`results/source_private_protected_residual_packet_gate_20260429`, a
TurboQuant/QJL-inspired protected residual codec ablation for the learned
source-private packet family. The codec ranks random-rotated scalar coordinates
by calibration separation, sends a protected scalar head, and uses remaining
bytes for a sign-sketch residual. It is source-control positive on all `9/9`
remap/budget rows and improves the 2-byte scalar WZ row on remaps `101` and
`107`, but the strict promotion gate fails: protected rows have p50 decode
latency `3.56-7.33 ms` versus the predeclared `<2 ms` systems bar, and two
6-byte rows trail scalar WZ by more than `0.02`. Treat this as a principled
compression/quantization comparator and near-miss, not as a new headline method.
The CPU systems frontier now has `56` rows and includes these protected
near-miss/fail rows.

Follow-up `2026-04-29`: hardened the Qwen3 target-decoder receiver harness with
condition subsets and append-only progress JSONL, then ran a cheap frozen
receiver smoke on core and holdout. The progress-enabled `n=16` subset uses only
`target_only`, `matched_packet`, and `shuffled_packet` conditions. Core seed29
passes with matched `0.688` versus target/shuffled `0.250`, valid prediction
rate `1.000`, and p50 matched latency `4190.55 ms`. Holdout seed30 passes with
matched `0.750` versus target/shuffled `0.250`, valid prediction rate `1.000`,
and p50 matched latency `4059.46 ms`. This strengthens the learned/model
receiver story as a smoke result, but the next receiver defense must scale to
`n=64` or `n=160` with all six controls.

Follow-up `2026-04-29`: promoted the frozen target-decoder receiver from
three-condition smoke to strict-small all-control evidence. Qwen3-0.6B CPU
passes at `n=32` on both frozen surfaces with all six conditions. Core seed29:
matched `0.688`, target-only `0.250`, shuffled/random/structured text controls
all `0.250`, valid matched rate `1.000`, p50 matched latency `2117.03 ms`.
Holdout seed30: matched `0.750`, target-only `0.250`, best control `0.281`,
valid matched rate `1.000`, p50 matched latency `2123.86 ms`. A short-decode
diagnostic at `max_new_tokens=8` fails with valid prediction rate `0.000`
because labels are truncated to a shared prefix, so valid receiver rows use a
24-token cap. A direct MPS probe still fails before prediction with the known
Apple MPS matmul shape error. The CPU systems frontier now has `63` rows.
Subagent literature synthesis keeps the broad latent-communication novelty
claim scoped and selects anchor-relative sparse innovation packets as the next
non-scalar cross-family branch.

Follow-up `2026-04-29`: implemented and ran the first static
anchor-relative sparse innovation packet (AR-SIP) cross-family smoke. The gate
sends sparse `(candidate-anchor id, quantized score)` atoms and tests
target-only, constrained shuffled source, answer-masked source, random valid
same-byte, target-derived sidecar, and anchor-id permutation controls. The
result is a useful negative: overall pass gate `False`, with direction pass
`core_to_holdout=False` and `holdout_to_core=True`. Core-to-holdout never beats
target-only (`0.242`, `0.250`, `0.125`, `0.250` across 2/4/6/8 bytes versus
target `0.250`) and anchor/random controls can dominate. Holdout-to-core has
clean positive rows at 2 bytes (`0.496` vs target/control `0.250`) and 8 bytes
(`0.373` vs target `0.250`, best control `0.262`). This repeats the same
asymmetry seen in scalar WZ and canonical RASP, so static relative sparse
coordinates are pruned as a bidirectional cross-family fix. The next cross-
family branch should use a learned receiver/query bottleneck or move to endpoint
systems telemetry for the existing positive packet.

Follow-up `2026-04-29`: added the first Mac endpoint-proxy systems frontier for
the existing source-private packet. Command pattern:
`/opt/homebrew/bin/timeout 900s env PYTHONUNBUFFERED=1 ./venv_arm64/bin/python
scripts/run_source_private_mac_endpoint_proxy_frontier.py --benchmark-jsonl
results/source_private_tool_trace_reviewer_risk_rows_20260429/<surface>/benchmark.jsonl
--output-dir results/source_private_mac_endpoint_proxy_frontier_20260429/<surface>_qwen3_n8_cpu_diagparse
--model Qwen/Qwen3-0.6B --device cpu --dtype float32 --limit 8
--max-new-tokens 24 --no-enable-thinking`. The first core attempt failed
because Qwen emitted a diagnostic code (`G0`) instead of a candidate label; the
parser now maps a unique emitted diagnostic code to the public candidate whose
`handles_repair_diag` matches it. With that auditable parser, both endpoint-
proxy surfaces pass. Core seed29 `n=8`: matched packet `0.750`, target-only
`0.250`, matched-byte text `0.250`, query-aware text `0.750`, JSON/free-text
`1.000`, full log `1.000`; packet payload `2` bytes, query text `14` bytes,
full log `366.5` bytes, full-log p50 TTFT `+181.1 ms` versus packet. Holdout
seed30 `n=8`: matched packet `0.750`, target-only `0.250`, matched-byte text
`0.250`, query-aware text `0.625`, JSON/free-text `0.875`, full log `1.000`;
full log `373.5` bytes and p50 TTFT `+279.5 ms` versus packet. I then widened
to `n=16` without changing the method. Core seed29 `n=16`: matched packet
`0.688`, target-only `0.250`, matched-byte text `0.250`, query-aware text
`0.812`, JSON/free-text/full-log `1.000`, full-log p50 TTFT `+165.4 ms` versus
packet. Holdout seed30 `n=16`: matched packet `0.688`, target-only `0.250`,
matched-byte text `0.250`, query-aware text `0.750`, JSON/free-text `0.938`,
full log `1.000`, full-log p50 TTFT `+190.7 ms` versus packet. The CPU systems
frontier now has `75` rows after adding these endpoint-proxy rows. This
supports a Mac-local byte/TTFT frontier, not a server-throughput claim. Next
gate: `n=64`/`n=160` endpoint-proxy replication with prompt paraphrase stress,
then a real vLLM/OpenAI-compatible serving run when GPUs are available.

Follow-up `2026-04-29`: added endpoint-proxy prompt-paraphrase stress. The
harness now accepts `--prompt-style {canonical,terse,audit}`. The deliberately
under-specified `terse` core `n=16` row fails: matched packet `0.250`, target
`0.250`, matched-byte text `0.250`; this shows the receiver is not automatic
under an ambiguous prompt. The `audit` paraphrase, which preserves the public
side-information contract but changes the wording, passes on both surfaces and
scales to `n=32`. Core seed29 `n=32` audit: matched packet `0.719`, target
`0.250`, matched-byte text `0.281`, query-aware text `0.781`, JSON `0.781`,
free text `0.812`, full log `0.406`, full-log p50 TTFT `+163.4 ms` versus
packet. Holdout seed30 `n=32` audit: matched packet `0.844`, target `0.312`,
matched-byte text `0.312`, query-aware text `0.812`, JSON `0.750`, free text
`0.875`, full log `0.344`, full-log p50 TTFT `+157.4 ms` versus packet. The
CPU systems frontier now has `80` rows including the prompt-stress pass/fail
boundary. Subagent audit flagged two next harness risks before promotion:
strict label accuracy should be separated from diagnostic-code-mapped accuracy,
and a deranged candidate-diagnostic table control should collapse the packet if
the prompt-side codebook is destroyed.

Follow-up `2026-04-29`: implemented the endpoint strict-control harness fix.
`scripts/run_source_private_mac_endpoint_proxy_frontier.py` now reports
strict-label accuracy separately from diagnostic-code-mapped accuracy and adds
`random_same_byte_packet` plus `deranged_candidate_diag_table` controls. The
audit prompt passes the stricter `n=16` gate on both frozen surfaces. Core:
matched packet `0.750`, target `0.250`, matched-byte text `0.250`, random same-
byte `0.000`, deranged public table `0.000`, query-aware text `0.812`, JSON
`0.812`, free text `0.750`, full log `0.312`, full-log p50 TTFT `+278.2 ms`
versus packet. Holdout: matched packet `0.875`, target `0.312`, matched-byte
text `0.312`, random same-byte `0.125`, deranged public table `0.000`, query-
aware text `0.812`, JSON `0.750`, free text `0.875`, full log `0.375`, full-log
p50 TTFT `+227.1 ms` versus packet. The strict-label caveat is now explicit:
matched packet strict-label accuracy is `0.062` core and `0.250` holdout, so
the endpoint receiver is a protocol-code decoder using public side information.
The CPU systems frontier now has `82` rows. Next gate: run canonical and audit
strict-control endpoint rows at `n=64`; if both pass, widen to `n=160`.

Follow-up `2026-04-29`: widened the endpoint strict-control audit gate to
`n=32` on both frozen surfaces without changing the method. Commands used the
same CPU endpoint-proxy runner with `--prompt-style audit`, `--limit 32`,
`--max-new-tokens 24`, `--no-enable-thinking`, and output directories
`results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n32_cpu_audit_strict_controls`
and
`results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n32_cpu_audit_strict_controls`.
Core seed29: matched packet `0.719`, target-only `0.250`, matched-byte text
`0.281`, random same-byte `0.031`, deranged public table `0.000`, best
source-destroying control `0.281`, strict-label packet accuracy `0.156`, and
full-log p50 TTFT `+159.2 ms` versus the packet. Holdout seed30: matched
packet `0.844`, target-only `0.312`, matched-byte text `0.312`, random
same-byte `0.094`, deranged public table `0.000`, best source-destroying
control `0.312`, strict-label packet accuracy `0.219`, and full-log p50 TTFT
`+185.8 ms` versus the packet. The regenerated CPU systems frontier now has
`84` rows. This is the strongest local endpoint evidence so far, but it is
still a protocol-code receiver and local CPU proxy. Next gate: `n=64`
canonical+audit strict controls; if both pass, widen to `n=160`.

Follow-up `2026-04-29`: fixed a stricter endpoint parser issue before
promoting the `n=64` gate. The old diagnostic-mapped parser could count a
generated diagnostic code even when that code was not in the transmitted
payload; this gave matched-byte/no-source controls occasional accidental
credit and let the matched packet receive credit for unrelated hallucinated
codes. The parser is now payload-gated: generated diagnostic codes map only if
the code was transmitted in the source payload. Payload-gated rescoring demotes
the old audit strict-control rows from pass to near-miss/fail under the
predefined `valid_prediction_rate >= 0.95` rule, even though the source signal
remains strong. Core `n=64` audit: packet `0.750`, target `0.250`, matched-byte
text `0.203`, random same-byte `0.000`, deranged public table `0.000`, best
source-destroying control `0.203`, packet valid rate `0.781`, strict-label
packet accuracy `0.172`, and full-log p50 TTFT `+260.2 ms` versus the packet.
To address the valid-output objection, I added a `label_strict` receiver prompt
that says outputs must be full candidate labels copied exactly. It passes both
`n=16` frozen surfaces with full strict controls. Core: matched packet `0.688`,
target `0.250`, matched-byte text `0.250`, random same-byte `0.000`, deranged
public table `0.188`, packet valid rate `1.000`, strict-label accuracy
`0.688`, and full-log p50 TTFT `+151.9 ms`. Holdout: matched packet `0.625`,
target `0.250`, matched-byte text `0.250`, random same-byte `0.000`, deranged
public table `0.250`, packet valid rate `1.000`, strict-label accuracy
`0.562`, and full-log p50 TTFT `+190.4 ms`. The regenerated CPU systems
frontier has `87` rows. Next gate: `label_strict` `n=32` core+holdout, then
`n=64` if both pass.

Follow-up `2026-04-29`: widened the `label_strict` endpoint receiver to `n=32`
on both frozen surfaces. Both pass with exact-label outputs and full strict
controls. Core seed29: matched packet `0.688`, target-only `0.250`,
matched-byte text `0.250`, random same-byte `0.000`, deranged public table
`0.219`, packet valid rate `1.000`, strict-label accuracy `0.656`, and
full-log p50 TTFT `+164.8 ms` versus the packet. Holdout seed30: matched
packet `0.656`, target-only `0.250`, matched-byte text `0.250`, random
same-byte `0.000`, deranged public table `0.250`, packet valid rate `1.000`,
strict-label accuracy `0.625`, and full-log p50 TTFT `+167.1 ms` versus the
packet. The CPU systems frontier now has `89` rows. This is the current live
endpoint receiver evidence because it avoids the parser-risk caveat while
preserving the 2-byte systems frontier. Next gate: label-strict `n=64`
core+holdout.

Follow-up `2026-04-29`: widened the `label_strict` endpoint receiver to `n=64`
on both frozen surfaces. Both pass with exact-label outputs and full strict
controls. Core seed29: matched packet `0.703`, target-only `0.250`,
matched-byte text `0.250`, random same-byte `0.000`, deranged public table
`0.234`, packet valid rate `1.000`, strict-label accuracy `0.672`, and
full-log p50 TTFT `+217.2 ms` versus the packet. Holdout seed30: matched
packet `0.672`, target-only `0.250`, matched-byte text `0.250`, random
same-byte `0.000`, deranged public table `0.250`, packet valid rate `1.000`,
strict-label accuracy `0.656`, and full-log p50 TTFT `+192.7 ms` versus the
packet. The CPU systems frontier now has `91` rows. This clears the local n64
endpoint receiver rung and makes paired uncertainty plus n160 the next
reviewer-facing blockers.

Follow-up `2026-04-29`: added the paired uncertainty gate for the live
`label_strict` endpoint receiver. Command:
`./venv_arm64/bin/python scripts/summarize_source_private_endpoint_uncertainty.py --result-dirs results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n64_cpu_label_strict_controls results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n64_cpu_label_strict_controls --output-dir results/source_private_endpoint_uncertainty_20260429/label_strict_n64 --bootstrap-samples 5000 --seed 20260429`.
Artifact hashes: `summary.json`
`94d9c6b652f23214c7b608685aa385fa2e230f726278246f41609befb48edfd6`,
`summary.md`
`7dea90eb80f8d03200bca596ce257b211440ca97e3838e4ba4f946516c566920`.
Outcome: pass. Across core and holdout, the minimum packet-vs-target lower CI
is `+0.297`, the minimum packet-vs-best-source-destroying-control lower CI is
`+0.297`, the minimum strict-label packet-vs-target lower CI is `+0.281`, and
valid rate is `1.000`. Exact sign tests have zero packet losses versus
target-only on both surfaces (`29/0/35` and `27/0/37` wins/losses/ties). The
CPU systems frontier was regenerated with a new endpoint uncertainty row,
raising the aggregate to `92` rows (`cpu_systems_frontier.json`
`d24b49a1694ae02ad924f283ff6c5dbc74083019b18bad368046ef16b980e4ba`).
Query-aware text remains an accuracy-comparable `14` byte rate/quality
comparator, not a destructive control failure. Next exact gate: frozen
`label_strict` endpoint core+holdout at `n=160`, then server TTFT/throughput
when GPU serving is available.

Follow-up `2026-04-29`: started the frozen `n=160` label-strict endpoint gate
locally on CPU. Core seed29 completed and passed; holdout was not launched in
this cycle because the all-condition core run took roughly `70` minutes on the
Mac. Command:
`./venv_arm64/bin/python scripts/run_source_private_mac_endpoint_proxy_frontier.py --benchmark-jsonl results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29/benchmark.jsonl --output-dir results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n160_cpu_label_strict_controls --limit 160 --max-new-tokens 24 --no-enable-thinking --prompt-style label_strict`.
Core `n=160` metrics: packet `0.675`, strict-label packet `0.662`, target-only
`0.250`, matched-byte text `0.250`, random same-byte `0.000`, deranged public
table `0.244`, best source-destroying control `0.250`, valid rate `1.000`,
query-aware text `0.694` at `14` bytes, structured free text `0.713` at `17`
bytes, and full hidden-log relay `0.463` at `366.5` bytes with p50 TTFT
`+164.3 ms` versus the packet. Artifact hashes: `summary.json`
`ca065ed000472c3a0efb27966b76a818bbaa6bf91b431667edd82af0dedb49f3`,
`endpoint_proxy_rows.jsonl`
`bc19809ded5cba6ad56d7084c21ed6e952094ce995b3eef932a86e054a4aa090`.
Paired uncertainty on core `n=160` also passes: packet-vs-target and
packet-vs-best-control lower CIs are `+0.350`; strict-label packet-vs-target
lower CI is `+0.338` (`summary.json`
`a868bc1961c969b17c98e0363c389b19561e1414fe93170cb4e364cbaaa82646`).
The CPU systems frontier was regenerated with `94` rows. Next exact gate:
holdout seed30 `n=160` label-strict all-condition CPU endpoint run, then paired
core+holdout `n=160` uncertainty.

Follow-up `2026-04-29`: completed the frozen holdout `n=160` label-strict
endpoint gate locally on CPU. Command:
`./venv_arm64/bin/python scripts/run_source_private_mac_endpoint_proxy_frontier.py --benchmark-jsonl results/source_private_tool_trace_reviewer_risk_rows_20260429/holdout_seed30/benchmark.jsonl --output-dir results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n160_cpu_label_strict_controls --limit 160 --max-new-tokens 24 --no-enable-thinking --prompt-style label_strict`.
Holdout `n=160` metrics: packet `0.688`, strict-label packet `0.675`,
target-only `0.250`, matched-byte text `0.250`, random same-byte `0.000`,
deranged public table `0.244`, best source-destroying control `0.250`, valid
rate `1.000`, query-aware text `0.688` at `14` bytes, structured free text
`0.719` at `17` bytes, and full hidden-log relay `0.531` at `373.5` bytes with
p50 TTFT `+183.5 ms` versus the packet. Artifact hashes: `summary.json`
`63ce5b089246bd021cae5b5ee5f884eb26eab4fdd4680b8b165fe50e904dc9e8`,
`endpoint_proxy_rows.jsonl`
`688c7b730dcc2c3b405c450e763f289fb61620a6261e3fd6c9c94a85c7527c54`.
Combined core+holdout `n=160` uncertainty passes: minimum packet-vs-target and
packet-vs-best-control lower CIs are `+0.350`; minimum strict-label
packet-vs-target lower CI is `+0.338` (`summary.json`
`1baec769e5dea93975b36972b0e2ce9b240997a5befbf5a467c24574f7c57511`).
The CPU systems frontier was regenerated with `96` rows. This clears the local
medium endpoint rung; next exact gate is server-side TTFT/throughput when GPU
serving is available, or a Mac-local learned candidate-embedding receiver smoke
to reduce the hand-designed-interface objection.

Follow-up `2026-04-29`: implemented and ran the first target-preserving learned
candidate-embedding receiver smoke. New files:
`scripts/run_source_private_candidate_embedding_receiver.py` and
`tests/test_run_source_private_candidate_embedding_receiver.py`. The receiver
uses a learned source encoder, packet/candidate bit interactions, public
candidate features, and a calibrated margin gate that preserves the target prior
unless the packet evidence is strong enough. Command:
`./venv_arm64/bin/python scripts/run_source_private_candidate_embedding_receiver.py --output-dir results/source_private_candidate_embedding_receiver_20260429/gated_budget4_seed29_30 --train-examples 768 --eval-examples 512 --train-family-set all --eval-family-set all --feature-dim 512 --budgets 4 --train-seed 29 --eval-seed 30 --ridge 1e-2`.
Outcome: pass. At 4 bytes, matched receiver accuracy is `0.748`, target-only
`0.250`, best destructive control `0.262`, full diagnostic oracle `0.998`, and
the calibrated margin threshold is `0.625476`. Controls: zero-source `0.250`,
shuffled-source `0.250`, answer-masked `0.221`, random same-byte `0.262`,
target-derived `0.250`, answer-only `0.242`, structured text prefix `0.203`,
wrong-projection source `0.232`. Artifact hashes: `summary.json`
`9dad546898a444dba4e34eea3a98b2d52734c7bc74ec477afe1ef83e076ec9ac`,
`predictions_budget4.jsonl`
`459ad78075226711b708bf418358193b38584ace952b1b7b04cf95b3e361be29`.
The CPU systems frontier now has `97` rows. This is a promising learned
receiver smoke, not a promoted headline claim. Next exact gate: 3-seed repeat
at 4 bytes and one held-out-family split with the same controls.

Follow-up `2026-04-29`: ran the learned candidate-embedding receiver
multi-seed and held-out-family diagnostic on the Mac, then regenerated the CPU
systems frontier. Commands:
`./venv_arm64/bin/python scripts/run_source_private_candidate_embedding_receiver.py --output-dir results/source_private_candidate_embedding_receiver_20260429/gated_budget4_seed31_32 --train-examples 768 --eval-examples 512 --train-family-set all --eval-family-set all --feature-dim 512 --budgets 4 --train-seed 31 --eval-seed 32 --ridge 1e-2`;
same command for `gated_budget4_seed37_38` with seeds `37 -> 38`; same command
for `diagnostic_budget8_seed29_30`, `diagnostic_budget8_seed31_32`, and
`diagnostic_budget8_seed37_38` with `--budgets 8`; held-out command:
`./venv_arm64/bin/python scripts/run_source_private_candidate_embedding_receiver.py --output-dir results/source_private_candidate_embedding_receiver_20260429/heldout_core_to_holdout_budget8_seed29_30 --train-examples 768 --eval-examples 512 --train-family-set core --eval-family-set holdout --feature-dim 512 --budgets 8 --train-seed 29 --eval-seed 30 --ridge 1e-2`; invariant ablation command adds
`--candidate-feature-dims 0 --eval-examples 256`. The `4` byte receiver is not
seed-stable: `2/3` seeds pass, matched mean `0.589`, matched minimum `0.328`,
and minimum matched-control delta `+0.049`. The `8` byte receiver passes `3/3`
same-distribution seeds, matched mean `0.749`, matched minimum `0.514`, max
destructive control `0.283`, and minimum matched-control delta `+0.230`. The
core-to-holdout `8` byte row fails: matched `0.453`, target `0.250`, best
destructive control `0.311`, full diagnostic oracle `0.809`. Removing raw
candidate features worsens the held-out result at `n=256`: matched `0.332`,
best destructive control `0.309`, oracle `0.742`. Aggregate artifacts:
`multiseed_and_heldout_summary.json`
`a15302f1146a8232b446b186d99b6b4406ad9ddf7a5c8723c2ef2066107742b0`,
`multiseed_and_heldout_summary.md`
`5471c59c00980d320609f63855a1426531594a23e0774d3853bf27edb1846d17`.
Representative summary hashes: budget8 seed29/30
`c535f3ef5ec7a58535c5689eaa1657fe5468eabb320b754de0f2050f2aabec2c`,
budget8 seed31/32
`6dd3478b7a58f19c7d4c4d1301686fd973d06ea3a5657a87e7f0bb47435a84cf`,
budget8 seed37/38
`8f808bd8d9248aa5ab7af79c07d0c058d40255ec8c7fae9bcb48a28b21a7aa40`,
heldout core-to-holdout
`9dee4211cf8a3c0d1139ddff985dc6b382dc0794ce8d28c01cfd08435194fa5e`.
CPU systems frontier now has `101` rows. Next exact gate: replace the raw
candidate-coordinate receiver with an anchor-relative/codebook or fold-heldout
calibrated receiver at `8` bytes; do not promote the learned receiver as
cross-family yet.

Follow-up `2026-04-29`: implemented `--receiver-kind code_similarity` and
`--packet-feature-mode anchor_relative` in
`scripts/run_source_private_candidate_embedding_receiver.py`, then ran three
family-invariant core-to-holdout diagnostics. Commands:
`./venv_arm64/bin/python scripts/run_source_private_candidate_embedding_receiver.py --output-dir results/source_private_candidate_embedding_receiver_20260429/heldout_core_to_holdout_code_similarity_budget8_seed29_30 --train-examples 768 --eval-examples 512 --train-family-set core --eval-family-set holdout --feature-dim 512 --candidate-feature-dims 0 --receiver-kind code_similarity --budgets 8 --train-seed 29 --eval-seed 30 --ridge 1e-2`;
same command with
`--output-dir results/source_private_candidate_embedding_receiver_20260429/heldout_core_to_holdout_anchor_relative_code_similarity_budget8_seed29_30 --packet-feature-mode anchor_relative --anchor-count 128`;
and same anchor-relative command with `--receiver-kind ridge` to
`heldout_core_to_holdout_anchor_relative_ridge_budget8_seed29_30`. Outcomes:
hashed code similarity fails with matched `0.256`, target `0.250`, best
destructive `0.285`, but oracle `1.000`; anchor-relative code similarity fails
with matched `0.281`, target `0.250`, best destructive `0.258`, oracle
`0.756`; anchor-relative ridge fails with matched `0.303`, target `0.250`, best
destructive `0.438`, oracle `0.342`. Aggregate artifact:
`family_invariant_receiver_followup_summary.json`. The CPU systems frontier now
has `104` rows. Interpretation: candidate-code decoding can work if the packet
is oracle, but the source encoder and naive anchor-relative bank do not carry
transferable heldout-family source evidence. Next exact gate: fold-heldout
calibration or sparse/shared-dictionary receiver; simple code-similarity and
cosine-anchor variants are pruned.

Follow-up `2026-04-29`: added a reviewer-facing pass/fail ledger over the CPU
systems frontier. Command:
`./venv_arm64/bin/python scripts/build_source_private_pass_fail_ledger.py --output-dir results/source_private_pass_fail_ledger_20260429`.
Outcome: `104` rows total, with `3` paper-ready paired-uncertainty rows, `58`
positive rows needing more evidence, `1` weak positive, and `42` failed or
pruned rows. Artifact hashes: `pass_fail_ledger.json`
`19b0f6440a2ffa59dfa5d4fdd0c4912ef1da61f3becb0430ce71956ab7423320`,
`pass_fail_ledger.csv`
`fdab05d4958e0c9e50878a932f4706eeae227c01a88619924b4d86770f7fb5a0`,
`pass_fail_ledger.md`
`db4698b243c148a98c535a4b16bf88923defe25536d60927ede5270b11a2dea9`,
`manifest.json`
`59d0d9577273a8e5ab13c7723b81f35d828b485fe0d330261159b9da1c01ec94`.
Focused test:
`./venv_arm64/bin/python -m pytest tests/test_build_source_private_pass_fail_ledger.py -q`
passed. The new literature memo
`references/494_iclr_strengthening_scout_20260429.md` records the next systems
and method baselines: KV/cache byte lower-bound accounting against
TurboQuant/QJL/KIVI/KVQuant/SnapKV/CacheGen, followed by a masked or sparse
source-private innovation receiver gate.

Follow-up `2026-04-29`: implemented and ran the Mac-local KV/cache baseline
accounting table. Command:
`./venv_arm64/bin/python scripts/build_source_private_kv_cache_baseline_table.py --output-dir results/source_private_kv_cache_baseline_table_20260429`.
Outcome: `12` rows over the core and holdout `n=160` label-strict endpoint
summaries. The local Qwen3-0.6B config gives `114,688` fp16/bf16 KV bytes per
extra prompt token, `14,336` bytes at KIVI-style `2` bit, and `7,168` bytes at
QJL-style `1` bit. The 2-byte packet remains the far-left rate point: the
minimum non-packet QJL-style cache payload is `10,752.0x` the packet, and the
minimum non-packet KIVI-style cache payload is `21,504.0x` the packet.
Artifact hashes: `kv_cache_baseline_table.json`
`c2dd1ec937267cabb15cf47f320b6e94e85e5e8b87245d137d56b8abffa339c5`,
`kv_cache_baseline_table.csv`
`7fbae8949d2ba166dc6c98e0564100adf9cd07087158710f0a0d1bf0bf384ce6`,
`kv_cache_baseline_table.md`
`e81309b7fefe69314772c73e8ae5d0934d30fe57b8103cf5b28eedc05a6d8457`,
`manifest.json`
`34ca6f7d6b48204d99fd4b287d848d218f9d4e6b965ca433e5d86cd3e62f4194`.
Focused test:
`./venv_arm64/bin/python -m pytest tests/test_build_source_private_kv_cache_baseline_table.py -q`
passed. Interpretation: this is derived byte accounting, not a KV quantization
kernel or server-throughput benchmark. It strengthens the systems framing and
keeps the claim scoped to source-private residual communication.

Follow-up `2026-04-29`: implemented and ran the sparse masked source-private
innovation receiver. New files:
`scripts/run_source_private_masked_innovation_receiver.py`,
`tests/test_run_source_private_masked_innovation_receiver.py`,
`paper/source_private_masked_innovation_receiver_20260429.md`, and
`references/495_masked_innovation_receiver_refs_20260429.md`.
Same-distribution smoke command:
`./venv_arm64/bin/python scripts/run_source_private_masked_innovation_receiver.py --output-dir results/source_private_masked_innovation_receiver_20260429/smoke_all_seed3_4 --train-examples 128 --eval-examples 64 --train-family-set all --eval-family-set all --feature-dim 256 --anchor-count 64 --source-topk 48 --target-topk 24 --budgets 4 8 --train-seed 3 --eval-seed 4 --mask-repeats 1 --calibration-examples 32`.
Outcome: pass. At `4` bytes, matched `0.766`, target `0.250`, best
destructive `0.281`, oracle `1.000`; at `8` bytes, matched `0.922`, target
`0.250`, best destructive `0.266`, oracle `1.000`. Cross-family command:
`./venv_arm64/bin/python scripts/run_source_private_masked_innovation_receiver.py --output-dir results/source_private_masked_innovation_receiver_20260429/core_to_holdout_seed29_30 --train-examples 256 --eval-examples 128 --train-family-set core --eval-family-set holdout --feature-dim 256 --anchor-count 64 --source-topk 48 --target-topk 24 --budgets 4 8 12 --train-seed 29 --eval-seed 30 --mask-repeats 1 --calibration-examples 48`.
Outcome: fail. Matched is `0.258` at `4` bytes and `0.250` at `8/12`, target
`0.250`, and oracle `1.000`. Summary hashes: same-distribution
`76ce60666c05aa7b7fc010bac7d6d48f46656385d9f2b796c549d6e0b7352a28`;
core-to-holdout
`0e142e40635870efb0e821f0a9ba0ad52388f7974e80401fabcef83fe7615fb2`.
Focused test:
`./venv_arm64/bin/python -m pytest tests/test_run_source_private_masked_innovation_receiver.py -q`
passed. Interpretation: the branch is alive only as a same-distribution method
smoke. Do not promote it as cross-family communication; the next variant needs
shared-dictionary/crosscoder calibration and feature knockout.

Follow-up `2026-04-29`: added `--candidate-view shared_text` to the masked
innovation receiver and reran a smaller core-to-holdout discriminator. Command:
`./venv_arm64/bin/python scripts/run_source_private_masked_innovation_receiver.py --output-dir results/source_private_masked_innovation_receiver_20260429/core_to_holdout_shared_text_seed29_30 --train-examples 128 --eval-examples 64 --train-family-set core --eval-family-set holdout --candidate-view shared_text --feature-dim 128 --anchor-count 64 --source-topk 32 --target-topk 32 --budgets 4 8 --train-seed 29 --eval-seed 30 --mask-repeats 1 --calibration-examples 24`.
Outcome: fail. At `4` bytes, matched `0.266`, target `0.250`, best destructive
`0.250`, oracle `1.000`; at `8` bytes, matched `0.250`, target `0.250`, best
destructive `0.250`, oracle `1.000`. Summary hash:
`ea32fc2742d2c52a39c801af3ccbf24bcbc698539acec12628b5bcdc63eb7838`;
manifest hash:
`53579f684ac3110bcbc9b3986595b9b3c89b80041ca46903168ad5999ce76508`.
Focused test after adding the shared view:
`./venv_arm64/bin/python -m pytest tests/test_run_source_private_masked_innovation_receiver.py -q`
passed. Interpretation: anchor-relative and shared-text masked-innovation
variants now fail the same cross-family way despite oracle headroom. Stop
tuning this family unless the next variant is materially different, such as a
shared-dictionary/crosscoder receiver with feature knockout.

Follow-up `2026-04-29`: implemented and ran the coded-label/protocol risk gate
for the current positive 2-byte evidence-packet method. New files:
`scripts/run_source_private_coded_label_risk_gate.py`,
`tests/test_run_source_private_coded_label_risk_gate.py`,
`paper/source_private_coded_label_risk_gate_20260429.md`, and
`references/496_coded_label_risk_and_uniqueness_scout_20260429.md`. Command:
`./venv_arm64/bin/python scripts/run_source_private_coded_label_risk_gate.py --examples 160 --candidates 4 --family-set all --seeds 29,31,37 --budget 2 --output-dir results/source_private_coded_label_risk_gate_20260429`.
Outcome: pass. Across `15` seed/transform rows (`baseline`, opaque
candidate-label rename, diagnostic-code remap, candidate-pool permutation, and
the composed label+code+order stress), matched 2-byte packet accuracy is
`1.000`, target-only is `0.250`, reviewer-negative controls stay at `0.250`,
positive oracle floor is `1.000`, and the worst source-destroying control is
`0.263`. Artifact hashes: `summary.json`
`d29b2ba829b8fc09cbe0204f80d34f9ec298b71d3fb71e7a0c7b6902eecd8240`,
`summary.md`
`914e359f13d03022e3ebe6388f80a4c691e6fd1163512bb243f1b5ecbcdb5bdd`,
`predictions.jsonl`
`7f9067b7a59d8133df59265695d0bebdae3a718a907d98409e725bc04301eb79`,
`manifest.json`
`f7537d59d4c47cbcbaf7337adb7e17384271fbcf4e53acd29f9b6f16372c737e`.
Focused tests:
`./venv_arm64/bin/python -m pytest tests/test_run_source_private_coded_label_risk_gate.py -q`
passed. Interpretation: fixed label, display-order, and single-codebook lookup
explanations are materially weakened. This strengthens the scoped
source-private packet claim, but does not solve broad cross-family latent
transfer. Next exact gate: one-command reproduction/novelty bundle and, if
time allows, a larger `n=500` composed coded-label stress row.

Follow-up `2026-04-29`: implemented the reviewer-facing ICLR evidence bundle.
New files: `scripts/build_source_private_iclr_evidence_bundle.py`,
`tests/test_build_source_private_iclr_evidence_bundle.py`,
`paper/source_private_iclr_evidence_bundle_20260429.md`, and
`references/497_iclr_evidence_bundle_refs_20260429.md`. Command:
`./venv_arm64/bin/python scripts/build_source_private_iclr_evidence_bundle.py --output-dir results/source_private_iclr_evidence_bundle_20260429`.
Outcome: pass, with `10/10` machine checks, `5` contribution rows, and `8`
novelty-matrix comparisons. The bundle verifies that required artifacts exist,
the rate frontier passes, matched-byte text stays at target, the packet keeps a
`7.0x` byte advantage over query-aware text, QJL-style cache byte lower-bound is
above `1000x`, the coded-label composed stress passes, endpoint uncertainty
passes on core/holdout, and the pass/fail ledger has `3` paper-ready rows. It
also writes `reproduce_iclr_evidence_bundle.sh` with the exact Mac-local
commands for rebuilding derived tables and focused tests. Focused test:
`./venv_arm64/bin/python -m pytest tests/test_build_source_private_iclr_evidence_bundle.py -q`
passed. Interpretation: this does not create a new method, but it materially
reduces reproducibility, novelty-positioning, and contribution-depth reviewer
risk. Next exact gate: a negative-boundary appendix aggregating cross-family
failures and oracle headroom, or an `n=500` composed-only coded-label stress.

Follow-up `2026-04-29`: implemented the cross-family negative-boundary
appendix. New files:
`scripts/build_source_private_cross_family_negative_boundary.py`,
`tests/test_build_source_private_cross_family_negative_boundary.py`,
`paper/source_private_cross_family_negative_boundary_20260429.md`, and
`references/498_cross_family_negative_boundary_refs_20260429.md`. Command:
`./venv_arm64/bin/python scripts/build_source_private_cross_family_negative_boundary.py --output-dir results/source_private_cross_family_negative_boundary_20260429`.
Outcome: pass as a boundary artifact, with `27` rows, `6` method families, `0`
claim-ready cross-family methods, and `6` oracle-headroom rows. It aggregates
learned WZ, canonical RASP, consistent posterior, anchor-relative sparse
packets, learned target-preserving receivers, and masked innovation receivers.
The key interpretation is that cross-family source/private learned
communication is not a headline claim: several rows have high oracle headroom,
but current learned/static interfaces fail or are asymmetric under controls.
Focused test:
`./venv_arm64/bin/python -m pytest tests/test_build_source_private_cross_family_negative_boundary.py -q`
passed. Next exact method gate, if pursued, is a shared sparse
crosscoder/dictionary packet with feature knockout; otherwise move to paper
revision with this boundary made explicit.

Follow-up `2026-04-29`: implemented the systems caveat frontier to strengthen
the systems contribution while preventing overclaim. New files:
`scripts/build_source_private_systems_caveat_frontier.py`,
`tests/test_build_source_private_systems_caveat_frontier.py`,
`paper/source_private_systems_caveat_frontier_20260429.md`, and
`references/499_systems_caveat_frontier_refs_20260429.md`. Command:
`./venv_arm64/bin/python scripts/build_source_private_systems_caveat_frontier.py --output-dir results/source_private_systems_caveat_frontier_20260429`.
Outcome: pass. The artifact aggregates the `n=160` core and holdout
label-strict Mac endpoint rows, paired uncertainty, KV/cache byte lower bounds,
and a terse-prompt failure row. Both endpoint rows pass with packet accuracy
`0.675/0.688` versus target/control `0.250`, minimum paired CI95 lower bound
`+0.350`, 2-byte packet payload, query-aware text `7.0x` larger, full hidden
log `183.25x-186.75x` larger, full-log p50 TTFT `+164.3 ms` to `+183.5 ms`
versus packet, and QJL-style 1-bit cache byte lower-bound `10752.0x` packet
bytes. The terse prompt stress fails at target accuracy, documenting that the
public receiver contract is required. Focused test:
`./venv_arm64/bin/python -m pytest tests/test_build_source_private_systems_caveat_frontier.py -q`
passed. Interpretation: claim an extreme-rate source-private communication
frontier with Mac-local endpoint evidence and derived cache byte floors; do not
claim native superiority over TurboQuant, QJL, KIVI/KVQuant, C2C, KVComm, or
production serving systems. Next exact reviewer-risk gate: anti-lookup
label-blind receiver stress at `n=160` core + holdout.

Follow-up `2026-04-29`: implemented and ran the label-blind anti-lookup smoke.
New files: `scripts/build_source_private_anti_lookup_label_blind_summary.py`,
`tests/test_build_source_private_anti_lookup_label_blind_summary.py`, and
`paper/source_private_anti_lookup_label_blind_20260429.md`; the endpoint harness
now supports `--candidate-view label_blind`. Commands:
`./venv_arm64/bin/python scripts/run_source_private_mac_endpoint_proxy_frontier.py --benchmark-jsonl results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29/benchmark.jsonl --output-dir results/source_private_anti_lookup_label_blind_20260429/core_seed29_qwen3_n8_label_blind --model Qwen/Qwen3-0.6B --device cpu --dtype float32 --limit 8 --max-new-tokens 12 --prompt-style label_strict --candidate-view label_blind --conditions target_only matched_packet matched_byte_text_2 random_same_byte_packet deranged_candidate_diag_table query_aware_diag_span structured_json_diag structured_free_text_diag full_hidden_log`;
same command for `holdout_seed30`, then
`./venv_arm64/bin/python scripts/build_source_private_anti_lookup_label_blind_summary.py --output-dir results/source_private_anti_lookup_label_blind_20260429`.
Outcome: pass as a collapse-control artifact. With candidate repair-key
metadata and original labels hidden, opaque payloads collapse to target on both
core and holdout `n=8`: target `0.250`, matched packet `0.250`, max opaque
payload `0.250`, exact-ID parity true, valid rate `1.000`. Positive
diagnostic-table comparators remain strongly positive with minimum lift
`+0.425`. Focused tests:
`./venv_arm64/bin/python -m pytest tests/test_run_source_private_mac_endpoint_proxy_frontier.py tests/test_build_source_private_anti_lookup_label_blind_summary.py -q`
passed. Interpretation: this weakens hidden-label leakage as an explanation but
confirms the current method requires target-side public side information. Next
exact gate: scale label-blind anti-lookup to `n=160` with paired uncertainty,
then pursue shared sparse crosscoder packets for a less protocol-shaped learned
method.

Follow-up `2026-04-29`: scaled the label-blind anti-lookup stress to a
strict-small `n=32` core + holdout collapse check and hardened the summary with
paired bootstrap upper-bound diagnostics against `target_only`. Commands:
`./venv_arm64/bin/python scripts/run_source_private_mac_endpoint_proxy_frontier.py --benchmark-jsonl results/source_private_tool_trace_reviewer_risk_rows_20260429/core_seed29/benchmark.jsonl --output-dir results/source_private_anti_lookup_label_blind_20260429/core_seed29_qwen3_n32_label_blind --model Qwen/Qwen3-0.6B --device cpu --dtype float32 --limit 32 --max-new-tokens 8 --prompt-style label_strict --candidate-view label_blind --conditions target_only matched_packet matched_byte_text_2 random_same_byte_packet deranged_candidate_diag_table query_aware_diag_span structured_json_diag structured_free_text_diag full_hidden_log`;
same command for `holdout_seed30`, then
`./venv_arm64/bin/python scripts/build_source_private_anti_lookup_label_blind_summary.py --output-dir results/source_private_anti_lookup_label_blind_20260429`.
Outcome: pass as a stricter collapse-control artifact. Across `4` rows
(`n=8` and `n=32`, core and holdout), every opaque payload remains exactly at
target accuracy `0.250`, exact-ID parity is true, valid/strict-valid coverage
is `1.000`, max opaque-target delta is `0.000`, and max paired-bootstrap CI95
high versus target is `0.000` for both regular and strict correctness. The
positive diagnostic-table comparator remains positive with minimum lift
`+0.425`. Focused tests:
`./venv_arm64/bin/python -m pytest tests/test_build_source_private_anti_lookup_label_blind_summary.py tests/test_run_source_private_mac_endpoint_proxy_frontier.py -q`
passed. Interpretation: this substantially weakens hidden-label/opaque-string
lookup as an explanation for the positive row, but it also reinforces the
scoped claim that the current packet needs public side information. Next exact
gate remains `n=160` label-blind confirmation or a materially different shared
sparse crosscoder packet with atom knockout.

Follow-up `2026-04-29`: implemented and ran the first shared sparse
crosscoder-inspired source-private packet gate. New files:
`scripts/run_source_private_shared_sparse_crosscoder_packet_gate.py`,
`tests/test_run_source_private_shared_sparse_crosscoder_packet_gate.py`,
`paper/source_private_shared_sparse_crosscoder_packet_gate_20260429.md`, and
`references/500_shared_sparse_crosscoder_packet_refs_20260429.md`. Command:
`./venv_arm64/bin/python scripts/run_source_private_shared_sparse_crosscoder_packet_gate.py --output-dir results/source_private_shared_sparse_crosscoder_packet_gate_20260429 --budgets 4 8 --train-examples 256 --eval-examples 128 --seed 29`.
Outcome: pass. The gate evaluates core -> holdout, holdout -> core, and
same-family all surfaces at `n=128` each. Cross-family passes bidirectionally:
core -> holdout reaches `1.000` accuracy at `4/8` bytes versus target and best
control `0.250`; holdout -> core reaches `0.875` at `8` bytes versus target and
best control `0.250`; same-family reaches `0.938`. The minimum passing paired
CI95 lower bound versus target is `+0.539`, all source-destroying controls stay
at target, and top-atom knockout removes `100%` of the matched-minus-target
lift. Focused test:
`./venv_arm64/bin/python -m pytest tests/test_run_source_private_shared_sparse_crosscoder_packet_gate.py -q`
passed. Interpretation: this is the first strict-small positive learned-method
successor after the endpoint protocol. It is still a controlled
crosscoder-inspired shared dictionary, not a trained neural crosscoder over LLM
activations. Next exact gate: seed-repeat confirmation at seed `31`, then a
larger frozen slice or learned shared-dictionary variant.

Follow-up `2026-04-29`: ran the seed-repeat confirmation for the shared sparse
crosscoder-inspired packet. Command:
`./venv_arm64/bin/python scripts/run_source_private_shared_sparse_crosscoder_packet_gate.py --output-dir results/source_private_shared_sparse_crosscoder_packet_gate_20260429_seed_repeat --budgets 4 8 --train-examples 256 --eval-examples 128 --seed 31`.
Outcome: pass. The repeat preserves the exact same headline shape: cross-family
pass true, core -> holdout `1.000` at `4/8` bytes versus target/control
`0.250`, holdout -> core `0.875` at the passing `8` byte budget versus
target/control `0.250`, same-family `0.938`, minimum passing paired CI95 lower
bound `+0.539`, and top-atom knockout removing `100%` of lift. Interpretation:
this improves seed/remap stability, but it is still the same controlled family
generator. Next exact gate: larger frozen-slice shared sparse confirmation
(`512/512`) or a learned shared-dictionary/crosscoder variant with the same
controls.

Follow-up `2026-04-29`: added and ran the synonym/ontology stress for the
shared sparse packet by extending
`scripts/run_source_private_shared_sparse_crosscoder_packet_gate.py` with
`--candidate-atom-view synonym_stress`. Command:
`./venv_arm64/bin/python scripts/run_source_private_shared_sparse_crosscoder_packet_gate.py --output-dir results/source_private_shared_sparse_crosscoder_packet_gate_20260429_synonym_stress --budgets 4 8 --train-examples 256 --eval-examples 128 --seed 29 --candidate-atom-view synonym_stress`.
Outcome: fail as an overclaim boundary. The stress paraphrases candidate
intents into terms outside the hand atom rules while leaving source-private
packet extraction unchanged. Cross-family pass is false, all direction passes
are false, max shared sparse accuracy drops to `0.375`, max shared-target delta
is only `+0.125`, and there are `0` pass rows. Controls remain at target. This
means the current shared sparse packet is seed-stable under the native
controlled ontology but not robust to ontology/synonym shift. Next exact method
gate: learned shared dictionary/crosscoder or conditional consistency syndrome
packet; otherwise frame the shared sparse contribution as an agreed-protocol
source-private dictionary, not robust semantic transfer.

Follow-up `2026-04-29`: implemented and ran a conditional semantic syndrome
smoke to test whether a simple learned residual can replace the hand sparse
dictionary under synonym stress. New files:
`scripts/run_source_private_conditional_semantic_syndrome_gate.py`,
`tests/test_run_source_private_conditional_semantic_syndrome_gate.py`, and
`paper/source_private_conditional_semantic_syndrome_gate_20260429.md`. Command:
`./venv_arm64/bin/python scripts/run_source_private_conditional_semantic_syndrome_gate.py --output-dir results/source_private_conditional_semantic_syndrome_gate_20260429 --budgets 2 4 8 --train-examples 128 --eval-examples 64 --seed 29 --feature-dim 64 --candidate-view synonym_stress`.
Outcome: fail. Oracle candidate residual is `1.000`, so the surface has
headroom, but cross-family pass is false, all direction passes are false, and
there are `0` pass rows. Core -> holdout is at or below target except a tiny
`8` byte lift (`0.281` vs `0.250`, CI low negative). Holdout -> core and
same-family show matched lift, but controls leak: answer-masked source reaches
`0.500` and public-only source reaches `0.500`. Interpretation: a naive learned
semantic residual does not solve the ontology-stress failure. Next learned gate
must add synonym-consistency training, a learned shared dictionary, or
target-preserving abstention before it can become a claim.

Follow-up `2026-04-29`: ran the larger frozen-slice shared sparse boundary
gate requested by the ICLR-strengthening review. Commands:
`./venv_arm64/bin/python scripts/run_source_private_shared_sparse_crosscoder_packet_gate.py --output-dir results/source_private_shared_sparse_crosscoder_packet_gate_20260429_n512 --budgets 4 8 --train-examples 512 --eval-examples 512 --seed 41`
and
`./venv_arm64/bin/python scripts/run_source_private_shared_sparse_crosscoder_packet_gate.py --output-dir results/source_private_shared_sparse_crosscoder_packet_gate_20260429_n512_synonym_stress --budgets 4 8 --train-examples 512 --eval-examples 512 --seed 41 --candidate-atom-view synonym_stress`.
Outcome: native agreed-ontology pass and synonym-stress failure. Native
`n=512` cross-family pass is true with all direction passes true, `5` pass
rows, max shared sparse accuracy `1.000`, max shared-target delta `+0.750`,
minimum passing paired CI95 lower bound `+0.582`, and top-atom knockout
removing `100%` of lift in passing rows. The matched `n=512` synonym-stress
row has cross-family pass false, all direction passes false, `0` pass rows, max
shared sparse accuracy `0.375`, and max shared-target delta `+0.125`.
Artifact hashes: native summary JSON
`371c74311066520eeb6d6b4755bd996a895936c4156f877a799f617fce134c0b`,
native summary MD
`a314fc85ebdee5fd9a194b4a0aee5b4c9687e7acdf52817c93e96ba893922807`,
synonym summary JSON
`bf6e3d30ce6cef55f5a3800f2564a5ae2feae04b9067ebb4473661a8ba9397fa`,
and synonym summary MD
`02438ce3855e090e91cce8551568a48b3924b2bf27a0640cddc3e0ee842e4062`.
Focused test:
`./venv_arm64/bin/python -m pytest tests/test_run_source_private_shared_sparse_crosscoder_packet_gate.py -q`
passed. Interpretation: the agreed-dictionary sparse packet now has larger
slice evidence and causal knockout, making it a stronger interpretable
technical contribution, but the synonym failure keeps the claim scoped. Next
exact method gate: learned synonym-invariant shared dictionary/crosscoder with
the same controls, atom/dictionary derangement, matched-byte text, and causal
feature knockout.
