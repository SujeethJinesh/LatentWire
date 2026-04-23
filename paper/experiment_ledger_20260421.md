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
