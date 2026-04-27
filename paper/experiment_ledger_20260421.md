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
