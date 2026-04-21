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
| Harness probes as competitor claims | LatentMAS Qwen2.5-0.5B `N=1` baseline/text-MAS probes | These verify wrapper plumbing only. Do not compare them against GSM70 method rows or cite them as fair LatentMAS results. |
| Naive full-stack composition | hub + sticky router + protected mixed-bit frontier + verifier stop toy | Individually plausible pieces can interfere. The leak-free stack scores `0.5938` versus raw pairwise `0.7344`; only oracle routing beats raw at `0.8229`. Validate interfaces before stacking. |

## Keep But Gate

| Lane | Current evidence | Promotion condition |
|---|---|---|
| Strict route selection + process repair | GSM70 `0.2000` vs target self-repair `0.1714`; SVAMP70 `0.5429` vs `0.5000`; zero observed repair harm on current slices | Needs matched token/latency/byte budgets, paired intervals, and LatentMAS/C2C/text controls on exact IDs. |
| Shared feature dictionary / crosscoder | Toy shared dictionary beats raw residual transport; feature+atom stack beats isolated components | Promote only with real route-pool feature IDs, atom recovery, and patch/quant-error calibration. |
| Route atoms / codebooks | Learned shared codebook beats raw ridge despite worse MSE | Require atom recovery, codebook usage/perplexity, dead-code rate, and task delta. |
| Byte/span tokenizer interface | Token-ID reconstruction fails while byte-span reconstruction succeeds on stress split | Run on real tokenizer pairs and include remap coverage before any downstream bridge claim. |
| Quant-error mixed-bit allocation | Recovers uniform-4-bit toy accuracy at lower achieved bpw | Stack with protected frontier selection; log bit histogram, outlier protection, help/harm, and false-prune. |
| Feature-routed projector bank | Toy feature routing reaches `0.9187` vs monolithic `0.8687`, close to oracle `0.9688` | Move into route-pool harness with random/confidence/oracle controls and matched bytes. |
| Hub dictionary bridge | Shared hub toy reaches `1.0000` accuracy and atom recovery with fewer adapters than pairwise; random hub fails | Promote only with real route-pool feature IDs, atom recovery, dead-feature rate, and hub-versus-pairwise scaling. |
| Sticky/feature router | Feature router reaches `0.9438`; sticky router keeps accuracy and raises perturb stability to `1.0000` | Promote only with route entropy, perturb stability, load-balance, random/confidence/oracle controls, and matched compute. |
| Verifier/process stop rules | Verifier-harm stop preserves high toy accuracy and reduces harm relative to blind refinement | Promote with real repair trajectories, stop reason, halt precision, missed-help, and over-refinement telemetry. |
| LatentMAS comparator | Wrapper exists; cached baseline/text-MAS `N=1` probes run; latent-MAS direct mode is runtime-blocked after MPS fallback | Fix latent-mode runtime and run bounded matched GSM/SVAMP baseline/text-MAS/latent-MAS before head-to-head claims. |
| Stack oracle route ceiling | In the composition toy, oracle routing reaches `0.8229` versus raw pairwise `0.7344` | Use this to debug route assignment and stop policy; do not cite as a method row. |

## Next Stack To Test

The next positive-method stack should be additive only after each component has
an interaction control:

1. Byte/span-normalized route atoms instead of token-ID transfer.
2. Hub dictionary/shared feature basis instead of `O(n^2)` pairwise bridges.
3. Sticky feature-routed projector bank instead of one monolithic bridge or
   confidence-only routing.
4. Quant-error protected mixed-bit frontier instead of flat precision.
5. Process/verifier stop rule instead of fixed-depth repair.
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
