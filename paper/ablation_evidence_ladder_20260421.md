# Ablation Evidence Ladder

Date: 2026-04-21

This table summarizes local telemetry for stack decisions. It separates toy-positive components, controls, and blockers so we do not rerun saturated ideas without changing the hypothesis.

| Lane | Method | Level | Status | Accuracy | Delta | MSE | Bytes | Atom recovery | Route acc | Stability | Over-refine | Bits | Help | Harm | Promotion gate |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| Hub/shared dictionary | `hub_shared_dictionary` | toy positive | present | 1.0000 | 0.3208 | 0.0199 | 20980.0000 | 1.0000 | - | - | - | - | 0.7458 | 0.0000 | Beat pairwise on held-out route pools with fewer adapters or bytes. |
| Pairwise bridge control | `pairwise_bridges` | toy control | present | 0.6792 | 0.4250 | 0.4595 | 33600.0000 | 0.6542 | - | - | - | - | 0.4250 | 0.0000 | Keep only as O(n^2) scaling baseline. |
| Multi-way canonical hub | `multiway_gpa_canonical (1-shot held-out family)` | toy positive / boundary | present | 1.0000 | 0.0000 | 0.1327 | - | - | - | - | - | - | - | - | Promote only if the low-shot gain survives on a real held-out-family split and still holds after adding a sparse shared dictionary. |
| Gauge-fix quotient bridge | `quotient_match_after_fix (1-shot held-out family)` | toy positive / boundary | present | 1.0000 | 0.0000 | 0.0796 | - | - | - | - | - | - | - | - | Promote only if the low-shot gain survives after composition with a sparse shared dictionary and still helps on real held-out route pools. |
| GPA sparse dictionary hub | `multiway_gpa_sparse_dictionary (1-shot held-out family)` | toy positive / boundary | present | 1.0000 | 0.0000 | 0.1171 | - | 0.1900 | - | - | - | - | 0.0000 | 0.0000 | Promote only if the low-shot gain survives tokenizer/interface shifts and the learned atoms become interpretable enough to justify a shared-basis story. |
| Sparse dictionary repair gate | `multiway_gpa_sparse_dictionary_repair (1-shot held-out family)` | toy blocker | present | 1.0000 | 0.0000 | 0.1171 | - | 0.1900 | - | - | - | - | 0.0000 | 0.0000 | Do not promote until the verifier-gated repair step fires on held-out examples and improves MSE or accuracy over the sparse-dictionary base. |
| Sticky feature routing | `sticky_paraphrase_stable_routing` | toy positive | present | 0.9438 | 0.0000 | 0.0243 | 16.0000 | - | 0.9875 | 1.0000 | - | - | 0.0000 | 0.0000 | Improve perturbation stability without lowering route-pool accuracy. |
| Confidence-only routing | `confidence_routing` | toy blocker | present | 0.3688 | -0.5750 | 1.5497 | 16.0000 | - | 0.2812 | 0.5437 | - | - | 0.0063 | 0.5813 | Do not rerun as sole router; only use as uncertainty feature. |
| Feature+atom stack | `stacked_feature_atom` | toy positive interaction | present | 0.8542 | 0.2083 | 1.7413 | 5920.0000 | 0.3948 | - | - | - | - | 0.2083 | 0.0000 | Test interaction terms, not isolated feature-only or atom-only branches. |
| Mixed-bit frontier | `quant_error_target_bpw_allocator` | toy positive | present | 1.0000 | 0.7750 | 0.0314 | 319.0000 | - | - | 1.0000 | - | 3:26, 8:6 | 0.7750 | 0.0000 | Preserve accuracy at lower bpw than flat precision with help/harm logged. |
| Verifier stop rule | `verifier_harm_stop` | toy positive / safety | present | 0.9625 | 0.0500 | 0.0614 | 40.4688 | - | - | - | 0.8687 | - | 0.0188 | 0.0188 | Reduce over-refinement and harm versus fixed-depth repair. |
| Naive component stack | `hub_sticky_frontier_verifier_stop` | toy interaction blocker | present | 0.5938 | -0.1406 | 1.1173 | 22964.0000 | 0.5286 | 0.6250 | 0.9219 | 0.4583 | - | 0.0000 | 0.1406 | Do not stack hub, router, frontier, and stop policy until each interface is validated. |
| Stack oracle routing | `oracle_router_control` | toy oracle headroom | present | 0.8229 | 0.0885 | 0.2609 | 22914.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | - | 0.0885 | 0.0000 | Use as route-quality ceiling, not as a method row. |
| Fixed-depth repair blocker | `fixed_4_step` | toy blocker | present | 0.9125 | -0.0500 | 0.0673 | 43.0000 | - | - | - | 0.9625 | - | 0.0312 | 0.0812 | Do not promote without a stop policy. |

## Read

- Promote hub dictionaries, sticky/feature routing, mixed-bit frontiers, and verifier stop rules only as an interaction stack with matched controls.
- The route-conditioned hub sweep shows that the current frontier and stop heuristics are not drop-in additive components: the best frontier gain is only `+0.0104`, oracle frontier is negative, and the stop rule never adds positive accuracy.
- The route-class patch follow-up shows that calibration-aware local protection only ties the current quant-error frontier and route-class frontier pruning is still negative, so the next fix should move up to the hub/interface or pruning-rule level.
- The multi-way canonical-hub follow-up adds a boundary clue rather than a full fix: at `1` shot/class the GPA-style shared basis helps on MSE, but direct family fitting retakes the lead once `2+` paired shots/class are available.
- The gauge-fix quotient follow-up sharpens the symmetry story: once head matching is scored after Procrustes alignment, quotient-aware matching becomes the best non-oracle method at `1` shot/class (`0.0796` MSE), beats direct few-shot fitting (`0.0985`), and recovers the true head correspondence exactly, but direct family fitting still retakes the lead once `2+` paired shots/class are available.
- The GPA sparse-dictionary follow-up sharpens that low-shot clue: at `1` shot/class the shared sparse dictionary is now the best non-oracle method on MSE (`0.1171` vs `0.1825` for direct few-shot and `0.2355` for canonical-only), but direct family fitting still retakes the lead once `2+` paired shots/class are available, and the current repair gate remains a pure no-op.
- Treat confidence-only routing and fixed-depth repair as blockers, not baselines to keep rerunning.
- Any real-route-pool promotion should carry the same telemetry columns: atom recovery, route stability, bit histogram, stop reason, help/harm, bytes, and latency.
