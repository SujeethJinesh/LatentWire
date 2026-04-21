# 371 Multi-Way Canonical Hub Recent References

Date: 2026-04-21

Scope: 2024-2026 primary references on multi-way alignment, generalized Procrustes / canonicalization, shared sparse dictionaries / crosscoders, gauge fixing, and symmetry-aware shared latent spaces for a held-out-family canonical hub experiment in LatentWire.

## Sources

- [Multi-Way Representation Alignment](https://arxiv.org/abs/2602.06205). Multi-way GPA plus geometry-corrected Procrustes gives a shared orthogonal universe across `M >= 3` models and is the cleanest direct anchor for a canonical hub.

- [Sparse Crosscoders for diffing MoEs and Dense models](https://arxiv.org/abs/2603.05805). Crosscoders jointly model multiple activation spaces with explicit shared features; this is the strongest recent shared-dictionary baseline for a multi-model hub.

- [Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes](https://arxiv.org/abs/2603.04426). Shows shared dictionaries can isolate localized asymmetric changes; useful when only a small portion of the hub should move between source and target families.

- [LUCID-SAE: Learning Unified Vision-Language Sparse Codes for Interpretable Concept Discovery](https://arxiv.org/abs/2602.07311). Shared sparse codes plus OT matching produce an interpretable latent dictionary; useful as a hub that is both canonical and explainable.

- [Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment](https://arxiv.org/abs/2502.03714). A universal overcomplete SAE reconstructs multiple models through one shared concept space; this is the cleanest sparse-dictionary counterpart to GPA.

- [SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability](https://arxiv.org/abs/2507.06265). Global TopK plus cross-reconstruction enforces identical active dimensions across streams; useful for measuring whether hub atoms are genuinely shared.

- [Variational Routing: A Scalable Bayesian Framework for Calibrated Mixture-of-Experts Transformers](https://arxiv.org/abs/2603.09453). Uncertainty-aware routing is a direct antidote to brittle confidence-only gating and gives a principled router for a canonical hub.

- [RASA: Routing-Aware Safety Alignment for Mixture-of-Experts Models](https://arxiv.org/abs/2602.04448). Routing-aware repair prevents degenerate optimization through routing bypasses; relevant when bridge stacking may be “fixing” the router instead of the representation.

- [Unification of Symmetries Inside Neural Networks: Transformer, Feedforward and Neural ODE](https://arxiv.org/abs/2402.02362). Gauge symmetries explain why apparently equivalent parameterizations can drift across stages; this is the best symmetry anchor for why stacked modules may fail.

- [Concept Bottleneck Models Without Predefined Concepts](https://arxiv.org/abs/2407.03921). Unsupervised concept discovery plus input-dependent concept selection gives an interpretability template for hub atoms and sparse shared codes.

## Why It Matters For Us

LatentWire’s canonical-hub question is not just whether models can be aligned, but whether a **single shared basis** can remain stable after stacking transport, routing, and repair. The recent literature suggests three failure modes:

1. **Gauge mismatch**: each bridge learns a different coordinate system even if the functions are equivalent.
2. **Dictionary mismatch**: dense transport hides reusable concepts that become visible only under sparse shared codes.
3. **Router mismatch**: confidence-only routing collapses or silently changes behavior instead of selecting the right hub atoms.

For a held-out-family experiment, the best evidence will come from showing that one canonical hub survives unseen model pairs while also preserving interpretability and compute-normalized performance.

## Concrete Ablations / Diagnostics

1. **GPA hub vs pairwise bridge.** Compare pairwise Procrustes, multi-way GPA, and GPA plus post-hoc geometry correction on a held-out-family split under the same byte budget.

2. **Dense hub vs sparse shared dictionary.** Compare a dense canonical hub, a USAE/Delta-Crosscoder style sparse dictionary, and a shared-plus-private dictionary with residual capacity.

3. **Frozen vs uncertainty-aware routing.** Compare hard confidence routing, variational routing, and fixed-routing-at-eval with repair allowed only after route selection; log route stability under paraphrases and noise.

4. **Gauge-fix between stages.** Insert an explicit re-canonicalization step between hub import and repair, then measure whether stacking gain improves or collapses.

## Diagnostics To Log

- `basis_condition_number`, `singular_spectrum`, `gauge_fix_residual`
- `held_out_pair_delta`, `shared_space_residual`, `pairwise_residual`
- `shared_atom_count`, `exclusive_atom_count`, `atom_sparsity`, `dead_atom_rate`, `atom_overlap`
- `route_entropy`, `route_stability`, `collapse_rate`, `route_help`, `route_harm`
- `verifier_granularity`, `verifier_calls`, `halt_confidence`, `false_halt`, `overthink_flag`
- `tokens_in`, `tokens_out`, `bytes`, `kv_bytes`, `latency_ms`, `correct`, `baseline_correct`

## Recommendation

Single best next experiment: **GPA-initialized shared hub + sparse dictionary on a held-out-family split, with routing frozen during evaluation and one verifier-gated repair step.** This is the cleanest way to tell gauge mismatch apart from transport mismatch before adding any more machinery.
