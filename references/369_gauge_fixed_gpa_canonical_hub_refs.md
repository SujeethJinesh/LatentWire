# 368 Gauge-Fixed GPA, Canonical Hub, And Shared Dictionary References

Date: 2026-04-21

Scope: gauge fixing, generalized Procrustes / GPA, multi-way representation alignment, and shared sparse dictionaries as the most plausible way to reduce pairwise alignment into a canonical hub.

## Working Hypothesis

LatentWire component stacking may fail because each bridge learns its own gauge. A later module then sees coordinates that are technically equivalent but not canonically comparable. The most promising repair is to:

1. Choose a shared hub basis first.
2. Fix the gauge of each source/target pair into that basis.
3. Only then ask whether a sparse dictionary or residual correction is needed.

If the hub is unstable under held-out pairs, the paper should stop chasing a universal bridge and instead emphasize pairwise routing plus verifier-gated repair.

## Sources

- **[Multi-Way Representation Alignment](https://arxiv.org/abs/2602.06205)**. `Core idea:` generalized Procrustes analysis can construct a shared orthogonal universe across `M >= 3` models, with geometry correction for retrieval mismatch. `Why it matters:` this is the clearest recent evidence that a canonical hub is the right object to test first. `LatentWire use:` use GPA to initialize a shared hub, then test whether a post-hoc correction is enough. `Telemetry:` basis condition number, singular spectrum, shared-space residual, held-out-pair delta, and route help/harm.

- **[Unification of Symmetries Inside Neural Networks: Transformer, Feedforward and Neural ODE](https://arxiv.org/abs/2402.02362)**. `Core idea:` parameter redundancies in transformers can be viewed as gauge symmetries. `Why it matters:` if the bridge basis is not gauge-fixed, stacking later modules can be coordinate-dependent and brittle. `LatentWire use:` add an explicit re-canonicalization step between bridge import and repair. `Telemetry:` gauge-fix residual, basis drift across stages, route stability, and stacking gain.

- **[Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment](https://arxiv.org/abs/2502.03714)**. `Core idea:` one overcomplete sparse dictionary can reconstruct and interpret multiple models. `Why it matters:` a canonical hub is more useful if the coordinates are also sparse and interpretable. `LatentWire use:` compare a GPA hub alone to a GPA-initialized sparse dictionary and to a dense bridge. `Telemetry:` code usage, dead-code rate, shared-vs-private atom split, reconstruction error, and held-out transfer.

- **[Sparse Crosscoders for diffing MoEs and Dense models](https://arxiv.org/abs/2603.05805)**. `Core idea:` shared features across activation spaces can be recovered with BatchTopK crosscoders. `Why it matters:` this is the strongest recent argument that multi-model shared structure can be sparse rather than dense. `LatentWire use:` use a sparse crosscoder on top of the shared hub and compare against a dense residual bridge. `Telemetry:` shared feature count, private feature count, fractional variance explained, and atom overlap.

- **[LUCID-SAE: Learning Unified Vision-Language Sparse Codes for Interpretable Concept Discovery](https://arxiv.org/abs/2602.07311)**. `Core idea:` shared sparse codes plus OT matching can produce a unified interpretable dictionary. `Why it matters:` OT gives a concrete alignment objective for turning a hub into a canonical shared codebook. `LatentWire use:` compare GPA-only, GPA+OT sparse hub, and ordinary SAE routing. `Telemetry:` OT cost, concept clustering, cross-family transfer, and dictionary interpretability.

- **[Concept Bottleneck Models Without Predefined Concepts](https://arxiv.org/abs/2407.03921)**. `Core idea:` unsupervised concepts plus input-dependent concept selection can make latent spaces interpretable without hand labels. `Why it matters:` route atoms should behave like discovered concepts, not opaque coordinates. `LatentWire use:` require each hub atom to have a simple interpretation and a consistent selection pattern. `Telemetry:` concept purity, selection entropy, active concept count, and error correction by concept.

- **[The geometry of hidden representations of large transformer models](https://arxiv.org/abs/2302.00294)**. `Core idea:` the most semantically rich layers often appear near intrinsic-dimension minima and can mirror symmetric autoencoder structure. `Why it matters:` a hub only helps if we choose the right layer window. `LatentWire use:` restrict hub construction to geometry-rich layers rather than all layers. `Telemetry:` layer window, intrinsic-dimension proxy, patch correlation, and held-out-pair gain.

- **[Geometric Deep Learning](https://arxiv.org/abs/2104.13478)**. `Core idea:` symmetry priors can reduce the search space by constraining the model to equivariant structure. `Why it matters:` it is the high-level anchor for treating gauge fixing as a structural prior rather than an optimization trick. `LatentWire use:` justify a canonical hub basis as symmetry reduction. `Telemetry:` symmetry mismatch, basis drift, and transfer gain after canonicalization.

## Three Concrete Ablations

1. **GPA hub vs pairwise bridge.** Compare pairwise Procrustes, multi-way GPA, and GPA plus residual correction on the same held-out pair split.

2. **GPA hub vs GPA-initialized sparse dictionary.** Compare a dense hub basis, a sparse crosscoder/SAE built on top of that basis, and a sparse dictionary with private residual capacity.

3. **Gauge-fix between stages.** Insert an explicit canonicalization step between bridge import and repair, then compare against no gauge-fix under identical compute and stop rules.

## Telemetry Fields

Every run should emit these fields where relevant:

- `method`, `source_model`, `target_model`, `dataset`, `example_id`, `seed`, `commit`
- `basis_type`, `shared_basis_id`, `basis_condition_number`, `basis_drift`, `gauge_fix_residual`
- `shared_atom_count`, `exclusive_atom_count`, `atom_sparsity`, `dead_atom_rate`, `atom_overlap`
- `shared_space_residual`, `pairwise_residual`, `held_out_pair_delta`
- `layer_window`, `intrinsic_dimension_proxy`, `route_stability`, `route_help`, `route_harm`
- `verifier_type`, `verifier_granularity`, `stop_reason`, `halt_confidence`, `false_halt`, `overthink_flag`
- `latency_ms`, `tokens_in`, `tokens_out`, `bytes`, `kv_bytes`, `correct`, `baseline_correct`

## Recommendation

The single best next experiment is **GPA-initialized shared hub + sparse dictionary on a held-out pair split**, with routing frozen during evaluation and only one verifier-gated repair step allowed. That isolates whether the bottleneck is gauge mismatch or transport mismatch before adding any extra machinery.
