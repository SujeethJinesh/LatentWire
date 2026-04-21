# Route / Transport Geometry After Span-ALM Failure

Context:
- `span-ALM` failed, so the output-space / likelihood-side bridge path is not the next cheap win.
- The remaining question is whether any **geometric shortcut** is still real in cross-model KV transport.
- This note focuses on query-conditioned route atoms, head/layer gauge symmetry, OT variants, CCA/Procrustes shortcuts, and KV orientation.

## 1) Query-conditioned route atoms

Hypothesis:
- The transport map may be better approximated by a small set of **route atoms** with query-dependent coefficients.
- If so, the problem is not a single static bridge, but a low-rank routing dictionary over a few canonical transport paths.

Recent references:
- [FLARE: Fast Low-rank Attention Routing Engine](https://arxiv.org/abs/2508.12594), 2025-08-18
- [Route-DETR: Pairwise Query Routing in Transformers for Object Detection](https://arxiv.org/abs/2512.13876), 2025-12-15
- [Geometric Routing Enables Causal Expert Control in Mixture of Experts](https://arxiv.org/abs/2604.14434), 2026-04-15

What to ablate:
- One static route vs 2 / 4 / 8 route atoms.
- Softmax-mixture routing vs Sinkhorn-selected route atoms.
- Route atoms trained on query features only vs query + transport residual.

Why this may still help:
- Query routing is more expressive than a single gate.
- If the current bridge only used a scalar or tiny local projector, it may have been too weak to express route selection.

Why it may already be saturated:
- We already exhausted query-conditioned local bridges and banks.
- If route atoms only reparameterize a weak bridge, they likely inherit the same floor.

Interpretation:
- Route atoms are the last plausible “cheap geometry” move **only if** they change the route geometry, not just the bridge amplitude.

## 2) Head / layer gauge symmetry

Hypothesis:
- Head identity can still be non-canonical after transport.
- Gauge freedom can show up as permutations, rotations, and basis changes inside heads or head groups.

Recent references:
- [Learnable Permutation for Structured Sparsity on Transformer Models](https://arxiv.org/abs/2601.22980), 2026-01-30
- [PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](https://arxiv.org/abs/2510.10136), 2025-10-11
- [Attention Layers Add Into Low-Dimensional Residual Subspaces](https://arxiv.org/abs/2508.16929), 2025-08-23
- [Low-Rank Key Value Attention](https://arxiv.org/abs/2601.11471), 2026-01-16

What to ablate:
- Identity head ordering vs exact permutation vs Sinkhorn-soft permutation.
- Rotation-only vs rotation + permutation.
- Shared low-rank basis vs separate per-head basis.

Why this may still help:
- `grouped_rotational_transport` is the only geometry shortcut that clearly moved the floor.
- A residual permutation on top of rotation is a plausible remaining gauge fix.

Why it may already be saturated:
- `grouped_permutation`, `grouped_canonical_transport`, and related canonicalized transport variants already plateaued.
- That suggests the gauge problem is real but not sufficient.

Interpretation:
- Keep gauge fixing as a comparator, but do not expect it to beat the current floor by itself.

## 3) Sinkhorn / OT variants

Hypothesis:
- The transport itself might still be too rigid if it is not solved as a softer coupling problem.
- A Sinkhorn / entropic OT coupling can be interpreted as a smoother route mixture, especially when heads are only partially matched.

Recent references:
- [PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](https://arxiv.org/abs/2510.10136), 2025-10-11
- [An in depth look at the Procrustes-Wasserstein distance: properties and barycenters](https://arxiv.org/abs/2507.00894), 2025-07-01
- [Procrustes Wasserstein Metric: A Modified Benamou-Brenier Approach with Applications to Latent Gaussian Distributions](https://arxiv.org/abs/2503.16580), 2025-03-20

What to ablate:
- Hard permutation vs Sinkhorn-soft permutation vs rectangular Sinkhorn.
- Cost from template distance vs cost from QK geometry vs cost from subspace overlap.
- Transport with and without residual correction.

Why this may still help:
- OT is the cleanest way to turn head matching into a geometric coupling problem.
- Rectangular OT can encode unmatched mass, which is useful when source and target heads are not in 1-to-1 correspondence.

Why it may already be saturated:
- We already have multiple transport variants in this repo, and the better ones plateaued.
- If the coupling is the issue, the gain should have shown up already in the transport family.

Interpretation:
- OT is still worth one more comparator, but only if the cost is genuinely better than the old static descriptors.

## 4) CCA / Procrustes shortcuts

Hypothesis:
- The remaining mismatch might be rigid: subspace overlap plus orthogonal alignment.
- If so, CCA/Procrustes should be enough after the right centering / whitening.

Primary references:
- [An in depth look at the Procrustes-Wasserstein distance: properties and barycenters](https://arxiv.org/abs/2507.00894), 2025-07-01
- [Procrustes Wasserstein Metric: A Modified Benamou-Brenier Approach with Applications to Latent Gaussian Distributions](https://arxiv.org/abs/2503.16580), 2025-03-20

Classic grounding:
- CCA/SVCCA/CKA are still the standard representation-similarity shortcuts, but they are diagnostics rather than a strong new method lane here.

What to ablate:
- Procrustes vs ridge vs reduced-rank vs CCA-derived shared basis.
- Shared low-rank basis with and without whitening.

Why this may still help:
- These are the simplest canonical alignment shortcuts.
- They are cheap and easy to interpret.

Why it looks saturated:
- `grouped_subspace_transport`, `grouped_shared_basis_transport`, and covariance variants already failed to move the floor enough.
- If the rigid shortcut were sufficient, the residual/low-rank transport family should already have shown it.

Interpretation:
- Use CCA/Procrustes as a sanity check, not as the next big method.

## 5) KV orientation

Hypothesis:
- Even after alignment, the KV cache may still be oriented incorrectly.
- Rotation is the remaining cheap symmetry that can preserve more structure than a raw permutation or basis fit.

Recent references:
- [Attention Layers Add Into Low-Dimensional Residual Subspaces](https://arxiv.org/abs/2508.16929), 2025-08-23
- [Low-Rank Key Value Attention](https://arxiv.org/abs/2601.11471), 2026-01-16

What to ablate:
- Orthogonal vs Hadamard vs DCT vs identity rotation.
- Rotation before whitening vs rotation after whitening.
- Rotation-only transport vs rotation + shared basis.

Why this is still live:
- `grouped_rotational_transport` is the only clear positive geometry result we have.

Why it may be saturated:
- Once rotation is already applied, the remaining gain from orientation alone seems bounded.

Interpretation:
- Keep rotation as the best cheap symmetry fix.

## Cheap ablations that are still distinct from the dead likelihood-mass family

1. Route-atom bank with 2, 4, and 8 atoms, using query-conditioned coefficients.
2. Rotation-only vs rotation + exact permutation vs rotation + Sinkhorn-soft permutation.
3. Rectangular Sinkhorn OT with costs built from QK geometry vs shared-basis subspace overlap.
4. Procrustes vs ridge vs reduced-rank alignment under the same layer map.
5. Rotation versus Hadamard/DCT as the head-orientation comparator.

## What the current evidence says

Ranked by plausibility:
1. **KV orientation / rotational gauge** remains the strongest live shortcut.
2. **Head permutation / soft matching** is plausible but likely secondary.
3. **Query-conditioned route atoms** could help only if they change the actual route geometry, not just the local bridge amplitude.
4. **Sinkhorn/OT refinements** are worth one clean comparator with a real geometric cost.
5. **CCA / Procrustes** should now be treated as diagnostics, not the next main lane.

## Decision rule

- If route-atom routing, permutation, and rotation all tie the floor, the problem is no longer a symmetry problem.
- At that point, the next move should be a module replacement or tokenizer/shared-interface transplant, not another local bridge.
