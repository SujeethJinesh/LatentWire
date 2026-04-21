# Gauge / Quotient / Transport References for LatentWire

Web check: 2026-04-21. Primary-source memo focused on 2025-2026 work that sharpens gauge fixing, quotient matching, shared latent spaces, Procrustes-style alignment, and transport between model representations.

## Sources

### Gauge structure and quotient geometry

| Date | Source | Link | Why it matters |
|---|---|---|---|
| 2025-10-29 | Complete Characterization of Gauge Symmetries in Transformer Architectures | [OpenReview](https://openreview.net/forum?id=KrkbYbK0cH) | Gives the cleanest recent statement of Transformer gauge redundancy, including head-wise `((GL(d_k))^h x (GL(d_v))^h) ⋊ S_h` symmetry. This makes raw-space bridge fitting a quotient problem, not a plain regression problem. |
| 2025-12 | Gauge Fiber Bundle Geometry of Transformers | [OpenReview](https://openreview.net/forum?id=YC9O7OyLFK) | Treats Transformer parameters as a principal bundle with gauge fibers and horizontal/vertical decomposition. The useful tool is not just the geometry, but the separation of gauge drift from function-changing drift. |
| 2025-12 | Curvature Meets Bispectrum: A Correspondence Theory for Transformer Gauge Invariants | [OpenReview](https://openreview.net/forum?id=GnjpMOXIkV) | Connects Fisher-Rao curvature on the quotient with bispectral invariants, which suggests LatentWire should report gauge-free diagnostics instead of only CKA or cosine. |

### Multiway alignment and Procrustes transport

| Date | Source | Link | Why it matters |
|---|---|---|---|
| 2026-02-05 | Multi-Way Representation Alignment | [arXiv](https://arxiv.org/abs/2602.06205) | Adapts Generalized Procrustes Analysis to build a shared orthogonal universe across multiple models. This is the strongest recent lead for a multi-model canonical hub before any bridge or router is learned. |
| 2025-10-15 | When Embedding Models Meet: Procrustes Bounds and Applications | [arXiv](https://arxiv.org/abs/2510.13406) | Gives alignment guarantees when pairwise dot products are approximately preserved, plus a simple Procrustes post-processing recipe. Useful as a bound-aware justification for canonicalization before transfer. |
| 2026-02-13 | Transporting Task Vectors across Different Architectures without Training | [arXiv](https://arxiv.org/abs/2602.12952) | Formalizes task-vector transport as a functional matching problem after orthogonal Procrustes alignment. This is close to LatentWire if we treat communication payloads as transportable updates rather than raw activations. |

### Latent transport and symmetry-compatible compression

| Date | Source | Link | Why it matters |
|---|---|---|---|
| 2026-03-24 | Probabilistic Geometric Alignment via Bayesian Latent Transport for Domain-Adaptive Foundation Models | [arXiv](https://arxiv.org/abs/2603.23783) | Introduces uncertainty-aware latent transport with a Bayesian operator over Wasserstein-type geodesics. Strong cue for route uncertainty, not just point-estimate routing, when LatentWire messages are ambiguous. |
| 2025-09 | Transformers as Optimal Transport: A Geometric Framework for Representation Evolution | [OpenReview](https://openreview.net/forum?id=IzAooxm1yv) | Shows attention can be viewed exactly as semi-relaxed entropic OT. The useful transfer is to phrase cross-model communication as transport on a quotient space, not only as basis matching. |

## Why These Are The Right Leads

- The gauge papers imply that a communication bridge can fail because it is learning in the wrong coordinates, not because the target space is intrinsically misaligned.
- The multiway Procrustes paper suggests a shared canonical hub is more defensible than pairwise one-off bridges when there are three or more related models or views.
- The Procrustes bounds paper is the cleanest modern argument for using orthogonal alignment as a certified preprocessing step before any learned transfer.
- The transport papers push the method toward route selection under uncertainty, which is relevant for LatentWire's repair and budget allocation logic.
- The OT view is useful because LatentWire currently has several things that behave like transport: head matching, token remapping, router selection, and byte budgeting.

## Concrete LatentWire Ablations

1. `gauge_fix_then_bridge`: canonicalize each head or layer into a quotient-friendly basis before training the bridge, then compare against the raw bridge at matched bytes and parameters.
2. `quotient_match_after_fix`: after gauge-fixing, solve residual head or channel permutation with Hungarian matching or OT, and log whether transport cost drops relative to the raw bridge.
3. `multiway_canonical_hub`: fit one shared orthogonal hub across all available models or views, then route through the hub instead of learning pairwise bridges.
4. `task_vector_transport`: treat a communication packet as a task vector or delta, align it with orthogonal Procrustes, and test whether transported deltas transfer better than activation regression.
5. `gauge_free_metrics`: report quotient-aware residuals, permutation-invariant costs, and curvature-style diagnostics alongside accuracy so that apparent wins are not parameterization artifacts.
6. `uncertainty_aware_routing`: replace hard router scores with a latent transport posterior or entropy-calibrated route mixture, then test whether uncertainty-aware messages reduce repair failures.
7. `canonicalize_then_compress`: first fix the gauge or align the hub, then apply sparsity, quantization, or codebook compression to the canonical coordinates.
8. `shared_hub_vs_pairwise`: compare a single multiway hub against separate pairwise adapters on the same byte budget, and log whether the hub helps most in the lowest-shot regime.

## Strongest Near-Term Reads For LatentWire

- Use `Multi-Way Representation Alignment` and `When Embedding Models Meet` as the justification for moving from pairwise bridges to a shared canonical hub.
- Use `Complete Characterization of Gauge Symmetries` and `Gauge Fiber Bundle Geometry` to justify quotient-aware evaluation and gauge-fixing before training or scoring bridges.
- Use `Curvature Meets Bispectrum` to add at least one gauge-invariant diagnostic rather than relying on raw cosine or CKA.
- Use `Transporting Task Vectors` and the OT paper to motivate transport-style payloads and route selection as first-class communication objects.
