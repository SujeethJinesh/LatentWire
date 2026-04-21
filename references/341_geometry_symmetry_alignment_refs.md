# Geometry, Symmetry, and Alignment References for LatentWire

This note is a paper-safe ablation map, not a claim sheet. The goal is to mine the math of latent alignment for cross-model communication: orthogonal gauge freedom, correlation-based similarity, OT/GW structure, symmetry/permutation effects, and geometry-aware interpretability. The most useful output from each source is a diagnostic or ablation we can run on route pools, repair traces, and hidden-state bridges.

## 1) Offline bilingual word vectors, orthogonal transformations and the inverted softmax
- Link: https://arxiv.org/abs/1702.03859
- Core mathematical idea: learn an orthogonal map between two vector spaces; orthogonality preserves inner products, norms, and relative geometry while reducing overfitting compared with unconstrained linear maps.
- What LatentWire should borrow: treat cross-model latent communication as a gauge-fixing problem; start with orthogonal alignment as the default bridge, then test whether any extra degrees of freedom are actually needed.
- Concrete ablations: orthogonal map vs unconstrained linear map vs affine map; with/without whitening; with/without norm preservation; initialize with Procrustes then fine-tune.
- Telemetry fields: Procrustes residual norm, singular-value spectrum of the learned map, cosine-distance preservation, layerwise route agreement before/after alignment.
- Claim risks: this can overstate transfer if the true mismatch is nonlinear, tokenization-dependent, or route-specific rather than a simple rotation.

## 2) A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings
- Link: https://arxiv.org/abs/1805.06297
- Core mathematical idea: self-learning iteratively bootstraps a mapping from an initial seed dictionary, alternating assignment and refinement to improve alignment.
- What LatentWire should borrow: a self-training loop for latent routes, where high-confidence matches or repair decisions become pseudo-seeds for the next alignment round.
- Concrete ablations: one-shot alignment vs iterative self-learning; pseudo-seeds from highest-confidence routes only vs thresholded confidence pools; hard vs soft assignment.
- Telemetry fields: pseudo-seed precision/recall, confidence calibration, alignment drift per iteration, entropy of assignment coupling, route-flip rate.
- Claim risks: self-training can amplify early mistakes and may look good only because the selection heuristic narrows the evaluation set.

## 3) Cross-Lingual Alignment of Non-Isomorphic Embeddings with Iterative Normalization
- Link: https://arxiv.org/abs/1906.01622
- Core mathematical idea: make alignment easier by enforcing unit norm and centering before the orthogonal fit; normalization compensates for non-isomorphic embedding geometry.
- What LatentWire should borrow: explicitly normalize hidden-state clouds before any alignment or repair operator, especially if route pools are distorted by scale drift.
- Concrete ablations: raw vs centered vs unit-normalized vs whitened hidden states; per-layer normalization schedules; pre-align normalization only vs pre+post normalization.
- Telemetry fields: mean shift, norm distribution, covariance anisotropy, whitening error, alignment loss before/after normalization.
- Claim risks: normalization can hide useful scale information, so gains may disappear when the downstream task depends on magnitude.

## 4) Deep Canonical Correlation Analysis
- Link: https://arxiv.org/abs/1312.5350
- Core mathematical idea: learn nonlinear projections so two views maximize correlation in a shared latent space; the objective focuses on shared signal rather than raw coordinate matching.
- What LatentWire should borrow: use CCA-style objectives for route-to-route or model-to-model bridges when the goal is shared semantics rather than exact vector recovery.
- Concrete ablations: Procrustes bridge vs linear CCA vs deep CCA-style bridge; correlation objective on route embeddings vs on token embeddings; shared-vs-private subspace split.
- Telemetry fields: canonical correlations by component, explained shared variance, reconstruction loss from shared/private split, per-layer correlation curves.
- Claim risks: correlation can be high even when the model preserves the wrong semantics, so this needs causal or task-level validation.

## 5) SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability
- Link: https://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability
- Core mathematical idea: SVD compresses each representation before CCA, making the similarity estimate more stable and less sensitive to redundant dimensions.
- What LatentWire should borrow: treat SVCCA as a stability diagnostic for route pools and hidden-state bridges, especially when some layers are highly redundant.
- Concrete ablations: raw CCA vs SVCCA; different SVD truncation thresholds; route-level SVCCA over time; repair traces with vs without SVD compression.
- Telemetry fields: retained singular mass, canonical-correlation profiles, rank collapse, layerwise similarity heatmaps, sensitivity to truncation.
- Claim risks: truncation can artificially smooth away differences and make alignment look better than it is.

## 6) Similarity of Neural Network Representations Revisited
- Link: https://proceedings.mlr.press/v97/kornblith19a.html
- Core mathematical idea: CKA provides a scale- and orthogonal-invariant similarity measure that is often more stable than raw CCA or linear regression for comparing representations.
- What LatentWire should borrow: use linear CKA as a primary diagnostic for whether repaired routes and aligned latents are actually converging across models and layers.
- Concrete ablations: CKA before/after each bridge component; CKA vs behavior gain; CKA on raw activations vs hidden deltas vs route logits.
- Telemetry fields: linear CKA, RBF CKA if needed, per-layer pairwise similarity matrices, similarity-vs-accuracy scatter plots.
- Claim risks: CKA is a similarity metric, not a success criterion; improved CKA does not guarantee improved cross-model communication.

## 7) Insights on representational similarity in neural networks with canonical correlation
- Link: https://papers.nips.cc/paper/7815-insights-on-representational-similarity-in-neural-networks-with-canonical-correlation
- Core mathematical idea: PWCCA weights canonical directions by their contribution to the representation, correcting the tendency of plain CCA to overcount weak, noisy directions.
- What LatentWire should borrow: use PWCCA-style weighting to detect whether only a few high-signal latent directions are carrying the whole bridge.
- Concrete ablations: PWCCA vs SVCCA vs CKA as selection criteria for repair candidates; route selection by weighted canonical directions; top-k shared directions only.
- Telemetry fields: direction weights, effective shared rank, signal-to-noise by canonical direction, selection stability over seeds.
- Claim risks: weighted similarity can still be gamed by a narrow subset of directions and may miss failure modes in the discarded subspace.

## 8) Gromov-Wasserstein Alignment of Word Embedding Spaces
- Link: https://arxiv.org/abs/1809.00013
- Core mathematical idea: align two spaces by preserving internal pairwise geometry rather than assuming direct pointwise correspondence; GW is the right tool when spaces are non-isomorphic.
- What LatentWire should borrow: use OT/GW when route pools or latent traces are structurally similar but not coordinate-aligned, which is likely closer to cross-model communication than raw Procrustes.
- Concrete ablations: GW on route-pool geometry vs token geometry; exact vs entropic GW; partial GW when only some routes should align; GW initialization followed by orthogonal refinement.
- Telemetry fields: GW coupling entropy, transport mass sparsity, within-space distance distortion, matched-route consistency, partial-match coverage.
- Claim risks: GW can be expensive and may fit structural noise if the underlying geometry is too coarse or if the route graph is unstable.

## 9) Gromov-Wasserstein unsupervised alignment reveals structural correspondences between humans and LLMs
- Link: https://www.nature.com/articles/s41598-024-65604-1
- Core mathematical idea: GW can uncover cross-system structural correspondences even when the two representations live in very different spaces and are not directly point-matched.
- What LatentWire should borrow: the evaluation mindset more than the algorithm itself; test whether LatentWire preserves structural correspondences across models, not just answer identity.
- Concrete ablations: human-like structural analogies for route pools; GW-based structural probes on hidden-state trajectories; compare GW matches against random and nearest-neighbor baselines.
- Telemetry fields: structural correspondence score, coupling stability, matched-neighborhood overlap, trajectory curvature before/after repair.
- Claim risks: structural correspondences are easier to over-interpret than direct accuracy gains, so any narrative must stay tied to task outcomes.

## 10) Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion
- Link: https://arxiv.org/abs/2502.00264
- Core mathematical idea: transformer fusion is not only about head permutation; rotation symmetry matters, so model combination should consider orthogonal reparameterizations at the weight level.
- What LatentWire should borrow: treat latent communication as a rotation-selection problem and check whether some repair gains come from latent basis choice rather than new information.
- Concrete ablations: head permutation only vs permutation+rotation; fixed orthogonal fusion vs learned orthogonal fusion; basis-frozen vs basis-adapted repairs.
- Telemetry fields: rotation angle distributions, basis-change magnitude, layerwise fusion error, head-matching stability across seeds.
- Claim risks: weight-space rotation benefits may not transfer directly to route-level hidden-state repair without an explicit bridge between the two settings.

## 11) ReFT: Representation Finetuning for Language Models
- Link: https://arxiv.org/abs/2404.03592
- Core mathematical idea: intervene directly on hidden representations with small learned transformations rather than updating the full model.
- What LatentWire should borrow: a disciplined intervention layer over hidden states or route summaries, especially when we want interpretability and low parameter count.
- Concrete ablations: fixed intervention vs learned intervention; one-token vs multi-token intervention; intervention at early vs late layers; intervention on route summaries vs token states.
- Telemetry fields: intervention norm, activation shift, downstream gain per intervention site, calibration drift, hidden-state change sparsity.
- Claim risks: representation interventions can look elegant while still being too task-specific or too expensive in generation-time compute.

## 12) Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models
- Link: https://arxiv.org/abs/2410.06981
- Core mathematical idea: sparse autoencoders can uncover monosemantic features that appear to recur across different LLMs, suggesting a shared feature basis beneath superficial model differences.
- What LatentWire should borrow: inspect whether route pools share a sparse feature basis and whether communication improves when we map between feature dictionaries instead of raw hidden coordinates.
- Concrete ablations: SAE features vs raw residual stream; feature-space alignment vs token-space alignment; universal features restricted to route-critical activations.
- Telemetry fields: feature sparsity, feature overlap across models, feature activation histograms, reconstruction error, monosemanticity proxies.
- Claim risks: universal features are still a hypothesis, and findings can be sensitive to SAE training choices and layer selection.

## 13) The Geometry of Multilingual Language Model Representations
- Link: https://arxiv.org/abs/2205.10964
- Core mathematical idea: multilingual LMs maintain a shared latent space while retaining language-sensitive axes; geometry is mixed, not fully collapsed into one universal basis.
- What LatentWire should borrow: expect a shared core plus model-specific residual axes; do not force full collapse if a decomposed shared/private factorization works better.
- Concrete ablations: shared-private decomposition; orthogonal complement constraints; cross-model communication through only the shared subspace; language/model-sensitive residual channels ablated out.
- Telemetry fields: shared-subspace dimensionality, language/model-sensitive axis energy, principal-angle spectra, transfer-vs-private leakage.
- Claim risks: multilingual geometry is a close analogy, not the same object as cross-model communication, so the shared/private story must be validated empirically.

## 14) Language Is Not All You Need: Aligning Perception with Language Models
- Link: https://arxiv.org/abs/2302.14045
- Core mathematical idea: multimodal systems need explicit alignment between perception and language spaces, typically through lightweight projectors or interface modules.
- What LatentWire should borrow: use projector-style interfaces between model latents and route latents, rather than assuming the hidden states are already in a commensurate basis.
- Concrete ablations: raw bridge vs projector bridge; symmetric vs asymmetric projectors; modality-style encoder-decoder interface for cross-model routes; frozen vs learned projector.
- Telemetry fields: projector rank, alignment loss, cross-space retrieval accuracy, route recovery after projection, stability under prompt perturbation.
- Claim risks: modality alignment can be too forgiving compared with same-modality model-to-model alignment, so the analogy may overestimate feasibility.

## 15) Traveling Words: A Geometric Interpretation of Transformers
- Link: https://arxiv.org/abs/2309.07315
- Core mathematical idea: transformer updates can be viewed as trajectories on a hypersphere, making residual-stream dynamics a geometric object rather than just a stack of linear layers.
- What LatentWire should borrow: track route trajectories, not only endpoints; some communication failures may be trajectory-path problems rather than endpoint misalignment.
- Concrete ablations: endpoint-only repair vs trajectory-aware repair; curvature regularization; arc-length or geodesic proxy losses; compare early-layer and late-layer trajectory matching.
- Telemetry fields: trajectory curvature, geodesic deviation, angular vs radial components of the update, path length to correction.
- Claim risks: geometric trajectories are easy to visualize but can tempt overinterpretation unless tied to route outcome and repair help/harm.

## 16) Transformers represent belief state geometry in their residual stream
- Link: https://arxiv.org/abs/2405.15943
- Core mathematical idea: residual streams can encode structured belief-state geometry, sometimes linearly, even when the latent object has nontrivial geometry.
- What LatentWire should borrow: treat route pools as structured belief states and test whether linear readouts already recover the relevant geometry before adding heavier alignment machinery.
- Concrete ablations: linear readout vs nonlinear readout of route belief state; belief-state probes before/after repair; compare classically aligned routes with causally relevant route probes.
- Telemetry fields: linear probe accuracy, belief-state separability, probe calibration, belief-state trajectory stability.
- Claim risks: belief-state structure may be task-specific and may not generalize to cross-model latent communication without extra alignment constraints.

## 17) Multi-Scale Manifold Alignment: A Unified Framework for Enhanced Explainability of Large Language Models
- Link: https://arxiv.org/abs/2505.20333
- Core mathematical idea: align global, intermediate, and local manifolds jointly so the alignment respects multiple semantic scales instead of flattening everything into one global map.
- What LatentWire should borrow: a multi-scale bridge for route pools, where coarse route identity, intermediate reasoning state, and local token evidence each get their own alignment pressure.
- Concrete ablations: single-scale vs multi-scale alignment; coarse-to-fine route repair; local-token supervision added only after global route agreement; scale-specific loss weights.
- Telemetry fields: scale-wise alignment error, coarse/intermediate/local agreement, semantic drift across scales, per-scale repair contribution.
- Claim risks: multi-scale objectives can hide which scale is actually driving gains, so the paper needs a strict scale-wise ablation table.

## Practical telemetry contract for LatentWire
- Track the same fields across every geometry/repair ablation: route accuracy, repair help, repair harm, oracle gap, selection entropy, calibration error, CKA/SVCCA/PWCCA, orthogonal residual norm, GW coupling entropy, singular-value spectra, and per-layer trajectory curvature.
- If a method improves geometry metrics but not held-out route accuracy, treat it as a diagnostic, not a claim.
- If a method improves accuracy but inflates compute, report matched-budget or bytes-per-correct-answer numbers before framing it as a paper-worthy method.
