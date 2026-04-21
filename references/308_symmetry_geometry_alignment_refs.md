# 308 Symmetry / Geometry / Alignment Memo

Primary-source ideas worth stealing for LatentWire:

- **Orthogonal Procrustes / gauge alignment**: fit the least-distorting rotation between source/target latent bases, then treat the residual as interference. This is the cleanest mathematical template for cross-model KV communication when you want to preserve norm and orientation.
  - [Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion](https://arxiv.org/abs/2502.00264)
  - [An empirical evaluation of functional alignment using inter-subject decoding](https://www.sciencedirect.com/science/article/pii/S1053811921009563)

- **CKA / SVCCA / RSA**: use them as geometry diagnostics, not as the method itself. They are good for asking whether two models share a comparable subspace, whether layer-to-layer transport is stable, and whether the learned bridge collapses to a low-rank overlap.
  - [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414)
  - [SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability](https://arxiv.org/abs/1706.05806)
  - [Neural Coding of Cognitive Control: The Representational Similarity Analysis Approach](https://pubmed.ncbi.nlm.nih.gov/33895065/)

- **Model stitching / functional interchangeability**: train a minimal adapter between representations and test whether downstream behavior survives. This is the closest analogue to LatentWire’s “can we communicate useful state across model boundaries?” question.
  - [Model Stitching: Looking For Functional Similarity Between Representations](https://arxiv.org/abs/2303.11277)

- **Permutation symmetries and model merging**: if hidden units are only defined up to permutation, then alignment should explicitly search over these symmetries before introducing richer transport. This is a useful baseline before any learned bridge.
  - [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/abs/2209.04836)
  - [Evolutionary optimization of model merging recipes](https://www.nature.com/articles/s42256-024-00975-8)

- **Optimal transport / Gromov-Wasserstein alignment**: use when geometry is only partially shared and you want soft correspondences rather than strict one-to-one matches. This is especially relevant if source and target heads disagree in rank, attention sparsity, or token salience.
  - [Using Optimal Transport as Alignment Objective for fine-tuning Multilingual Contextualized Embeddings](https://arxiv.org/abs/2110.02887)
  - [Graph Optimal Transport for Cross-Domain Alignment](https://arxiv.org/abs/2006.14744)
  - [Unsupervised alignment in neuroscience: Introducing a toolbox for Gromov-Wasserstein optimal transport](https://www.sciencedirect.com/science/article/pii/S0165027025000846)

- **Representation universality / shared latent bases**: the working hypothesis is that some cross-model transfer is possible because models reuse a common latent geometry, but only after correcting for rotation, permutation, or scale.
  - [Are neural network representations universal or idiosyncratic?](https://www.nature.com/articles/s42256-025-01139-y)
  - [Dimensions underlying the representational alignment of deep neural networks with humans](https://www.nature.com/articles/s42256-025-01041-7)

- **Sparse dictionary / SAE alignment**: if the bridge is too dense or interference-heavy, project both models into a sparse feature basis first, then align feature atoms rather than raw residual stream coordinates.
  - [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/)
  - [Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders](https://arxiv.org/abs/2410.20526)
  - [Improving Dictionary Learning with Gated Sparse Autoencoders](https://arxiv.org/abs/2404.16014)

LatentWire-specific ablation ideas:

1. **Rotation-only vs rotation + permutation vs OT**: start with Procrustes, then add head permutation matching, then soft OT. Report whether each step improves accuracy, calibration, and interference.
2. **Gauge stability across layers**: measure whether a single orthogonal basis works across all layers or whether each layer needs its own gauge. If the bridge drifts by depth, that is a structural blocker.
3. **Orientation-span telemetry**: log singular spectrum concentration, cosine drift, pairwise subspace overlap, and signed alignment determinant. These tell us whether the transport is preserving a coherent basis or just overfitting one head.
4. **Interference telemetry**: log residual norm ratio, gate entropy, head-usage entropy, and expert collapse rate. A good bridge should reduce interference without degenerating to one dominant path.
5. **Sparse-before-align vs align-before-sparse**: compare SAE/dictionary compression before alignment against alignment in the raw latent basis. If sparse features align better, the paper should pivot toward feature transport rather than coordinate transport.
6. **Controlled symmetry breaking**: test whether small diagonal scaling, signed permutations, or low-rank residual corrections help after orthogonal alignment. This is the right place to test whether the remaining mismatch is scale, sign, or genuine semantic interference.

Telemetry to keep interpretable:

- `orientation_span_avg`
- `gauge_det_sign`
- `singular_value_entropy`
- `cka_layerwise`
- `svcca_layerwise`
- `rsa_headwise`
- `subspace_overlap_topk`
- `residual_norm_ratio`
- `gate_entropy`
- `expert_usage_entropy`
- `route_atom_keep_fraction`
- `route_atom_score_entropy`

Practical takeaway:

- If orthogonal alignment closes most of the gap, LatentWire should emphasize gauge correction and symmetry handling.
- If performance only improves after sparse or OT-style transport, the method should be reframed as geometry-preserving feature routing rather than simple adapter fitting.
- If none of the above helps beyond diagnostics, the key blocker is likely semantic mismatch, not coordinate mismatch, and the next move should be richer latent interfaces instead of deeper linear algebra.
