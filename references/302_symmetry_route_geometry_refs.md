# Symmetry, Routing, and Geometry References

Context:
- LatentWire’s recent negatives suggest the main bottleneck is not just bridge capacity, but how information is *routed*, *matched*, and *preserved* under quotient symmetries.
- The relevant questions are: which symmetries are real, which are gauge artifacts, and whether the bridge should align heads, layers, tokens, or latent bottlenecks.
- This memo only records primary sources, the math hypotheses they motivate, and the ablations that are worth running next.

## 1) Routing, head interaction, and sparse composition

- [Multi-Head Attention: Collaborate Instead of Concatenate](https://arxiv.org/abs/2006.16362) (Cordonnier et al., 2020). Core point: head outputs are redundant enough that sharing or composing query/key projections can recover efficiency. For LatentWire, this is a direct warning that naïvely concatenated head states may be the wrong object to transport.
- [Alleviating the Inequality of Attention Heads for Neural Machine Translation](https://arxiv.org/abs/2009.09672) (Sun et al., 2020). Head imbalance is real and model-dependent. This supports headwise diagnostics before any bridge is fit.
- [Alignment Attention by Matching Key and Query Distributions](https://arxiv.org/abs/2110.12567) (Zhang et al., 2021). Explicitly regularizes per-head key/query distributions; useful as a minimal alignment baseline when the bridge seems to fail because Q/K geometry is miscalibrated.
- [Improving Transformers with Dynamically Composable Multi-Head Attention](https://arxiv.org/abs/2405.08553) (Xiao et al., 2024). Heads can be dynamically composed rather than treated as independent channels. This makes “route atoms” plausible: the bridge may need compositional head mixing instead of a dense global map.
- [Optimizing Knowledge Distillation in Transformers: Enabling Multi-Head Attention without Alignment Barriers](https://arxiv.org/abs/2502.07436) (Bing et al., 2025). Head-count mismatch is an alignment barrier by itself. Any cross-model bridge should treat head cardinality as a first-class variable, not an implementation detail.
- [Routing Manifold Alignment Improves Generalization of Mixture-of-Experts LLMs](https://arxiv.org/abs/2511.07419) (2025). Routing geometry itself can be aligned; the route manifold is a target, not just the token manifold.
- [Multilingual Routing in Mixture-of-Experts](https://arxiv.org/abs/2510.04694) (2025). Layerwise routing can be language-specific in early/late blocks while mid layers share more structure. For LatentWire, this argues for layer-local routing rather than a single all-layer gate.
- [Unifying Mixture of Experts and Multi-Head Latent Attention for Efficient Language Models](https://arxiv.org/abs/2508.01261) (2025). Shared always-on experts plus sparse specialists is a good template for a communication bridge: separate stable transport from specialized transport.

Concrete hypotheses:
- Head mixing before routing can destroy separability; route decisions should be made per head or per small head-group.
- Sparse, query-conditioned route atoms should outperform a dense bridge at equal byte budget if the task really has factorized modes.
- A shared expert plus specialist experts is likely better than a pure softmax mixture because it reduces mean-collapse.

Concrete ablations:
- `headwise_route_atom`: route atoms are fit per attention head before concatenation, with route entropy and collision counts logged.
- `shared_plus_specialist_route`: one always-on shared atom plus one or more routed specialist atoms.
- `layer_local_route_only`: compare early/mid/late layer routing instead of one global route bank.
- `head_imbalance_probe`: compare per-head cosine/CKA/variance before and after the bridge, then correlate with answer flips.

## 2) Permutation, gauge, and equivalence-class alignment

- [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414) (Kornblith et al., 2019). Linear CKA is a strong similarity statistic, but it is about representational structure, not functional equivalence.
- [Reliability of CKA as a Similarity Measure in Deep Learning](https://arxiv.org/abs/2210.16156) (Davari et al., 2022). CKA can be manipulated under simple transformations without big functional change. This is the main cautionary result: high CKA is not evidence of a useful bridge.
- [Fine-Tuned Transformers Show Clusters of Similar Representations Across Layers](https://arxiv.org/abs/2109.08406) (Phang et al., 2021). Layer clusters are stable enough that layerwise matching is meaningful, but not enough to justify a single global alignment map.
- [Tracing Representation Progression: Analyzing and Enhancing Layer-Wise Similarity](https://arxiv.org/abs/2406.14479) (Jiang et al., 2024). Layerwise cosine and CKA can track progression, but the more interesting signal is whether similarity rises where prediction becomes linearly useful.
- [Unification of Symmetries Inside Neural Networks: Transformer, Feedforward and Neural ODE](https://arxiv.org/abs/2402.02362) (Hashimoto et al., 2024). Parametric redundancies can be understood as gauge symmetries.
- [Transformer models are gauge invariant: A mathematical connection between AI and particle physics](https://arxiv.org/abs/2412.14543) (van Nierop, 2024). Useful as a reminder that many transformer parameterizations are not unique.
- [CASK: A Gauge Covariant Transformer for Lattice Gauge Theory](https://arxiv.org/abs/2501.16955) (Nagai et al., 2025). If the task has a symmetry group, the architecture should respect it rather than fight it.
- [Optimal Symmetries in Binary Classification](https://arxiv.org/abs/2408.08823) (Ngairangbam and Spannowsky, 2024). Bigger symmetry groups are not automatically better; choose the subgroup that matches the task.

Concrete hypotheses:
- A bridge can look strong under CKA while still being functionally irrelevant because CKA is invariant to many nuisance transformations.
- The right alignment object may live on a quotient space: head permutations, orthogonal gauges, or layer-wise equivalence classes.
- Over-regularizing to global symmetry may erase task-specific asymmetry; the best bridge may be only locally equivariant.

Concrete ablations:
- `gauge_free_vs_gauge_fixed`: compare free bridges against versions with explicit permutation/orthogonal gauge normalization.
- `cka_vs_functional_gain`: log CKA, Procrustes residual, and downstream accuracy together and look for mismatch cases.
- `layerwise_quotient_alignment`: align layers independently on a quotient metric, not with one monolithic transform.
- `symmetry_ablation_controls`: random orthogonal / random permutation controls that preserve norm but destroy geometry.

## 3) Procrustes, optimal transport, and head matching

- The internal note `references/161_transformers_as_optimal_transport_a_geometric_framework_for_representation_evolution.pdf` is useful background for OT-style representation matching, but no stable primary-source link was verified in current search. Do not cite it externally until the source is confirmed.
- [Gromov-Wasserstein unsupervised alignment reveals structural correspondences between the color similarity structures of humans and large language models](https://arxiv.org/abs/2308.04381) (Kawakita et al., 2023). GW alignment is useful when the two spaces do not share coordinates but do share relational structure.
- [Neural Entropic Gromov-Wasserstein Alignment](https://arxiv.org/abs/2312.07397) (Wang and Goldfeld, 2023). Soft structural matching scales better than exact discrete matching and is a natural bridge candidate when head counts differ.
- [Representational Alignment Across Model Layers and Brain Regions with Hierarchical Optimal Transport](https://arxiv.org/abs/2510.01706) (Shah and Khosla, 2025). Global soft couplings plus neuron-level transport plans are a better analogy for LatentWire than independent layer-by-layer matching.
- [Alignment Attention by Matching Key and Query Distributions](https://arxiv.org/abs/2110.12567) (Zhang et al., 2021). This is also relevant here because matching key/query distributions is a simple transport-like baseline.
- [Dual-Space Knowledge Distillation with Key-Query Matching for Large Language Models with Vocabulary Mismatch](https://arxiv.org/abs/2603.22056) (Tsiapali et al., 2026). Cross-tokenizer KD becomes easier when the Q/K spaces are aligned explicitly.

Concrete hypotheses:
- If source and target heads are only identifiable up to permutation, then OT should beat fixed index matching.
- Procrustes should help when the spaces are mostly related by rotation/reflection; OT should help when there is local mass splitting; a hybrid should help when both are true.
- A low OT residual should predict useful transfer only if the transported representation still supports the task, so residuals need to be paired with behavior.

Concrete ablations:
- `procrustes_then_ot`: first solve a rigid basis alignment, then a soft mass transport.
- `ot_vs_permutation_match`: compare OT, greedy head matching, and fixed-index heads at equal parameter budget.
- `hierarchical_ot_bridge`: align layers globally and heads locally with a two-level transport plan.
- `residual_behavior_correlation`: correlate transport residuals with answer flips, entropy shift, and controlled-slice accuracy.

## 4) Latent bottlenecks, query routing, and cross-modal communication

- [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206) (Jaegle et al., 2021). A small latent set can absorb very large inputs through iterative cross-attention.
- [Perceiver-VL: Efficient Vision-and-Language Modeling with Iterative Latent Attention](https://arxiv.org/abs/2211.11701) (Tang et al., 2022). Latent attention gives a clean efficiency/performance curve and is a strong benchmark for any shared bottleneck.
- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) (Li et al., 2023). Q-Former is the canonical “small query bottleneck between frozen experts” design.
- [RegionBLIP: A Unified Multi-modal Pre-training Framework for Holistic and Regional Comprehension](https://arxiv.org/abs/2308.02299) (Zhou et al., 2023). Freezing the bottleneck and specializing only the adapter is a useful control for LatentWire.
- [LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models](https://arxiv.org/abs/2502.02406) (Chang and Venkataraman, 2025). The query block can be much smaller than the KV block; communication should therefore be query-aware.
- [Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution](https://arxiv.org/abs/2510.00636) (2025). Future-query utility is a better pruning signal than source saliency alone.
- [Seeing the Forest and the Trees: Query-Aware Tokenizer for Long-Video Multimodal Language Models](https://arxiv.org/abs/2511.11910) (2025). Query-aware token selection is exactly the type of evidence gate LatentWire needs if the bottleneck is the right abstraction.

Concrete hypotheses:
- If cross-model communication is the bottleneck, then a tiny query-conditioned latent set should preserve answer-relevant structure while dropping nuisance detail.
- The bottleneck should be evaluated as a router, not only as a translator.
- Query-aware selection should beat source-only selection if the target model’s future query distribution matters.

Concrete ablations:
- `latent_kv_bottleneck`: compress source K/V into a small latent basis, translate the latent, then reconstruct the target side.
- `query_pool_transport`: let a small learned query pool read source states before emitting target-side messages.
- `byte_probe_bridge`: evaluate transported states through a tokenizer-independent byte probe.
- `future_query_weighted_kv`: weight bridge atoms by predicted future utility instead of current source saliency.

## 5) What to log so the next round is interpretable

- Route geometry: atom histogram, route entropy, dead-atom count, top-k margin, and head collision rate.
- Alignment geometry: CKA, CCA/SVCCA, Procrustes residual, OT cost, and per-layer cosine.
- Bottleneck health: reconstruction error, byte-probe likelihood, token-family KL, and norm drift.
- Behavior: controlled-slice accuracy, paired flip rate against target-alone, answer entropy shift, repetition rate, and extraction failure rate.
- Symmetry controls: random permutation, random orthogonal basis, and shuffled head-order ablations.

## 6) Priority stack

1. `headwise_route_atom` because it tests whether route collisions are the actual failure mode.
2. `procrustes_then_ot` because rigid-plus-soft alignment is the cleanest quotient-space baseline.
3. `latent_kv_bottleneck` because it directly tests whether a smaller shared interface is more stable than full-state transport.
4. `byte_probe_bridge` because tokenizer mismatch is likely part of the failure even when geometry looks good.
