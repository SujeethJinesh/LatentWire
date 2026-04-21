# Symmetry / Alignment / Steering References for LatentWire

This memo collects primary sources that are directly useful for cross-model latent/KV transport when the main question is not “can we compress harder?” but “what symmetry, orientation, or shortcut structure is still left to exploit?”

The working hypothesis for LatentWire is:

- Some of the gap is pure orientation mismatch.
- Some of the gap is permutation / gauge ambiguity.
- Some of the gap is structural mismatch that no linear post-hoc map can remove.
- Sparse, interpretable bases may make the shortcut structure easier to isolate than dense residual activations.

## 1) Orthogonal Procrustes and transport-friendly alignment

| Source | Primary link | Why it matters for LatentWire | Direct ablation it suggests |
|---|---|---|---|
| **Unsupervised Alignment of Embeddings with Wasserstein Procrustes** | https://arxiv.org/abs/1805.11222 | Canonical orthogonal-plus-permutation alignment. This is the cleanest baseline for “same geometry, different orientation.” | Compare `identity`, `orthogonal Procrustes`, and `Wasserstein Procrustes` on the same hidden/KV pair, holding transport budget fixed. |
| **Quantized Wasserstein Procrustes Alignment of Word Embedding Spaces** | https://arxiv.org/abs/2212.02468 | Shows that qWP can approximate a Procrustes/OT alignment more cheaply via quantization. | Test whether a quantized or subsampled alignment pass is enough before any learned bridge is trained. |
| **When Embedding Models Meet: Procrustes Bounds and Applications** | https://arxiv.org/abs/2510.13406 | A recent theory-first statement of when orthogonal alignment should work at all. Useful for deciding whether the bridge problem is actually rigid or not. | Use its conditions as a checklist: if they fail, stop trying to force rigid alignment and switch to OT / stitching / residual methods. |

## 2) CCA / CKA and representational similarity diagnostics

| Source | Primary link | Why it matters for LatentWire | Direct ablation it suggests |
|---|---|---|---|
| **SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability** | https://arxiv.org/abs/1706.05806 | Gives a low-rank correlation lens for layer pairing and subspace overlap. | Use SVCCA to select candidate layer pairs before fitting transport, rather than assuming monotone depth alignment. |
| **Insights on Representational Similarity in Neural Networks with Canonical Correlation** | https://arxiv.org/abs/1806.05759 | Introduces PWCCA and improves over raw SVCCA when signal/noise separation matters. | Compare `SVCCA-picked` vs `PWCCA-picked` vs `manual` pairings for the bridge. |
| **Similarity of Neural Network Representations Revisited** | https://arxiv.org/abs/1905.00414 | The clean CKA reference; important because it is invariant to many linear reparameterizations but still exposes representational correspondence. | Treat CKA as a diagnostic, not a method: ask whether high CKA actually predicts better KV transport. |
| **google/svcca repo** | https://github.com/google/svcca | Official code for SVCCA/PWCCA diagnostics. | Reuse the same diagnostics to score layer pairs before and after any bridge transform. |

## 3) Gromov-Wasserstein / OT for structural alignment

| Source | Primary link | Why it matters for LatentWire | Direct ablation it suggests |
|---|---|---|---|
| **Gromov-Wasserstein Learning for Graph Matching and Node Embedding** | https://arxiv.org/abs/1901.06003 | Uses GW to align relational structure, not just pointwise coordinates. This is the right mental model if heads or layers should be matched by interaction pattern rather than raw embedding distance. | Model heads as nodes in a graph and compare `pointwise Procrustes` vs `GW structural match`. |
| **Neural Entropic Gromov-Wasserstein Alignment** | https://arxiv.org/abs/2312.07397 | Shows a scalable neural EGW estimator; useful if the bridge should match relational structure with soft transport. | Compare hard one-to-one matching to entropic soft transport on the same latent graph. |
| **Transformer Fusion with Optimal Transport** | [paper](https://arxiv.org/abs/2310.05719), [repo](https://github.com/graldij/transformer-fusion) | Directly on-model for cross-architecture fusion with OT; good bridge-language for LatentWire. | Evaluate whether OT fusion improves over a simple aligned projector when architectures differ. |
| **FlashSinkhorn: IO-Aware Entropic Optimal Transport** | https://arxiv.org/abs/2602.03067 | A systems-level entropic-OT reference. Useful if a soft-transport implementation becomes the bottleneck rather than the math. | Use it as the efficiency baseline for any entropic transport solver in the bridge. |

## 4) Permutation and gauge symmetry in neural nets

| Source | Primary link | Why it matters for LatentWire | Direct ablation it suggests |
|---|---|---|---|
| **Git Re-Basin: Merging Models modulo Permutation Symmetries** | https://arxiv.org/abs/2209.04836 | The canonical “same function, different neuron order” reference. | Compare `no permutation`, `weight matching`, and `Git Re-Basin` before any latent transport. |
| **Weight-space symmetry in deep networks gives rise to permutation saddles** | https://arxiv.org/abs/1907.02911 | Shows that permutation symmetry is not a minor nuisance; it creates real flat valleys / saddles. | If a bridge only works after permutation fixing, that is evidence the mismatch is symmetry-driven, not semantic. |
| **Maximal Gauge Symmetry in Transformer Architectures** | https://openreview.net/forum?id=K1df8mmncF | A strong transformer-specific gauge statement, including attention and RoPE/GQA/MQA structure. | Test whether a gauge-fixed head/channel basis outperforms naive per-layer alignment. |
| **Unification of Symmetries Inside Neural Networks: Transformer, Feedforward and Neural ODE** | https://arxiv.org/abs/2402.02362 | Frames transformer redundancies as gauge symmetries, which is exactly the right language for “orientation is not canonical.” | Use it to justify explicit gauge fixing before comparing transport maps. |
| **Transformer models are gauge invariant** | https://arxiv.org/abs/2412.14543 | A recent and compact transformer-gauge argument. | Use as a sanity check that the transport basis is not unique even when outputs are identical. |

## 5) Model stitching and functional compatibility

| Source | Primary link | Why it matters for LatentWire | Direct ablation it suggests |
|---|---|---|---|
| **Revisiting Model Stitching to Compare Neural Representations** | https://arxiv.org/abs/2106.07682 | Stitching is a functional test, not just a similarity score. It can reveal whether two layers are actually interoperable. | Compare a bridge trained to minimize reconstruction loss vs a stitch layer trained to preserve end-task performance. |
| **Model Stitching: Looking For Functional Similarity Between Representations** | https://arxiv.org/abs/2303.11277 | Extends stitching across shape / architecture mismatch. | Use as a comparator for cross-architecture latent transport, especially when Procrustes fails. |
| **samuela/git-re-basin repo** | https://github.com/samuela/git-re-basin | Official code for permutation-based model merging. | Reuse its matching logic to test whether LatentWire’s best map is closer to a merge permutation than a dense projector. |

## 6) Representation alignment and concept transfer

| Source | Primary link | Why it matters for LatentWire | Direct ablation it suggests |
|---|---|---|---|
| **Text-To-Concept (and Back) via Cross-Model Alignment** | https://arxiv.org/abs/2305.06386 | A strong example that a linear alignment can make two model spaces interoperable enough for concept transfer. | Test whether LatentWire can recover a concept-level shared subspace before attempting full token-level transport. |
| **Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts** | https://arxiv.org/abs/2501.02009 | Directly about cross-LLM concept transfer with a linear bridge. | Use as evidence for a concept-space transfer baseline, separate from exact KV transport. |
| **minyoungg/platonic-rep repo** | https://github.com/minyoungg/platonic-rep | Provides alignment metrics and code for comparing model representations across modalities. | Use its metrics as a diagnostic for whether LatentWire is preserving shared neighborhoods or just cosine similarity. |

## 7) Activation steering and sparse autoencoders

| Source | Primary link | Why it matters for LatentWire | Direct ablation it suggests |
|---|---|---|---|
| **Representation Engineering: A Top-Down Approach to AI Transparency** | https://arxiv.org/abs/2310.01405 | The clean top-down steering reference: manipulate population-level representations, not individual neurons. | Compare steering in raw residual space vs an aligned latent basis. |
| **Steering Llama 2 via Contrastive Activation Addition** | https://arxiv.org/abs/2312.06681 | A simple, strong activation-steering baseline that is close in spirit to transport-plus-residual edits. | Use contrastive steering vectors as a shortcut baseline against the bridge itself. |
| **Sparse Autoencoders Find Highly Interpretable Features in Language Models** | https://arxiv.org/abs/2309.08600 | The canonical SAE feature-disentangling paper. Sparse bases may make source/target orientation cleaner. | Compare transport in raw activations vs SAE latents vs SAE-derived steering directions. |
| **Scaling and evaluating sparse autoencoders** | https://arxiv.org/abs/2406.04093 | Adds scaling laws and quality metrics; useful for deciding whether SAE capacity is enough. | Sweep SAE width/sparsity and test whether better monosemanticity improves transport fidelity. |
| **openai/sparse_autoencoder repo** | https://github.com/openai/sparse_autoencoder | Official SAE code and GPT-2-small feature visualizer. | Use as a concrete implementation template for feature-space transport. |
| **SAELens repo** | https://github.com/decoderesearch/SAELens | The current standard tooling stack for SAE analysis and visualization. | Use it to inspect whether transport failure comes from feature entanglement or basis mismatch. |
| **steering-vectors repo** | https://github.com/steering-vectors/steering-vectors | A practical repo for steering vectors / representation engineering. | Use as the shortcut baseline: if a steering vector solves the task, the bridge may be overkill. |

## What this means for LatentWire

The strongest pattern across the sources is that a bridge should probably be factored into three pieces:

1. A symmetry-fixing step: permutation / gauge / orientation.
2. A structural alignment step: CKA/SVCCA/GW/stitching.
3. A residual or shortcut path: steering / delta / skip connection.

If LatentWire only learns a dense map, it risks mixing these three together and hiding which one is actually carrying the signal.

## Top LatentWire ablations to run next

1. **Permutation vs rotation vs both**
   - Compare `identity`, `permutation only`, `orthogonal only`, and `permutation + orthogonal` before the bridge.
   - This isolates whether the transport failure is mostly ordering, orientation, or both.

2. **Gauge-fixed vs ungauged head bases**
   - Apply a deterministic head/channel gauge fix, then fit the bridge.
   - If gauge fixing helps more than extra bridge capacity, the bottleneck is symmetry, not expressivity.

3. **Stitching loss vs reconstruction loss**
   - Train one bridge to reconstruct activations and another to preserve downstream task behavior through a stitch layer.
   - If the stitch layer wins, the bridge is preserving function better than raw feature geometry.

4. **SAE basis vs dense residual basis**
   - Transport in raw residual space, SAE latent space, and SAE steering-direction space.
   - If SAE space wins, the issue is feature entanglement, not just a bad linear map.

5. **Shortcut-only vs bridge-only vs bridge-plus-residual**
   - Test `residual copy`, `transported delta`, and `delta + residual skip`.
   - This separates “the bridge is learning the answer” from “the bridge is only correcting a shortcut.”
