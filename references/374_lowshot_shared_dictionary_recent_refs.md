# Low-Shot Shared Dictionaries and Related Recent References

## Sources

- [Quantifying Feature Space Universality Across Large Language Models via Sparse Autoencoders (2024)](https://arxiv.org/abs/2410.06981) - Tests whether SAE feature spaces are similar across independently trained LLMs up to rotation-invariant structure.
- [Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment (2025)](https://arxiv.org/abs/2502.03714) - Learns one shared SAE that reconstructs multiple models, directly matching the shared-dictionary idea.
- [Route Sparse Autoencoder to Interpret Large Language Models (2025)](https://arxiv.org/abs/2503.08200) - Adds routing over layers to a shared SAE, a useful reference for low-shot multi-layer sharing.
- [Sparse Autoencoders, Again? (2025)](https://proceedings.mlr.press/v267/lu25w.html) - Re-examines canonical SAE assumptions and compares against newer alternatives, useful for regularizer and objective choices.
- [Sparse Crosscoders for diffing MoEs and Dense models (2026)](https://arxiv.org/abs/2603.05805) - Uses a shared crosscoder dictionary with explicit shared/private features, which is close to the decomposition LatentWire wants.
- [Group Crosscoders for Mechanistic Analysis of Symmetry (2024)](https://arxiv.org/abs/2410.24184) - Learns dictionaries over transformed inputs under a symmetry group, relevant when transfer should be invariant to family-level changes.
- [Multi-Way Representation Alignment (2026)](https://arxiv.org/abs/2602.06205) - Moves from pairwise to multi-way alignment with a global orthogonal universe, which is the right abstraction for held-out-family transfer.
- [From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel Corpora (2025)](https://aclanthology.org/2025.emnlp-main.374/) - Data-side analog showing that multi-way aligned supervision beats unaligned training for semantic transfer.
- [SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability (2025)](https://arxiv.org/abs/2507.06265) - Uses global TopK and cross-reconstruction to enforce a unified latent space across models and modalities.
- [COSPADI: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning (2025)](https://arxiv.org/abs/2509.22075) - Shows calibration-guided sparse dictionary factorization is a strong alternative to low-rank approximations and a useful regularization analogue.

## Why It Matters For LatentWire

- The recent direction is converging on shared latent bases rather than pairwise bridges. For LatentWire, that means testing whether a canonical dictionary is a better backbone than another round of bridge tuning.
- Held-out-family transfer likely depends on geometry, support reuse, and basis stability, not just lower reconstruction error. We should measure whether the learned dictionary is actually shared or just overfitting each family with a sparse wrapper.
- Multi-way alignment papers suggest using a global initialization or regularizer first, then learning sparse, family-specific corrections on top.
- Crosscoder papers imply that reporting shared-vs-private atom structure is as important as reporting aggregate accuracy or MSE.
- The strongest low-shot story will probably come from an interpretable shared basis plus a light transfer rule, not from a more complex frontier heuristic.

## Concrete Ablations / Diagnostics

- Run a shot sweep for held-out families: 1, 2, 4, 8 shots per family, and compare pairwise bridge, shared canonical hub, and multi-way GPA/GCPA initialization under the same parameter budget.
- Compare `L1`, TopK, BatchTopK, global TopK, and route-aware TopK, then log shared-atom reuse, support overlap, and held-out-family reconstruction.
- Add dictionary regularizers one at a time: orthogonality, mutual coherence, low-rank shared basis, column sparsity, and calibration-guided sparse coding.
- Separate shared and private atoms explicitly and report how much transfer comes from each side.
- Track geometry metrics: principal angles, subspace overlap, CKA/SVCCA, support Jaccard, and pre/post alignment error.
- Add a byte-level or tokenizer-free fallback ablation to separate interface simplification from actual latent alignment.

