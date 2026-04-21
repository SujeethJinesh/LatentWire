# Recent Cross-Model Interface References for LatentWire

Web check: 2026-04-21. This memo is a focused source list for 2024-2026 methods that could improve cross-model reasoning/communication without falling back to a plain adapter or an affine Procrustes map. The emphasis is on model stitching, universality, feature dictionaries, activated adapters/cache reuse, test-time refinement, tokenizer/vocabulary adaptation, and multimodal connector transfer.

## Primary sources

- **[Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models](https://arxiv.org/abs/2410.06981)**. `Core idea:` SAE dictionaries expose shared features across models and make latent similarity measurable instead of implicit. `LatentWire ablation:` train a source-model SAE on the interface layer, then use SAE feature mass, overlap, or persistence to choose which bridge channels survive. `Telemetry to log:` feature sparsity, feature overlap, reconstruction error, top-feature stability across prompts, repair help/harm, and feature-level agreement between source and target. `Failure mode to watch:` a cleaner feature basis does not guarantee the protected frontier is the causally useful one.

- **[Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment](https://arxiv.org/abs/2502.03714)**. `Core idea:` a shared sparse dictionary can reconstruct activations from multiple models and recover cross-model concepts in one space. `LatentWire ablation:` replace the current bridge selection head with a universal dictionary and ask whether a single universal basis reduces repair entropy under source/target mismatch. `Telemetry to log:` shared-dictionary utilization, concept purity, reconstruction loss by model, cross-model transfer accuracy, and dead-feature rate. `Failure mode to watch:` universal dictionaries can overfit to broad semantic overlap while missing pair-specific transport quirks.

- **[Transferring Features Across Language Models With Model Stitching](https://arxiv.org/abs/2506.06609)**. `Core idea:` affine mappings between residual streams are enough to transfer features and even transfer SAE weights across model scales. `LatentWire ablation:` use a stitched source-to-target feature map as initialization for the latent bridge, then compare trained-from-scratch vs stitch-initialized vs stitch-frozen variants. `Telemetry to log:` stitching residual, feature overlap, SAE transfer quality, downstream recovery, and calibration under source/target scale mismatch. `Failure mode to watch:` stitching may prove representational similarity without proving the bridge improves communication under a fixed byte budget.

- **[TokAlign: Efficient Vocabulary Adaptation via Token Alignment](https://arxiv.org/abs/2506.03523)**. `Core idea:` vocabulary replacement can be done by aligning token IDs and rearranging embeddings, which makes token-level knowledge transfer cheaper and more stable than naive re-tokenization. `LatentWire ablation:` perform TokAlign-style token remapping before bridge training, then test whether the latent interface still needs as many learned correction parameters. `Telemetry to log:` remap purity, token overlap, token-length reduction, perplexity recovery, bridge bytes saved, and downstream accuracy under the same decode budget. `Failure mode to watch:` token alignment can hide interface weakness if gains come entirely from making the input easier, not the bridge better.

- **[zip2zip: Inference-Time Adaptive Vocabularies for Language Models via Token Compression](https://arxiv.org/abs/2506.01084)** and **[AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation](https://arxiv.org/abs/2503.19693)**. `Core idea:` dynamically adapting or compressing the vocabulary can reduce token count and inference cost without necessarily retraining the whole model. `LatentWire ablation:` replace the current bridge tokenization with a compressed/hypertoken vocabulary and measure whether communication quality degrades less than byte count drops would suggest. `Telemetry to log:` input/output token shrinkage, hypertoken reuse rate, decode latency, exact-match accuracy, and bridge stability under long prompts. `Failure mode to watch:` better token compression does not imply better semantic transport if the latent interface cannot decode the new token regime.

- **[Activated LoRA: Fine-tuned LLMs for Intrinsics](https://research.ibm.com/publications/activated-lora-fine-tuned-llms-for-intrinsics)** and **[Cross-Modal Adapter: Parameter-Efficient Transfer Learning Approach for Vision-Language Models](https://arxiv.org/abs/2404.12588)**. `Core idea:` adapters can be activated or blended conditionally, and cross-modal transfer often benefits from modality-specific cache/adapter paths rather than one always-on branch. `LatentWire ablation:` add an activated-adapter head on top of the bridge and compare always-on, input-gated, and repair-triggered activation. `Telemetry to log:` activation rate, gate entropy, cache reuse rate, switch cost, repair gain by regime, and false activation / missed activation. `Failure mode to watch:` conditional adapters may win on efficiency while leaving the actual cross-model interface unresolved.

- **[AWT: Transferring Vision-Language Models via Augmentation, Weighting, and Transportation](https://arxiv.org/abs/2407.04603)** and **[Libra: Building Decoupled Vision System on Large Language Models](https://arxiv.org/abs/2405.10140)**. `Core idea:` multimodal transfer works better when augmentation, weighting, and transport are separated, and when the cross-modal bridge is distinct from the unimodal experts. `LatentWire ablation:` split the interface into unimodal expert paths plus a cross-model bridge, then test augmentation/weighting/transport separately instead of as one monolith. `Telemetry to log:` expert utilization, bridge-vs-expert contribution, modality-pair asymmetry, cross-modal loss, and per-regime accuracy. `Failure mode to watch:` decoupling can improve modularity without improving the bridge itself unless the bridge remains the bottleneck.

- **[DeAL: Decoding-time Alignment for Large Language Models](https://arxiv.org/abs/2402.06147)** and **[MDPO: Conditional Preference Optimization for Multimodal Large Language Models](https://arxiv.org/abs/2406.11839)**. `Core idea:` test-time alignment and preference-based refinement can rescue outputs after the forward pass already looked reasonable. `LatentWire ablation:` add a repair/refinement stage after the latent exchange and compare single-pass vs iterative decode-time alignment with matched token budgets. `Telemetry to log:` refinement iterations, score drift, recovery gain, confidence shift, and whether the repair stage helps more on hard pairs than easy pairs. `Failure mode to watch:` test-time refinement can overfit the scorer while masking a weak interface.

- **[Generalization from Starvation: Hints of Universality in LLM Knowledge Graph Learning](https://arxiv.org/abs/2410.08255)**. `Core idea:` universality can appear as stitchable representations under resource pressure, suggesting that some latent interfaces are genuinely affine or nearly affine. `LatentWire ablation:` test whether the hardest source-target pairs become more stitchable after forcing a low-resource latent bottleneck, versus after unconstrained bridge training. `Telemetry to log:` affine residual, stitch quality, path generalization, bottleneck sensitivity, and budget-vs-fidelity curves. `Failure mode to watch:` universality is a hypothesis about representation geometry, not a guarantee that a communication protocol will be learnable.

## LatentWire ablations to prioritize

1. **SAE feature selector vs raw activation selector.** Keep the bridge fixed, then choose protected frontier nodes by SAE feature mass, feature persistence, and top-k activation magnitude.
2. **Universal dictionary warm start.** Initialize the bridge from a shared dictionary or stitched SAE basis, then compare against training the bridge from scratch.
3. **Tokenizer-first interface repair.** Apply TokAlign or vocab compression before bridge training, then check whether the remaining bridge can be shallower and smaller.
4. **Activated repair path.** Gate a low-rank repair adapter only when the bridge confidence is low, and measure whether conditional activation beats always-on correction.
5. **Iterative decode-time refinement.** Add a short test-time alignment loop after the bridge, and compare one-pass, two-pass, and verifier-guided repair under the same total token budget.

## Telemetry fields to log

- `source_model`
- `target_model`
- `tokenizer_pair`
- `bridge_type`
- `bridge_init`
- `dictionary_type`
- `selection_method`
- `activation_gate_entropy`
- `shared_feature_overlap`
- `stitch_residual`
- `remap_purity`
- `token_shrink_ratio`
- `repair_iterations`
- `repair_gain`
- `repair_harm`
- `exact_match`
- `accuracy_per_byte`
- `accuracy_per_token`
- `latency_ms`
- `decode_tokens`
- `byte_budget`
- `repair_budget`

## What this memo should be used for

- If the next experiment is about **what to keep**, start with SAE / universal-dictionary selectors.
- If the next experiment is about **what to initialize**, start with model stitching or OT-style transfer.
- If the next experiment is about **what to tokenize**, start with TokAlign or adaptive vocabulary compression.
- If the next experiment is about **what to repair**, start with activated adapters or a short test-time refinement loop.
- If the next experiment is about **what to prove**, log the full telemetry contract above so the bridge can be audited later.
