# Tokenizer / Latent Interface Ablation Hypotheses for LatentWire

Primary-source memo for the next set of LatentWire ablations. The organizing question is which interface removes the most communication friction per byte: tokenizer/vocab bridges, learned adapters, multimodal latent interfaces, diffusion/refinement-style updates, or explicit representation alignment.

## 1) Tokenizer and vocabulary bridges

- `Cross-Tokenizer Distillation via Approximate Likelihood Matching` ([paper](https://arxiv.org/abs/2503.20083)) and `Cross-Chain of Thought Distillation via Optimal Transport Alignment for Language Models with Different Tokenizers` ([paper](https://arxiv.org/abs/2502.16806)) show that mismatched vocabularies are not just a nuisance; they are a transfer bottleneck that needs explicit alignment.
- `TokAlign: Efficient Vocabulary Adaptation via Token Alignment` ([paper](https://arxiv.org/abs/2506.03523)) is the clearest tokenizer-remapping baseline because it updates embeddings through token alignment rather than full retraining.
- `Cross-Tokenizer LLM Distillation through a Byte-Level Interface` ([paper](https://arxiv.org/abs/2604.07466)) is the strongest “shared substrate” idea: move to a common byte interface when token IDs disagree.
- Transferable idea: decouple semantic transfer from tokenizer compatibility by aligning or replacing the surface vocabulary, then measure whether downstream transport gets easier.
- Concrete LatentWire ablation/metric: compare `shared tokenizer`, `token-ID remap`, `byte-level bridge`, and `source-tokenizer + adapter` at matched byte budget; report task accuracy, token fragmentation rate, cross-tokenizer KL, and bytes moved.
- Risk / guardrail: tokenizer gains can be illusory if they simply reduce fragmentation while worsening reasoning; always keep a same-tokenizer control and a fixed-budget text relay baseline.

## 2) Learned adapters and zero-init interfaces

- `LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention` ([paper](https://arxiv.org/abs/2303.16199)) is still the cleanest adapter reference because it injects new information without destabilizing the frozen backbone.
- `LLaMA-Adapter V2` ([paper](https://arxiv.org/abs/2304.15010)) extends that idea to visual instruction, which is useful as an existence proof for lightweight cross-domain coupling.
- `Model-Aware Tokenizer Transfer (MATT)` ([paper](https://arxiv.org/abs/2510.21954)) is a stronger adapter hypothesis than pure embedding similarity because it distills attention influence, not just lexical overlap.
- Transferable idea: use small, zero-init or gated adapters as an interface layer that learns when to pass, suppress, or reshape source-model state before it reaches the target.
- Concrete LatentWire ablation/metric: test `no-op`, `zero-init adapter`, `gated adapter`, and `attention-influence adapter` while logging accuracy, adapter norm growth, gate entropy, and source-target hidden-state CKA / Procrustes error.
- Risk / guardrail: adapters can become hidden re-training of the backbone; cap adapter parameter count, freeze the base models, and require gains under a parameter-budget and latency budget.

## 3) Multimodal latent interfaces

- `Multimodal Latent Language Modeling with Next-Token Diffusion` ([paper](https://arxiv.org/abs/2412.08635), [GitHub](https://github.com/sanowl/Multimodal-Latent-Language-Modeling-with-Next-Token-Diffusion)) is the best primary source for a unified latent interface over heterogeneous modalities.
- `Latent Sketchpad: Sketching Visual Thoughts to Elicit Multimodal Reasoning in MLLMs` ([paper](https://arxiv.org/abs/2510.24514)) shows that a latent scratchpad can be made interpretable if the latent state can be rendered back into human-readable form.
- `LatentLM`’s `sigma-VAE` + next-token diffusion setup is especially relevant if LatentWire needs a continuous bottleneck instead of a token bridge.
- Transferable idea: represent intermediate communication as a structured latent block, not as free-form text, then decode only the final state or a small rendered summary.
- Concrete LatentWire ablation/metric: compare `text relay`, `latent relay`, and `latent relay + renderer` while holding the source and target backbones fixed; report task accuracy, latent length, reconstruction fidelity, and interpretable render agreement when available.
- Risk / guardrail: latent interfaces can improve compression while hurting interpretability; require a decode-back sanity check and an ablation where the renderer is removed at inference.

## 4) Diffusion and iterative refinement

- `LatentLM` ([paper](https://arxiv.org/abs/2412.08635)) and `Latent Sketchpad` ([paper](https://arxiv.org/abs/2510.24514)) both support the same high-level pattern: keep a compact latent state, then refine it iteratively instead of committing in one shot.
- `LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning` ([paper](https://arxiv.org/abs/2510.04573)) is the direct text-reasoning analogue: refine a block of latent thought tokens with iterative denoising.
- Transferable idea: if direct one-pass transport is unstable, use a small number of refinement steps and make the stop rule explicit.
- Concrete LatentWire ablation/metric: sweep `1 / 2 / 4 / 8` refinement steps under the same wall-clock budget; log accuracy, stepwise entropy, update norm, rollback rate, and early-stop step.
- Risk / guardrail: more steps can hide an underpowered interface; every refinement sweep needs a matched no-op control with the same compute budget.

## 5) Cross-model representation alignment and symmetry

- `Cache-to-Cache (C2C)` ([paper](https://arxiv.org/abs/2510.03215), [GitHub](https://github.com/thu-nics/C2C)) is the closest primary baseline for direct semantic transfer through KV projection and fusion.
- `KVCOMM` ([paper](https://arxiv.org/abs/2510.03346)) and `Q-KVComm` ([paper](https://arxiv.org/abs/2512.17914)) sharpen the distinction between communication quality and cache compression quality.
- `Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion` ([paper](https://arxiv.org/abs/2502.00264)) motivates rotation-aware alignment before fusion or transport.
- Transferable idea: communication should preserve the geometry of the source representation, not just its values; align, then transport, then compress.
- Concrete LatentWire ablation/metric: compare `unaligned`, `permutation-aligned`, `rotation-aligned`, and `cache-projected` transport; report accuracy, JS/KL between source and transported states, CKA / Procrustes error, and route-value overlap.
- Risk / guardrail: alignment can overfit a pair of checkpoints; require cross-seed and cross-model-family evaluation before claiming a general interface.

## Highest-priority LatentWire ablations

- `byte-level bridge` vs `token remap` vs `shared tokenizer`, because vocabulary mismatch is the cheapest failure mode to test and the easiest way to expose interface friction.
- `zero-init/gated adapter` vs `no-op`, because a small learned interface is the cleanest way to test whether transport needs learnable control.
- `latent relay` vs `text relay`, because a structured latent block is the strongest alternative to free-form communication.
- `iterative refinement` with `1/2/4/8` steps, because it separates interface quality from one-shot decoding instability.
- `rotation-aligned cache projection` vs `unaligned projection`, because symmetry-aware alignment is the most direct representation-level hypothesis for cross-model transport.

## Guardrails for the paper

- Keep one strict same-tokenizer / text-relay control in every family of runs.
- Track bytes moved, latency, and parameter count alongside accuracy so interface improvements are not confused with extra compute.
- Report at least one interpretability metric whenever a latent or adapter path is used, even if the metric is imperfect.
- Use the same evaluation split and decoding budget for every interface family.
- Treat any gain as provisional until it holds across at least two source-target pairs or seeds.
