# Multimodal, Diffusion, and Latent-Connector References

Primary-source memo for LatentWire ideas that transfer across multimodal connectors,
latent-token bridges, diffusion / flow-matching generators, and tokenizer or vocab
alignment. The goal is to mine mechanisms that can be turned into clean, interpretable
ablations rather than importing full architectures wholesale.

| Source | Primary links | Transferable mechanism | Concrete LatentWire ablation |
|---|---|---|---|
| **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** | [paper](https://arxiv.org/abs/2301.12597) | A lightweight bridge module can align a frozen source encoder to a frozen target model with far fewer trainable parameters than end-to-end tuning. | Compare LatentWire’s current transport to a frozen source/target setup with a small learned projector only; ablate projector depth, rank, and whether it acts on K, V, or both. |
| **Flamingo: a Visual Language Model for Few-Shot Learning** | [paper](https://arxiv.org/abs/2204.14198) | Gated cross-attention blocks are a principled way to inject external modality signals without rewriting the base model. | Replace or augment the current bridge with gated cross-attention between source cache summaries and target hidden states; compare always-on, gated, and layer-sparse variants at matched bytes. |
| **Perceiver-VL: Efficient Vision-and-Language Modeling with Iterative Latent Attention** | [paper](https://arxiv.org/abs/2211.11701), [code](https://github.com/zinengtang/Perceiver_VL) | A small latent bottleneck can absorb high-dimensional inputs while preserving useful cross-modal structure. | Add a learnable latent token bank as an intermediate cache-compression layer before transport; sweep latent count, update frequency, and whether latents are shared across layers. |
| **DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models** | [paper](https://arxiv.org/abs/2405.20985) | Compression and abstraction should be separated; compress first, let the LLM do semantics. | Test whether LatentWire should compress source caches into fewer transport tokens before any semantic routing; compare patch-level / token-level / latent-level compression with the same downstream budget. |
| **Libra: Building Decoupled Vision System on Large Language Models** | [paper](https://arxiv.org/abs/2405.10140) | Cross-modal bridges can be injected directly into attention keys / values rather than only through a final projector. | Add bridge transforms on K only, V only, and K+V separately; measure paired flips, reconstruction error, and whether K/V-specific transport helps on longer contexts or only on short prompts. |
| **Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers** | [paper](https://arxiv.org/abs/2405.05945) | Flow-matching / diffusion-style iterative refinement can operate over latent token spaces with zero-initialized attention and learnable placeholders. | Try iterative refinement of transported latent states for 1, 2, and 4 denoising-style steps before decoding; compare against one-shot transport under equal compute and report whether refinement helps only difficult or ambiguous cases. |
| **Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens** | [paper](https://arxiv.org/abs/2504.14666) | Discrete timestep tokens can make compressed latent spaces more recursive and easier for an LLM to reason over. | Add timestep / stage tokens to LatentWire’s transported latent stream and test whether recurrence markers improve cache transport, especially when source and target tokenizers differ. |
| **FOCUS: Effective Embedding Initialization for New Tokenizers** | [paper](https://arxiv.org/abs/2305.14481) | Vocabulary and embedding alignment can be improved by copying shared tokens and composing unseen tokens from similar anchors. | For source-target model pairs with tokenizer mismatch, ablate shared-token copying, embedding composition for missing tokens, and vocab remapping before any cache transport. |
| **X-InstructBLIP: A Framework for aligning X-Modal instruction-aware representations to LLMs and Emergent Cross-modal Reasoning** | [paper](https://arxiv.org/abs/2311.18799) | A small instruction-aware projection can generalize across multiple modalities, but projector choice matters under low-data or multi-modal settings. | Compare linear projector, Q-Former-like latent queries, and a small MLP bridge on the same LatentWire training slice; log transfer quality separately for in-domain and OOD prompts. |
| **MDPO: Conditional Preference Optimization for Multimodal** | [paper](https://arxiv.org/abs/2406.11839) | Preference optimization can fail when the conditioning signal is not enforced strongly enough. | If LatentWire adds preference-style fine-tuning, include a conditioning-strength ablation that checks whether the source cache actually changes outcomes versus a language-only shortcut. |

## LatentWire-Ready Ablations

- Latent-token bridge: replace direct cache transport with a small learned latent bank, then decode back into K/V.
- Projector family sweep: linear, 2-layer MLP, low-rank adapter, Q-Former-like latent queries, and gated cross-attention bridge.
- K-only vs V-only vs K+V transport: isolate whether the benefit comes from retrieval structure, content, or both.
- Iterative refinement: one-shot transport versus 2- and 4-step flow-matching / denoising over the latent transport state.
- Tokenizer and vocab alignment: shared-token copying, embedding composition for OOV tokens, and remapped vocab transport before any cache bridge.

## Guardrails

- Always keep compute matched when comparing latent bridges to direct transport; extra refinement steps must be budgeted explicitly.
- Report paired flips, reconstruction error, and route entropy alongside raw accuracy so a gain can be interpreted mechanistically.
- Any claim that “better alignment” helps must be checked against a no-alignment control, a random-latent control, and a slot-permuted control.
- If a connector win only appears on one source-target pair, treat it as a hypothesis generator, not a paper claim.

## Practical Reading Order

1. BLIP-2 and Flamingo for the basic bridge / gated-cross-attention pattern.
2. Perceiver-VL and DeCo for latent compression versus semantic abstraction.
3. Libra for K/V-specific bridge injection.
4. Lumina-T2X and discrete diffusion timestep tokens for iterative latent refinement.
5. FOCUS for tokenizer and vocab alignment before any transport experiments.
