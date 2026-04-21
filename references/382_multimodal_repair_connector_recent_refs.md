# Multimodal Repair / Connector Latest References for LatentWire

Web check: 2026-04-21. This memo prioritizes 2025-2026 primary sources that look most relevant to LatentWire's positive-method search: connector quality, latent repair, iterative communication, token-free or latent-token reasoning, diffusion-style latent decoding, and adapter/bridge architectures.

## Sources

### Connectors, bridges, and learnable query interfaces

- [BREEN: Bridge Data-Efficient Encoder-Free Multimodal Learning with Learnable Queries](https://arxiv.org/abs/2503.12446) (2025-03-16, WACV 2026). Learnable queries sit between image and text tokens and are distilled from CLIP. Strong evidence that bridge quality is about query selection and alignment, not just connector depth.
- [RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models](https://arxiv.org/abs/2512.06811) (2025-12-07). Dual-branch adapter with a reconstruction path that preserves general knowledge while adapting. Good template for “repair + preserve” rather than plain residual tuning.
- [Bifrost-1: Bridging Multimodal LLMs and Diffusion Models with Patch-level CLIP Latents](https://arxiv.org/abs/2508.05954) (2025-08-08, NeurIPS 2025). Uses patch-level CLIP latents as the shared bridge variable between an MLLM and a diffusion model. Useful if LatentWire should expose a higher-bandwidth latent port instead of a single projector.
- [OmniBridge: Unified Multimodal Understanding, Generation, and Retrieval via Latent Space Alignment](https://arxiv.org/abs/2509.19018) (2025-09-23). A unified multimodal framework built around latent alignment, not modality-specific handoffs. Relevant as a “shared hub” baseline for cross-model communication.
- [AsymLoRA: Unlocking the Power of Multimodal LLMs via Asymmetric LoRA](https://openreview.net/forum?id=E2T8wulSb9) (2025, workshop submission). Separates shared cross-modal structure from task-specific adaptation. Useful if LatentWire needs asymmetric update rules for sender vs receiver.
- [MoLoRA: Composable Specialization via Per-Token Adapter Routing](https://arxiv.org/abs/2603.15965) (2026-03-16). Per-token routing over multiple adapters. Strong inspiration for token-/channel-level bridge routing instead of sequence-level all-or-nothing adapters.

### Iterative communication, repair, and test-time adaptation

- [Test-Time Warmup for Multimodal Large Language Models](https://arxiv.org/abs/2509.10641) (2025-09-12). Adapts an MLLM per test instance using weak auxiliary data. Good evidence that test-time warmup can unlock latent multimodal reasoning without retraining the whole stack.
- [Identity Bridge: Enabling Implicit Reasoning via Shared Latent Memory](https://arxiv.org/abs/2509.24653) (2025-09-29). Zero-hop identity supervision reshapes latent geometry and helps compositional reasoning. Relevant if LatentWire needs an explicit “identity / pass-through / self-consistency” repair target.
- [MILR: Improving Multimodal Image Generation via Test-Time Latent Reasoning](https://arxiv.org/abs/2509.22761) (2025-09-26). Runs joint reasoning over image and text in a unified latent vector space at test time. Strong evidence that latent repair can be done after training, not only during it.
- [Reasoning Within the Mind: Dynamic Multimodal Interleaving in Latent Space](https://arxiv.org/abs/2512.12623) (2025-12-14). Interleaves modalities directly in latent space during reasoning. Good reference for alternating “repair / verify / continue” loops.
- [Multimodal Latent Reasoning via Hierarchical Visual Cues Injection](https://arxiv.org/abs/2602.05359) (2026-02-05). Recursive internal loops with hierarchical visual cue injection. Useful if LatentWire should support multi-stage repair with coarse-to-fine latent reinjection.
- [LanteRn: Latent Visual Structured Reasoning](https://arxiv.org/abs/2603.25629) (2026-03-26). Generates and attends to continuous visual thought embeddings during inference. Direct inspiration for latent-only reasoning steps before final decoding.
- [System-1.5 Reasoning: Traversal in Language and Latent Spaces with Dynamic Shortcuts](https://arxiv.org/abs/2505.18962) (2025-05-25). Early-exit and step-reuse shortcuts in latent space. Good control baseline for “repair only when needed” scheduling.

### Token-free, latent-token, and diffusion-language models

- [Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning](https://arxiv.org/abs/2502.03275) (2025-02-05). Mixes latent tokens with text tokens inside the reasoning trace. Strong support for hybrid discrete/latent communication rather than forcing one representation.
- [Enhancing Latent Computation in Transformers with Latent Tokens](https://arxiv.org/abs/2505.12629) (2025-05-19). Introduces non-interpretable latent tokens that steer decoding through attention. A direct low-overhead latent-bridge baseline.
- [Reasoning with Latent Tokens in Diffusion Language Models](https://arxiv.org/abs/2602.03769) (2026-02-03). Shows latent token count can be a controllable quality/compute dial and can be ported back into autoregressive models.
- [Evo: Autoregressive-Diffusion Large Language Models with Evolving Balance](https://arxiv.org/abs/2603.06617) (2026-03-20). Bridges AR and diffusion generation in one latent trajectory. Useful if LatentWire wants a mixed refinement/generation schedule.
- [Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](https://arxiv.org/abs/2510.03206) (2025-10-03). Treats latent and token spaces jointly, with a single model denoising in the combined space.
- [DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models](https://arxiv.org/abs/2512.15713) (2025-12-17). Drop-in AR-to-diffusion translation for multimodal models; useful for a “repair pass” that does not replace the backbone.
- [STAR-LDM: Stop-Think-AutoRegress Language Diffusion Model](https://arxiv.org/abs/2602.20528) (2026-02-24). Explicit stop/think/autoregress control is very close to a bridge + verifier + release schedule.

## Why It Matters For LatentWire

- The connector literature is converging on one point: the bridge is usually a query-selection, routing, or latent-state-shaping problem, not just a projection-capacity problem.
- The iterative latent-reasoning papers suggest LatentWire should stop treating communication as a one-shot serialization step. A repair loop, warmup phase, or latent interleaving stage is now a mainstream design pattern.
- The latent-token and diffusion papers make a useful claim for our paper: the boundary between “communication” and “reasoning” can move into latent space, which gives us a legitimate positive-method lane if the bridge can preserve structure and recover from errors.
- The adapter papers argue for asymmetric roles: one branch or token path can preserve general knowledge while another branch specializes or repairs. That is a better fit than symmetric end-to-end mapping when source and target models are mismatched.
- For interpretability, the most useful papers expose an explicit latent port, learnable queries, or a controllable stop/repair policy. Those are easier to diagnose than black-box dense projectors.

## Concrete LatentWire Ablations

1. Compare `static projector` vs `learnable-query bridge` vs `reconstruction-branch adapter` at matched bytes and matched downstream compute.
2. Add a `latent-repair loop` with 0, 1, 2, and 4 correction rounds, and gate each round with a verifier or confidence estimate.
3. Test `text-only`, `latent-only`, and `hybrid latent+text` bridge payloads, with the same total bandwidth.
4. Swap the current bridge for a `per-token router` over a small adapter set, and compare sequence-level routing against token-level routing.
5. Add an `identity-bridge` auxiliary loss so the bridge must preserve zero-hop identity and not just task-specific outputs.
6. Compare `one-shot handoff` against `warmup + handoff` and `interleaved latent reasoning`, holding wall-clock budget fixed.
7. Add `diffusion-style refinement` as a bridge-side repair pass before final decode, then compare against standard AR refinement.
8. Log bridge diagnostics that reveal whether we are learning a real communication channel: query entropy, retained-latent sparsity, reconstruction error, stop rate, and whether gains survive source/target permutation stress tests.
