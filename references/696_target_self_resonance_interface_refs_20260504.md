# References: Target Self-Resonance Interface Pivot

Date: 2026-05-04

## Why These References Matter

The latest local ledger rules out shallow receiver switching as the next
highest-value path. The live branch is now a compact learned target-side
interface: first prove that frozen target models can use a small learned slot or
soft-prefix interface to improve their own decision surface under zero-source
controls, then ask whether source latents can populate that same interface
without leaking selected labels, candidate IDs, source logits, or target-cache
lookups.

## Compact Target-Side Interfaces

- Li and Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation,"
  ACL-IJCNLP 2021.
  https://aclanthology.org/2021.acl-long.353/

- Lester, Al-Rfou, and Constant, "The Power of Scale for
  Parameter-Efficient Prompt Tuning," EMNLP 2021.
  https://aclanthology.org/2021.emnlp-main.243/

- Mu, Li, and Goodman, "Learning to Compress Prompts with Gist Tokens,"
  NeurIPS 2023.
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html

- Chevalier et al., "Adapting Language Models to Compress Contexts,"
  EMNLP 2023.
  https://arxiv.org/abs/2305.14788

- Ge et al., "In-context Autoencoder for Context Compression in a Large
  Language Model," arXiv 2023.
  https://arxiv.org/abs/2307.06945

Boundary: these works learn task prompts, prompt summaries, or context memory
slots for a single model. LatentWire's next gate should not claim novelty for
soft tokens or prompt compression. The unique claim must be a source-private
communication test: target-side slots are learned and stress-tested first, then
source-derived signals must add paired gain over the target-resonant zero-source
interface at the same byte/slot budget.

## Query Bottlenecks And Frozen-Backbone Connectors

- Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs &
  Outputs," arXiv 2021.
  https://arxiv.org/abs/2107.14795

- Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning,"
  NeurIPS 2022.
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html

- Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
  Image Encoders and Large Language Models," ICML 2023.
  https://proceedings.mlr.press/v202/li23q.html

Boundary: Q-Former and Perceiver-style resamplers are the closest architectural
precedent for learned bottleneck queries between frozen systems. LatentWire is
only distinct if the interface is evaluated as a cross-model communication
channel with source-destroying, label-blind, same-family, and strict
cross-family controls, not merely as another frozen-backbone adapter.

## Direct Latent/KV Communication Comparators

- Fu et al., "Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models," ICLR 2026.
  https://openreview.net/forum?id=LeatkxrBCi
  https://arxiv.org/abs/2510.03215

- Shi et al., "KVComm: Enabling Efficient LLM Communication through Selective
  KV Sharing," ICLR 2026.
  https://openreview.net/forum?id=F7rUng23nw

- Ye et al., "KVCOMM: Online Cross-context KV-cache Communication for
  Efficient LLM-based Multi-agent Systems," arXiv 2025.
  https://arxiv.org/abs/2510.12872

- Dery et al., "Latent Space Communication via K-V Cache Alignment,"
  arXiv 2026.
  https://arxiv.org/abs/2601.06123

Boundary: these are high-dimensional cache/KV sharing or alignment methods.
They are the mandatory accuracy/latency/exposure comparators. The LatentWire
claim can only be different if it moves a much smaller bounded packet or slot
state and proves the source signal is consumed beyond the target-only slot
cache.

## Activation Steering And Representation Engineering

- Li et al., "Inference-Time Intervention: Eliciting Truthful Answers from a
  Language Model," NeurIPS 2023.
  https://arxiv.org/abs/2306.03341

- Turner et al., "Steering Language Models With Activation Engineering,"
  arXiv 2023/2024.
  https://arxiv.org/abs/2308.10248

- Zou et al., "Representation Engineering: A Top-Down Approach to AI
  Transparency," arXiv 2023/2025.
  https://arxiv.org/abs/2310.01405

- Rimsky et al., "Steering Llama 2 via Contrastive Activation Addition,"
  ACL 2024.
  https://aclanthology.org/2024.acl-long.828/

Boundary: activation steering changes a model's behavior with internal
directions, usually without a source model. LatentWire should use these as
controls and telemetry inspiration: steering-vector norm, layer localization,
and off-target degradation. The positive claim must require source-conditioned
incremental information, not a target-only behavior vector.

## Sparse Autoencoders And Crosscoders

- Huben et al., "Sparse Autoencoders Find Highly Interpretable Features in
  Language Models," ICLR 2024.
  https://openreview.net/forum?id=F76bwRSLeK

- Lindsey et al., "Sparse Crosscoders for Cross-Layer Features and Model
  Diffing," Transformer Circuits, 2024.
  https://transformer-circuits.pub/2024/crosscoders/index.html

- Anthropic, "Scaling Monosemanticity: Extracting Interpretable Features from
  Claude 3 Sonnet," Transformer Circuits, 2024.
  https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html

- Thasarathan et al., "Universal Sparse Autoencoders: Interpretable Cross-Model
  Concept Alignment," arXiv 2025/2026.
  https://arxiv.org/abs/2502.03714

Boundary: SAEs/crosscoders are interpretability and shared-concept tools, not
by themselves communication evidence. They become useful if slot activations are
interpretable, sparse, and causally tied to answer changes under atom-shuffle,
top-atom knockout, and wrong-source controls.

## Diffusion And Iterative Refinement Inspiration

- Ho, Jain, and Abbeel, "Denoising Diffusion Probabilistic Models,"
  NeurIPS 2020.
  https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html

- Austin et al., "Structured Denoising Diffusion Models in Discrete
  State-Spaces," NeurIPS 2021.
  https://arxiv.org/abs/2107.03006

- Li et al., "Diffusion-LM Improves Controllable Text Generation,"
  arXiv 2022.
  https://arxiv.org/abs/2205.14217

- Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback,"
  arXiv 2023.
  https://arxiv.org/abs/2303.17651

Boundary: diffusion/refinement is an algorithmic analogy for iterative
target-state denoising, not a novelty claim. If used, the gate should log
per-step target loss, no-source refinement gain, source-conditioned refinement
gain, and whether later steps amplify or erase source information.

## Quantization And Systems Comparators

- Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization
  with Zero Overhead," arXiv 2024.
  https://arxiv.org/abs/2406.03482

- Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache,"
  ICML 2024.
  https://arxiv.org/abs/2402.02750

- Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate," ICLR 2026.
  https://openreview.net/forum?id=tO3ASKZlok
  https://arxiv.org/abs/2504.19874

Boundary: these works compress vectors/KV while preserving geometric or
attention fidelity. They do not solve source-private semantic communication.
LatentWire must compare against them only in a native systems harness or as
byte-exposure floors, and must avoid claiming throughput superiority until the
native NVIDIA/vLLM/SGLang rows are filled.

## Recommended Next Gate

Run `target_self_resonance_slot_gate` before any new cross-family widening:

1. Train a tiny target-side slot interface on official train only with the
   target frozen. Inputs are public candidate/context features plus `k` learned
   slots or queries. No source model is present.
2. Freeze the interface and measure zero-source self-resonance on a larger
   frozen slice with seed repeats, paired CIs, wrong-row controls, and
   candidate-text/option-order permutations.
3. Add a source encoder that can only populate the same `k` slots at a fixed
   byte/slot budget. Evaluate source-present minus zero-source-slot, not
   source-present minus raw target.
4. Falsify with shuffled-source slots, source-label destruction, target-cache
   lookup prevention, same-byte text, packet-only, C2C/KVComm-style cache
   baselines where feasible, and one strict non-Qwen source-target pair.

The paper framing should be:

> A bounded target-native interface for testing whether source-private signals
> can improve an already calibrated target decision surface.

Do not frame it as:

- a new soft prompt method;
- prompt/context compression;
- universal latent language;
- KV-cache sharing;
- activation steering;
- SAE/crosscoder interpretability alone;
- native systems superiority.
