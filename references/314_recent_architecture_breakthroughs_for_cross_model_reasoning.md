# Recent Architecture Breakthroughs for Cross-Model Reasoning

Primary-source memo for the newest architecture ideas that are most likely to move LatentWire: latent reasoning, iterative refinement, multimodal projector design, MoE routing, explicit memory, tokenizer/vocab adaptation, cache fusion, and geometry-preserving transport.

The goal is not to add more complexity by default. The goal is to isolate which interface actually preserves task-relevant signal per byte.

| Source | Primary links | Transferable idea | Concrete LatentWire ablation / metric | Risk / guardrail |
|---|---|---|---|---|
| **Latent Reasoning in LLMs as a Vocabulary-Space Superposition** | [paper](https://arxiv.org/abs/2510.15522) | Latent reasoning can be treated as a structured object in vocabulary space rather than as free-form hidden state. The useful signal is not the final token alone, but the latent superposition that precedes it. | Compare `text relay` vs `latent relay` vs `latent relay + vocab-space projection` under matched byte budget. Log answer accuracy, latent-step entropy, decode-back fidelity, and token-level KL to the target. | Guard against a fake win from simply using more latent capacity. Keep latent length and wall-clock fixed across controls. |
| **LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning** | [paper](https://arxiv.org/abs/2510.04573) | Iterative refinement over a latent block can recover mistakes that one-shot transport cannot. Bidirectional masks are the main mechanism, not just extra steps. | Sweep `1 / 2 / 4 / 8` refinement steps for the communication block. Log accuracy, rollback rate, update norm, stepwise entropy, and wall-clock at equal compute. | Do not claim refinement is intrinsically better unless matched against a no-op control with the same budget. |
| **DeepSeek-V3 Technical Report** | [paper](https://arxiv.org/abs/2412.19437), [code](https://github.com/deepseek-ai/DeepSeek-V3) | MoE routing and multi-token prediction show that compute should be allocated selectively, with sparse routing and load balancing as first-class design choices. | Replace the current fixed transport path with a sparse router over source channels / heads. Compare dense vs sparse vs load-balanced routing. Log routing entropy, expert utilization, bytes per answer, and accuracy. | Keep the total activated budget fixed. Otherwise a routing gain can just be a compute gain. |
| **Qwen2.5-VL Technical Report** | [paper](https://arxiv.org/abs/2502.13923) | Dynamic resolution and multimodal position encoding show that input representation and positional treatment matter as much as backbone scale. | Add a dynamic token-budget transport variant: fixed-resolution vs adaptive-resolution message packing, plus position-aware transport. Log accuracy, token fragmentation, and robustness to reordered / spatially perturbed inputs. | Separate resolution effects from total-token effects. Matched token budget is mandatory. |
| **HyperLLaVA: Dynamic Visual and Language Expert Tuning for Multimodal Large Language Models** | [paper](https://arxiv.org/abs/2403.13447) | Static projector layers are often too rigid; dynamic projector / expert tuning can adapt the interface without full backbone retraining. | Compare `static projector`, `zero-init adapter`, `gated projector`, and `hyperprojector` variants on the same transport path. Log adapter norm growth, gate entropy, CKA to target states, and task accuracy. | Freeze the base models and cap adapter parameters so the result is not just hidden fine-tuning. |
| **LM2: Large Memory Models** | [paper](https://arxiv.org/abs/2502.06049) | An explicit memory module with cross-attention and gating can store context that is hard to preserve through a plain decode path. | Add a small memory bank to the transport layer and compare `no-memory`, `memory tokens`, and `gated memory`. Log long-context accuracy, read/write entropy, retrieval recall, and reconstruction loss. | Keep the memory budget fixed. Memory tokens should not become an unbounded context extension. |
| **TokAlign: Efficient Vocabulary Adaptation via Token Alignment** | [paper](https://arxiv.org/abs/2506.03523) | Vocabulary mismatch is a transport problem, not just a preprocessing detail. Aligning token IDs and embeddings can make transfer cheaper than full remapping. | Compare `shared tokenizer`, `token-ID remap`, and `byte-level bridge` at equal bytes and identical prompts. Log token fragmentation, OOV rate, token-level KL, and end-task accuracy. | Tokenization gains can be cosmetic. Require the same byte budget and the same source-target pair in every run. |
| **Cache-to-Cache (C2C)** | [paper](https://arxiv.org/abs/2510.03215), [code](https://github.com/thu-nics/C2C) | KV cache fusion is a direct semantic communication channel: transfer source cache state instead of re-encoding everything as text. | Compare `text relay`, `selective KV sharing`, and `full KV fusion`. Log accuracy, transported layers, route-value overlap, and end-to-end latency. | Cache fusion can overfit a single model pair. Validate on at least one heterogeneous pair and one long-context pair. |
| **KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction** | [paper](https://arxiv.org/abs/2505.23416) | Cache quality depends on what the model can reconstruct from the compressed cache, not just on raw compression ratio. | Add query-agnostic KV compression to the transport path and compare it against query-aware compression under the same byte budget. Log accuracy, reconstruction loss, bytes moved, and latency. | A compression win is not a communication win unless the downstream task stays fixed. |
| **TurboQuant** | [paper](https://arxiv.org/abs/2504.19874) | Better coordinates can matter more than lower precision. Geometry-preserving transforms can reduce distortion before quantization or transport. | Compare `identity`, `random orthogonal`, `learned orthogonal`, and `TurboQuant-style` transforms before cache/message compression. Log cosine drift, Procrustes error, reconstruction loss, and accuracy per byte. | Do not conflate better geometry with more parameters. The transform must be cost-neutral or explicitly budgeted. |

## Highest-priority LatentWire ablations

- `latent relay` vs `text relay` vs `latent relay + vocab-space projection`, because latent reasoning is the cleanest way to test whether transport should happen before surface realization.
- `1 / 2 / 4 / 8` refinement steps with a matched no-op control, because iterative correction is the simplest way to separate transport quality from one-shot decoding failure.
- `static projector` vs `gated projector` vs `hyperprojector`, because multimodal systems repeatedly show that the interface itself is often the bottleneck.
- `shared tokenizer` vs `token-ID remap` vs `byte-level bridge`, because vocabulary mismatch is a cheap and interpretable failure mode.
- `text relay` vs `selective KV sharing` vs `full KV fusion`, because cache fusion is the most direct test of whether the source model’s internal state is more useful than its text output.

## What to log in every run

- Communication budget in bytes, tokens, active layers, active heads, and end-to-end latency.
- Reconstruction loss or decode-back fidelity for any latent or cache-based interface.
- Routing entropy, expert utilization, protected-channel fraction, and route-value overlap.
- Token fragmentation, OOV rate, and token-level KL for any tokenizer or vocab experiment.
- Alignment quality: CKA, Procrustes error, cosine drift, and per-step update norm.
- A same-budget no-op control for every family of runs.

## Guardrails for interpretation

- Match byte budget before comparing methods.
- Match wall-clock budget before comparing iterative methods.
- Keep the backbone frozen unless the experiment explicitly studies adaptation.
- Report at least one interpretability metric whenever the interface is latent, routed, or compressed.
- Treat any win as provisional until it holds across at least two source-target pairs or seeds.
