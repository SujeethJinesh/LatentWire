# Multimodal Bridge Inspiration for LatentWire

Goal: collect recent projector / adapter / latent-space bridge ideas that are likely to transfer to cross-model KV communication without retraining a full model.

The main pattern across these papers is consistent:

- keep the backbone frozen;
- make the bridge do the alignment work;
- supervise the bridge in the target model’s own representational space;
- if possible, use query-conditioned or refinement-based signals rather than only next-token loss.

That is exactly the design pressure we want for the next LatentWire branch if span-level approximate likelihood alone is not enough.

## Best primary sources

| Paper | Date | Link | What to steal for KV communication |
|---|---:|---|---|
| LangBridge: Interpreting Image as a Combination of Language Embeddings | 2025-03-25 | https://arxiv.org/abs/2503.19404 | The cleanest tiny-bridge template: map inputs into linear combinations of the target LLM’s vocabulary embeddings. This is the most direct analogue of projecting transported KV into a target-native basis. |
| How Visual Representations Map to Language Feature Space in Multimodal LLMs | 2025-06-13 | https://arxiv.org/abs/2506.11976 | Freeze both backbones and train only a linear adapter into the LLM’s existing representational space. The important lesson is that the bridge should land in the target space, not let the target model re-specialize around the bridge. |
| BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models | 2025-08-09 | https://arxiv.org/abs/2508.06895 | Use refined internal embeddings as direct supervision for the projector, plus angle/logit matching. This is the strongest “teach the bridge with the target’s own refined representation” recipe. |
| MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models | 2025-05-15 | https://arxiv.org/abs/2505.10526 | Add a lightweight projector, then self-distill from target-generated outputs. Good analogue for a frozen transport path with a tiny learned correction layer and teacher-generated targets. |
| UniFusion: Vision-Language Model as Unified Encoder in Image Generation | 2025-10-14 | https://arxiv.org/abs/2510.12789 | Layerwise Attention Pooling compresses multi-layer frozen features into a compact conditioning interface. This maps well to pooling transported KV into a small conditioning module before decoding. |
| Whisfusion: Parallel ASR Decoding via a Diffusion Transformer | 2025-08-09 | https://arxiv.org/abs/2508.07048 | A lightweight cross-attention adapter bridges a frozen encoder to a different decoder. This is a strong template for a small interface module that sits between two frozen systems. |
| Speech-Omni-Lite: Portable Speech Interfaces for Vision-Language Models | 2026-03-10 | https://arxiv.org/abs/2603.09627 | A small projector plus token generator can be plug-and-play across frozen backbones. This is a good portability signal for a KV-side interface module. |

## Most transferable mechanism

If we want a single design pattern to borrow next, it is:

**LangBridge-style shared embedding basis projector + BASIC-style direct target-space supervision.**

Why this is the best fit:

- it is materially different from another residual bank or a bigger adapter stack;
- it can be applied on top of frozen transport;
- it turns the bridge into an explicit target-space projector rather than a latent correction gadget;
- it gives us a clean training signal: target embedding geometry, logit distributions, and optionally teacher-generated outputs.

For LatentWire, the closest analogue would be:

1. keep the current transport path frozen;
2. attach a tiny projector after transport;
3. supervise it against the target model’s refined hidden / embedding space;
4. optionally add self-distilled teacher outputs as a weak auxiliary.

## Practical interpretation for LatentWire

These papers suggest that if transport-only branches keep plateauing, the next useful interface is probably not:

- more banked residual experts;
- more static alignment;
- or another offline-fit canonicalization tweak.

Instead, the next plausible improvement is a **small shared-interface projector** that is trained to live in the target representation space and optionally uses a refinement or self-distillation signal.

That gives us a cleaner bridge story for the paper:

- transport first;
- canonicalize if helpful;
- then use a tiny projector to land in the target’s space;
- supervise with a stronger teacher than gold next-token likelihood.

## Notes on transferability

The strongest ideas here are most likely to transfer to a KV bridge when they:

- preserve frozen backbones;
- use direct latent supervision rather than only output-level supervision;
- compress multi-layer signals into a compact conditioning interface;
- and keep the bridge small enough that it is still interpretable.

The most implementable path remains a low-rank projector on top of frozen transport, supervised by target-space losses and optionally target-generated teacher outputs.
