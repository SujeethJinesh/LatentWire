# Interface Redesign / Connector References

Date: `2026-04-22`

## Why This Memo Exists

The current real-lane evidence keeps saying the same thing:

- direct latent alignment is only weakly alive
- the best same-pair gain is narrow and saturates quickly
- text relay can actively poison the target

That pushes the next method branch toward interface redesign rather than more
direct basis surgery.

## Strongest Connector / Bottleneck References

- [Cross-Tokenizer LLM Distillation through a Byte-Level Interface](https://arxiv.org/abs/2604.07466)
  The cleanest tokenizer-agnostic bridge reference; strongest direct argument
  for byte-level transport under mismatch.
- [TokAlign: Efficient Vocabulary Adaptation via Token Alignment](https://arxiv.org/abs/2506.03523)
  Useful if the bottleneck is vocabulary adaptation rather than latent geometry.
- [Dense Connector for MLLMs](https://arxiv.org/abs/2405.13800)
  Strong recent example of a cheap learned connector using richer multi-layer
  features.
- [Ovis: Structural Embedding Alignment for Multimodal Large Language Model](https://arxiv.org/abs/2405.20797)
  Relevant because it structurally aligns continuous and discrete embeddings
  instead of only applying a projection.
- [InstructBLIP](https://arxiv.org/abs/2305.06500)
  Best instruction-aware Q-Former reference.
- [BLIP-2](https://arxiv.org/abs/2301.12597)
  Canonical frozen-backbone + learned query bottleneck recipe.
- [Flamingo](https://arxiv.org/abs/2204.14198)
  Perceiver Resampler remains the classic learned latent bottleneck for frozen
  transfer.
- [Perceiver-VL](https://arxiv.org/abs/2211.11701)
  Stronger latent-resampler reference with iterative latent attention.
- [SelfCP: Compressing Over-Limit Prompt via the Frozen Large Language Model Itself](https://arxiv.org/abs/2405.17052)
  Good analogue for a frozen-model compressor that turns long inputs into dense
  transport states.
- [Denoising Diffusion Bridge Models](https://arxiv.org/abs/2309.16948)
  Clean bridge-model reference for paired-distribution translation instead of
  direct alignment.
- [Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](https://arxiv.org/abs/2510.20819)
  Good recent latent-bridge reference if we pivot toward distribution-bridge
  training.
- [Bifrost-1: Bridging Multimodal LLMs and Diffusion Models with Patch-level CLIP Latents](https://openreview.net/forum?id=z0WhTwZscg)
  Useful example of an explicit bridge between heterogeneous model families.

## Repo-Relevant Read

1. The smallest decisive connector experiment is a frozen 16-query bottleneck
   between source and target, trained at matched parameter budget against the
   current live dynalign residual lane.
2. If a learned query bottleneck fails, the fallback should be byte-level or
   tokenizer-aligned transport rather than more direct latent geometry work.
3. Cross-family survival matters more than same-pair polish for this lane.
