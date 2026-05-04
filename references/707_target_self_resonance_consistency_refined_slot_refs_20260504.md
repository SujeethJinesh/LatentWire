# Target Self-Resonance Consistency-Refined Slot References

Date: 2026-05-04

This memo records the literature boundary after the failed TinyLlama-to-Qwen
consistency-refined slot gate.

## Refinement And Consistency Inspiration

- Consistency Models introduce direct noise-to-data mappings for one-step or
  few-step generation, motivating the idea of a learned refinement map, but
  they are image/audio/video generative models rather than cross-LLM
  communication receivers:
  https://arxiv.org/abs/2303.01469
- Latent Consistency Models apply the same family of ideas in latent diffusion
  space for fast few-step image generation:
  https://arxiv.org/abs/2310.04378
- The current LatentWire gate only borrows the refinement metaphor. It does
  not solve a probability-flow ODE and should not be framed as a diffusion
  contribution.

## Soft Prefix And Prompt Compression Boundary

- Prefix-Tuning and Prompt Tuning show that learned continuous prompts over a
  frozen language model are established tools:
  https://arxiv.org/abs/2101.00190
  https://arxiv.org/abs/2104.08691
- Gist tokens compress prompts into reusable learned tokens for context and
  compute efficiency:
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html
- Therefore the novelty cannot be "soft tokens plus refinement." It must be
  source-specific communication under wrong-source, zero-source, target-derived,
  and candidate-roll controls.

## Model-To-Model Communication Competitors

- Cache-to-Cache projects and fuses source KV cache into a target model and is
  the closest semantic communication competitor:
  https://openreview.net/forum?id=LeatkxrBCi
- KVComm selectively shares KV pairs and directly targets efficient inter-LLM
  collaboration:
  https://openreview.net/forum?id=F7rUng23nw
- KVCOMM reuses and aligns cross-context KV cache segments for multi-agent
  prefill speedups:
  https://arxiv.org/abs/2510.12872
- LatentWire must be framed as lower-byte task-level source-private packets
  and target-native receiver controls, not as raw KV-cache sharing.

## Representation And Common-Basis Boundary

- Relative representations explicitly target latent-space communication and
  model stitching through anchor-relative geometry:
  https://openreview.net/forum?id=SrC-nwieGJ
- Sparse autoencoders expose interpretable activation features and motivate a
  future sparse/common-basis source code:
  https://arxiv.org/abs/2309.08600
- A future positive method should therefore show either source-specific target
  behavior or interpretable feature transfer. Generic hidden regression is not
  enough.

## Systems And Quantization Boundary

- QJL compresses KV cache with a Johnson-Lindenstrauss transform plus sign-bit
  quantization and reports memory/runtime wins:
  https://arxiv.org/abs/2406.03482
- TurboQuant uses random rotation plus scalar quantization for near-optimal
  vector distortion, with KV-cache quantization as a key application:
  https://arxiv.org/abs/2504.19874
- These are systems baselines and inspirations for byte accounting. They do not
  make a high-dimensional source-hidden receiver a systems contribution unless
  LatentWire demonstrates task success at a smaller packet budget.

## Reviewer Boundary

The failed gate says that a one-step target-native refinement module can match
or slightly improve frozen-slot accuracy while still failing communication.
For a positive claim, the matched-source row must beat:

```text
frozen target slots
zero-source refinement
wrong-source refinement
candidate-roll refinement
target-derived refinement
refinement-step shuffle
source top1/source-rank controls
same-budget text/candidate-only packet
C2C/KVComm-style state-sharing baselines
```
