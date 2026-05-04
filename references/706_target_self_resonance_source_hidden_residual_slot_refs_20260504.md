# Target Self-Resonance Source-Hidden Residual Slot References

Date: 2026-05-04

This memo records the literature boundary after the failed TinyLlama-hidden to
Qwen soft-prefix residual gate.

## Soft Prefix And Prompt Compression Boundary

- Prefix-Tuning optimizes continuous task-specific prefixes while keeping the
  language model frozen:
  https://arxiv.org/abs/2101.00190
- Prompt Tuning and P-Tuning variants make the same broad reviewer risk clear:
  learned continuous prompt vectors by themselves are not novel:
  https://arxiv.org/abs/2104.08691
  https://arxiv.org/abs/2103.10385
  https://arxiv.org/abs/2110.07602
- Gist tokens, AutoCompressors, and ICAE compress context into learned tokens,
  memory vectors, or soft prompts. LatentWire must differ by proving
  source-conditioned cross-model information beyond target-only compression:
  https://arxiv.org/abs/2304.08467
  https://arxiv.org/abs/2305.14788
  https://arxiv.org/abs/2307.06945

## Query Bottleneck And Bridge Architecture Boundary

- Perceiver-style latent arrays, Flamingo resamplers, and BLIP-2's Q-Former
  show that learned query bottlenecks are standard bridge modules between
  frozen components:
  https://arxiv.org/abs/2103.03206
  https://arxiv.org/abs/2204.14198
  https://arxiv.org/abs/2301.12597
- A target-native receiver therefore needs strict wrong-source and target-only
  controls. Otherwise reviewers can interpret it as ordinary query resampling
  or prompt compression.

## Representation Alignment And SAE Boundary

- Relative representations and representation similarity methods motivate a
  common-basis bridge, but they also make naive hidden regression non-novel:
  https://arxiv.org/abs/2209.15430
  https://arxiv.org/abs/1706.05806
  https://arxiv.org/abs/1905.00414
- Sparse autoencoders can expose interpretable activation features, and recent
  work argues some SAE feature spaces can be compared across LLMs:
  https://arxiv.org/abs/2309.08600
  https://arxiv.org/abs/2410.06981

## Communication And Systems Competitors

- C2C is the closest semantic communication competitor because it projects and
  fuses KV-cache state across models:
  https://openreview.net/forum?id=LeatkxrBCi
- KVComm / KVCOMM / Q-KVComm are mandatory comparators for any model-to-model
  communication claim that moves KV or hidden state:
  https://arxiv.org/abs/2510.03346
  https://arxiv.org/abs/2510.12872
  https://arxiv.org/abs/2512.17914
- vLLM/PagedAttention and SGLang/RadixAttention define native serving baselines
  for TTFT, TPOT, goodput, and KV-cache memory behavior:
  https://arxiv.org/abs/2309.06180
  https://arxiv.org/abs/2312.07104

## Quantization And Rate-Distortion Boundary

- KIVI, KVQuant, QJL, and TurboQuant make low-bit state transport a crowded
  systems space. LatentWire needs a task/communication win before claiming a
  compression contribution:
  https://arxiv.org/abs/2402.02750
  https://arxiv.org/abs/2401.18079
  https://arxiv.org/abs/2406.03482
  https://arxiv.org/abs/2504.19874

## Diffusion / Refinement Inspiration

- Consistency models and latent diffusion motivate iterative latent refinement,
  but they are not direct baselines for this gate:
  https://arxiv.org/abs/2303.01469
  https://arxiv.org/abs/2112.10752

## Decision Boundary

The failed residual-slot gate shows that raw source hidden summaries are not
enough. Future claims must show that source-conditioned latents beat:

```text
frozen target slots
zero-source target slots
wrong-source slots
candidate-roll source slots
target-derived slots
source top1/source-rank controls
same-budget text relay
C2C/KVComm-style state-sharing baselines
```

The strongest next framing is not "soft prompts work." It is "a held-out
source-conditioned encoder can populate target-native slots with information
that target-only soft prompts and wrong-source controls cannot reproduce."
