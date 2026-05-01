# HellaSwag Validation[0:7168] References

This memo supports `paper/source_private_hellaswag_validation7168_20260501.md`
and the refreshed ICLR evidence bundle. The local result is a positive dense
packet validation extension, not a new common-basis or quantization method.

## Local Claim

The `2B` raw / `5B` framed dense hidden-innovation packet passes seven
contiguous HellaSwag validation slices, rows `0:7168`, while preserving the
source-private boundary: no source text, KV cache, raw hidden vector, or raw
score vector is transmitted.

## Primary Related Work Boundaries

- Prefix tuning learns continuous task-specific prefix vectors for frozen
  language models; LatentWire does not transmit prefix or soft-prompt tokens.
  Source: https://arxiv.org/abs/2101.00190
- Prompt/P-Tuning variants optimize continuous prompts; this is parameter- or
  prompt-efficient conditioning, not per-example source-private packet transfer.
  Sources: https://arxiv.org/abs/2104.08691 and https://arxiv.org/abs/2110.07602
- Adapter modules add persistent task parameters; LatentWire does not install
  adapter modules or transmit adapter weights. Source: https://arxiv.org/abs/1902.00751
- Sparse autoencoders and universal sparse autoencoders learn interpretable or
  cross-model sparse features; our cheap SAE scout failed and the positive claim
  is the fixed-byte packet/control result. Sources: https://arxiv.org/abs/2309.08600
  and https://arxiv.org/abs/2502.03714
- Sparse Crosscoders and relative representations are prior art for shared
  feature/anchor-coordinate alignment; LatentWire should not claim common-basis
  novelty. Sources: https://transformer-circuits.pub/2024/crosscoders/index.html
  and https://arxiv.org/abs/2209.15430
- C2C and KVComm directly communicate source KV/cache state; LatentWire differs
  by sending only a byte-scale decision packet, but native comparisons remain
  pending. Sources: https://arxiv.org/abs/2510.03215 and
  https://arxiv.org/abs/2510.03346
- QJL and TurboQuant are vector/KV-cache quantization methods; they are
  comparator byte floors, not our contribution. Sources:
  https://arxiv.org/abs/2406.03482 and https://arxiv.org/abs/2504.19874
- vLLM/PagedAttention and SGLang/RadixAttention define the serving systems
  context for future native rows. Sources: https://arxiv.org/abs/2309.06180
  and https://arxiv.org/abs/2312.07104

## Citation Use

Use these references to make the novelty boundary precise:

> LatentWire is not a learned prompt, adapter, SAE/common-basis, KV-cache
> communication, or KV quantization method. It is a fixed-byte source-private
> packet protocol with destructive controls showing that the packet carries more
> useful information than label-copy or score-only shortcuts on HellaSwag.
