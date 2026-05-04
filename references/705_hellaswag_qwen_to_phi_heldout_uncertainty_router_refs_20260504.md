# HellaSwag Qwen-To-Phi Held-Out Uncertainty Router References

Date: 2026-05-04

This memo records the literature boundary after the failed held-out
uncertainty-router gate.

## Closest Communication Baselines

- C2C / Cache-to-Cache is the closest cross-model communication threat: it
  learns source-to-target KV-cache projection and fusion. LatentWire differs
  only if we preserve a source-private, byte-scale packet boundary rather than
  transmitting KV state:
  https://openreview.net/forum?id=LeatkxrBCi
  https://arxiv.org/abs/2510.03215
- KVCOMM / KVComm / Q-KVComm are mandatory systems comparators because they
  communicate by sharing or compressing KV-cache state:
  https://arxiv.org/abs/2510.12872
  https://arxiv.org/abs/2510.03346
  https://arxiv.org/abs/2512.17914

## Router And Fusion Non-Novelty Risks

- DExperts, GeDi, FUDGE, contrastive decoding, and Proxy-Tuning all show that
  output distributions can be steered by external experts, discriminators, or
  smaller-model logits. A LatentWire receiver that is just source-logit fusion
  is not novel:
  https://arxiv.org/abs/2105.03023
  https://arxiv.org/abs/2009.06367
  https://arxiv.org/abs/2104.05218
  https://arxiv.org/abs/2210.15097
  https://arxiv.org/abs/2401.08565
- Selective prediction and learning-to-defer motivate the uncertainty-router
  selection criterion, but they also set the reviewer bar: a router must beat
  always-accept, always-defer, and shortcut baselines with paired uncertainty:
  https://arxiv.org/abs/1705.08500
  https://arxiv.org/abs/1807.07215

## Prefix, Soft-Token, And Target-Interface Boundary

- Prefix-Tuning and Prompt Tuning optimize continuous vectors for frozen
  language models:
  https://arxiv.org/abs/2101.00190
  https://arxiv.org/abs/2104.08691
- Gist tokens, AutoCompressors, and ICAE compress context into learned memory
  or soft-token interfaces:
  https://arxiv.org/abs/2304.08467
  https://arxiv.org/abs/2305.14788
  https://arxiv.org/abs/2307.06945

The next target self-resonance branch must therefore be framed as
per-example communication from source evidence into a frozen target-native
interface, not as generic prompt tuning.

## Systems And Quantization Boundary

- vLLM/PagedAttention and SGLang/RadixAttention define the serving baseline
  surface for native TTFT/TPOT/goodput/memory claims:
  https://arxiv.org/abs/2309.06180
  https://arxiv.org/abs/2312.07104
- KIVI, KVQuant, QJL, and TurboQuant are the relevant low-bit KV/vector
  compression pressure tests. They make "quantized state transport" non-novel
  by itself:
  https://arxiv.org/abs/2402.02750
  https://arxiv.org/abs/2401.18079
  https://arxiv.org/abs/2406.03482
  https://openreview.net/forum?id=tO3ASKZlok
  https://arxiv.org/abs/2504.19874

## Decision Boundary

The failed router shows that coarse score-level candidate packets are
saturated on the current Qwen-to-Phi gate. The still-viable novelty claim must
move to a richer target-native latent interface, with strict controls against:

```text
source-index copying
source-logit fusion
target-only soft-prefix slots
wrong-source / shuffled-source packets
KV-cache sharing baselines
matched-byte text relay
```

The systems claim remains byte/exposure accounting only until native
vLLM/SGLang/C2C/KVComm/TurboQuant rows are run on NVIDIA hardware.
