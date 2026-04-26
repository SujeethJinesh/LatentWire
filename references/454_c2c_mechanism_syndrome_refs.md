# C2C Mechanism Syndrome References

Date: `2026-04-26`

## Why This Memo Exists

The SVAMP32 C2C-derived 1-byte syndrome sidecar cleared as a bound, but source
hidden and learned source-token predictors failed. This memo supports the next
diagnostic: use C2C prefill projector traces and cache residual summaries as a
deployable source/cache signal, without parsing C2C final answers.

## Primary Sources

- [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://arxiv.org/abs/2510.03215)
  Problem: direct baseline and teacher mechanism. Mechanism: project and fuse
  source KV cache into target cache, making projector residuals a natural
  distillation target/input. Role: baseline and mechanism source.
- [C2C OpenReview entry](https://openreview.net/forum?id=LeatkxrBCi)
  Problem: fair comparator status and review-facing framing. Mechanism:
  cache-fusion communication rather than text relay. Role: baseline and paper
  positioning.
- [Official C2C repository](https://github.com/thu-nics/C2C)
  Problem: implementation details for projector tracing. Mechanism: trainable
  frozen-model projector/fuser; local code exposes scalar/gate and residual
  instrumentation points. Role: implementation reference.
- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
  Problem: if C2C residual traces become useful but too large, byte/latency
  compression is the next systems gate. Mechanism: rotation plus scalar
  quantization and a 1-bit residual correction for inner-product fidelity.
  Role: compression inspiration and future systems ablation, not a direct
  baseline yet.
- [SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406)
  Problem: unstable/outlier-heavy cache features may not be linearly readable.
  Mechanism: learned rotations reduce quantization error for weights,
  activations, and KV cache. Role: inspiration for canonicalized residual
  features before sidecar coding.
- [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079)
  Problem: key/value residuals have different sensitivity and outlier behavior.
  Mechanism: KV-specific quantization choices including key-channel handling and
  outlier treatment. Role: systems and residual-feature ablation inspiration.
- [BLIP-2 / Q-Former](https://arxiv.org/abs/2301.12597)
  Problem: compactly extract useful signal from a frozen source interface.
  Mechanism: lightweight query bottleneck between frozen backbones. Role:
  connector inspiration if raw C2C residual summaries are not readable.
- [Flamingo](https://arxiv.org/abs/2204.14198)
  Problem: condition a frozen language model on an external representation with
  low interference. Mechanism: Perceiver-style resampling and gated
  cross-attention. Role: stacked connector inspiration after source-derived
  signal exists.

## Practical Read

The C2C mechanism branch should be promoted only if pre-generation projector
or residual traces recover the clean syndrome IDs while zero/shuffle/label
shuffle/target-only/slots-only controls recover none. If scalar/residual
summaries remain below target-only with `0/6` clean source-necessary IDs, do
not scale this branch; either instrument a richer layer/token-level residual
representation with a crisp hypothesis, or pivot to the next live branch:
source-control contrastive bottlenecks on a surface with measured headroom.
