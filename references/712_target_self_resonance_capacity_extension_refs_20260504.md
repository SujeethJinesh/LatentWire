# Reference Memo 712: Target Self-Resonance Capacity Extension

Date: 2026-05-04

## Local Result Boundary

The target self-resonance soft-prefix oracle capacity probe now passes on
HellaSwag validation `0:64` after adding slices `32:48` and `48:64`. The two
new slices both reach `1.000000` agreement with Qwen full-prompt decisions,
with mean KL `0.000060` and `0.000181`. Across `0:64`, optimized soft prefixes
reach `0.937500` full-prompt agreement and `0.003533` mean KL, while chunk,
zero, random, shuffled, and candidate-deranged controls remain far worse.

Boundary: this is an oracle capacity result. It shows that compact
target-native continuous inputs can induce behaviorally equivalent target
states, but it does not yet show a generalizing source-private encoder or a
systems-efficient packet.

## Closest Prior Work And Boundary

- Prefix tuning and prompt tuning optimize continuous conditioning vectors for
  frozen models. Our current oracle result is close to per-example prefix
  optimization, so it must not be claimed as novel by itself. The future
  novelty requires source-derived, instance-specific packets and strict
  destructive controls.
  Sources: https://arxiv.org/abs/2101.00190,
  https://arxiv.org/abs/2104.08691

- Gist tokens, AutoCompressors, ICAE-style context compression, and LLMLingua
  compress text contexts or prompts. LatentWire's defensible branch is not
  human-readable text compression; it is target-state induction from allowed
  source evidence.
  Sources: https://arxiv.org/abs/2304.08467,
  https://arxiv.org/abs/2310.05736,
  https://arxiv.org/abs/2307.06945,
  https://arxiv.org/abs/2404.04997

- Knowledge distillation and logit composition motivate matching the target's
  full-prompt behavior via score/logit KL, but they generally use training-time
  targets or live score fusion. LatentWire needs fixed-byte inference-time
  source evidence, not raw score transport.
  Sources: https://arxiv.org/abs/1503.02531,
  https://arxiv.org/abs/2105.03023,
  https://arxiv.org/abs/2401.08565

- Relative representations, sparse autoencoders, and sparse crosscoders address
  common-basis and feature alignment. They are relevant to the next encoder,
  but a shared basis is only useful if it causally improves held-out receiver
  behavior under source-row and atom-shuffle controls.
  Sources: https://arxiv.org/abs/2209.15430,
  https://arxiv.org/abs/2309.08600,
  https://transformer-circuits.pub/2024/crosscoders/index.html

- Continuous latent reasoning methods such as Coconut and Soft Thinking support
  the broader idea that continuous tokens can carry reasoning state. They do
  not solve cross-model source-private communication, and some recent analyses
  explicitly warn that latent tokens may be hard to interpret or control.
  Sources: https://arxiv.org/abs/2412.06769,
  https://arxiv.org/abs/2505.15778,
  https://arxiv.org/abs/2509.12875,
  https://arxiv.org/abs/2509.19170

- Diffusion Transformers, consistency models, and latent consistency models
  motivate iterative denoising/refinement as a future encoder repair strategy:
  generate a noisy latent prefix, then repeatedly refine it toward a target
  behavior state. This should remain a next-branch inspiration, not a claim
  about the current oracle run.
  Sources: https://arxiv.org/abs/2212.09748,
  https://arxiv.org/abs/2303.01469,
  https://arxiv.org/abs/2310.04378,
  https://arxiv.org/abs/2510.04573

- C2C, KVComm/KVCOMM, QJL, KIVI, KVQuant, and TurboQuant define the mandatory
  systems and state-transport boundaries. C2C/KVComm move dense KV/cache state;
  QJL/KIVI/KVQuant/TurboQuant compress KV/vector state. LatentWire remains
  distinct only if it keeps source text/KV/hidden/raw-score exposure false and
  proves a small source-derived packet can recover target behavior.
  Sources: https://arxiv.org/abs/2510.03215,
  https://arxiv.org/abs/2411.02820,
  https://arxiv.org/abs/2510.03346,
  https://arxiv.org/abs/2406.03482,
  https://arxiv.org/abs/2402.02750,
  https://arxiv.org/abs/2401.18079,
  https://arxiv.org/abs/2504.19874

## Consequence For The Next Gate

The next experiment should not match all hidden activations. It should train a
selective logit-resonance encoder:

- target: full-prompt candidate score/logit distribution;
- emitted state: `8` target-native soft slots normalized to target embedding
  RMS;
- losses: logit KL plus small top1/top2 stabilizer, optional late-layer
  projected feature loss only if it improves held-out controls;
- controls: chunk, slots-only, zero, random same-norm, shuffled-row,
  candidate-deranged, wrong-source/source-row-shuffle, target-derived, and
  same-byte random;
- pass: held-out agreement/KL improves over target-only controls, answer
  accuracy beats fixed hybrid/source-index rows with paired CI, and destructive
  controls fail to match.

If this encoder cannot generalize while oracle prefixes remain strong, the
paper should frame target resonance as a capacity finding and keep ICLR work on
hold until a source-specific encoder is found.
