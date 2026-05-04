# Event-Triggered Packet Gate Reference Refresh

Date: 2026-05-04

## Why This Branch Exists

The packet-only innovation diagnostic removed the zero-source intercept but
also lost held-out lift. The event-triggered branch tests whether a selective
receiver can preserve the behavior-atom signal while abstaining on risky,
corrupted, wrong-row, or target-derived packets.

The branch fails the strict ARC n16 gate. It improves harm control but still
cannot separate matched packets from same-source-choice wrong-row, candidate
roll, top-atom knockout, source-index/rank/score, and Qwen-substitution
controls. This weakens post-hoc gating and promotes a decoder/basis objective
that trains corruptions to decode to no-op.

## Selective Prediction And Deferral

- [Selective Classification for Deep Neural
  Networks](https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks.pdf)
  frames prediction with an explicit reject option and risk/coverage tradeoff.
  SRP should report accepted coverage, accepted accuracy, helps, harms, and
  full-slice accuracy whenever a gate can abstain.
- [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) is a
  direct precedent for learning selective classifiers. The SRP receiver differs
  only if abstention is tied to source-private packet causality and destructive
  packet controls.
- [Learning to Defer to an Expert](https://proceedings.mlr.press/v119/mozannar20b.html)
  and [Learning to Defer to Multiple
  Experts](https://arxiv.org/abs/2210.16955) motivate estimating when another
  information source is worth querying or trusting. SRP's source packet is the
  external expert, but the paper must show the gate is not only cherry-picking.

## Event-Triggered And Corruption Training Priors

- [A survey on recent advances in event-triggered communication and
  control](https://www.sciencedirect.com/science/article/abs/pii/S0020025518303062)
  motivates acting only when an event condition justifies communication or
  control. SRP can use this analogy precisely: packet application is an event,
  abstention is target-only control.
- [Stacked Denoising
  Autoencoders](https://jmlr.csail.mit.edu/papers/v11/vincent10a.html)
  motivate learning representations that recover useful signal under
  corruption. For SRP, the relevant adaptation is stronger: corrupted packets
  should decode to no-op rather than merely remain stable.
- [Consistency Models](https://arxiv.org/abs/2303.01469) motivate one-step
  mappings that are stable under perturbations. SRP should not claim diffusion
  novelty, but consistency-style corruptions are useful for packet robustness.

## Sparse Basis Competitors And Next Branch

- [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410),
  [Sparse Crosscoders](https://transformer-circuits.pub/2024/crosscoders/),
  [Dedicated Feature Crosscoders](https://arxiv.org/abs/2602.11729), and
  [Delta-Crosscoder](https://arxiv.org/abs/2603.04426) make it unsafe to claim
  common sparse bases as novel. The remaining SRP novelty is using a
  source-private sparse atom packet with strict causal controls.
- The next defensible gate is a behavior-loss BatchTopK/DFC packet objective:
  train shared, source-private, and target-private atom banks; transmit only
  source-private/shared innovation atoms; train wrong-row, same-source-choice,
  atom-shuffle, coefficient-shuffle, target-derived, and candidate-roll packets
  to decode to no-op.

## Dense And Quantized Systems Baselines

- [C2C](https://openreview.net/forum?id=LeatkxrBCi) and
  [KVComm](https://openreview.net/forum?id=F7rUng23nw) remain the main dense
  communication baselines. SRP should compete on source exposure and bytes
  until native serving rows exist.
- [TurboQuant](https://arxiv.org/abs/2504.19874),
  [KVQuant](https://arxiv.org/abs/2401.18079),
  [QJL](https://arxiv.org/abs/2406.03482), and
  [KIVI](https://arxiv.org/abs/2402.02750) set strong low-bit KV byte floors.
  Event-triggered SRP should not make GPU throughput or HBM claims without
  vLLM/SGLang/NVIDIA measurement.

## Benchmark Controls

The event-triggered branch strengthens the need for risk/coverage reporting:

- full-slice matched accuracy;
- accepted coverage;
- accepted helps and harms;
- no-act fallback accuracy;
- paired bootstrap against every required shortcut/destructive control;
- zero-source and same-source-choice wrong-row collapse;
- top-atom knockout degradation rather than improvement.

The current row fails because same-source-choice wrong-row, candidate roll,
top-atom knockout, and Qwen substitution remain too competitive. That is a
basis/decoder failure, not just a threshold calibration failure.
