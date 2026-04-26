# Source-Surface Blocker Interface References

Date: `2026-04-26`

## Why This Memo Exists

The current blocker is not only routing: the Qwen2.5-Math -> Qwen3 source
signal is sparse and unstable across SVAMP surfaces. This memo records primary
sources that motivate the next bounded branch: stronger source-derived
interfaces with strict source-destroying controls.

## Primary Sources

- [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://arxiv.org/abs/2510.03215)
  Problem: strongest direct semantic-communication baseline. Mechanism:
  learned projection/fusion of source KV cache into target KV cache. Experiment
  impact: any positive row must compare against C2C on exact frozen IDs and
  report bytes/latency. Role: baseline and headroom teacher.
- [BLIP-2 / Q-Former](https://arxiv.org/abs/2301.12597)
  Problem: bridging frozen heterogeneous systems with a compact trainable
  interface. Mechanism: learned query transformer extracts target-consumable
  information from frozen encoder states. Experiment impact: revive
  rate-capped query/resampler source sidecars before wider benchmark expansion.
  Role: method inspiration.
- [Relative Representations Enable Zero-Shot Latent Space Communication](https://arxiv.org/abs/2209.15430)
  Problem: raw coordinates can be brittle under isometries and scale changes.
  Mechanism: represent examples by similarities to anchors. Experiment impact:
  test anchor-relative source sidecars or local-neighborhood codes instead of
  global decoded-feature thresholds. Role: method and theory support.
- [Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment](https://arxiv.org/abs/2502.03714)
  Problem: dense activations make it hard to prove source-specific transfer.
  Mechanism: shared sparse concept space across models. Experiment impact:
  top-k sparse source sidecars offer an interpretable alternative to dense KV
  transport. Role: method and interpretability support.
- [DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving](https://arxiv.org/abs/2411.02820)
  Problem: same-family KV reuse can fail unless critical layers are handled
  carefully. Mechanism: identify critical KV layers and selectively recompute.
  Experiment impact: layer-localized reuse/recompute should be a systems
  ablation if C2C-like transport is revived. Role: baseline and systems
  inspiration.
- [KV Cache Transform Coding for Compact Storage in LLM Inference](https://arxiv.org/abs/2511.01815)
  Problem: low-rate transport must be compared to strong compression baselines.
  Mechanism: transform coding with decorrelation, quantization, and entropy
  coding for KV states. Experiment impact: measure whether source sidecars beat
  compressed KV transport at matched bytes. Role: systems baseline.

## Practical Read

The next experiment should not keep tuning shallow decoded-feature routers.
Either find a stronger source surface first or test a stronger source interface
on the best existing surface:

1. rate-capped learned query/resampler source sidecar
2. anchor-relative or sparse-dictionary source code
3. C2C/KV-compression baseline with matched IDs and byte accounting

Promotion still requires zero-source, shuffled-source, target-only/slots-only,
text relay, C2C, exact-ID parity, and paired uncertainty.
