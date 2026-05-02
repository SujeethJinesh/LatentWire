# ARC Candidate-Syndrome Connector References

Date: 2026-05-02

## Local Evidence

- Gate:
  `results/source_private_arc_challenge_candidate_syndrome_connector_gate_20260502/candidate_syndrome_connector_gate.json`
- Decision: a learned candidate scorer over cached TinyLlama packet features
  and source-score shapes does not repair the strict ARC TinyLlama-vs-Qwen
  disagreement surface.
- Selected primary view: `tiny_score_shape_connector`.
- Frozen test selected-primary/Qwen/oracle: `0.288/0.317/0.586`.
- Frozen test selected-primary delta and CI95 low: `-0.029`, `-0.091`.
- Paired-family diagnostic test accuracy: `0.316`, which ties but does not
  beat Qwen-substituted packets and is not a source-private primary claim.

## Common-Basis And Connector Motivation

- Relative representations. Use to motivate anchor-defined shared coordinates
  and the boundary that our failed connector is only score-geometry based:
  `https://arxiv.org/abs/2209.15430`.
- Perceiver IO. Use as the architectural precedent for learned latent queries
  over structured inputs:
  `https://arxiv.org/abs/2107.14795`.
- BLIP-2 / Q-Former. Use as a strong precedent for a lightweight querying
  transformer that bridges frozen encoders and frozen language models:
  `https://arxiv.org/abs/2301.12597`.
- Flamingo. Use as a precedent for cross-attention style bridges between
  frozen modality-specific backbones:
  `https://arxiv.org/abs/2204.14198`.

## Feature Dictionaries And Cross-Model Alignment

- SAE universality. Use to motivate testing whether interpretable feature
  dictionaries can form a cross-model basis:
  `https://arxiv.org/abs/2410.06981`.
- Universal sparse autoencoders. Use as a direct related-work boundary for
  cross-model concept alignment:
  `https://arxiv.org/abs/2502.03714`.
- Sparse crosscoders. Use as a candidate mechanism for mapping between model
  feature spaces:
  `https://arxiv.org/abs/2603.05805`.
- Transformers as optimal transport. Use as a mathematical lens for
  representation movement and transport-style connector objectives:
  `https://openreview.net/forum?id=IzAooxm1yv`.

## Boundaries Against Prompt And Cache Transfer

- Prefix tuning. Boundary: prefix tuning learns continuous virtual tokens for
  one model; this gate uses fixed cached packet/score features and does not
  inject a prompt prefix:
  `https://arxiv.org/abs/2101.00190`.
- Prompt tuning. Boundary: soft prompts are task parameters for a frozen model,
  not source-private inter-model communication packets:
  `https://arxiv.org/abs/2104.08691`.
- C2C. Boundary: C2C projects and fuses source KV cache state, while this gate
  intentionally does not transmit source KV caches:
  `https://arxiv.org/abs/2510.03215`.
- KVComm. Boundary: selective KV sharing is a high-rate source-state baseline,
  not a fixed-byte source-private packet:
  `https://arxiv.org/abs/2510.03346`.
- KVCOMM. Boundary: offset-aligned KV-cache reuse is a systems competitor and
  source-exposure baseline, not the same protocol:
  `https://arxiv.org/abs/2510.12872`.
- TurboQuant. Boundary: TurboQuant is an online vector/KV quantization baseline
  for byte and distortion accounting; it does not create a cross-model semantic
  packet by itself:
  `https://arxiv.org/abs/2504.19874`.

## Paper Implication

This memo supports a negative ablation and a narrowing decision. Cached
candidate-level connectors are not enough; the next ICLR branch should use a
true hidden-state/query-resampler/SAE/crosscoder common-basis connector, or a
stronger alternate source-family run under the same frozen ARC disagreement
surface.
