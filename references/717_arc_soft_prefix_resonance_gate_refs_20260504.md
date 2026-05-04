# ARC Soft-Prefix Resonance Gate References

Date: 2026-05-04

Purpose: support the strict ARC source-conditioned target-native soft-prefix
resonance gate and clarify novelty risks.

## Closest Prior Work

- Li and Liang, 2021, "Prefix-Tuning: Optimizing Continuous Prompts for
  Generation." arXiv: https://arxiv.org/abs/2101.00190
  - Relevance: learned continuous prefixes / virtual tokens for frozen language
    models. Our gate must differ by being per-example and source-conditioned,
    not a static task prefix.

- Lester, Al-Rfou, and Constant, 2021, "The Power of Scale for
  Parameter-Efficient Prompt Tuning." arXiv: https://arxiv.org/abs/2104.08691
  - Relevance: soft prompts are a strong baseline and novelty risk.

- Fu et al., 2026, "Cache-to-Cache: Direct Semantic Communication Between
  Large Language Models." OpenReview ICLR 2026:
  https://openreview.net/forum?id=LeatkxrBCi
  - Relevance: strongest direct competitor for non-text LLM communication.
    Our distinction must be receiver-native soft prefixes/logit resonance rather
    than source-KV projection/fusion.

## Systems / Compression Framing

- Zandieh et al., 2025, "TurboQuant: Online Vector Quantization with
  Near-optimal Distortion Rate." arXiv: https://arxiv.org/abs/2504.19874
  - Relevance: systems-side framing for quantized vectors, inner-product
    distortion, and KV-cache compression. This motivates future compression of
    learned soft-prefix packets but is not itself the communication method.

## Consistency / Denoising Motivation

- Song et al., 2023, "Consistency Models." arXiv:
  https://arxiv.org/abs/2303.01469
  - Relevance: motivates one-step denoising/refinement, but current
    contrastive refinement failed source-specific controls on the n8 ARC gate.

## Positioning Rule For The Paper

The contribution should be described as a strict operational communication
gate: a per-example source-conditioned encoder emits receiver-native continuous
inputs and is tested against target-only, wrong-row, candidate-roll,
source-index/rank/score, same-byte text, and source-family substitution
controls. Do not claim novelty merely from "soft prefixes" or "latent
communication"; those are already covered by prompt/prefix tuning and C2C-style
KV communication.
