# Hidden-Atom Decoder Packet Reference Refresh

Date: 2026-05-04

This memo records the targeted related-work refresh for the
target-conditioned hidden-atom Sparse Resonance Packet gate.

## Dense KV / Activation Communication Competitors

- Fu et al. (2026), "Cache-to-Cache: Direct Semantic Communication Between
  Large Language Models." OpenReview: https://openreview.net/forum?id=LeatkxrBCi
  and arXiv: https://arxiv.org/abs/2510.03215
- Shi et al. (2026), "KVComm: Communication-Efficient Collaborative Inference
  via KV Cache Sharing." OpenReview: https://openreview.net/forum?id=F7rUng23nw
  and arXiv: https://arxiv.org/abs/2510.03346
- Ramesh and Li (2025), "Communicating Activations Between Language Model
  Agents." https://arxiv.org/abs/2501.14082

Use in paper: these own dense KV or raw-activation communication. SRP's safe
lane is source-private packetization: atom IDs and low-bit coefficients, not
raw source hidden states, raw activations, or source KV.

## Sparse Feature Bases, Crosscoders, And Transcoders

- Lindsey et al. (2024), "Crosscoders." Transformer Circuits:
  https://transformer-circuits.pub/2024/crosscoders/index.html
- Jiralerspong and Bricken (2026), "Cross-Architecture Crosscoders."
  https://openreview.net/forum?id=YXB8uigyOg
- Kassem et al. (2026), "Delta-Crosscoder."
  https://arxiv.org/abs/2603.04426
- Dumas et al. (2025), "Robust Crosscoder Chat-Tuning."
  https://arxiv.org/abs/2504.02922
- Dunefsky, Chlenski, and Nanda (2024), "Transcoders."
  https://arxiv.org/abs/2406.11944
- Ameisen et al. (2025), "Circuit Tracing: Revealing Computational Graphs in
  Language Models." https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Marks et al. (2024), "Sparse Feature Circuits."
  https://arxiv.org/abs/2403.19647
- Lan et al. (2024), "SAE Feature-Space Universality."
  https://arxiv.org/abs/2410.06981
- Karvonen et al. (2025), "SAEBench."
  https://proceedings.mlr.press/v267/karvonen25a.html

Use in paper: do not claim sparse bases, cross-model features, or transcoders
as novel. Use them as basis-learning machinery and evaluate downstream
source-private communication with causal controls: atom shuffle, coefficient
shuffle, top-atom knockout, wrong-row packets, and target-derived packets.

## Quantization And Systems Byte Floors

- KIVI, 2-bit asymmetric KV quantization:
  https://arxiv.org/abs/2402.02750
- KVQuant, sub-4-bit KV cache quantization:
  https://arxiv.org/abs/2401.18079
- TurboQuant, online vector/KV quantization:
  https://arxiv.org/abs/2504.19874
- CacheGen, compressed KV streaming:
  https://arxiv.org/abs/2310.07240
- vLLM/PagedAttention:
  https://arxiv.org/abs/2309.06180
- DistServe:
  https://arxiv.org/abs/2401.09670

Use in paper: report SRP payload bytes, framed bytes, cache-line/DMA bytes,
and decode FLOPs. Dense KV comparisons must include fp16 and quantized
byte-floor rows; do not claim native GPU throughput without measurement.

## Benchmark And Statistical Controls

- ARC-Challenge: Clark et al. (2018), "Think you have Solved Question
  Answering? Try ARC, the AI2 Reasoning Challenge."
  https://arxiv.org/abs/1803.05457
- MCQ option-order sensitivity: Pezeshkpour and Hruschka (2024).
  https://arxiv.org/abs/2308.11483
- Option-ID / answer-order bias: Zheng et al. (2023).
  https://arxiv.org/abs/2309.03882
- Benchmark leakage cards: Xu et al. (2024).
  https://arxiv.org/abs/2404.18824
- Data/order contamination tests: Oren et al. (2023).
  https://arxiv.org/abs/2310.17623
- Paired significance: Dror et al. (2018).
  https://aclanthology.org/P18-1128/
- Bootstrap significance: Koehn (2004).
  https://aclanthology.org/W04-3250/
- Sanity checks for explanations: Adebayo et al. (2018).
  https://arxiv.org/abs/1810.03292
- Calibration: Guo et al. (2017), "On Calibration of Modern Neural Networks."
  https://arxiv.org/abs/1706.04599
- Distillation/soft targets: Hinton et al. (2015), "Distilling the Knowledge in
  a Neural Network." https://arxiv.org/abs/1503.02531

Use in paper: source-hidden atom packets must beat source-choice, calibrated
source-score quantization, same-byte text, target-derived packets, and
destructive atom controls with paired uncertainty. Top-atom knockout improving
over matched is a hard negative for causal atom semantics.

## Claim Boundary

Reviewer-safe wording:

> SRP does not claim to discover universal SAE/crosscoder atoms. It tests
> whether compact source-hidden atom IDs and low-bit coefficients can act as a
> source-private communication packet whose target-conditioned decoder improves
> downstream decisions beyond source-choice, source-score, same-byte text,
> target-cache, and destructive atom controls.

Current ARC hidden-atom decoder result: partial signal but negative. It beats
target-only on a tiny n16 scout, but ties source-score/text controls and loses
to top-atom knockout and Qwen substitution. The next branch should train atoms
for target behavior residuals directly, not PCA reconstruction.
