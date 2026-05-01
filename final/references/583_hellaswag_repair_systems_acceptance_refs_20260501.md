# HellaSwag Repair Systems Acceptance References

Web check: 2026-05-01. Scope: primary sources for the HellaSwag repair
acceptance card, the source-label-copy threat model, and systems comparison
boundaries.

## Local Result

Artifact:
`results/source_private_hellaswag_repair_systems_acceptance_card_20260501/`

- pass gate as audit artifact: `True`
- method gate pass: `False`
- systems audit pass: `True`
- native queue allowed: `False`
- best HellaSwag repair delta versus source-label-copy: `+0.001953`
- trained label-bias control rows recorded: `2`
- best HellaSwag repair delta versus trained label-bias copy when available:
  `-0.009766`
- strict promotion threshold: `+0.02` versus source-label-copy, with paired
  CI95 low greater than `0` when available, and `+0.02` versus trained
  label-bias copy when that control is available

## Benchmark And MCQ Control Motivation

- Zellers et al., HellaSwag: Can a Machine Really Finish Your Sentence?,
  ACL 2019. HellaSwag is the adversarial continuation benchmark used for this
  frozen validation repair surface. https://arxiv.org/abs/1905.07830
- Zhao et al., Calibrate Before Use, ICML 2021. Contextual calibration
  motivates treating answer priors and label-copy shortcuts as first-class
  controls rather than incidental baselines. https://arxiv.org/abs/2102.09690
- Zheng et al., Large Language Models Are Not Robust Multiple Choice
  Selectors, ICLR 2024. MCQ option-ID and selection bias make source-label and
  rank-copy controls especially important for HellaSwag-style gates.
  https://arxiv.org/abs/2309.03882

## Systems And Communication Baselines

- Fu et al., Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models, ICLR 2026. C2C is a direct semantic communication baseline
  through source KV-cache projection/fusion; the local acceptance card does not
  claim to beat it natively. https://arxiv.org/abs/2510.03215
- Shi et al., KVComm: Enabling Efficient LLM Communication through Selective
  KV Sharing, ICLR 2026. KVComm shares selected KV pairs/layers; its access
  model exposes source KV state and must be compared on native serving metrics
  before any win is claimed. https://arxiv.org/abs/2510.03346
- Ye et al., KVCOMM: Online Cross-context KV-cache Communication for Efficient
  LLM-based Multi-agent Systems, 2025. KVCOMM is a KV-cache reuse/communication
  neighbor for multi-agent systems with different source-state exposure.
  https://arxiv.org/abs/2510.12872
- Zandieh et al., QJL: 1-Bit Quantized JL Transform for KV Cache Quantization
  with Zero Overhead, 2024. QJL motivates the one-bit sign-sketch byte-floor
  comparator, not a defeated native baseline. https://arxiv.org/abs/2406.03482
- Zandieh et al., TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate, 2025. TurboQuant motivates future low-bit vector/packet
  sketches and the 3.5-bit state floor, but the local card only reports byte
  accounting. https://arxiv.org/abs/2504.19874
- Kwon et al., Efficient Memory Management for Large Language Model Serving
  with PagedAttention, SOSP 2023. vLLM/PagedAttention defines the native
  serving substrate where TTFT, TPOT, goodput, peak memory, and HBM traffic
  remain pending. https://arxiv.org/abs/2309.06180

## Safe Boundary

The acceptance card is a reviewer-risk control and systems audit. It should be
cited to explain why HellaSwag is not a headline positive method yet, why the
paper is not hiding failed repairs, and why native systems time should wait
until a repair clears the strict label-copy gate. It is not evidence that
cross-model latent communication has been solved on HellaSwag.
