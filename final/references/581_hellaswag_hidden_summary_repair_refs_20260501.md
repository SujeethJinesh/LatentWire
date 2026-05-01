# HellaSwag Hidden-Summary Repair References

Web check: 2026-05-01. Scope: primary sources for the failed HellaSwag
source-hidden summary repair gate and the safe novelty boundary.

## Local Result

Artifact:
`results/source_private_hellaswag_hidden_summary_repair_probe_20260501_qwen05_train512_validation1024/`

- hidden packet: `0.413086`
- source-label copy: `0.461914`
- hidden packet minus source-label copy: `-0.048828`
- CI95 vs source-label copy: `[-0.086499, -0.012695]`
- source top-2 oracle: `0.715820`
- selected packet: `2B` raw / `5B` framed

## Benchmark

- Zellers et al., HellaSwag: Can a Machine Really Finish Your Sentence?,
  ACL 2019. HellaSwag was built with adversarial filtering; the paper reports
  high human accuracy while then-state-of-the-art models struggled.
  https://arxiv.org/abs/1905.07830

## Calibration And Label-Copy Controls

- Zhao et al., Calibrate Before Use, ICML 2021. Contextual calibration shows
  few-shot LM predictions are sensitive to answer priors and prompt artifacts.
  https://arxiv.org/abs/2102.09690
- Zheng et al., Large Language Models Are Not Robust Multiple Choice
  Selectors, ICLR 2024. MCQ option-position bias is a direct threat model for
  label and rank-copy repair packets. https://arxiv.org/abs/2309.03882
- Geifman and El-Yaniv, Selective Classification for Deep Neural Networks,
  NeurIPS 2017. Relevant framing for "trust source vs repair" gates, but the
  local hidden-summary gate did not learn a useful selector.
  https://arxiv.org/abs/1705.08500

## Latent And Hidden-State Communication Boundaries

- Fu et al., Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models, ICLR 2026. C2C projects and fuses source KV caches into a
  target model; this is a high-rate internal-state baseline, not a same-byte
  source-private packet. https://arxiv.org/abs/2510.03215
- Ye et al., KVCOMM: Online Cross-context KV-cache Communication for Efficient
  LLM-based Multi-agent Systems, 2025. KVCOMM aligns and reuses KV caches for
  multi-agent prefill; it is a systems/cache reuse baseline with a different
  threat model. https://arxiv.org/abs/2510.12872
- Sun et al., Enhancing Latent Computation in Transformers with Latent Tokens,
  2025. Latent tokens can steer autoregressive decoding with lightweight
  trained auxiliary tokens; hidden/latent steering is therefore not novel by
  itself. https://arxiv.org/abs/2505.12629
- He, Welleck, and Fried, Reasoning with Latent Tokens in Diffusion Language
  Models, 2026. Diffusion language models reveal latent-token-style joint
  reasoning and AR models can benefit from auxiliary multi-token prediction.
  This motivates denoising/refinement-style future probes, not a current claim.
  https://arxiv.org/abs/2602.03769
- Su et al., Token Assorted, 2025. Hybrid text/latent reasoning traces show
  compressed latent reasoning is active prior art; local novelty must be the
  fixed-byte source-private packet and controls, not "latent tokens."
  https://arxiv.org/abs/2502.03275

## Quantization And Systems Inspiration

- Zandieh et al., QJL: 1-Bit Quantized JL Transform for KV Cache Quantization
  with Zero Overhead, 2024. QJL uses JL projection plus sign-bit quantization
  for KV-cache compression; useful as a sign-sketch analogy, not a direct
  semantic communication baseline. https://arxiv.org/abs/2406.03482
- Zandieh et al., TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate, 2025. TurboQuant uses random rotations, scalar quantizers,
  and a QJL residual to improve vector/KV quantization; useful for future
  packet sketches, not a claim this run satisfies. https://arxiv.org/abs/2504.19874
- Kwon et al., Efficient Memory Management for Large Language Model Serving
  with PagedAttention, SOSP 2023. vLLM/PagedAttention is the serving baseline
  for native TTFT/TPOT/throughput/KV accounting once the method has positive
  evidence. https://arxiv.org/abs/2309.06180

## Safe Boundary

The hidden-summary HellaSwag gate is negative. It should be cited as a
falsification of simple train-only last-layer hidden repair, not as positive
latent communication evidence. A future revived branch needs a materially
different learner: layer sweep, CCA/Procrustes/OT common-basis alignment,
denoising residual prediction, or cross-model supervision.
