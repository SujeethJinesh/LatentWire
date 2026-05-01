# HellaSwag Hidden-Innovation Repair References

Web check: 2026-05-01. Scope: primary sources for the HellaSwag
hidden-innovation repair branch, its uniqueness boundary, and the
systems/quantization/denoising inspirations that should be cited without
overclaiming.

## Local Result

Artifact:
`results/source_private_hellaswag_hidden_innovation_repair_probe_20260501_qwen05_train512_validation1024/`

- pass gate: `True`
- frozen validation rows: `1024`
- scored/hidden train rows: `512`
- selected view: `score_hidden_residual`
- selected eval accuracy: `0.499023`
- best source/trained label-copy controls: `0.461914` / `0.458984`
- delta versus best label-copy: `+0.037109`
- paired CI95 versus best label-copy: `[+0.009766, +0.061060]`
- zero/wrong/candidate-roll hidden controls: `0.461914` / `0.395508` /
  `0.360352`
- packet boundary: `2B` raw / `5B` framed; no source text, source KV, raw
  hidden vectors, or raw score vectors transmitted.

## Benchmark And Latent-Communication Context

- Zellers et al., HellaSwag: Can a Machine Really Finish Your Sentence?,
  ACL 2019. HellaSwag is adversarially filtered commonsense continuation,
  making it the right anti-shortcut stress surface for source-label-copy
  controls. https://arxiv.org/abs/1905.07830
- Moschella et al., Relative Representations Enable Zero-Shot Latent Space
  Communication, ICLR 2023. Anchor-relative coordinates are the closest
  common-basis latent-communication precedent; our current hidden-innovation
  packet uses source hidden states internally but does not transmit a raw
  aligned latent vector. https://arxiv.org/abs/2209.15430
- Fu et al., Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models, ICLR 2026. C2C projects and fuses source KV cache into the
  target cache and reports accuracy/latency gains versus text. It is a
  required competitor, but it exposes source KV state and is not a fixed-byte
  source-private packet. https://arxiv.org/abs/2510.03215
- Shi et al., KVComm: Enabling Efficient LLM Communication through Selective
  KV Sharing, ICLR 2026. KVComm selectively shares KV pairs/layers, so it is a
  direct internal-state communication baseline, not the same threat model as
  our source-private candidate/confidence packet.
  https://openreview.net/forum?id=F7rUng23nw

## Quantization And Denoising Inspirations

- Zandieh et al., TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate, 2025. TurboQuant uses random rotation plus scalar
  quantization and a QJL residual correction for unbiased inner products. Cite
  it as inspiration for residual/sign-source-state systems baselines, not as a
  claim that LatentWire invents vector quantization.
  https://arxiv.org/abs/2504.19874
- Ho et al., Denoising Diffusion Probabilistic Models, 2020. DDPM motivates
  the denoising framing: infer a clean decision from corrupted/partial latent
  evidence. Our repair is a one-step discriminative denoiser, not a diffusion
  generator. https://arxiv.org/abs/2006.11239
- Peebles and Xie, Scalable Diffusion Models with Transformers, 2022. DiT is
  relevant as evidence that transformer backbones can denoise latent patches
  at scale, but it is not a direct LLM-to-LLM communication method.
  https://arxiv.org/abs/2212.09748

## Safe Novelty Boundary

The unique claim is not "new latent communication", "new quantization", or
"new diffusion". The defensible claim is narrower:

> A train-only, fixed-byte, source-private hidden-innovation repair packet
> improves a HellaSwag receiver beyond source-label-copy and trained-label-copy
> controls while transmitting only a candidate/confidence record, not source
> text, KV cache, raw hidden states, or raw score vectors.

This is now a live method branch, but it is not ICLR-headline-ready until it
survives seed/fold stability, strict same-family/cross-family separation, and
native vLLM/SGLang systems measurement.
