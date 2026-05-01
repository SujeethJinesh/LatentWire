# HellaSwag Validation[0:8192] References

This memo supports `paper/source_private_hellaswag_validation8192_20260501.md`
and the refreshed ICLR evidence bundle. The local result is a positive dense
packet validation extension, not a new common-basis, prompt-tuning,
quantization, or latent-reasoning method.

## Local Claim

The `2B` raw / `5B` framed dense hidden-innovation packet passes eight
contiguous HellaSwag validation slices, rows `0:8192`, while preserving the
source-private boundary: no source text, KV cache, raw hidden vector, or raw
score vector is transmitted.

## Primary Related Work Boundaries

- Prefix tuning and P-Tuning learn soft prompt/prefix parameters; LatentWire
  does not insert prompt tokens or transmit soft prompts. Sources:
  https://arxiv.org/abs/2101.00190 and https://arxiv.org/abs/2110.07602
- Adapter modules add persistent task parameters; LatentWire does not install
  adapters or transmit adapter weights. Source: https://arxiv.org/abs/1902.00751
- Sparse autoencoders, universal sparse autoencoders, sparse crosscoders, and
  relative representations cover shared sparse/common-coordinate representation
  learning. LatentWire should not claim common-basis novelty. Sources:
  https://arxiv.org/abs/2410.06981, https://arxiv.org/abs/2502.03714,
  https://transformer-circuits.pub/2024/crosscoders/index.html, and
  https://arxiv.org/abs/2209.15430
- C2C, KVComm, KVCOMM, and Q-KVComm are close non-text model communication
  baselines because they communicate or reuse source KV/cache state. LatentWire
  differs by not transmitting source KV or raw state, but native comparisons
  remain pending. Sources: https://arxiv.org/abs/2510.03215,
  https://arxiv.org/abs/2510.03346, https://arxiv.org/abs/2510.12872, and
  https://arxiv.org/abs/2512.17914
- QJL, TurboQuant, KIVI, and KVQuant are low-bit source-state or KV-cache
  compression baselines, not the current LatentWire contribution. Sources:
  https://arxiv.org/abs/2406.03482, https://arxiv.org/abs/2504.19874,
  https://arxiv.org/abs/2402.02750, and https://arxiv.org/abs/2401.18079
- Diffusion transformers and latent-reasoning methods motivate denoising or
  hidden-state repair, but LatentWire's current evidence is cross-model
  fixed-byte communication. Sources: https://arxiv.org/abs/2212.09748 and
  https://arxiv.org/abs/2502.12134

## Citation Use

Use these references to draw the boundary precisely:

> LatentWire is a fixed-byte source-private packet protocol with destructive
> controls showing useful per-example hidden innovation beyond label-copy and
> score-only shortcuts. It is not a learned prompt, adapter, SAE/common-basis,
> KV-cache communication, KV quantization, diffusion-transformer, or latent-CoT
> method.
