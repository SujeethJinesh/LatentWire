## X2I: Seamless Integration of Multimodal Understanding into Diffusion Transformer via Attention Distillation

- Title: `X2I: Seamless Integration of Multimodal Understanding into Diffusion Transformer via Attention Distillation`
- Link: https://arxiv.org/abs/2503.06134
- Why it matters here:
  - useful cross-architecture bridge reference for transferring behavior through attention distillation rather than direct coordinate matching
  - supports the idea that a small bridge can be supervised by target attention behavior even when the source and target internal spaces are not naturally aligned

Most transplantable mechanism:
- train a small bridge with attention-behavior supervision rather than relying only on latent-state regression

Immediate use in our setting:
- keep the transport path fixed
- use target-side attention/readout behavior as the supervision target for the bridge instead of adding more static transport penalties
