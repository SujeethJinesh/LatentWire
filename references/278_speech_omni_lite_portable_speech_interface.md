# Speech-Omni-Lite

- Title: `Speech-Omni-Lite: Portable Speech Interfaces for Vision-Language Models`
- Date: 2026-03-10
- Link: https://arxiv.org/abs/2603.09627
- Why it matters here:
  - strong recent example of a small plug-in projector/interface working across frozen multimodal backbones
  - useful if the next LatentWire step should look like a portable interface module rather than a bridge-specific residual tweak

Most transplantable mechanism:
- keep the backbone frozen and learn only a compact input-side interface projector that can be reused across tasks and upstream conditions

Immediate use in our setting:
- supports the case for trying a genuinely modular attention/projector replacement between transported source-side signals and the frozen target model
