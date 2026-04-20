## DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation

- Title: `DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation`
- Link: https://arxiv.org/abs/2602.21669
- Why it matters here:
  - useful next-step reference for moving the bridge teacher above plain latent regression and cheap local attention losses
  - directly relevant if the next bridge branch weights token-level supervision by teacher importance or uses sequence-alignment losses across non-identical tokenizations

Most transplantable mechanism:
- weighted prediction-level distillation with a stronger token-importance signal than uniform per-token KL

Immediate use in our setting:
- keep the current grouped transport frozen
- add a weighted teacher loss on calibration prompts so the bridge spends more capacity on the tokens where the target model's predictive mass is concentrated
