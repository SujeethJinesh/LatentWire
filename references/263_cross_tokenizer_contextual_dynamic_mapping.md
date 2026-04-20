# Contextual Dynamic Mapping

- Title: `Enhancing Cross-Tokenizer Knowledge Distillation with Contextual Dynamical Mapping`
- Date: 2025-02-16
- Link: https://arxiv.org/abs/2502.11104
- Why it matters here:
  - strongest recent direct reference for output-side distillation when teacher and student token spaces do not align cleanly
  - useful if the next bridge should be trained against dynamic output-side supervision instead of static latent regression

Most transplantable mechanism:
- use contextual dynamic mapping to match teacher and student targets before applying output-side distillation

Immediate use in our setting:
- use it as the main anchor if the next bridge branch becomes a prediction-level or likelihood-style teacher with dynamic token/output alignment
