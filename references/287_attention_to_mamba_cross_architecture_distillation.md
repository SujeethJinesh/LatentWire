# Attention to Mamba: A Recipe for Cross-Architecture Distillation

- Date: 2026-04-01
- Link: https://arxiv.org/abs/2604.14191

## Why it matters here

- It is a strong recent example of **staged module replacement** rather than a
  small local adapter tweak.
- The useful transplant is the two-step view:
  first learn an intermediate interface, then swap in the final module.

## Potential use in LatentWire

- If the token-remapping lane still stalls, the next architectural pivot can
  be a **more global module replacement** after transport:
  - first align upstream tokens/spans,
  - then hand the better-aligned signal to a stronger attention/module block.

## Current read

- This is the most relevant recent support for the alternative live lane:
  **more global module replacement**, not another tiny local bridge variant.
