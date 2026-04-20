# AweDist: Attention-aware Embedding Distillation for New Input Token Embeddings

- Title: `AweDist: Attention-aware Embedding Distillation for New Input Token Embeddings`
- Date: `2025-05-26`
- Link: `https://arxiv.org/abs/2505.20133`

## Why it matters

- It treats token adaptation as an attention-aware distillation problem rather
  than only a vocabulary-matching problem.
- That is useful for LatentWire because the current live blocker is not just
  alignment solving, but how supervision is attached to mismatched token
  boundaries.

## Transplantable idea

- Distill new token representations using:
  - the original-tokenization teacher,
  - attention-aware supervision,
  - and a lightweight learned embedding/interface layer.

## Use in our stack

- Best fit as inspiration for a next teacher-side branch:
  - keep `dynalign`,
  - add token/span interaction-aware supervision,
  - and test whether attention-aware token targets can move the controlled
    slice above the current `0.1000` floor.
