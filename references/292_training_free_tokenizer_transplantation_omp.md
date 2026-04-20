# Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit

- Title: `Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit`
- Date: `2025-06-07`
- Link: `https://arxiv.org/abs/2506.06607`

## Why it matters

- It treats tokenizer adaptation as a sparse reconstruction problem rather than
  a full retraining problem.
- That is relevant to LatentWire because the current live blocker looks
  increasingly upstream of the local bridge and sensitive to mismatched token
  boundaries.

## Transplantable idea

- Build a tokenizer-side compatibility layer before the bridge:
  - reconstruct target-side token behavior from sparse source-side anchors,
  - then fit the current bridge on top of that remapped interface.

## Use in our stack

- Best fit as a cheap next-step control beside `dynalign`:
  - compare the current token-mixture teacher against a sparse
    tokenizer-transplant style remapping,
  - keep the downstream module fixed,
  - test whether a more explicit tokenizer-side interface is enough to improve
    the controlled slice.
