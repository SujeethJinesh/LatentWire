# Scoop Check Addendum: ResQ, Salfati, CMPQ

Date: 2026-05-27

## Summary

This addendum records the post-Stage-1 scoop-check citations added to the
OutlierMigrate paper. Each citation was verified against its arXiv abstract
page before integration.

| Citation | Verification Status | Source | Integration |
|---|---|---|---|
| ResQ | PASS with attribution correction | https://arxiv.org/abs/2412.14363 | Related Work, quantization/rotation paragraph |
| Quantization Dominates Rank Reduction for KV-Cache Compression | PASS | https://arxiv.org/abs/2604.11501 | Discussion, boundary-discontinuity mechanism |
| CMPQ | PASS | https://arxiv.org/abs/2410.13056 | Related Work, quantization/rotation paragraph |

## Verification Notes

- ResQ resolves to arXiv:2412.14363, title "ResQ: Mixed-Precision
  Quantization of Large Language Models with Low-Rank Residuals." The verified
  author list is Utkarsh Saxena, Sayeh Sharify, Kaushik Roy, and Xin Wang. The
  queue note attributed this to "Saha et al."; the paper and bibliography use
  the verified Saxena et al. metadata.
- Salfati resolves to arXiv:2604.11501, title "Quantization Dominates Rank
  Reduction for KV-Cache Compression," author Samuel Salfati. The abstract
  states the softmax-Fisher perturbation comparison used in the discussion.
- CMPQ resolves to arXiv:2410.13056, title "Channel-Wise Mixed-Precision
  Quantization for Large Language Models," authors Zihan Chen, Bike Xie,
  Jundong Li, and Cong Shen. The abstract states channel-wise mixed precision
  based on activation distributions, matching the related-work differentiator.

## Paper Changes

- Added `saxena2025resq`, `salfati2026kvquantization`, and `chen2025cmpq` to
  the inline bibliography and `bibliography.bib`.
- Added ResQ and CMPQ differentiators to Section 2.
- Added Salfati's softmax-Fisher result as mechanism-level support for why
  hard channel dropping/switching can be worse than bounded quantization noise.
- Updated the LLM Use Disclosure and hallucination-audit appendix to record
  that these citations were added after a user-flagged scoop check and arXiv
  verification.
