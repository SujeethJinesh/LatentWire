# Citation Additions: ResQ, Salfati, CMPQ

Timestamp: 2026-05-27T10:30Z

## Decision

The post-Stage-1 citation additions were verified and integrated.

## Verified Sources

- ResQ: arXiv:2412.14363 resolves to "ResQ: Mixed-Precision Quantization of
  Large Language Models with Low-Rank Residuals" by Utkarsh Saxena, Sayeh
  Sharify, Kaushik Roy, and Xin Wang. The queue note's "Saha et al." attribution
  was incorrect and was not used.
- Salfati: arXiv:2604.11501 resolves to "Quantization Dominates Rank Reduction
  for KV-Cache Compression" by Samuel Salfati.
- CMPQ: arXiv:2410.13056 resolves to "Channel-Wise Mixed-Precision
  Quantization for Large Language Models" by Zihan Chen, Bike Xie, Jundong Li,
  and Cong Shen.

## Paper Changes

- Added ResQ and CMPQ to Related Work under quantization and rotations.
- Added Salfati's softmax-Fisher result to the Discussion mechanism paragraph.
- Added all three entries to the inline bibliography and `bibliography.bib`.
- Updated the LLM Use Disclosure and hallucination-audit appendix to record
  the post-audit citation additions.
- Wrote `docs/scoop_check_addendum.md`.

## Verification

- Rebuilt `experimental/outlier_migrate/paper/outlier_migrate_colm2026.pdf`
  successfully with TeX warnings only.
- Page check via `pypdf`: 13 total pages, with references starting on page 8.
- Mirrored paper source, PDF, and bibliography into `release/paper/`.
