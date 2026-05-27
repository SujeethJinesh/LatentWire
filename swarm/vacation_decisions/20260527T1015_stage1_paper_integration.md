# Stage 1 Paper Integration

Timestamp: 2026-05-27T10:15Z

## Decision

Stage 1 results were integrated into the OutlierMigrate workshop draft and the
release paper mirror.

## Integrated Evidence

- E2 cross-prompt replication: MATH-500 strict set-leaving is 0.561 on
  Granite-Small, 0.651 on DeepSeek-R1-Distill, and 0.675 on Falcon-H1; all are
  within 0.02 absolute of their AIME-2025 references.
- E1 narrowed KL/FFT diagnostic: DeepSeek-R1-Distill and Falcon-H1 both select
  sublinear square-root KL fits across the tested regimes, with spectral
  entropy 0.850 and 0.891 and 100-token autocorrelation.
- E3 composition: ParoQuant+M11b top-10 on Granite-Small is sub-additive,
  median recovery 0.565, trailing ParoQuant alone by 0.189 and the
  ParoQuant+random control by 0.233.
- E15 BCa/Holm correction: Nemotron M11b top-5/top-10 remain corrected-positive
  channel-set evidence; Granite M11b remains a wide-CI partial signal; ParoQuant
  remains a descriptive high-median rotation baseline; E3 composition remains
  non-positive.
- E4 and E5 were integrated only as limitations/provenance. No format-axis
  benchmark claim or Qwen3 scale-up claim is authorized.

## Verification

- Rebuilt `experimental/outlier_migrate/paper/outlier_migrate_colm2026.pdf`
  successfully with TeX warnings only.
- Mirrored `paper.tex`, `paper.pdf`, and updated figure PDFs into
  `release/paper/`.
- Page check via `pypdf`: 13 total PDF pages, with references starting on page
  8, so the main body remains at the 8-page target.

## Next Queue Item

The structural audit documents already exist. The next active queue item is the
zero-GPU citation-addition pass for ResQ, Salfati, and CMPQ, followed by the
math formalization pass.
