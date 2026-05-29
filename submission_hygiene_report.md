# Submission Hygiene Report

Date: 2026-05-29

## Actions

- Genericized visible provenance paths in the paper appendix. The detailed provenance table now uses anonymized artifact identifiers instead of distinctive internal repository paths.
- Reworded the Reproducibility Statement so it describes the anonymized FAST_VERIFY bundle rather than listing internal source paths.
- Reconciled the systems-cost aggregate arithmetic in the main text and Table 9 caption:
  `8 modules x 10 active experts x 1536 output rows x 32 columns = 3.93M MACs/token`.
- Confirmed the DecDEC prior-work recall statement against the OSDI 2025 paper. The relevant passage says DecDEC reaches around 80% recall relative to Exact, while Static falls to around 30% or below.
- Checked PDF metadata with `pypdf`: metadata contains Creator, Producer, and CreationDate only; no Author field was present.
- Refreshed the final supplementary pack and reran a conservative scan. It contains no `.git` directories, no email hits, no absolute workspace-root paths, no repo/user-name hits, no experiment-tracker/token markers, and no original project-codename path names. The only GitHub URLs found are public upstream references to `z-lab/paroquant`.
- Final supplementary pack size: 44 MiB.

## Remaining Intentional Limitations

- The supplement still contains anonymized relative artifact names and model identifiers because reviewers need those to inspect provenance.
- The ParoQuant-style implementation remains a local reproduction following the reported scaled pairwise-rotation design, not official code.
- The systems-cost section remains analytical and is not a profiler/kernel claim.
