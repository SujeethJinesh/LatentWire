# COLM Final Bundle

Date: 2026-05-02

This folder is the self-contained COLM submission bundle for the current
LatentWire draft.

## Current Readiness

- COLM workshop: plausible but borderline. The package is buildable,
  citation-audited, and evidence-audited, but the story must stay narrow.
- ICLR full paper: not ready. The positive method is still same-family and the
  systems result is byte/exposure accounting, not native serving throughput.
- Current paper story: fixed-byte, source-private packets transfer useful
  same-family evidence on ARC-Challenge and OpenBookQA; strict cross-family
  Phi-3 replacement and Mac-local learned connector repairs fail.
- Exact blocker: reviewers may interpret the packet as compressed source-choice
  routing rather than broad latent communication. A stronger receiver-family or
  cross-family positive is still needed for ICLR.

## Folder Layout

- `paper/`: submission TeX, BibTeX, compiled PDF, COLM style files, and figures.
- `paper/template/`: the unzipped COLM 2026 template files kept for style audit.
- `evidence/results/`: frozen result directories used by the paper.
- `evidence/memos/`: readiness, gate-tree, and connector memos.
- `evidence/excluded_diagnostics/`: relevant late diagnostics that inform
  reviewer risk but are not claimed in the current PDF.
- `code/scripts/`: scripts needed to reproduce the reported rows.
- `code/tests/`: tests covering the scripts and reported artifacts.
- `references/`: citation audit source memo.
- `audits/`: build/test logs, claim audit, citation audit, and reviewer panel.

## Verification Commands

From the repository root:

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_build_source_private_arc_challenge_fourier_anchor_syndrome_gate.py \
  tests/test_build_source_private_arc_challenge_seed_stability.py \
  tests/test_build_source_private_arc_challenge_source_family_cache_falsification.py \
  tests/test_analyze_source_private_arc_cross_family_failure_decomposition.py \
  tests/test_build_source_private_systems_boundary_figure_table.py \
  tests/test_build_source_private_arc_challenge_candidate_syndrome_connector_gate.py \
  tests/test_build_source_private_arc_challenge_hidden_query_mlp_cache_connector_gate.py
```

From `colm_final/paper`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error latentwire_colm2026.tex
bibtex latentwire_colm2026
pdflatex -interaction=nonstopmode -halt-on-error latentwire_colm2026.tex
pdflatex -interaction=nonstopmode -halt-on-error latentwire_colm2026.tex
```

Expected status is recorded in `audits/test_report.txt` and
`audits/build_report.txt`.

## Submission Caveats

- Do not claim universal latent language, solved cross-family transfer, or GPU
  throughput gains.
- Treat random-anchor success as evidence for shared public coordinates, not
  semantic anchor names.
- Treat the systems result as object-size and exposure accounting only.
- The next experiment to de-risk reviews is a direct source-choice/index
  baseline plus a receiver-family gate that must beat packet-only and
  target-only with paired confidence intervals.
- HellaSwag diagnostics are packaged as excluded diagnostics. They should not be
  cited as current PDF evidence unless the paper is revised to include and audit
  those rows.
