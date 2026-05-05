# COLM Final Bundle

Date: 2026-05-02

This folder is the repo-root reproducibility bundle for the current LatentWire
COLM draft. It includes the paper, copied evidence, copied scripts/tests, and
audit reports; full reruns still assume the repository root and the local
`./venv_arm64` environment.

## Current Readiness

- COLM workshop: integrated COLM_v3 draft ready for internal/reviewer
  circulation after a human copyedit and page-budget pass. The package is
  buildable, citation-audited, test-audited, source-index-audited,
  evidence-audited, and now aligned with the generated COLM_v3 review packet.
- ICLR full paper: not ready. The positive method is still same-family and the
  systems result is byte/exposure accounting, not native serving throughput.
- Current paper story: source-private byte-scale packets can transfer narrow
  same-family candidate evidence under strict destructive controls; the paper
  also shows where apparent wins collapse into source-choice, cross-family, or
  cached-connector failures.
- Exact COLM_v3 blocker: human copyedit, page-budget review, final PDF/table
  placement, and final consistency check against
  `results/latentwire_colm_v3_review_packet_20260505/`.
- Exact ICLR blocker: the direct source-index/source-rank audit shows the
  current packet does not beat explicit selected-candidate communication. A
  stronger receiver-family, calibrated source-score, or cross-family positive is
  still needed for ICLR.

## Folder Layout

- `paper/`: submission TeX, BibTeX, compiled PDF, COLM style files, and figures.
- `paper/template/`: the unzipped COLM 2026 template files kept for style audit.
- `evidence/results/`: frozen result directories used by the paper.
- `evidence/inputs/`: small input splits/caches needed for the acceptance audit.
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
  tests/test_build_source_private_arc_challenge_hidden_query_mlp_cache_connector_gate.py \
  tests/test_build_colm_acceptance_baseline_audit.py
```

From `colm_final/paper`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error latentwire_colm2026.tex
bibtex latentwire_colm2026
pdflatex -interaction=nonstopmode -halt-on-error latentwire_colm2026.tex
pdflatex -interaction=nonstopmode -halt-on-error latentwire_colm2026.tex
```

Expected status is recorded in `audits/test_report.txt`,
`audits/build_report.txt`, and `audits/reproducibility_report.md`.

The COLM_v3 review packet can be regenerated from the repository root with:

```bash
./venv_arm64/bin/python scripts/build_latentwire_colm_v3_review_packet.py
```

## Submission Caveats

- Do not claim universal latent language, solved cross-family transfer, or GPU
  throughput gains.
- Treat random-anchor success as evidence for shared public coordinates, not
  semantic anchor names.
- Treat the systems result as object-size and exposure accounting only.
- Treat the QJL-related systems row as an internal 1-bit-per-KV-element
  accounting floor, not a native QJL performance comparison.
- The next experiment to de-risk reviews is a receiver-family or calibrated
  source-score gate that must beat source-index/source-rank, same-budget text,
  target-only, and destructive controls with paired confidence intervals.
- HellaSwag diagnostics are packaged as excluded diagnostics. They should not be
  cited as current PDF evidence unless the paper is revised to include and audit
  those rows.
