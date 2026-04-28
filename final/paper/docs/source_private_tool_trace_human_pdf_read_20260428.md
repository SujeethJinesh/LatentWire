# Source-Private Tool-Trace Human PDF Read

- date: `2026-04-28`
- status: final source/package hygiene pass complete
- live branch: explicit source-private tool-trace packet handoff
- scale rung: large frozen slice plus final human-read gate

## Starting Status

Current ICLR readiness: ready for scoped manuscript submission packaging after
final source hygiene. The evidence supports the narrow protocol-method claim,
not broad latent transfer.

Current paper story: source-private tool traces are compressed into explicit
rate-capped diagnostic packets and decoded with candidate-side decoder
information.

Exact blocker entering this gate: source upload hygiene and final wording
around the target-decoder smoke ablation.

## Fixes Applied

- Localized figure paths by copying the two PDF figures into
  `paper/iclr2026/figures/` and updating the LaTeX includes.
- Removed absolute user paths from the public figure manifest.
- Refreshed figure-manifest commit provenance.
- Softened the target-decoder smoke conclusion so it is a limited sanity check,
  not evidence of a learned target bridge.
- Kept candidate-side information consistently framed as decoder side
  information.

## Audit Result

The manuscript/PDF contains no author leakage found by the final audit: the
source uses `Anonymous Authors`, no acknowledgments are present, and no user
paths appear in the manuscript body.

The ICLR template/demo files under `paper/iclr2026/` should not be uploaded as
part of the final source bundle because they contain template/demo content and
are not referenced by this paper.

## Remaining Artifact Note

No blocker for manuscript submission. For an external artifact package, archive
or force-add decisive raw JSON/JSONL inputs and predictions that are currently
summarized by tracked manifests but ignored under `results/` or `.debug/`.

## Submission Package Follow-Up

`source_private_tool_trace_submission_upload_20260428` built a clean source
bundle at:

- `paper/iclr2026/submission/source_private_tool_trace_iclr_source_20260428.zip`

The bundle includes only the manuscript TeX, bibliography, ICLR style files,
math/style dependencies, local PDF figures, and compiled PDF. It excludes the
ICLR template/demo TeX, BibTeX, and PDF files.

The bundle was extracted into `.debug/` and compile-tested with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error source_private_tool_trace.tex
```

The extracted bundle compiled successfully with no overfull boxes, undefined
references, citation warnings, or BibTeX warnings.

## Next Exact Gate

`source_private_tool_trace_artifact_release_choice_20260428`: decide whether
the external artifact release should archive ignored raw JSON/JSONL
inputs/predictions or ship only tracked paper-facing summaries/manifests.
