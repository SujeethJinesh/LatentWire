# Source-Private Tool-Trace Submission Upload Manifest

- date: `2026-04-28`
- status: source bundle built and compile-tested
- live branch: explicit source-private tool-trace packet handoff
- scale rung: final submission packaging

## Bundle

- path: `paper/iclr2026/submission/source_private_tool_trace_iclr_source_20260428.zip`
- size: `360709` bytes
- sha256: `cb1595cfd1ebb6cc5441143dbed2a00cce3bacd6b5aa43f2cca179c82862478e`

## Included Files

The zip contains:

- `source_private_tool_trace/source_private_tool_trace.tex`
- `source_private_tool_trace/source_private_tool_trace.bib`
- `source_private_tool_trace/source_private_tool_trace.pdf`
- `source_private_tool_trace/iclr2026_conference.sty`
- `source_private_tool_trace/iclr2026_conference.bst`
- `source_private_tool_trace/math_commands.tex`
- `source_private_tool_trace/natbib.sty`
- `source_private_tool_trace/fancyhdr.sty`
- `source_private_tool_trace/figures/source_private_setup.pdf`
- `source_private_tool_trace/figures/rate_curve.pdf`

The ICLR template/demo `.tex`, `.bib`, and `.pdf` files are intentionally
excluded from the submission bundle.

## Compile Test

Command:

```bash
rm -rf .debug/submission_package_compile_check_20260428
mkdir -p .debug/submission_package_compile_check_20260428
unzip -q paper/iclr2026/submission/source_private_tool_trace_iclr_source_20260428.zip \
  -d .debug/submission_package_compile_check_20260428
cd .debug/submission_package_compile_check_20260428/source_private_tool_trace
latexmk -pdf -interaction=nonstopmode -halt-on-error source_private_tool_trace.tex
```

Result: compile passed.

Log audit:

```bash
rg -n "Overfull|undefined|Citation|LaTeX Warning|Package natbib Warning|Warning--" \
  .debug/submission_package_compile_check_20260428/source_private_tool_trace/source_private_tool_trace.log
```

Result: no matches.

## Decision

The paper manuscript/source bundle is ready for upload. The remaining choice is
external artifact release scope: tracked summaries/manifests only versus
archiving ignored raw JSON/JSONL inputs and predictions.
