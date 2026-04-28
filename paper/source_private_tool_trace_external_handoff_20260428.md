# Source-Private Tool-Trace External Handoff

- date: `2026-04-28`
- gate: `source_private_tool_trace_external_handoff_20260428`
- status: ready for external upload
- handoff memo created in commit: `8262ee33dc777cbe8f52238c8da848d68368a6d3`
- provenance wording fixed in commit: `b2fc75a6f1ecbe3c07076b35720bb6cb5db423f6`

## Current Readiness

The scoped source-private diagnostic-packet paper is ready for manuscript upload
and artifact handoff. No additional method evidence is required for the current
claim boundary.

The remaining work is external venue mechanics: upload the manuscript PDF/source
bundle and attach or mirror the artifact archive.

## Upload Files

Primary manuscript PDF:

- path: `paper/iclr2026/source_private_tool_trace.pdf`
- bytes: `226708`
- SHA256: `97e460ddb3919b6b3373e12a1d01a64d64912102a8e6d2efd3301eb16cd326fe`

Compile-tested source bundle:

- path: `paper/iclr2026/submission/source_private_tool_trace_iclr_source_20260428.zip`
- bytes: `360709`
- SHA256: `cb1595cfd1ebb6cc5441143dbed2a00cce3bacd6b5aa43f2cca179c82862478e`
- contents: manuscript TeX, bibliography, ICLR style/BibTeX files, style
  dependencies, local PDF figures, and compiled PDF
- excludes: ICLR demo/template files and build intermediates

External artifact archive:

- path: `paper/artifacts/source_private_tool_trace_artifacts_20260428.zip`
- bytes: `7548312`
- SHA256: `64153e44dd5b41a30e54ffa5cdb0d95ca5498c2345c13da015a9f9f076c0121f`
- contents: compiled PDF, source upload bundle, decisive raw JSON/JSONL
  results, target-decoder benchmark inputs, figure/rate data, and readout memos
- manifest: `results/source_private_tool_trace_artifact_release_20260428/manifest.md`

## Pre-Upload Checks Already Passed

Source bundle scratch compile:

```bash
cd .debug/submission_package_compile_check_20260428/source_private_tool_trace
latexmk -pdf -interaction=nonstopmode -halt-on-error source_private_tool_trace.tex
```

Warning audit against the extracted compile log found no overfull boxes,
undefined references, citation warnings, or BibTeX warnings.

Artifact archive integrity:

```bash
unzip -t paper/artifacts/source_private_tool_trace_artifacts_20260428.zip
```

Result: all files tested `OK`.

Release local-path hygiene: the release tree, artifact archive, and
baseline-pack readout were scanned for absolute user/home path markers. Result:
no matches.

## Claim Boundary To Preserve In Submission Text

The paper should be submitted as a scoped positive method:

- Claim: explicit source-private diagnostic packets can transfer hidden
  tool-trace evidence to a target-side decoder with candidate side information
  at very low rate.
- Do not claim: learned latent transfer, universal cross-model communication,
  raw-log repair, or unconstrained program repair.
- Target-decoder LLM row remains a smoke ablation, not the headline receiver.
- Systems claim is rate/transport efficiency at comparable control validity,
  not broad deployment speedup.

## Post-Upload Sanity Checks

After upload, verify:

- the PDF preview is `7` pages and uses the intended figures;
- the source bundle is accepted by the venue system;
- the artifact archive is attached or mirrored and its SHA256 matches this memo;
- any artifact link in the submission points to the same archive version;
- the abstract/metadata do not broaden the scoped claim beyond diagnostic-code
  source-private communication.

## Next Gate

`external_submission_confirmation_20260428`: record the uploaded file IDs/URLs,
artifact location, and any portal-side warnings or required format changes.
