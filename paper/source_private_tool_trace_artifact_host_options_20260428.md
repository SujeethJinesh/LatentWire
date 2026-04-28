# Source-Private Tool-Trace Artifact Host Options

- date: `2026-04-28`
- gate: `external_submission_confirmation_20260428`
- status: host choice required before external confirmation

## Current State

The local manuscript/source/artifact package is complete and pushed. GitHub CLI
is authenticated, the remote is `SujeethJinesh/LatentWire`, and the repository is
currently public.

No GitHub release exists yet.

## Anonymity Caveat

Because the current GitHub repository URL includes the account/repo identity and
is public, publishing a public GitHub release may be inappropriate for a
double-blind ICLR submission unless the venue permits non-anonymous artifacts or
the submission is already non-anonymous.

For a double-blind submission, prefer an anonymous artifact route:

- OpenReview anonymous supplementary material, if available;
- an anonymous artifact archive/link supported by the venue;
- a separate anonymized repository or archival record that removes author,
  account, path, and commit identity.

## Public GitHub Release Path

If public GitHub hosting is acceptable, the prepared command is:

```bash
gh release create source-private-tool-trace-artifacts-20260428 \
  paper/artifacts/source_private_tool_trace_artifacts_20260428.zip \
  paper/source_private_tool_trace_upload_checksums_20260428.sha256 \
  paper/iclr2026/submission/source_private_tool_trace_iclr_source_20260428.zip \
  paper/iclr2026/source_private_tool_trace.pdf \
  --title "Source-Private Tool-Trace Artifacts 2026-04-28" \
  --notes-file paper/source_private_tool_trace_release_notes_20260428.md
```

The release should then be verified with:

```bash
gh release view source-private-tool-trace-artifacts-20260428 --json url,assets,tagName
```

and the resulting URL should be copied into
`paper/source_private_tool_trace_submission_confirmation_20260428.md`.

## Anonymous Host Path

If anonymity is required, upload these files through the venue-approved
anonymous artifact mechanism:

- `paper/iclr2026/source_private_tool_trace.pdf`
- `paper/iclr2026/submission/source_private_tool_trace_iclr_source_20260428.zip`
- `paper/artifacts/source_private_tool_trace_artifacts_20260428.zip`
- `paper/source_private_tool_trace_upload_checksums_20260428.sha256`

Then update `paper/source_private_tool_trace_submission_confirmation_20260428.md`
with the anonymous URL or submission attachment ID and the SHA256 verification
result.

## Recommendation

Do not publish a public GitHub release until the anonymity requirement is known.
The repo is otherwise ready for either host path.
