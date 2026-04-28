# Source-Private Tool-Trace Submission Confirmation

- date: `2026-04-28`
- gate: `external_submission_confirmation_20260428`
- status: blocked on external portal/artifact-host action

## Current Readiness

The local repo is ready for external submission. The paper PDF, compile-tested
source bundle, and artifact archive are present, hashed, documented, and pushed.

## Files To Upload

- manuscript PDF:
  `paper/iclr2026/source_private_tool_trace.pdf`
- source bundle:
  `paper/iclr2026/submission/source_private_tool_trace_iclr_source_20260428.zip`
- artifact archive:
  `paper/artifacts/source_private_tool_trace_artifacts_20260428.zip`

Hashes are recorded in
`paper/source_private_tool_trace_external_handoff_20260428.md` and the checksum
sidecar `paper/source_private_tool_trace_upload_checksums_20260428.sha256`.
For double-blind artifact upload, use
`paper/source_private_tool_trace_anonymous_upload_checksums_20260428.sha256`
instead.

## Blocker

External confirmation cannot be completed from the local repo alone. It needs
one of:

- the ICLR/OpenReview submission portal upload state;
- an artifact host URL or release ID;
- user-provided confirmation that the three upload files above were accepted.

Until then, the submission status should be treated as locally upload-ready but
not externally confirmed.

GitHub release hosting is technically available, but the repository is public
and non-anonymized. The host decision and anonymity caveat are documented in
`paper/source_private_tool_trace_artifact_host_options_20260428.md`.
The preferred double-blind artifact payload is
`paper/artifacts/source_private_tool_trace_artifacts_anonymous_20260428.zip`.
A transfer bundle containing the PDF, source zip, anonymous artifact zip, and
checksum sidecar is available at
`paper/artifacts/source_private_tool_trace_anonymous_submission_bundle_20260428.zip`.

## Record After Upload

Once uploaded, update this memo with:

- final submission ID or OpenReview forum URL;
- uploaded artifact URL or release ID;
- host type: anonymous venue upload, anonymized artifact host, or public GitHub
  release;
- whether the files were uploaded separately or via the anonymous transfer
  bundle;
- portal-side PDF/source warnings, if any;
- whether the uploaded archive SHA256 matches the local handoff memo;
- whether all uploaded files match
  `paper/source_private_tool_trace_upload_checksums_20260428.sha256`;
- the final commit used for submission.

## Next Gate

`external_submission_confirmation_20260428`: replace this blocker note with the
actual portal/artifact-host confirmation.
