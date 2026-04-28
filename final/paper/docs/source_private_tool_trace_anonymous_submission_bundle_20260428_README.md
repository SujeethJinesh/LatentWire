# Anonymous Submission Bundle README

- date: `2026-04-28`
- purpose: transfer checklist for venue upload

## Files

- `source_private_tool_trace.pdf`: manuscript PDF.
- `source_private_tool_trace_iclr_source_20260428.zip`: compile-tested
  manuscript source bundle.
- `source_private_tool_trace_artifacts_anonymous_20260428.zip`: anonymous
  artifact archive for double-blind upload.
- `source_private_tool_trace_anonymous_upload_checksums_20260428.sha256`:
  checksum sidecar for the three upload files.

## Verify

From this directory:

```bash
shasum -a 256 -c source_private_tool_trace_anonymous_upload_checksums_20260428.sha256
```

Expected result: all three upload files report `OK`.

## Upload Guidance

Use the individual files above in the submission portal. The enclosing transfer
bundle is only for moving the files together; the portal may require the PDF,
source zip, and artifact zip as separate uploads.

After upload, record the submission ID or URL, artifact attachment URL or ID,
portal warnings, and checksum verification in
`paper/source_private_tool_trace_submission_confirmation_20260428.md`.
