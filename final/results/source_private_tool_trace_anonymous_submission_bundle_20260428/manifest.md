# Anonymous Submission Transfer Bundle Manifest

- date: `2026-04-28`
- gate: `anonymous_submission_bundle_20260428`
- status: transfer bundle built and verified
- bundle: `paper/artifacts/source_private_tool_trace_anonymous_submission_bundle_20260428.zip`
- bundle bytes: `8137764`
- bundle SHA256: `6d771cfc41ad4acfb8d500a6841e06e28781b3ce2549fc368dff0a1ed666e377`

## Contents

The bundle root is `source_private_tool_trace_anonymous_submission_20260428/`
and contains:

- `README.md`
- `source_private_tool_trace.pdf`
- `source_private_tool_trace_iclr_source_20260428.zip`
- `source_private_tool_trace_artifacts_anonymous_20260428.zip`
- `source_private_tool_trace_anonymous_upload_checksums_20260428.sha256`

## Verification

The bundle was tested with:

```bash
unzip -t paper/artifacts/source_private_tool_trace_anonymous_submission_bundle_20260428.zip
```

Result: all files tested `OK`.

The extracted bundle was scanned for account names, local paths, GitHub URLs,
and the project-name marker. Result: no matches.

Internal checksums were verified from inside the extracted bundle:

```bash
shasum -a 256 -c source_private_tool_trace_anonymous_upload_checksums_20260428.sha256
```

Result: manuscript PDF, source zip, and anonymous artifact zip all reported
`OK`.

## Use

This is a transfer convenience bundle. If the venue requires separate uploads,
use the individual files inside the bundle rather than the enclosing bundle
itself.
