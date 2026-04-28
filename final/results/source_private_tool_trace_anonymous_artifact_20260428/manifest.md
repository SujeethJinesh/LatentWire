# Anonymous Source-Private Tool-Trace Artifact Manifest

- date: `2026-04-28`
- gate: `anonymous_artifact_handoff_20260428`
- status: anonymous artifact archive built and integrity-checked
- archive: `paper/artifacts/source_private_tool_trace_artifacts_anonymous_20260428.zip`
- archive bytes: `7553609`
- archive SHA256: `02b1dbd73dea1332976e60a255def4a470f6c91416f2603cbd8631270be3790a`

## Purpose

This archive is the preferred artifact payload for a double-blind venue route.
It is derived from `paper/artifacts/source_private_tool_trace_artifacts_20260428.zip`
but removes the one project-name marker found in an included paper memo and uses
an anonymous archive root.

## Checks

Archive integrity:

```bash
unzip -t paper/artifacts/source_private_tool_trace_artifacts_anonymous_20260428.zip
```

Result: all files tested `OK`; no compressed-data errors detected.

Anonymity string audit:

```bash
rg -n "Sujeeth|sujeeth|Jinesh|jinesh|LatentWire|github.com|/Users|Desktop" \
  .debug/anonymous_artifact_verify_20260428
```

Result: no matches.

Upload checksum sidecar:

```bash
shasum -a 256 -c paper/source_private_tool_trace_anonymous_upload_checksums_20260428.sha256
```

## Recommendation

For anonymous submission, upload this archive instead of the public-repo
artifact archive:

- `paper/artifacts/source_private_tool_trace_artifacts_anonymous_20260428.zip`

The non-anonymous public-repo artifact archive remains available for internal
release or a non-anonymous GitHub release route.
