# Vacation Decision - M11 Oversized Raw Artifact

## Situation

The completed M11 result packet contains:

- `experimental/outlier_migrate/phase9/results/om_phase9_m11_granite_small_vac12_20260516T010728Z/activation_magnitudes.jsonl.gz`

This file is `444204543` bytes. Git LFS is not installed on this node, and
GitHub's normal push path rejects files above 100MB. The file is already
covered by the packet's local `artifact_hashes.json` and by `metrics.json`.

## Options Considered

1. Force-add the full file and attempt push.
   - Rejected because it would almost certainly fail the GitHub push and
     create avoidable history cleanup work.
2. Install/configure Git LFS mid-sprint.
   - Rejected because the repository is not currently configured for LFS and
     changing storage policy during an autonomous sprint is higher risk than
     preserving local artifacts and pushing reduced metadata.
3. Commit the reduced packet and leave the oversized raw artifact in-place on
   `/workspace`, with its SHA recorded in the committed metadata.
   - Chosen because it preserves auditability on the active pod and keeps the
     pushed repository visible to the human.

## Decision

Commit all M11 packet files except the oversized raw activation-magnitude dump.
The raw artifact remains in the workspace at its original path, and its SHA is
recorded in `artifact_hashes.json` and `metrics.json`.

## Invalidating Condition

If the human wants every raw activation dump present in Git history, they
should configure Git LFS or another artifact store, then re-add this artifact
from the preserved workspace path before the pod is destroyed.
