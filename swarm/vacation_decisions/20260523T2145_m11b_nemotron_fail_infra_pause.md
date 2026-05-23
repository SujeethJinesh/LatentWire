# Vacation Decision - M11b Nemotron FAIL_INFRA Pause

## Situation

The M11b Nemotron replication runner completed all scoring regimes for
`om_phase9_m11b_nemotron_20260520T2013Z`, including `bf16`, `static_1pct`,
`m11b_top1`, `m11b_top5`, `m11b_top10`, and `static_top10`. The checker then
returned:

`FAIL_INFRA_M11B_NEMOTRON`

The checker reason is:

`excluded_tensors.by_regime mismatch`

The packet contains metrics and per-trace rows, but `artifact_complete` is
false in both `checker_result.json` and `artifact_check.json`.

## Options Considered

1. Treat the numeric metrics as interpretable and proceed with paper
   integration.
2. Patch the artifact/checker mismatch locally and rerun the checker.
3. Pause the framing decision, preserve the packet, and wait for human review.

## Decision

I chose option 3. The human's pre-authorization explicitly said that if a
Nemotron infrastructure failure or conflicting evidence occurs, I should pause
and write a progress note rather than improvise major strategic decisions.

The numeric metrics are recorded, but I will not use them to choose PASS,
AMBIGUOUS, or KILL framing until the artifact mismatch is reviewed.

## Current Numeric Readout, Not Treated As Accepted

- `m11b_top1` median recovery: -2.600041328227093
- `m11b_top5` median recovery: -0.9202140086605652
- `m11b_top5` CI95: [-16.239006269077635, 4.628823538409526]
- `m11b_top10` median recovery: -2.7949459243213406
- `static_top10` median recovery: -0.2916786834232309
- no-recoverable-static-gap fraction: 0.5
- included positive-gap traces: 6/12

These values suggest the run would not support a positive-method replication,
but because the packet is checker-failed, this is diagnostic only.

## Artifact Preservation

The completed run directory remains at:

`experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_20260520T2013Z`

A full out-of-repo backup with SHA256 sums was created at:

`/workspace/outlier_migrate_artifact_backups/om_phase9_m11b_nemotron_20260520T2013Z_final/`

## What Would Invalidate This Decision

If the human determines that the missing `static_1pct` entry in
`excluded_tensors.by_regime` is a harmless metadata/checker defect and
authorizes accepting or patching the packet, the numeric results can be
reclassified under the preregistered decision rule.
