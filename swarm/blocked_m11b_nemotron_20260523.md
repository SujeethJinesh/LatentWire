# Blocked: M11b Nemotron Replication Checker Failure

## Status

`FAIL_INFRA_M11B_NEMOTRON`

The M11b Nemotron replication completed all scoring regimes, but the checker
marked the artifact incomplete.

## Run Directory

`experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_20260520T2013Z`

## Checker Failure

`checker_result.json` reports:

```json
{
  "artifact_complete": false,
  "decision": "FAIL_INFRA_M11B_NEMOTRON",
  "reasons": [
    "excluded_tensors.by_regime mismatch"
  ]
}
```

The `excluded_tensors.json` file has `by_regime` entries for:

- `m11b_top1`
- `m11b_top5`
- `m11b_top10`
- `static_top10`

The shared checker expects entries for every non-BF16 regime, including
`static_1pct`.

## Numeric Metrics Present But Not Accepted

The run wrote `metrics.json`, `per_trace_metrics.json`, `bootstrap_ci.json`,
and `control_metrics.json`. Because artifact completeness failed, these are
diagnostic rather than accepted paper results.

Key diagnostic values:

- `m11b_top1` median recovery: -2.600041328227093
- `m11b_top5` median recovery: -0.9202140086605652
- `m11b_top5` CI95: [-16.239006269077635, 4.628823538409526]
- `m11b_top10` median recovery: -2.7949459243213406
- `static_top10` median recovery: -0.2916786834232309
- no-recoverable-static-gap fraction: 0.5
- included positive-gap traces: 6/12

## Backup

Final packet backup:

`/workspace/outlier_migrate_artifact_backups/om_phase9_m11b_nemotron_20260520T2013Z_final/`

The backup includes `SHA256SUMS.txt`.

## Action Required

Human review is needed to decide whether this is:

1. A harmless metadata/checker issue that can be patched and rechecked without
   rerunning GPU work.
2. A genuine artifact incompleteness requiring the run to remain
   `FAIL_INFRA`.

Per the pre-authorization instructions, paper integration and framing decisions
are paused rather than improvised from this checker-failed packet.
