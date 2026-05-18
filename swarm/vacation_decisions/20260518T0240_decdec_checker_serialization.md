# Vacation Decision: DecDEC Checker Serialization Bug

## Situation

The Granite-Small DecDEC baseline runner completed all scoring regimes and
printed the baseline summary, but exited non-zero during checker evaluation.
The traceback showed `FileExistsError` in
`experimental/outlier_migrate/phase9/check_om_phase9_decdec_baseline.py`:
`write_json()` called `path.parent.mkdir(parents=True)` for the already
existing run directory when writing `checker_result.json`.

The score caches are present for all regimes:
`bf16`, `static_1pct`, `decdec_reactive_top1_proxy`, `static_top10`, and
`random_reactive_top1`.

## Options Considered

1. Treat the run as `FAIL_INFRA_DECDEC_BASELINE` and move on.
2. Manually write a checker result from the printed metrics.
3. Patch the checker serialization helper to use `exist_ok=True`, then rerun
   the checker mechanically on the completed packet.

## Decision

I chose option 3. This preserves the scientific decision rule and fixes only
an infrastructure serialization bug. The checker already computed the packet
summary; it failed only while creating an already-existing parent directory.

## Invalidation Condition

If rerunning the checker after this fix produces any artifact-completeness
error or changes the computed metrics inconsistently with `metrics.json`, this
decision is invalid and the packet should be treated as an infrastructure
failure until reviewed.
