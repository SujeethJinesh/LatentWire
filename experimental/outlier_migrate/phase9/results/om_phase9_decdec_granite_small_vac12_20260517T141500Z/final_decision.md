# DecDEC Algorithmic Baseline Final Decision

Run ID: `om_phase9_decdec_granite_small_vac12_20260517T141500Z`

Decision: `PASS_DECDEC_BASELINE_REPORTED`

This is a descriptive baseline packet, not a positive-method gate. The checker
requires artifact completeness and reports the recovery statistics for the
algorithmic DecDEC proxy.

## Primary Result

- Model: `ibm-granite/granite-4.0-h-small`
- Snapshot: `b8c0982bab7fde4eb48110f5a069527c008fab39`
- Trace count: 12 AIME-2025 traces
- Included traces with positive static gap: 8
- No-recoverable-static-gap fraction: `0.3333333333333333`
- Bootstrap seed: `20260528`
- Scoring position: `10000`
- Scoring window: `512` tokens

| Regime | Median recovery | CI95 low | CI95 high |
| --- | ---: | ---: | ---: |
| DecDEC reactive top-1% proxy | `-0.07003478125820645` | `-8.495693501232758` | `0.6741763703501289` |
| Static top-10% matched-budget control | `-0.22827336820997218` | `-68.56236265688855` | `0.6174752804074675` |
| Random reactive top-1% control | `-1.2626049261250778` | `-219.01468794770764` | `-0.8667338887624265` |

The DecDEC proxy is less damaging than the random reactive control and the
static top-10% control by median recovery, but its own median recovery is
negative and its confidence interval crosses zero. It does not recover the
BF16-vs-static-1% W4A16 gap on this Granite-Small slice.

## Interpretation

This result satisfies the reviewer-comparison requirement at the algorithmic
level: per-step reactive top-1% channel selection was tested under the same
recovery metric as the Phase 9 methods. It does not support a positive
method claim. In the current mechanism framing, it weakens the hypothesis that
reactive endpoint channel selection alone solves the gap, while preserving the
narrower observation that real magnitude-selected reactive channels are less
bad than random reactive channels.

The packet intentionally does not claim a full systems reproduction of DecDEC:
it omits DecDEC's CUDA kernel, CPU staging path, and throughput claims. It is
an algorithmic recovery comparison only.

## Artifact Trail

- Preregistration: `experimental/outlier_migrate/phase9/preregister_om_phase9_decdec_baseline.md`
- Metrics: `metrics.json`
- Controls: `control_metrics.json`
- Per-trace metrics: `per_trace_metrics.json`
- Checker result: `checker_result.json`
- Artifact check: `artifact_check.json`

An initial checker write-path bug created `infra_error.json` after the runner
had already produced all score caches. The checker was patched to use
`exist_ok=True` when writing JSON into an existing run directory, then rerun
successfully. The final checker decision is recorded in `checker_result.json`
and `artifact_check.json`.
