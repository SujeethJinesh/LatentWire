# M11b Nemotron Partial Rerun Feasibility

Date: 2026-05-24

## Conclusion

**Path C is feasible.** The failure is isolated to the `static_1pct`
baseline scoring path. The existing `m11b_top1`, `m11b_top5`,
`m11b_top10`, and `static_top10` score caches were produced through the
quantized scoring path and can be preserved for recovery recalculation.

Recommended path: patch the runner/checker to score only corrected
`static_1pct` with W4A16 quantization, write a derived result packet that
reuses the existing BF16 traces, activation magnitudes, protected sets,
and M11b/static-top10 score caches, then recompute metrics and checker
outputs. Do not mutate the original failed packet except as a source
artifact reference.

Estimated cost: **about 10-11 GPU hours** for one 12-trace scoring regime,
based on the observed Nemotron regime timings.

## Question Answered

The key question was whether the same bug invalidated activation capture
or M11b scoring across all regimes.

Answer: **No.** The activation artifact is a calibration artifact and was
captured before any quantized scoring. The M11b regimes then used that
calibration evidence to build protected sets and scored fresh models via
`apply_quantization(...)`. Only `static_1pct` was scored through the
baseline loop that skipped `apply_quantization(...)`.

## Evidence

### Runner Code Path

In `experimental/outlier_migrate/phase9/run_om_phase9_m11b_budget_scaling.py`,
the M11b runner has two distinct paths.

The `bf16` and `static_1pct` baseline loop calls `score_targets(...)`
directly when no reusable cache exists:

```python
for regime, source_regime in [("bf16", "bf16"), ("static_1pct", "static_1pct")]:
    cached = m11_runner.read_score_cache_any(...)
    ...
    all_scores[regime] = phase4_runner.score_targets(...)
```

That path is correct for `bf16` but wrong for `static_1pct`, because it
does not call `phase4_runner.apply_quantization(...)`.

By contrast, the M11b and `static_top10` regimes are scored via
`score_regime(...)`:

```python
excluded = phase4_runner.apply_quantization(model, protected_sets, regime)
scores = phase4_runner.score_targets(...)
```

The failed Nemotron run did not use a reusable score run:

```json
"reuse_score_run_dir": null
```

Therefore `static_1pct` fell onto the broken direct-scoring path, while
the M11b regimes used the quantized path.

### Artifact Evidence

Run directory:

`experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_20260520T2013Z`

The checker failed only on the missing quantization metadata:

```json
{
  "artifact_complete": false,
  "decision": "FAIL_INFRA_M11B_NEMOTRON",
  "reasons": ["excluded_tensors.by_regime mismatch"]
}
```

`excluded_tensors.json` contains quantization records for:

| Regime | Quantized tensors | Excluded tensors / modules |
|---|---:|---:|
| `m11b_top1` | 6052 | 76 |
| `m11b_top5` | 6052 | 76 |
| `m11b_top10` | 6052 | 76 |
| `static_top10` | 6052 | 76 |

It does not contain `static_1pct`, which is exactly the expected signature
of the baseline-loop bug.

`protected_sets.json` does contain `static_1pct`, `m11b_top1`,
`m11b_top5`, `m11b_top10`, and `static_top10` protected sets. Example
protected counts for layer 0:

| Regime | Layer-0 protected channels |
|---|---:|
| `static_1pct` | 27 |
| `m11b_top1` | 27 |
| `m11b_top5` | 135 |
| `m11b_top10` | 269 |
| `static_top10` | 269 |

This means the missing static baseline can be reconstructed without
recapturing activation magnitudes or rebuilding protected sets.

### Activation Artifact

The activation manifest contains a complete 100-position cadence through
position 10000:

- rows: `62400`
- positions: `100, 200, ..., 10000`
- source reuse: `null`

This artifact is a calibration artifact used to construct protected sets.
It is not a per-regime W4A16 score artifact. That is expected for M11b:
the quantization happens during scoring via `apply_quantization(...)`.

Therefore the activation artifact is reusable for Path C. The right
question is not whether it was "W4A16 captured"; the right question is
whether the M11b score regimes applied W4A16 quantization. They did.

### Score Cache Completeness

Existing score caches contain all 12 prompts for:

- `bf16`
- `m11b_top1`
- `m11b_top5`
- `m11b_top10`
- `static_top10`

The `static_1pct` cache exists but is invalid because it was scored
without W4A16 quantization. It should be replaced in the derived packet.

### Observed Timing

Observed completed-scoring spans in the failed Nemotron run:

| Regime | Completed prompts | Span |
|---|---:|---:|
| `static_1pct` invalid baseline | 12 | 10.06 h |
| `m11b_top1` | 12 | 9.55 h |
| `m11b_top5` | 12 | 8.99 h |
| `m11b_top10` | 12 | 9.07 h |
| `static_top10` | 12 | 9.19 h |

Measured from the previous regime's final completion to each regime's
final completion, the static-like regimes took about 10.2-11.0 hours
including model load overhead.

## Path C Implementation Sketch

Do not run this until human approval.

1. Patch `run_om_phase9_m11b_budget_scaling.py` or add a narrow salvage
   script so `static_1pct` is scored through:

   ```python
   score_regime(..., regime="static_1pct", ...)
   ```

2. Create a derived result directory, for example:

   `experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_YYYYMMDDTHHMMZ`

3. Copy or reference from the failed packet:

   - `activation_magnitudes.jsonl.gz`
   - `activation_magnitude_manifest.json`
   - `bf16_traces.jsonl.gz`
   - `bf16_trace_manifest.json`
   - `protected_sets.json`
   - `protected_trajectories.json`
   - score caches for `bf16`, `m11b_top1`, `m11b_top5`, `m11b_top10`,
     and `static_top10`

4. Regenerate only:

   - corrected `score_cache/static_1pct.json`
   - `excluded_tensors.json` with all non-BF16 regimes, including
     corrected `static_1pct`
   - `per_trace_metrics.json`
   - `metrics.json`
   - `control_metrics.json`
   - `bootstrap_ci.json`
   - `artifact_hashes.json`
   - `artifact_check.json`
   - `checker_result.json`

5. Run the checker and only then classify the Nemotron replication as
   PASS, AMBIGUOUS, or KILL.

## Recommendation

Choose **Path C** if the human is willing to spend about 10-11 GPU hours.
It is much cheaper than a 30-40 GPU-hour full rerun and preserves the valid
M11b regime scores already produced.

If no further GPU spend is acceptable, choose Path A and report the
Nemotron replication as `FAIL_INFRA`, not as KILL or PASS.

I do **not** recommend Path B full rerun unless the derived static-only
salvage run fails or the human wants a single monolithic packet for audit
simplicity.

