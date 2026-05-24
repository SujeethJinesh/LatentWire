# M11b Nemotron FAIL_INFRA Diagnostic

## Conclusion

Classification: **B) SUBSTANTIVE**

The missing `excluded_tensors.by_regime.static_1pct` field is not harmless
bookkeeping. It reflects that the Nemotron run scored the `static_1pct`
baseline without applying the W4A16 quantization/protection path. Therefore
the per-trace recovery ratios in the Nemotron packet are not trustworthy as
paper results.

## Evidence

### 1. Checker failure

The checker returned:

```json
{
  "artifact_complete": false,
  "decision": "FAIL_INFRA_M11B_NEMOTRON",
  "reasons": [
    "excluded_tensors.by_regime mismatch"
  ]
}
```

The shared M11b checker expects `excluded_tensors.by_regime` to include every
non-BF16 regime:

- `static_1pct`
- `m11b_top1`
- `m11b_top5`
- `m11b_top10`
- `static_top10`

The Nemotron packet includes only:

- `m11b_top1`
- `m11b_top5`
- `m11b_top10`
- `static_top10`

### 2. Runner code path

In `experimental/outlier_migrate/phase9/run_om_phase9_m11b_budget_scaling.py`,
the baseline loop handles `bf16` and `static_1pct` separately from the
quantized M11b regimes:

```python
for regime, source_regime in [("bf16", "bf16"), ("static_1pct", "static_1pct")]:
    cached = m11_runner.read_score_cache_any(...)
    ...
    all_scores[regime] = phase4_runner.score_targets(...)
```

If no reusable cache exists, this path calls `score_targets` directly. It does
not call `phase4_runner.apply_quantization(model, protected_sets, regime)`.

By contrast, `m11b_top1`, `m11b_top5`, `m11b_top10`, and `static_top10` go
through `score_regime(...)`, which does call:

```python
excluded = phase4_runner.apply_quantization(model, protected_sets, regime)
```

### 3. Granite comparison

The validated Granite M11b packet reused `static_1pct` scores from an earlier
M2 packet:

`experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z`

Granite `excluded_tensors.by_regime` includes `static_1pct`, and the ultimate
source packet (`om_phase9_m2_granite_small_vac12_finalized_20260514T233800Z`)
has actual static quantization metadata for `static_1pct`:

- `static_1pct` excluded tensors: 118
- `static_1pct` quantized tensors: 325

The Nemotron run did not use `--reuse-score-run-dir`, so it generated
`static_1pct` through the direct unquantized scoring path:

```json
"reuse_score_run_dir": null
```

Nemotron `excluded_tensors.by_regime` has no `static_1pct` entry, while its
quantized regimes each show:

- excluded tensors: 76
- quantized tensors: 6052

This is exactly the pattern expected if `static_1pct` skipped quantization
while the other regimes applied it.

### 4. Numerical plausibility check

The Nemotron `static_1pct` gaps are also consistent with an invalid baseline.
They compare BF16 to FP16-autocast scoring rather than BF16 to W4A16
static-1% protection:

- positive-gap traces: 6/12
- median static gap across all traces: `-9.328365123129068e-06`
- median positive static gap: `0.00019686357391979215`
- max positive gap: `0.19950103445258005`
- min gap: `-0.21615205021922734`

These gaps are not a reliable W4A16 static-protection denominator. The
reported recoveries are therefore ratios against the wrong baseline:

- `m11b_top5` median recovery: `-0.9202140086605652`
- `m11b_top10` median recovery: `-2.7949459243213406`
- `static_top10` median recovery: `-0.2916786834232309`

These diagnostic numbers suggest poor transfer, but they cannot be used as
accepted paper results without a correct `static_1pct` denominator.

## Practical Implication

The Nemotron packet should remain `FAIL_INFRA_M11B_NEMOTRON`.

The appropriate paper status is:

- Do not claim M11b replicated on Nemotron.
- Do not claim a valid Nemotron KILL from this packet.
- Document the attempted replication as blocked by a baseline-scoring bug
  unless the human authorizes a corrected partial rerun.

## Rerun Cost / Trade-off

A full rerun is not necessary if the human wants to salvage the packet.
The existing packet already contains:

- activation magnitudes
- BF16 traces
- BF16 scores
- M11b top-1/top-5/top-10 scores
- static-top10 scores

The likely salvage path is:

1. Patch the runner/checker so `static_1pct` is scored through
   `apply_quantization`.
2. Rerun only the `static_1pct` regime on the same 12 traces.
3. Recompute `per_trace_metrics.json`, `metrics.json`, `control_metrics.json`,
   `bootstrap_ci.json`, `artifact_hashes.json`, `artifact_check.json`, and
   `checker_result.json`.

Estimated GPU cost: approximately one scoring regime, similar to
`static_top10`, roughly 9-11 GPU hours at the observed 50-minute-per-prompt
cadence plus model load overhead.

Per the human's instruction, I am not running this corrected baseline without
approval.
