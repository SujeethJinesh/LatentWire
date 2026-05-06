# HybridKernel Profiler Analysis Gate

Status: **PENDING native profiler data.**

No native speed or overhead claim is allowed.

## Model Summary

| Model/config | Runs | Batch | Mean avoidable share | Mean gain UB | Median gain UB | IQR | Bootstrap 95% CI | Min gain UB | Clears? |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| pending | 0 | -- | -- | -- | -- | -- | -- | -- | no |

Definitions: `attention_ssm_boundary_ms` is the measured boundary-local cost from the native profiler pass. `matched_non_boundary_ms` is the matched local control cost. The avoidable share is `max(boundary - control, 0) / total_step_ms`; the recoverable-gain upper bound additionally multiplies by the assumed recoverable fraction.
