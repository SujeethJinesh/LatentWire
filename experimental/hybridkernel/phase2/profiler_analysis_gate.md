# HybridKernel Profiler Analysis Gate

Status: **PENDING native profiler data.**

No native speed or overhead claim is allowed.

## Model Summary

| Model | Runs | Mean avoidable share | Mean recoverable gain UB | Min gain UB | Clears all-run 3% gate? |
|---|---:|---:|---:|---:|---|
| pending | 0 | -- | -- | -- | no |

Definitions: `attention_ssm_boundary_ms` is the measured boundary-local cost from the native profiler pass. `matched_non_boundary_ms` is the matched local control cost. The avoidable share is `max(boundary - control, 0) / total_step_ms`; the recoverable-gain upper bound additionally multiplies by the assumed recoverable fraction.
