# HybridKernel Profiler Analysis Gate

Status: **WEAKLY ALIVE: profiler evidence is nonzero but below the prototype gate.**

Collect more repeated traces or narrow to the largest boundary anomaly.

## Model Summary

| Model/config | Runs | Batch | Mean avoidable share | Mean gain UB | Median gain UB | IQR | Bootstrap 95% CI | Min gain UB | Clears? |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| synthetic-granite-fixture / bfloat16 / graph=True / control=matched_transformer_block | 3 | b1 p128 d64 r16 | 2.00% | 1.20% | 1.20% | 0.00% | [1.20%, 1.20%] | 1.20% | no |

Definitions: `attention_ssm_boundary_ms` is the measured boundary-local cost from the native profiler pass. `matched_non_boundary_ms` is the matched local control cost. The avoidable share is `max(boundary - control, 0) / total_step_ms`; the recoverable-gain upper bound additionally multiplies by the assumed recoverable fraction.
