# HybridKernel Profiler Analysis Gate

Status: **KILL or shelve: native profiler summaries show less than 1% recoverable gain.**

Do not spend kernel implementation time without a new profiler anomaly.

## Model Summary

| Model/config | Runs | Batch | Mean avoidable share | Mean gain UB | Median gain UB | IQR | Bootstrap 95% CI | Primary 95% CI | Min primary gain UB | Clears? |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| ibm-granite/granite-4.0-h-tiny / bfloat16 / graph=True / control=granite_hybrid_attention_ssm_boundary_windows | 3 | b1 p128 d64 r16 | 0.00% | 0.00% | 0.00% | 0.00% | [0.00%, 0.00%] | [0.00%, 0.00%] | 0.00% | no |
| ibm-granite/granite-4.0-h-tiny / bfloat16 / graph=True / control=granite_same_model_non_boundary_ssm_to_ssm_or_attention_internal_windows | 3 | b1 p128 d64 r16 | 0.00% | 0.00% | 0.00% | 0.00% | [0.00%, 0.00%] | [0.00%, 0.00%] | 0.00% | no |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 / bfloat16 / graph=True / control=nemotron_h_attention_adjacent_boundary_windows | 3 | b1 p128 d64 r16 | 0.00% | 0.00% | 0.00% | 0.00% | [0.00%, 0.00%] | [0.00%, 0.00%] | 0.00% | no |

Definitions: `attention_ssm_boundary_ms` is the measured boundary-local cost from the native profiler pass. `matched_non_boundary_ms` is the matched local control cost. The avoidable share is `max(boundary - control, 0) / total_step_ms`; the recoverable-gain upper bound additionally multiplies by the assumed recoverable fraction.
