# SVAMP32 Process-Trace Sidecar Smoke Manifest

- date: `2026-04-27`
- status: `process_trace_sidecar_pruned`
- scale rung: `smoke`
- target set: `results/svamp32_target_sampling_clean6_20260427/sampled_clean6_target_set.json`
- candidate-pool prerequisite: target/no-source clean oracle `2/6`
- sidecar script: `scripts/materialize_svamp_process_trace_sidecars.py`

## Variants

| Variant | Sidecar Dir | Selector | Status |
|---|---|---|---|
| `process_trace` | `process_trace_sidecars/` | `top_selector_process_trace.md` | fail, zero-margin target-label collapse |
| `process_trace_sample_pref` | `process_trace_sidecars_sample_pref/` | `top_selector_process_trace_sample_pref.md` | fail, control clean union `1` |
| `process_trace_predonly_no_t2t` | `process_trace_sidecars_predonly_no_t2t/` | `top_selector_process_trace_predonly_no_t2t.md` | fail, matched clean `0`, control clean union `1` |

## Key Metrics

- best matched clean correct: `0/6`
- best source-necessary clean: `0`
- best control clean union: `1`
- best accepted harm: `0`
- prediction-only no-`t2t` accepted matched IDs:
  `1d50b408c8f5cd2c`, `3e8a5691f5443495`
- prediction-only no-`t2t` random clean correct:
  `575d7e83d84c1e67`
- prediction-only no-`t2t` selected value in unmasked source numbers: `5/6`
- prediction-only no-`t2t` effective rank: `31.270460`

## Decision

Prune deterministic hand-built process-trace similarity sidecars on this slice.
The observed failure is not low-rank collapse; it is lack of source-necessary
candidate selection after answer masking and controls.
