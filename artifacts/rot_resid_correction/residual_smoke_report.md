# DriftRot Residual-Correction Smoke Report

## Status

Decision: `KILL_TOP8X32_MOE_RESIDUAL_DIAGNOSTIC`.

The top-8-module, 32-column-per-module MoE residual correction is not promoted
to 3-trace smoke. After fixing an invalid first run, the valid prompt-4
diagnostic worsened the tight ParoQuant tail trace.

## Artifacts

| Artifact | Path |
|---|---|
| Delta-column materializer | `experimental/outlier_migrate/phase9/materialize_om_driftrot_delta_columns.py` |
| Delta-column cache | `artifacts/rot_resid_correction/delta_columns_granite_tail4_top8x32_20260528T2048Z/` |
| Residual smoke runner | `experimental/outlier_migrate/phase9/run_om_driftrot_residual_subset.py` |
| Invalid first diagnostic | `experimental/outlier_migrate/phase9/results/om_driftrot_residual_granite_tail4_diag_20260528T2052Z/` |
| Valid diagnostic | `experimental/outlier_migrate/phase9/results/om_driftrot_residual_granite_tail4_diag_fixed_20260528T2116Z/` |

The invalid first diagnostic is marked
`FAIL_INFRA_CLOSURE_CAPTURE_INVALID`: the initial runner used loop-captured
patched-forward state. The fixed runner binds each module's original forward
and residual tensors through a factory closure.

## Valid Diagnostic Result

Model: Granite-4.0-H-Small. Trace: `opencompass_AIME2025_I_4`.

| Regime | Perplexity | Recovery |
|---|---:|---:|
| BF16 | 1.0368 | n/a |
| Static top-1% | 1.0375 | 0.000 |
| Original ParoQuant `[0.25, 4.0]` | 1.0579 | -26.37 |
| Tight ParoQuant `[0.5, 2.0]` | 1.0329 | 5.94 |
| Tight ParoQuant + residual correction | 1.0470 | -12.29 |

The residual correction improves over original ParoQuant on this specific tail
trace, but it is much worse than tight ParoQuant alone. Because tight
ParoQuant already fixed the trace and residual correction reintroduces a large
loss, this candidate does not satisfy the DriftRot criterion: improve or
robustify static rotation beyond the ParoQuant/tight-clip reference.

## Interpretation

The selected MoE residual columns are not a safe tail-control mechanism. They
restore selected input columns for high-residual MoE input projections, but in
the nonlinear MoE path that local correction can still amplify downstream loss.
The Python-level expert-bank wrapper is also slow enough that this path would
need a kernel even if it passed.

Residual correction remains a possible future direction only with a stricter
bounded design, such as coefficient-shrunk correction, per-layer calibration
acceptance tests, or KLLOOK-gated columns. It is not the next live branch under
the current queue.
