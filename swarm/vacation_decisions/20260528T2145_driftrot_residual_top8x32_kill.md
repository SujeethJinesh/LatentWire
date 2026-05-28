# DriftRot Residual-Correction Top-8x32 Diagnostic

Date: 2026-05-28T21:45Z

## Decision

`KILL_TOP8X32_MOE_RESIDUAL_DIAGNOSTIC`

Do not run the planned 3-trace Granite smoke for the current residual-correction
candidate.

## Evidence

Artifacts produced:

- residual norms:
  `artifacts/rot_resid_correction/residual_cache_granite_tight_20260528T2025Z/`
- activation EMA:
  `artifacts/rot_resid_correction/activation_ema_granite_tail4_top8_20260528T2030Z/`
- candidate pool:
  `artifacts/rot_resid_correction/residual_candidate_pool_tail4_top8.json`
- selected DeltaW columns:
  `artifacts/rot_resid_correction/delta_columns_granite_tail4_top8x32_20260528T2048Z/`
- valid diagnostic run:
  `experimental/outlier_migrate/phase9/results/om_driftrot_residual_granite_tail4_diag_fixed_20260528T2116Z/`

Granite prompt 4:

| Regime | Recovery |
|---|---:|
| Original ParoQuant `[0.25, 4.0]` | -26.37 |
| Tight ParoQuant `[0.5, 2.0]` | 5.94 |
| Tight ParoQuant + residual correction | -12.29 |

The valid residual-correction run is better than original ParoQuant on this
tail trace, but it is much worse than tight ParoQuant alone. It does not meet
the DriftRot criterion of improving or robustifying rotation beyond the
reference rotation configuration.

## Infra Note

The first diagnostic run
`om_driftrot_residual_granite_tail4_diag_20260528T2052Z` is invalid and marked
`FAIL_INFRA_CLOSURE_CAPTURE_INVALID`. The initial wrapper captured loop
variables in patched-forward closures. The fixed runner binds per-module state
through factory closures.

## Consequence

Residual correction is demoted. Reopen it only with a new bounded design, such
as coefficient-shrunk residual correction, per-layer calibration acceptance, or
KLLOOK-gated columns. The next live rotation-specific gates are clip/CVaR
confirmation or surface/branch diagnostics.
