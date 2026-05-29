# Decision: Granite M-SURFACE Cheap Hooks Do Not Lower Drift

Timestamp: 2026-05-29T01:00Z

## Run

- Experiment: Granite M-SURFACE two-trace hook sanity diagnostic
- Run directory: `experimental/outlier_migrate/phase9/results/om_phase9_msurface_granite_sanity_20260529T0032Z`
- Prompt indices: `[0, 1]`
- Decode positions: `[100, 20000]`
- Surfaces:
  - `post_block_residual_block_output`
  - `mamba_out_projection_input`
  - `attention_out_projection_input`

## Result

| Surface | Mean strict set-leaving | Median strict set-leaving | Mean delta vs same-layer post-block |
|---|---:|---:|---:|
| post-block residual output | 0.578 | 0.585 | 0.000 |
| Mamba `out_proj` input | 0.887 | 0.902 | +0.311 |
| attention `o_proj` input | 0.780 | 0.817 | +0.183 |

Decision: `KILL_OR_DEFER_SURFACE_NO_LOWER_DRIFT`.

## Interpretation

The cheap internal module-hook surfaces do not explain away block-output drift.
They are worse than the same-run post-block control, so there is no reason to
promote this M-SURFACE branch to endpoint scoring or to run the analogous
Falcon surface diagnostic from the same hook family.

## Next Gate

Only reopen surface placement with a different local tensor target, such as
SSM B/C internals, or with a KLLOOK-backed hypothesis that the cheap hook
surfaces are the wrong candidate pool. Otherwise keep the positive-method
search focused on rotation/protocol evidence and other higher-yield gates.
