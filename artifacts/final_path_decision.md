# Final Bounded Positive-Method Gate After K-RES Kill

Created: 2026-05-29T02:56:52Z
Git commit: `05d80f6cc3354edbf47338443984fbf2c52c8c78`

## Inputs

| Gate | Status | Consequence |
|---|---|---|
| K-RES failure audit | `VALID_IMPLEMENTATION_KEEP_K_RES_PROXY_KILLED` | The current top-8x32 residual-energy proxy was validly implemented and worsened Granite tail recovery (-12.29) versus tight ParoQuant (5.94). Do not rerun this proxy. |
| Restricted residual KLLOOK oracle | `NOT_EXECUTED_NO_ROTATED_RESIDUAL_KLLOOK_RUNNER` | No residual-oracle pass exists. Existing M-KLLOOK code is original-basis channel protection and is not reused. Residual correction is not promoted. |
| M-SURFACE | `MEASURED_CHEAP_SURFACES_NO_PROMOTE__SSM_BC_INCOMPLETE` | Hookable Granite projection-input surfaces drift more than post-block. SurfaceRot/SurfaceProtect is not promoted; SSM input/B/C remain future work. |
| Falcon M-BRANCH / BranchRot | `DEFER_NO_BRANCH_LOCAL_CACHE_NO_PROMOTE` | Falcon branch-local surfaces are feasible but unmeasured. No BranchRot promotion. |

## Decision Tree Result

- A. KLLOOK residual oracle passes: **No.** Not executed; no evidence to overturn K-RES kill.
- B. M-SURFACE passes: **No.** Measured cheap surfaces do not pass.
- C. M-BRANCH passes: **No.** No branch-local measurement exists.
- D. All current positive gates fail or lack promotion evidence: **Yes.**

## Recommended Path

Stop the current DriftRot positive-method search. The publishable path is the mechanism/regime paper:

- channel-set drift measurement,
- failure taxonomy for static, hard-switch, EMA, predictive, residual, surface, and branch screens,
- rotation as strongest baseline, without claiming ParoQuant as ours,
- explanation that rotation removes basis dependence while channel identity drifts,
- negative residual/surface/branch screens,
- systems cost model for hypothetical residual correction as future method scaffolding.

A future residual method would require a new preregistered rotated-basis residual KLLOOK runner and held-out confirmation. It should not block the current paper path.

## Final Status Lines

K_RES_AUDIT_STATUS=VALID_IMPLEMENTATION_KEEP_K_RES_PROXY_KILLED
KLLOOK_ORACLE_STATUS=NOT_EXECUTED_NO_ROTATED_RESIDUAL_KLLOOK_RUNNER_RESIDUAL_NOT_PROMOTED
MSURFACE_STATUS=MEASURED_CHEAP_SURFACES_NO_PROMOTE__SSM_BC_INCOMPLETE
MBRANCH_STATUS=DEFER_NO_BRANCH_LOCAL_CACHE_NO_PROMOTE
RECOMMENDED_NEXT_ACTION=STOP_CURRENT_POSITIVE_METHOD_SEARCH_AND_FINALIZE_MECHANISM_REGIME_PAPER
PAPER_PATH=MECHANISM_REGIME_PAPER
