# Repro Audit: Final Bounded Gate Sources

Created: 2026-05-29T02:56:52Z
Git commit: `05d80f6cc3354edbf47338443984fbf2c52c8c78`

Primary sources:

- K-RES audit: `artifacts/k_res_audit/`
- Fixed residual endpoint run: `experimental/outlier_migrate/phase9/results/om_driftrot_residual_granite_tail4_diag_fixed_20260528T2116Z`
- M-SURFACE source artifact: `artifacts/msurface/`
- M-SURFACE run: `experimental/outlier_migrate/phase9/results/om_phase9_msurface_granite_sanity_20260529T0032Z`
- Falcon branch diagnostic source: `artifacts/k_branch/`
- Final decision: `artifacts/final_path_decision.md`

No new GPU job was launched for residual KLLOOK or BranchRot in this gate. That absence is itself recorded in the corresponding `decision.json` files and should not be represented as a measured pass or measured kill.
