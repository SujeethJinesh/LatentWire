# KILLED: HBSM Sensitivity Heterogeneity

Date killed: 2026-05-07

Evidence reviewed at head: `5c31c3685b11502eb3d72c514602fe0019c8b980`

## What Was Tried

HBSM tested whether sensitivity in current hybrid reasoners is enriched at
hybrid boundary layers and whether cheaper no-forward-pass predictors can
forecast that sensitivity.

The live B1 scouts perturbed Granite Tiny layer outputs under MXFP4-style noise
and checked whether boundary layers were overrepresented among the measured
top-sensitivity layers.

## Why It Died

The preregistered B1 sensitivity-heterogeneity evidence went the wrong way:

- two-prompt packet:
  `experimental/shared/results/hbsm_prompt2_sensitivity_20260507/hbsm_gate_packet/`
- decision: `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`
- Fisher p-value: `1.0`
- boundary top-decile count: `0`
- non-boundary top-decile count: `1`
- cheap-predictor Spearman: `-0.667`

The one-prompt smoke was also weak, with cheap-predictor Spearman `-0.476`.
This kills HBSM as an active COLM mechanism or sensitivity-discovery branch
under the current broad hypothesis. Do not scale B1 or send it to GPU as-is.

## Audit Trail

Primary stop manifest:

- `experimental/hbsm/phase2/b1_prompt2_repro_manifest_20260507.md`

Reviewer pack:

- `experimental/hbsm/paper/reviewer_pack.md`

## Salvage Value

The scaffold remains useful for future work:

- B1 row-packet builder and checker
- measured top-decile derivation from raw drift rows
- KL/outlier/predictor comparator controls
- negative/control appendix evidence for hybrid sensitivity claims

## Revival Condition

Revival requires a new preregistered narrower mechanism hypothesis before any
new rows are inspected. The new hypothesis must explain why both the one-prompt
and two-prompt Granite Tiny B1 scouts should reverse, and resource-limited
packets must remain non-promotable.

