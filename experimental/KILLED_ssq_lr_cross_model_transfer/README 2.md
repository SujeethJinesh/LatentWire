# KILLED: SSQ-LR Cross-Model Transfer

Date killed: 2026-05-07

Evidence reviewed at head: `5c31c3685b11502eb3d72c514602fe0019c8b980`

## What Was Tried

SSQ-LR tested whether recurrent SSM state in hybrid reasoners could be
quantized below FP16 with a frozen, no-retuning recipe that preserved reasoning
quality and reduced counted state memory.

The live candidate was the frozen `mixed_int3_mxfp4_low_error_25pct` recipe on
layers `0,30`, selected on Granite Tiny after Mac-local S1/S2 gates and then
tested for S3 transfer to Granite 350M.

## Why It Died

The preregistered S3 transfer contract failed:

- frozen Granite Tiny recipe: `mixed_int3_mxfp4_low_error_25pct` on layers
  `0,30`
- transfer model: `ibm-granite/granite-4.0-h-350m`
- result: the transfer replay selected an INT8 fallback at only `1.984x`,
  below the state-memory gate
- layer-0 rescue diagnostics also failed two-model S3: recipes that pass one
  Granite model do not pass the other without retuning

This kills SSQ-LR as an active COLM positive-method branch under the current
preregistered hypothesis. Do not GPU-promote the current recipe.

## Audit Trail

Primary stop manifest:

- `experimental/ssq_lr/phase2/s3_transfer_repro_manifest_20260507.md`

Reviewer pack:

- `experimental/ssq_lr/paper/reviewer_pack.md`

## Salvage Value

The scaffold remains useful for future work:

- strict S1/S2/S3 packet contracts
- recurrent-state tensor provenance checks
- byte-accounting and same-byte controls
- local Granite Tiny and Granite 350M replay harnesses

## Revival Condition

Revival requires a new preregistration before any new rows are inspected. The
new branch must freeze a fresh held-out prompt surface, a single layer-selection
rule, a single recipe-selection rule, the counted `>=4x` state-memory
threshold, and the no-retuning S3 transfer contract.

