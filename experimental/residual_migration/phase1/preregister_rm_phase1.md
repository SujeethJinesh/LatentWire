# Residual Migration Phase 1 Preregistration

**Frozen on**: 2026-05-07
**Gated by**: residual_migration_phase0 PASS (either path)

## Hypothesis

Phase 0 finding (whichever direction) replicates on Granite-4.0-H-Small
with layer-stratified ablation.

## Falsifiable predictions

For **replicates** Phase 0 path:
- **PASS**: AIME-2025 accuracy drop <1.5% on Granite-4.0-H-Small with
  bootstrap CI upper bound < 1.5%.
  Decision string: PASS_RM_PHASE1_REPLICATED_AT_SCALE
- **KILL**: drop ≥1.5%. Phase 0 was small-model artifact.
  Decision string: KILL_RM_PHASE1_FAILED_AT_SCALE

For **rejects-for-hybrids** Phase 0 path:
- **PASS**: drop >3% on Granite-4.0-H-Small with CI lower bound > 3%.
  Adds layer-stratified attribution: which layers drive the dependence?
  Decision string: PASS_RM_PHASE1_HYBRIDS_DEPEND_AT_SCALE
- **KILL**: drop ≤3%. Phase 0 negative result was small-model artifact.
  Decision string: KILL_RM_PHASE1_NEGATIVE_FAILED

## Model

- ibm-granite/granite-4.0-h-small

## Promptset

- 24 AIME-2025 traces (deterministic, 0-23)
- SHA-256: <COMMITTED TO RESULT PACKET>

## Procedure

Phase 0 procedure on the larger model, plus:
- Layer-stratified ablation: clip residual-stream outliers in only
  the first half, only the second half, only attention layers, only
  Mamba layers; record drop for each ablation set.
- This stratification is for attribution only; does not change the
  Phase 1 pass/kill decision (which is on the full ablation as in
  Phase 0).

## Forbidden inputs

- Must not change clipping threshold from 95th percentile.
- Must not select layer-stratification subsets based on observed drop.

## On pass

Append cross_model_validation_residual_migration to queue.yml.
Trigger paper drafting.

## On kill

Write experimental/KILLED_residual_migration_phase1_failed/README.md.
