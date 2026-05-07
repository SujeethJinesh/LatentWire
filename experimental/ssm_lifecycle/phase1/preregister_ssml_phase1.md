# SSM-State Lifecycle Phase 1 Preregistration

**Frozen on**: 2026-05-07
**Gated by**: ssm_lifecycle_phase0 PASS

## Hypothesis

Phase 0 state-aging finding replicates on Granite-4.0-H-Small with
longer traces and admits an age-conditioned compression recipe.

## Falsifiable predictions

- **PASS**: KS p<0.01 on ≥50% of Mamba layers AND drift ratio ≥2× on
  ≥50% of Mamba layers in Granite-4.0-H-Small with traces extended to
  decode position 50000. Decision string:
  PASS_SSML_PHASE1_STATE_AGES_AT_SCALE
- **KILL**: thresholds not met. Decision string:
  KILL_SSML_PHASE1_STATE_STABLE_AT_SCALE

## Model

- ibm-granite/granite-4.0-h-small

## Promptset

- 24 AIME-2025 traces (deterministic, 0-23)
- Decode positions: {100, 500, 1000, 5000, 10000, 25000, 50000}
- SHA-256: <COMMITTED TO RESULT PACKET>

## Procedure

Phase 0 procedure scaled up. Add age-conditioned compression recipe
prototype:
- For each Mamba layer with confirmed aging, compress state at
  position >5000 to 4-bit; keep state at position ≤5000 at 8-bit.
- Measure AIME-2025 accuracy with this recipe vs full BF16 baseline.
- Recipe must achieve <2% accuracy drop OR is reported as failed
  recipe (paper still valid as characterization).

## Forbidden inputs

- Must not change KS p-value or drift threshold from Phase 0.
- Must not change the position grid post-hoc.

## On pass

Append cross_model_validation_ssm_lifecycle to queue.yml.
Trigger paper drafting (characterization + recipe).

## On kill

Write experimental/KILLED_ssm_lifecycle_phase1_failed/README.md with
explicit note distinguishing this kill from SSQ-LR's kill.
