# OutlierMigrate Phase 1 Preregistration

**Frozen on**: 2026-05-07
**Gated by**: outlier_migrate_phase0 PASS (either path)

## Hypothesis

Phase 0 finding (whichever direction) replicates on a larger hybrid
model with formal statistical testing.

## Falsifiable predictions

For **dynamic** Phase 0 path:
- **PASS**: ≥5% migration on Granite-4.0-H-Small with bootstrap CI
  lower bound > 5%, replicating Phase 0 effect direction.
  Decision string: PASS_OM_PHASE1_REPLICATED_AT_SCALE
- **KILL**: <5% migration at scale.
  Decision string: KILL_OM_PHASE1_FAILED_AT_SCALE

For **static** Phase 0 path (negative result):
- **PASS**: <5% migration on Granite-4.0-H-Small confirms static-outliers
  finding at scale.
  Decision string: PASS_OM_PHASE1_STATIC_AT_SCALE
- **KILL**: ≥5% migration at scale. Phase 0 was a small-model artifact;
  scale flips the finding.
  Decision string: KILL_OM_PHASE1_NEGATIVE_FAILED

## Model

- ibm-granite/granite-4.0-h-small

## Promptset

- Source: AIME-2025
- Count: 24 traces (scaled from 12)
- Selection: deterministic, prompts 0-23
- SHA-256: <COMMITTED TO RESULT PACKET>

## Procedure

Identical to Phase 0, but on the larger model with twice the traces
and decode positions extended to {100, 500, 1000, 5000, 10000, 20000}.

## Forbidden inputs

- Must not condition on Phase 0 numerical magnitude when interpreting
  Phase 1 results.
- Must not change procedure beyond what's specified above.

## On pass

Append cross_model_validation_outlier_migrate to queue.yml.
Trigger paper drafting.

## On kill

Write experimental/KILLED_outlier_migrate_phase1_failed/README.md.
The Phase 0 result is still publishable as a small-model finding but
not as a generalizable claim about hybrid reasoners.
