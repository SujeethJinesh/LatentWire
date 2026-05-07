# OutlierMigrate Phase 0 Preregistration

**Frozen on**: 2026-05-07
**Frozen by**: human author (before any data observed)

## Hypothesis

In current 2026 hybrid Mamba-Transformer reasoning models (specifically
Granite-4.0-H-Tiny), the channels carrying high-magnitude activations
during decode are not stationary. Outlier channels migrate across decode
positions in long reasoning traces.

## Falsifiable predictions

Two valid pass paths and one kill path:

- **PASS (dynamic outliers)**: ≥5% of top-1% high-magnitude channels
  migrate >2 channel positions between decode position 100 and 10000.
  Decision string: PASS_OM_PHASE0_DECODE_TIME_MIGRATION
- **PASS (static outliers)**: <5% migrate. Negative result; outliers are
  static; existing static-protection recipes are validated for hybrids.
  Decision string: PASS_OM_PHASE0_STATIC_OUTLIERS
- **KILL**: 1-5% migration (effect too small to characterize cleanly).
  Decision string: KILL_OM_PHASE0_AMBIGUOUS_EFFECT_SIZE

## Model

- ibm-granite/granite-4.0-h-tiny

## Promptset

- Source: AIME-2025
- Count: 12 reasoning traces
- Selection: deterministic, prompts indexed 0-11 from canonical AIME-2025
  release (HF: ibm-research/AIME-2025 or equivalent fixed source)
- SHA-256 of concatenated prompts: <COMPUTED BY SHARED DUMP PASS, COMMITTED
  TO RESULT PACKET BEFORE ANALYSIS>

## Procedure

1. Run experimental/shared/run_phase0_branch.py --branch outlier_migrate
2. The shared dump pass produces per-channel activation magnitudes at every
   transformer layer at decode positions {100, 500, 1000, 5000, 10000} for
   each of the 12 traces.
3. For each layer, identify the top-1% channels by mean magnitude at
   position 100. Track their channel-magnitude rank at positions
   {500, 1000, 5000, 10000}.
4. Compute migration metric: fraction of top-1% channels at position 100
   whose rank at position 10000 differs by more than 2 positions.
5. Aggregate across layers, weighting equally.
6. The checker reports the aggregated migration fraction and applies the
   decision rule above.

## Statistical readout

- Bootstrap (n=1000) over the 12 traces to compute 95% CI on the
  migration fraction.
- Pass threshold (≥5%) requires CI lower bound > 5%.
- Kill threshold (1-5%) is on the point estimate.
- Static-pass threshold (<5%) requires CI upper bound < 5%.

## Forbidden inputs

- Must not condition on any results from other Phase 0 branches.
- Must not exclude traces post-hoc based on observed migration.
- Must not change the {100, 500, 1000, 5000, 10000} position grid after
  observing data.

## On pass (dynamic)

Append outlier_migrate_phase1 to queue.yml. Phase 1 scales to
Granite-4.0-H-Small with formal cross-trace testing.

## On pass (static)

Append outlier_migrate_phase1_negative to queue.yml. Phase 1 confirms
static finding at scale and frames the negative-result paper.

## On kill

Write experimental/KILLED_outlier_migrate_ambiguous/README.md with the
full kill manifest. No paper.
