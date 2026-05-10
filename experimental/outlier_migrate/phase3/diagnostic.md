# OutlierMigrate Phase 3 Diagnostic

Date: 2026-05-10
Run: `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z`
Checker: `experimental/outlier_migrate/phase3/check_phase3_intervention.py`
Decision: `KILL_OM_PHASE3_INTERVENTION_FAILS`

## Summary

The preregistered migration-aware static protection intervention did not pass.
The result packet is artifact-complete, but the primary union-protection regime
had median recovery `0.000000000000` with bootstrap 95% CI
`[0.000000000000, 0.711143244199]`. This violates the pass rule
(`median recovery >= 0.50` and `CI lower > 0.30`) and triggers the kill rule
because median recovery is below `0.20`.

The mandatory controls did not trigger the human stop condition. Static 2%
protection also had median recovery `0.000000000000`, and magnitude-average
protection had median recovery `0.061276118929`. Magnitude averaging exceeded
union by `0.061276118929`, below the stop threshold of `0.10`.

## What Failed

The Phase 3 hypothesis was that replacing position-100-only protection with a
union of top-1% channels across decode positions `{100, 1000, 5000, 10000}`
would recover a measurable fraction of the BF16-vs-static-protection
perplexity gap at decode position 10000. The packet shows that this simple
static-set intervention is not robust:

- Primary union protection: median recovery `0.000000000000`.
- Static 2% matched-budget control: median recovery `0.000000000000`.
- Magnitude-average control: median recovery `0.061276118929`.
- Sparse grid `{100, 5000, 10000}`: median recovery `0.000000000000`.
- Dense grid `{100, 500, 1000, 2000, 5000, 7500, 10000}`: median recovery
  `0.000000000000`.

Several individual traces have positive recovery, but the median is zero
across all union grid variants. The method therefore cannot support a
positive-method claim in its current static-union form.

## Mechanistic Interpretation

The earlier Phase 0/1/2 migration packets show that most of the original
rank-migration signal is strict set-leaving: channels that begin inside the
top-1% protected set later leave that set. That finding remains useful for
diagnosing why static position-100 protection is brittle.

Phase 3 shows that simply protecting the union of migrated channels is not
sufficient. The likely failure modes are:

- Per-channel membership is not the only relevant quantity; scale calibration
  and within-set rank changes also matter.
- The protected union may include the right channels but use stale or
  mismatched quantization scales at the scoring position.
- Layer-local effects matter: layer-stratified analysis shows migration in
  attention and SSM/Mamba layers for Granite, and attention, SSM/Mamba, and MoE
  layers for Nemotron-3.
- Some traces have little recoverable static gap under the simple symmetric
  INT4 setup, limiting the possible median recovery.

This supports reframing the paper as a characterization plus negative
intervention result: decode-time outlier migration is real and cross-family in
the measured packets, but a static union protection set is not enough to make a
defensible positive method.

## Paper Consequence

The paper should not claim migration-aware static protection as a successful
intervention. Section 5 should present the preregistered negative result and
the mandatory controls. The discussion should state that adaptive online
schemes, such as periodic refresh of protection sets or decode-position-aware
scale updates, are plausible future directions but were not authorized or
validated in this sprint.

Conditional Phase 3 follow-ups are skipped under the sprint rules:

- Nemotron-3 intervention check: skipped because Phase 3 did not pass on
  Granite-4.0-H-Tiny.
- Within-set rank-shuffling pilot intervention: skipped because Phase 3 did
  not pass.
- Decode-length scaling curve: skipped because Phase 3 did not pass with
  strong recovery.
