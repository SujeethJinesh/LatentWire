# OutlierMigrate Committee Review: Phase 2 Partial Pass Round 2

Date: 2026-05-09

Reviewed files:

- `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`
- `experimental/outlier_migrate/paper/reviewer_pack.md`
- `experimental/outlier_migrate/paper/committee_reviews/20260509_phase2_partial_pass_round1.md`
- `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/metrics.json`
- `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/artifact_check.json`
- `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/migration_decomposition.md`

## COLM Area Chair

Score: 7/10.

The revised draft no longer overclaims: it clearly frames OutlierMigrate as
an observational measurement candidate, not a positive-method or full
cross-architectural validation paper. The Phase 2 Nemotron-3 partial pass is
reported honestly, including `full_cross_validation_complete=false`, deferred
Qwen3.6/Kimi validation, and exact strict decomposition numbers. The three
requested metrics appear in the results: strict set-leaving, within-set rank
shuffling, and original migration fraction.

The strongest contribution is now clear: static top-1% channel protection is
empirically unstable in long decode on Granite and partially transfers to
Nemotron-3, with strict set-leaving 0.533713200380 on Nemotron-3. The draft
also appropriately treats QMamba/OuroMamba as prior dynamic-outlier evidence
rather than pretending this is first-of-kind.

Camera-ready candidate blockers remain. The paper needs either an accepted
measurement-paper framing despite the project positive-method goal, or a
follow-on migration-aware intervention / completed Qwen3.6/Kimi validation.
Also, the requested file name `profiler_metrics.json` is absent; the packet
uses `metrics.json`, which is artifact-checked, but the naming mismatch should
be cleaned up in review materials.

## MLSys Reviewer

Score: 6/10.

Reproducibility is strong for the measured claim: the packet includes model
provenance, prompt SHA, activation artifact SHA, command metadata, seeds,
logs, metrics, bootstrap CI, and artifact completeness. The strict
set-leaving decomposition is exactly the right systems-relevant readout for
static channel-protection methods, and the paper now separates it from the
original conflated migration metric.

Systems contribution is still thin. The paper motivates why static protection
could fail, but does not implement a protection refresh schedule, quantizer,
routing policy, kernel, latency/memory tradeoff, or accuracy-preserving
intervention. The revised draft does not pretend otherwise, which is good, but
it means this is not yet a systems positive-method paper.

Fixable issues before camera-ready candidate: yes. Add an explicit note that
the strict decomposition is post-hoc interpretability, not the preregistered
gate metric; verify all 2025/2026 citations; and resolve the
`profiler_metrics.json` vs `metrics.json` naming mismatch in reviewer-facing
materials.

## Adversarial Reviewer

Score: 6/10.

The revised draft avoids the main overclaim. It does not claim full
cross-architectural validation, does not claim Kimi/Qwen3.6 exhibit migration,
and does not claim a positive method. The limitations are appropriately sharp.

Remaining attacks: the pass threshold is very low relative to observed
effects, so reviewers may ask why 5% was meaningful. The strict set-leaving
analysis is post-hoc, so it must remain labeled as interpretability, not
preregistered success evidence. Nemotron-3 is helpful but does not replace
Qwen3.6/Kimi, especially because the discussion leans on gated delta/KDA-style
mechanisms. No intervention means the static-protection implication remains a
motivation, not demonstrated failure of existing methods.

Explicit answers:

- Overclaim full cross-architectural validation or positive-method status: no.
- Fixable issues before camera-ready candidate: yes.
- Stop condition fires: no p-hacking, no cherry-picking, no preregistration
  amendment scope creep, and no evidence of a previously passing paper test
  failing.

## Follow-Up Applied Before Commit

- The paper and reviewer pack now state that strict set-leaving is post-hoc
  interpretability, not the preregistered gate metric.
- The reviewer-facing materials now state that OutlierMigrate Phase 2 emits
  `metrics.json`, not a separate `profiler_metrics.json`.
- These wording fixes do not make the paper a camera-ready candidate because
  the missing intervention and deferred Qwen3.6/Kimi validation remain
  substantive blockers.
