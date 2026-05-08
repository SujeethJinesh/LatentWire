# OutlierMigrate Reviewer Pack

- status: Phase 0 dynamic-outlier gate passed; not camera-ready
- current decision: `PASS_OM_PHASE0_DECODE_TIME_MIGRATION`
- current paper readiness: not COLM/ICLR-ready; Phase 1 scale validation required

## Paper Link

- Draft TeX: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`
- Draft PDF: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.pdf`

## Current Claim

OutlierMigrate Phase 0 supports only this narrow claim: on
`ibm-granite/granite-4.0-h-tiny`, top-1% high-magnitude activation channels at
decode position 100 are not rank-stationary by decode position 10000 under the
preregistered migration metric.

This is not a camera-ready paper, not a full positive-method claim, and not
cross-model or Phase 1 evidence.

## Strongest Evidence

| Gate item | Exact value | Decision relevance |
|---|---:|---|
| Phase 0 checker decision | `PASS_OM_PHASE0_DECODE_TIME_MIGRATION` | dynamic-outlier pass |
| Migration fraction | 0.8178385416666667 | clears threshold >= 0.05 |
| Bootstrap 95% CI | [0.797265625, 0.8368489583333334] | CI lower > 0.05 |
| Dynamic pass threshold | point >= 0.05 and CI lower > 0.05 | preregistered rule |
| Model commit | `791e0d3d28c86e106c9b6e0b4cecdee0375b6124` | fixed snapshot |
| Model | `ibm-granite/granite-4.0-h-tiny` | Phase 0 only |
| Trace count | 12 deterministic AIME-2025 traces | small screen |
| Decode positions | 100, 500, 1000, 5000, 10000 | fixed before analysis |
| Top-channel rule | top 1% at decode position 100 | fixed before analysis |

## Artifact Paths

- Phase 0 preregistration: `experimental/outlier_migrate/phase0/preregister_om_phase0.md`
- Phase 0 result packet: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z`
- Checker output: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/checker_result.json`
- Metrics: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/metrics.json`
- Artifact completeness: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/artifact_check.json`
- Model provenance: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/model_provenance.json`
- Prompt manifest: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/prompt_manifest.json`
- Activation artifact: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/activation_magnitudes.jsonl.gz`

## Reviewer Risks

- Phase 0 is a small screen, not a validated method.
- There is no cross-model, same-family scale, or cross-family falsification
  evidence in this packet.
- The result says outlier ranks migrate; it does not show that any
  migration-aware intervention improves quality, latency, memory, or robustness.
- The current uncertainty is bootstrap over 12 traces and should not be treated
  as final paper-strength evidence.
- Do not tune positions, top-channel fraction, or rank-delta threshold on this
  packet; those were preregistered for this gate.

## Saturated / Alive / Next Branch

- saturated: the Phase 0 decision surface is closed and passed.
- alive: dynamic outlier migration in Granite hybrid decode traces.
- weakened: the static-outlier explanation on this exact Phase 0 surface.
- not established: cross-model transfer, Phase 1 scale stability, or a positive
  intervention method.
- next exact gate: run the preregistered Phase 1 scale validation for
  OutlierMigrate before making any submission-level method claim.
