# OutlierMigrate Reviewer Pack

- status: Phase 0 dynamic-outlier gate passed; Phase 1 replicated at scale; not camera-ready
- current decision: `PASS_OM_PHASE1_REPLICATED_AT_SCALE`
- current paper readiness: not COLM/ICLR-ready; cross-model validation and audits required

## Paper Link

- Draft TeX: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`
- Draft PDF: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.pdf`

## Current Claim

OutlierMigrate now supports this narrow claim: on two same-family Granite
hybrid model sizes, top-1% high-magnitude activation channels at decode
position 100 are not rank-stationary by the final preregistered decode
position under the preregistered migration metric.

This is a strong positive-method finding, but it is not a camera-ready paper,
not cross-model evidence, and not yet evidence that a migration-aware
intervention improves quality, latency, memory, or robustness.

## Strongest Evidence

| Gate item | Phase 0 exact value | Phase 1 exact value | Decision relevance |
|---|---:|---:|---|
| Checker decision | `PASS_OM_PHASE0_DECODE_TIME_MIGRATION` | `PASS_OM_PHASE1_REPLICATED_AT_SCALE` | dynamic-outlier pass replicated at scale |
| Migration fraction | 0.8178385416666667 | 0.843165650406504 | clears threshold >= 0.05 |
| Bootstrap 95% CI | [0.797265625, 0.8368489583333334] | [0.8334349593495936, 0.8511432926829268] | CI lower > 0.05 |
| Preregistered dynamic threshold | point >= 0.05 and CI lower > 0.05 | point >= 0.05 and CI lower > 0.05 | preregistered rule |
| Artifact complete | true | true | artifact gate passed |
| Model | `ibm-granite/granite-4.0-h-tiny` | `ibm-granite/granite-4.0-h-small` | same-family scale validation |
| Model commit | `791e0d3d28c86e106c9b6e0b4cecdee0375b6124` | `b8c0982bab7fde4eb48110f5a069527c008fab39` | fixed snapshots |
| Prompt SHA | `sha256:2f27c54baa8448e033d6e82f53f775dc6abe38188e4f1e5c0b97e3c74fe7c1dd` | `sha256:aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e` | fixed prompt packets |
| Activation artifact SHA | `sha256:2783aa329d82e8f43ea4f342e52f0a25fc283b0b8971880f2da093739474d687` | `sha256:326a04351e0dda28cf919fe4745d7d9341011db13d940a06cf9c8470633003e1` | fixed activation artifacts |
| Trace count | 12 | 24 | Phase 1 doubles traces |
| Layer count | 40 | 40 | same layer count, larger hidden size in Phase 1 |
| Decode positions | 100, 500, 1000, 5000, 10000 | 100, 500, 1000, 5000, 10000, 20000 | Phase 1 extends decode surface |
| Bootstrap seed | 20260508 | 20260508 | recorded seed |

## Artifact Paths

- Phase 0 preregistration: `experimental/outlier_migrate/phase0/preregister_om_phase0.md`
- Phase 0 result packet: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z`
- Phase 0 checker output: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/checker_result.json`
- Phase 0 metrics: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/metrics.json`
- Phase 0 artifact completeness: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/artifact_check.json`
- Phase 0 model provenance: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/model_provenance.json`
- Phase 0 prompt manifest: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/prompt_manifest.json`
- Phase 0 activation manifest: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/activation_magnitude_manifest.json`
- Phase 0 activation artifact: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/activation_magnitudes.jsonl.gz`
- Phase 1 preregistration: `experimental/outlier_migrate/phase1/preregister_om_phase1.md`
- Phase 1 result packet: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z`
- Phase 1 checker output: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/checker_result.json`
- Phase 1 metrics: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/metrics.json`
- Phase 1 artifact completeness: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/artifact_check.json`
- Phase 1 model provenance: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/model_provenance.json`
- Phase 1 prompt manifest: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/prompt_manifest.json`
- Phase 1 activation manifest: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/activation_magnitude_manifest.json`
- Phase 1 activation artifact: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/activation_magnitudes.jsonl.gz`

## Related-Work Sources Added

- Kimi Linear primary source: `https://arxiv.org/abs/2510.26692`
- Qwen3.6 official model card: `https://huggingface.co/Qwen/Qwen3.6-35B-A3B`
- Gated Delta Networks primary source: `https://arxiv.org/abs/2412.06464`
- Gated Linear Attention primary source: `https://arxiv.org/abs/2312.06635`

Use these sources only for theoretical motivation and architectural context.
Do not claim that Kimi Linear, Qwen3.6, GLA, or RWKV-7 empirically exhibit
OutlierMigrate migration until those models are measured.

## Reviewer Risks

- Phase 0 and Phase 1 are same-family Granite evidence, not cross-model
  validation.
- The paper does not yet include a strict cross-family falsification pair,
  contamination audit, or independent seed repeat beyond the recorded bootstrap
  procedure.
- The result says outlier ranks migrate; it does not show that any
  migration-aware intervention improves quality, latency, memory, or robustness.
- The mechanism discussion is theoretical. Channel-wise gating, KDA,
  Qwen3.6-style Gated DeltaNet, and GLA-style mechanisms plausibly explain why
  top-magnitude channels can change over decode, but no unmeasured model should
  be described as empirically positive.
- Do not tune positions, top-channel fraction, rank-delta threshold, or model
  selection on these packets; the Phase 0 and Phase 1 surfaces were
  preregistered.

## Saturated / Alive / Next Branch

- saturated: Phase 0 and Phase 1 decision surfaces are closed and passed.
- alive: same-family dynamic outlier migration in Granite hybrid decode traces.
- promoted: the dynamic-outlier hypothesis within Granite-family hybrids.
- weakened: the static-outlier explanation on the Phase 0 and Phase 1 surfaces.
- not established: cross-model transfer, delta-rule linear-attention validation,
  RWKV-7/GLA generalization, or a positive intervention method.
- next exact gate: preregistered cross-model validation with audits, including
  delta-rule linear-attention hybrids such as Qwen3.6 and Kimi Linear, before
  making any submission-level method claim.
