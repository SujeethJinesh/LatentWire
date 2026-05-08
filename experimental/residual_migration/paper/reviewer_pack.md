# Residual Migration Reviewer Pack

- status: Phase 0 mechanical PASS after checker; not a paper-ready positive method
- current decision: `PASS_RM_PHASE0_RETHINKING_REPLICATES`
- current paper readiness: not COLM/ICLR-ready; Phase 1 headroom/oracle safeguards are required

## Paper Link

- Draft TeX: `experimental/residual_migration/paper/residual_migration_colm2026.tex`
- Draft PDF: `experimental/residual_migration/paper/residual_migration_colm2026.pdf`

## Current Claim

Residual Migration Phase 0 supports only this narrow gate claim: on the frozen
12-trace packet in
`experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z`,
the checker returned `PASS_RM_PHASE0_RETHINKING_REPLICATES` because the
bootstrap CI upper bound for accuracy drop was 0.0, below the preregistered
replicates threshold of 0.015.

This is not a capability-preservation result. Baseline accuracy was 0.0 and
ablation accuracy was 0.0, so the zero drop cannot distinguish a harmless
residual intervention from a no-headroom evaluation surface.

## Strongest Evidence

| Gate item | Exact value | Artifact path |
|---|---:|---|
| Checker decision | `PASS_RM_PHASE0_RETHINKING_REPLICATES` | `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/checker_result.json` |
| Artifact complete | true | `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/artifact_check.json` |
| Baseline accuracy | 0.0 | `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/metrics.json` |
| Ablation accuracy | 0.0 | `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/metrics.json` |
| Accuracy drop | 0.0 | `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/metrics.json` |
| Bootstrap 95% CI | [0.0, 0.0] | `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/bootstrap_ci.json` |
| Preregistered pass-replicates threshold | CI upper < 0.015 | `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/checker_result.json` |
| Model | `ibm-granite/granite-4.0-h-tiny` | `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/model_provenance.json` |
| Model snapshot commit | `791e0d3d28c86e106c9b6e0b4cecdee0375b6124` | `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/model_provenance.json` |
| Prompt SHA | `sha256:2f27c54baa8448e033d6e82f53f775dc6abe38188e4f1e5c0b97e3c74fe7c1dd` | `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/prompt_manifest.json` |
| Generation artifact SHA | `sha256:5ce5d78cd10ec8a5fe92502d23624919fa286900f3ca769fd4f6979457eee8ad` | `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/artifact_hashes.json` |
| Trace count | 12 | `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/metrics.json` |
| Generation length | frozen 2048 `max_new_tokens` | `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/command_metadata.json` |
| Ablation | every discovered transformer layer forward pre-hook; values above per-layer/per-token-position 95th percentile clipped to threshold preserving sign | `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/ablation_config.json` |

## Artifact Paths

- Phase 0 result packet: `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z`
- Checker output: `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/checker_result.json`
- Metrics: `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/metrics.json`
- Bootstrap CI: `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/bootstrap_ci.json`
- Artifact completeness: `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/artifact_check.json`
- Model provenance: `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/model_provenance.json`
- Prompt manifest: `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/prompt_manifest.json`
- Ablation config: `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/ablation_config.json`
- Artifact hashes: `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/artifact_hashes.json`
- Generation artifact: `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/generations.jsonl`
- Command metadata: `experimental/residual_migration/phase0/results/rm_phase0_20260508T161556Z/command_metadata.json`

## Reviewer Risks

- The Phase 0 checker PASS is mechanical and headroom-limited.
- Baseline accuracy was 0.0; the zero drop cannot support a capability-preservation claim.
- The packet is single-model and 12-trace only.
- There is no oracle/headroom diagnostic yet.
- There is no cross-family falsification pair yet.
- There is no latency, bytes, robustness, or deployable residual-quantization recipe claim.

## Saturated / Alive / Next Branch

- saturated: Phase 0 checker surface is closed and passed under the registered replicates rule.
- alive: residual-migration investigation as a gated branch.
- weakened: any immediate capability-preservation interpretation of Phase 0.
- not established: residual quantization, cross-model communication, quality preservation, latency or memory benefit.
- next exact gate: Phase 1 must include headroom/oracle safeguards before any residual-quantization recipe claim.
