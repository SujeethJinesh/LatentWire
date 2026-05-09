# Swarm Final Report Draft

Status: draft while `cross_model_validation_outlier_migrate` is in flight.
Do not treat this file as final until the Phase 2 checker result and committee
reviews are incorporated.

## Executive Status

- Primary positive-method candidate: OutlierMigrate.
- Safe fallback paper: ThoughtFlow-FP8 falsification methodology.
- Current active queue entry: `cross_model_validation_outlier_migrate`.
- Current run: `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z`.
- Current Phase 2 scope: partial Nemotron-3 validation only; Qwen3.6 and Kimi
  Linear are deferred by the no-vLLM-upgrade authorized window.
- Camera-ready final: none; human review required.
- Camera-ready candidate: none yet.

## Portfolio

| Project | Status | Paper posture | Next human-visible decision |
| --- | --- | --- | --- |
| OutlierMigrate | Phase 0 PASS, Phase 1 PASS, Phase 2 Nemotron-3 partial run in flight | Positive-method candidate, not candidate-ready yet | Branch on Phase 2 checker result |
| ThoughtFlow-FP8 | Paper-polish gate PASS/buildable | Falsification-methodology fallback, not final | Human copyedit and venue-framing review |
| HybridKernel | KILL_HYBRIDKERNEL_BELOW_SHELF | No paper | Preserve artifacts; diagnostic only |
| Decode Microkernel | Phase 0/1 PASS, Phase 2 FAIL_INFRA | Deferred engineering integration | Human decides whether to fund real serving integration |
| Residual Migration | Phase 0 PASS, Phase 1 KILL_RM_PHASE1_FAILED_AT_SCALE | No paper | Preserve artifacts; diagnostic only |
| SSM-State Lifecycle | KILL_SSML_PHASE0_STATE_STABLE | No paper | Preserve artifacts; diagnostic only |
| SSM Shape-Conditioned Codec | KILL_SSC_PHASE0_NO_CODEC_GAIN | No paper | Preserve artifacts; no further mining |
| Cross-Layer Error | KILL_CLE_BOUND_LOOSE | No paper | Preserve artifacts; diagnostic only |

## OutlierMigrate Evidence

Granite-family gates:

- Phase 0 run: `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z`.
- Phase 0 decision: `PASS_OM_PHASE0_DECODE_TIME_MIGRATION`.
- Phase 0 migration fraction: `0.8178385416666667`, CI95
  `[0.797265625, 0.8368489583333334]`.
- Phase 1 run: `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z`.
- Phase 1 decision: `PASS_OM_PHASE1_REPLICATED_AT_SCALE`.
- Phase 1 migration fraction: `0.843165650406504`, CI95
  `[0.8334349593495936, 0.8511432926829268]`.

Strict set-membership decomposition:

- Phase 0: strict set-leaving `0.634244791667`, within-set rank shuffling
  `0.175260416667`, original migration fraction `0.817838541667`.
- Phase 1: strict set-leaving `0.566234756098`, within-set rank shuffling
  `0.270934959350`, original migration fraction `0.843165650407`.
- Report paths:
  `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/migration_decomposition.md`
  and
  `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/migration_decomposition.md`.

Phase 2 status:

- Active model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Deferred models: `Qwen/Qwen3.6-35B-A3B` and
  `moonshotai/Kimi-Linear-48B-A3B-Instruct`.
- Phase 2 checker result: pending.
- Required branch after checker:
  - PASS: update paper with Nemotron-3 partial cross-family confirmation and
    keep Qwen3.6/Kimi deferred.
  - KILL or ambiguous: demote paper to Granite-family characterization and do
    not claim cross-architectural validation.
  - INFRA: write a blocked note under the no-upgrade policy.

## Killed Branches

Kill manifests exist and now include explicit non-publication rationale:

- `experimental/KILLED_hybridkernel_below_shelf/README.md`
- `experimental/KILLED_residual_migration_phase1_failed/README.md`
- `experimental/KILLED_ssm_lifecycle_state_stable/README.md`
- `experimental/KILLED_ssm_shape_codec_no_gain/README.md`
- `experimental/KILLED_cross_layer_error_bound_loose/README.md`

## GPU Hours And Cost

- `swarm/state.json` recorded `gpu_hours_used=9.84` before the active
  Nemotron-3 partial run.
- Active run started at `2026-05-08T23:17:27Z`.
- Final GPU-hour accounting will be updated after the checker-backed Phase 2
  outcome.
- Cost estimate formula: total GPU hours × `$1.89/hr`.

## Committee Status

- OutlierMigrate latest committee scores before Phase 2 were below the
  camera-ready-candidate threshold; further review is required after the Phase
  2 branch update.
- ThoughtFlow-FP8 is buildable and reviewer-pack-current, but still needs
  human final framing review.

## Human Decisions On Landing

1. Review the Phase 2 Nemotron-3 checker outcome and the corresponding paper
   framing.
2. Decide whether to permit a vLLM upgrade or alternate runtime for Qwen3.6
   and Kimi Linear validation.
3. Decide whether OutlierMigrate is submission-track as a partial
   cross-family positive-method paper, a Granite-family characterization, or
   a hold for stronger validation.
4. Decide whether ThoughtFlow-FP8 should be submitted as the safe fallback
   falsification paper.
5. Decide whether Decode Microkernel merits real serving-integration
   engineering after its Phase 2 infra block.

## Reproducibility Scaffold

OutlierMigrate must cite:

- commit SHA after final Phase 2 commit;
- RTX PRO 6000 Blackwell GPU;
- exact runtime hours;
- model snapshot commits from `model_provenance.json`;
- prompt SHAs from `prompt_manifest.json`;
- exact runner and checker commands;
- `artifact_check.json` paths for every claimed number.

ThoughtFlow-FP8 must cite:

- commit SHA after final paper-polish commit;
- build command and PDF path;
- reviewer pack path;
- all result packet paths used by paper claims.
