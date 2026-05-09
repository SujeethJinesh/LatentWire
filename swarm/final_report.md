# Swarm Final Report Draft

Status: authorized-window landing report draft after partial Nemotron-3 PASS
and ThoughtFlow fallback-candidate audit. Do not treat this as human-approved
final; it is the machine-readable handoff for the returning human.

## Executive Status

- Primary positive-method candidate: OutlierMigrate.
- Safe fallback paper: ThoughtFlow-FP8 falsification methodology.
- Current active work: OutlierMigrate Phase 3 intervention sprint.
- Phase 3 preregistration commit: `c0031574`.
- Phase 3 runner/checker commit: `fc394bcb`.
- Completed run: `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z`.
- Phase 2 scope: partial Nemotron-3 validation only; Qwen3.6 and Kimi
  Linear are deferred by the no-vLLM-upgrade authorized window.
- Camera-ready final: none; human review required.
- Camera-ready candidate: ThoughtFlow-FP8 fallback candidate only, under the
  falsification-methodology workshop framing. It is not a positive-method
  candidate.
- Candidate commit: `e25a45d6c56151a31ef7f788d3d8d515eb46d649` for the
  ThoughtFlow fallback candidate review/audit packet.

## Portfolio

| Project | Status | Paper posture | Next human-visible decision |
| --- | --- | --- | --- |
| OutlierMigrate | Phase 0 PASS, Phase 1 PASS, partial Phase 2 Nemotron-3 PASS | Positive-method candidate, not candidate-ready yet | Committee review and human decision on deferred cross-validation |
| ThoughtFlow-FP8 | Paper-polish gate PASS/buildable | Falsification-methodology fallback, not final | Human copyedit and venue-framing review |
| HybridKernel | KILL_HYBRIDKERNEL_BELOW_SHELF | No paper | Preserve artifacts; diagnostic only |
| Decode Microkernel | Phase 0/1 PASS, Phase 2 FAIL_INFRA | Deferred engineering integration | Human decides whether to fund real serving integration |
| Residual Migration | Phase 0 PASS, Phase 1 KILL_RM_PHASE1_FAILED_AT_SCALE | No paper | Preserve artifacts; diagnostic only |
| SSM-State Lifecycle | KILL_SSML_PHASE0_STATE_STABLE | No paper | Preserve artifacts; diagnostic only |
| SSM Shape-Conditioned Codec | KILL_SSC_PHASE0_NO_CODEC_GAIN | No paper | Preserve artifacts; no further mining |
| Cross-Layer Error | KILL_CLE_BOUND_LOOSE | No paper | Preserve artifacts; diagnostic only |

## OutlierMigrate Evidence

Phase 3 intervention status:

- Preregistration:
  `experimental/outlier_migrate/phase3/preregister_om_phase3_intervention.md`
  committed and pushed at `c0031574` before any Phase 3 quantization run.
- Runner/checker/grid-sensitivity tooling committed and pushed at `fc394bcb`.
- No Phase 3 intervention result has been observed yet.
- Current GPU-hour cap for this sprint: cumulative `40`.

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
- Interpretation guard: strict set-leaving is post-hoc interpretability for
  static channel-protection relevance; it is not the preregistered gate
  criterion.
- Report paths:
  `experimental/outlier_migrate/phase0/results/om_phase0_20260508T011824Z/migration_decomposition.md`
  and
  `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/migration_decomposition.md`.

Phase 2 status:

- Active model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Deferred models: `Qwen/Qwen3.6-35B-A3B` and
  `moonshotai/Kimi-Linear-48B-A3B-Instruct`.
- Phase 2 checker result:
  `PARTIAL_PASS_OM_PHASE2_NEMOTRON3_ONLY_QWEN36_KIMI_DEFERRED`.
- Phase 2 migration fraction: `0.820809591642925`, CI95
  `[0.7865325261158594, 0.8544931149097815]`.
- Phase 2 strict set-leaving: `0.533713200379867`, CI95
  `[0.4674145299145299, 0.5999821937321937]`.
- Phase 2 within-set shuffling: `0.26908238366571696`, CI95
  `[0.23884140550807217, 0.2990562678062678]`.
- Interpretation: Nemotron-3 confirms the rank-migration signal under a
  partial cross-family check. It is not full cross-validation because
  Qwen3.6/Kimi are deferred.

## Killed Branches

Kill manifests exist and now include explicit non-publication rationale:

- `experimental/KILLED_hybridkernel_below_shelf/README.md`
- `experimental/KILLED_residual_migration_phase1_failed/README.md`
- `experimental/KILLED_ssm_lifecycle_state_stable/README.md`
- `experimental/KILLED_ssm_shape_codec_no_gain/README.md`
- `experimental/KILLED_cross_layer_error_bound_loose/README.md`

Verified on 2026-05-09: each listed manifest contains the decision string,
date, measured value/threshold summary, artifact SHA references, and a
non-publication rationale paragraph.

## GPU Hours And Cost

- `swarm/state.json` recorded `gpu_hours_used=9.84` before the active
  Nemotron-3 partial run.
- Active run started at `2026-05-08T23:17:27Z`.
- Active run completed at `2026-05-09T06:40:49Z`.
- Successful-run delta: `7.3894` GPU-hours.
- Updated `swarm/state.json` GPU hours: `17.2294`.
- Cost estimate at updated state: `17.2294 * $1.89/hr = $32.56`.

## Committee Status

- OutlierMigrate Phase 2 second committee review:
  `experimental/outlier_migrate/paper/committee_reviews/20260509_phase2_partial_pass_round2.md`.
  Scores: COLM `7/10`, MLSys `6/10`, adversarial `6/10`.
- No stop condition fired. The revised draft does not overclaim full
  cross-architectural validation or positive-method status.
- OutlierMigrate is not a camera-ready candidate: fixable review concerns
  remain, and the core blocker is substantive rather than wording-only
  (missing intervention plus deferred Qwen3.6/Kimi validation).
- ThoughtFlow-FP8 additional polish review:
  `experimental/thoughtflow_fp8/paper/committee_reviews/20260509_polish_round2.md`.
  Scores under falsification-methodology framing: COLM `7/10`, MLSys `7/10`
  as a diagnostic note (`3/10` as systems), adversarial `7/10`.
- ThoughtFlow-FP8 local checker returned `PASS_THOUGHTFLOW_PAPER_BUILDABLE`;
  owned tests passed with `70 passed, 1 warning`.
- ThoughtFlow-FP8 reproducibility audit:
  `experimental/thoughtflow_fp8/paper/committee_reviews/20260509_reproducibility_audit.md`.
  It passes the local fallback-candidate checks.
- ThoughtFlow-FP8 remains not camera-ready final. It is now marked as a
  fallback camera-ready candidate only as a falsification-methodology workshop
  note, pending human title/framing/citation review. It does not count as a
  positive-method candidate.

## Human Decisions On Landing

1. Review the partial Phase 2 Nemotron-3 PASS and the corresponding paper
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

- Phase 2 packet commit SHA:
  `3e2fdc8d3adbfbf67da6b02cda16bb0f58e2229b`;
- RTX PRO 6000 Blackwell GPU;
- exact runtime hours;
- model snapshot commits from `model_provenance.json`;
- prompt SHAs from `prompt_manifest.json`;
- exact runner and checker commands;
- `artifact_check.json` paths for every claimed number.

ThoughtFlow-FP8 must cite:

- fallback candidate commit SHA:
  `e25a45d6c56151a31ef7f788d3d8d515eb46d649`;
- build command and PDF path;
- reviewer pack path;
- all result packet paths used by paper claims.

## Phase 3 Intervention Sprint Status (2026-05-09)

Human authorized a 10-12 hour OutlierMigrate Phase 3 intervention sprint to
convert the paper from characterization toward a positive-method claim.

Completed before the first Phase 3 quantization run:

- Phase 3 preregistration authored and pushed:
  `experimental/outlier_migrate/phase3/preregister_om_phase3_intervention.md`
  at commit `c0031574`.
- Phase 3 core runner/checker/test tooling authored and pushed at commit
  `fc394bcb`.
- Sprint state/progress note pushed at commit `cda8cc92`.

Active:

- Core Granite-4.0-H-Tiny Phase 3 intervention run:
  `experimental/outlier_migrate/phase3/results/om_phase3_20260509T174000Z`.
- Started at `2026-05-09T17:38:58Z`.
- First activation-capture batch completed at `2026-05-09T17:47:43Z`.
- No Phase 3 decision metrics have been produced yet.

Completed no-GPU analysis:

- Layer-stratified migration analysis generated from existing Phase 0/1/2
  packets:
  `experimental/outlier_migrate/phase3/results/layer_stratified_migration.md`.
- The analysis reports strict set-leaving, within-set rank shuffling, and
  original migration by layer type.
- Key layer-type means:
  - Phase 0 Granite-Tiny: attention strict set-leaving `0.618490`, SSM/Mamba
    `0.635995`; original migration `0.798177` / `0.820023`.
  - Phase 1 Granite-Small: attention strict set-leaving `0.557165`,
    SSM/Mamba `0.567243`; original migration `0.843242` / `0.843157`.
  - Phase 2 Nemotron-3 partial: attention strict set-leaving `0.563014`,
    MoE `0.526503`, SSM/Mamba `0.533280`; original migration `0.832562` /
    `0.820384` / `0.818170`.

Pending:

1. Let the core Phase 3 intervention run complete.
2. Run `experimental/outlier_migrate/phase3/check_phase3_intervention.py` on
   the packet.
3. Stop and block if either mandatory control outperforms union protection by
   more than `0.10` median recovery.
4. Integrate Phase 3 outcome into the paper and run committee review.
