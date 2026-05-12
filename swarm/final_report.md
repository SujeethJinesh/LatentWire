# Swarm Final Report Draft

Status: Phase 5' landing report draft after `DYNAMIC_REGIME_TRANSFORMER`.
Do not treat this as human-approved final; it is the machine-readable handoff
for the returning human.

## Executive Status

- Primary positive-method candidate: none after Phase 4.
- Safe fallback paper: ThoughtFlow-FP8 falsification methodology.
- Current active work: Experiment D decomposition analysis, followed by
  Experiment E threshold sensitivity.
- Phase 3 preregistration commit: `c0031574`.
- Phase 3 runner/checker commit: `fc394bcb`.
- Phase 4 final commit: `9ec75b19`.
- Phase 5' preregistration commit: `c0a50c89`.
- Phase 5' runner/checker commit: `d93ae055`.
- Completed run: `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z`.
- Phase 2 scope: partial Nemotron-3 validation only; Qwen3.6 and Kimi
  Linear are deferred by the no-vLLM-upgrade authorized window.
- Phase 5' run:
  `experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z`.
- Phase 5' decision: `DYNAMIC_REGIME_TRANSFORMER`.
- Phase 5' migration fraction: `0.8393787202380952`, CI95
  `[0.8278459821428571, 0.8507254464285714]`.
- Phase 5' strict set-leaving: `0.6705729166666666`, CI95
  `[0.6526227678571428, 0.6902901785714286]`.
- Phase 5' within-set rank shuffling: `0.16573660714285715`, CI95
  `[0.15411086309523808, 0.1773623511904762]`.
- Experiment D decomposition analysis:
  `experimental/outlier_migrate/decomposition_analysis/`.
- Experiment D packets: Phase 0 Granite-Tiny, Phase 1 Granite-Small,
  partial Phase 2 Nemotron-3, and Phase 5' pure Transformer.
- Experiment D outputs: `kendall_tau_by_position.json`,
  `component_decomposition.md`, `component_decomposition.json`,
  `cross_tabulation.json`, and `trace_difficulty_regression.json`.
- Experiment E threshold sensitivity:
  `experimental/outlier_migrate/decomposition_analysis/threshold_sensitivity.md`
  and `threshold_sensitivity.pdf`.
- Experiment E headline: top-1% decomposition is stable through top-2% for
  all landed packets; top-5% changes the component split and should be
  reported as a sensitivity result, not used as the main operating point.
- Phase 6 RSPR status: skipped/blocked by the measurable-gap execution
  condition; see `swarm/blocked_phase6_testbed_selection.md`.
- Interpretation update: pure-Transformer R1-Distill-Qwen-1.5B shows
  migration at essentially the Granite/Nemotron scale. OutlierMigrate must no
  longer frame the main measurement as Mamba-2-specific; the defensible story
  is broader decode-time rank dynamics, with Mamba-2 layer-uniformity and
  cross-family measurements as supporting structure.
- Camera-ready final: none; human review required.
- Camera-ready candidate: ThoughtFlow-FP8 fallback candidate only, under the
  falsification-methodology workshop framing. It is not a positive-method
  candidate.
- Candidate commit: `e25a45d6c56151a31ef7f788d3d8d515eb46d649` for the
  ThoughtFlow fallback candidate review/audit packet.

## Portfolio

| Project | Status | Paper posture | Next human-visible decision |
| --- | --- | --- | --- |
| OutlierMigrate | Phase 0 PASS, Phase 1 PASS, partial Phase 2 Nemotron-3 PASS, Phase 3 KILL, Phase 4 KILL, Phase 5' DYNAMIC | Broader decode-time rank-dynamics characterization plus negative-intervention candidate, not positive method | Human decision on whether to accept broader framing or require a new positive method |
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

Phase 5' pure-Transformer control:

- Preregistration:
  `experimental/outlier_migrate/phase5_prime/preregister_om_phase5_prime_transformer_control.md`
  committed and pushed at `c0a50c89` before inference.
- Runner/checker:
  `experimental/outlier_migrate/phase5_prime/run_om_phase5_prime_transformer_control.py`
  and
  `experimental/outlier_migrate/phase5_prime/check_om_phase5_prime_transformer_control.py`
  committed and pushed at `d93ae055`.
- Result packet:
  `experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z`.
- Checker decision: `DYNAMIC_REGIME_TRANSFORMER`.
- Artifact status: `artifact_complete=true`.
- Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, HuggingFace snapshot
  `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
- Migration fraction: `0.8393787202380952`, CI95
  `[0.8278459821428571, 0.8507254464285714]`.
- Strict set-leaving: `0.6705729166666666`, CI95
  `[0.6526227678571428, 0.6902901785714286]`.
- Within-set rank shuffling: `0.16573660714285715`, CI95
  `[0.15411086309523808, 0.1773623511904762]`.
- Interpretation: Phase 5' broadens the rank-migration measurement beyond
  measured Mamba-2 hybrids. This is useful but it undercuts any paper claim
  that migration is explained primarily by Mamba-2 state dynamics.

Experiment D decomposition formalization:

- Output directory:
  `experimental/outlier_migrate/decomposition_analysis/`.
- Machine-readable Kendall output:
  `experimental/outlier_migrate/decomposition_analysis/kendall_tau_by_position.json`.
- Component report:
  `experimental/outlier_migrate/decomposition_analysis/component_decomposition.md`.
- Cross-tab:
  `experimental/outlier_migrate/decomposition_analysis/cross_tabulation.json`.
- Trace-difficulty regression:
  `experimental/outlier_migrate/decomposition_analysis/trace_difficulty_regression.json`.
- Component table headline:
  - Phase 0 strict set-leaving `0.634244791667`; within-set shuffling
    `0.175260416667`.
  - Phase 1 strict set-leaving `0.566234756098`; within-set shuffling
    `0.270934959350`.
  - Phase 2 strict set-leaving `0.533713200380`; within-set shuffling
    `0.269082383666`.
  - Phase 5' strict set-leaving `0.670572916667`; within-set shuffling
    `0.165736607143`.
- Interpretation: strict set-leaving remains the dominant component across
  all landed packets, including the pure-Transformer control. Rank-shuffling
  is still nontrivial, but the pure-Transformer control shifts a larger share
  into strict set-leaving than Granite-Small or Nemotron-3.

Experiment E threshold sensitivity:

- Output report:
  `experimental/outlier_migrate/decomposition_analysis/threshold_sensitivity.md`.
- Output figure:
  `experimental/outlier_migrate/decomposition_analysis/threshold_sensitivity.pdf`.
- Thresholds evaluated: top-0.5%, top-1%, top-2%, top-5%.
- Stability rule: both strict set-leaving and within-set rank-shuffling must
  remain within `0.10` absolute fraction of their top-1% values.
- Stable thresholds by packet:
  - Phase 0 Granite-Tiny: top-0.5%, top-1%, top-2%.
  - Phase 1 Granite-Small: top-0.5%, top-1%, top-2%.
  - Phase 2 Nemotron-3: top-1%, top-2%.
  - Phase 5' Transformer: top-0.5%, top-1%, top-2%.
- Interpretation: the top-1% operating point is not a cherry-picked
  threshold. The component split is stable through top-2% across all landed
  packets, while top-5% admits enough lower-magnitude channels to change the
  set-leaving/rank-shuffling balance.

Phase 6 RSPR gate outcome:

- Block/skip note: `swarm/blocked_phase6_testbed_selection.md`.
- Gate condition: Phase 4 needed fewer than 25% no-gap traces and no
  measurement-design kill.
- Observed Phase 4 no-gap fraction: `0.375`.
- Decision: `SKIP_OM_PHASE6_TESTBED_NOT_MEASURABLE`.
- No Phase 6 preregistration was authored and no RSPR inference was run.
- Interpretation: this preserves the positive-method bar. RSPR may still be
  a plausible future method, but it needs a fresh measurable quantization
  testbed authorization before any data.

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

Aborted before decision metrics:

- Initial Granite-4.0-H-Tiny Phase 3 intervention run:
  `experimental/outlier_migrate/phase3/results/om_phase3_20260509T174000Z`.
- Started at `2026-05-09T17:38:58Z`.
- Aborted at `2026-05-09T17:56:06Z`.
- Two activation-capture batches completed; no protected sets, quantized
  scoring, per-trace metrics, checker output, or decision metrics were
  produced.
- Reason: independent protocol review found that the checker did not enforce
  the control-stop exit, did not recompute protected sets/recoveries from raw
  artifacts, and the runner only quantized `nn.Linear` modules inside the layer
  stack. The abort occurred before any Phase 3 intervention result was
  observed.
- GPU-hour delta charged to the sprint: `0.2856`.

Corrective patch completed and pushed:

- Tighten `check_phase3_intervention.py` to recompute protected sets from
  activation rows, recompute recoveries from perplexities, validate the
  scoring window, enforce SmoothQuant/AWQ bans, and exit nonzero on
  control-stop packets.
- Tighten `run_phase3_intervention.py` to quantize expert-bank 3D weights and
  remove the blanket outside-layer exclusion. Tied input/output embedding heads
  remain excluded with explicit rationale to avoid confounding prompt
  embeddings with layer protection.
- Commit: `7e895e4f`.

Second run status:

- Run `experimental/outlier_migrate/phase3/results/om_phase3_20260509T180200Z`
  completed activation capture, protected-set construction, BF16 trace
  generation, and BF16 scoring.
- It failed before any quantized recovery metrics were produced, at the start
  of `static_1pct` scoring.
- Error:
  `RuntimeError: Expected conv_state.scalar_type() == input_type to be true`.
- Diagnosis: `causal_conv1d_update` requires the recurrent convolution cache
  dtype to match the input dtype; FP16 autocast made the quantized-regime input
  FP16 while the hybrid cache remained BF16.
- The source run has no `per_trace_metrics.json`, `metrics.json`,
  `checker_result.json`, or `artifact_check.json`; it is not a decision packet.
- GPU-hour delta charged to the sprint: `2.3370`.

Second corrective patch:

- Align hybrid cache `conv_states` and `ssm_states` to FP16 during FP16
  quantized-regime scoring.
- Add `--reuse-prequant-run-dir` so a fresh run can reuse only the valid
  activation/protected-set/BF16-trace artifacts from a failed pre-metric run.
- The reuse path refuses source runs that already contain metric or checker
  outputs.

Third run status:

- Reuse run `experimental/outlier_migrate/phase3/results/om_phase3_20260509T202600Z`
  completed BF16 scoring from reused activation/BF16-trace artifacts.
- It again failed before quantized recovery metrics, at `static_1pct` initial
  scoring, with the same causal-conv1d dtype assertion. The cache-casting patch
  did not affect the Granite fast-path prompt forward early enough.
- GPU-hour delta charged to the sprint: `0.8533`.

Third corrective patch:

- Disable GraniteMoeHybrid's optional Mamba CUDA fast path only during FP16
  quantized-regime scoring.
- Rationale: the upstream causal-conv1d update kernel requires
  `conv_state.scalar_type() == input_type`; the torch fallback avoids this
  fused-kernel dtype assertion while preserving the fixed model, trace set,
  protected sets, and scoring target.

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

Completed Phase 3 decision packet:

- Final run:
  `experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z`.
- Started at `2026-05-09T21:18:55Z`; completed at
  `2026-05-10T05:05:27Z`.
- GPU-hour delta charged to the sprint: `7.7756`.
- New cumulative GPU hours used: `28.4808`.
- Estimated spend at `$1.89/hr`: `$53.79`.
- Checker exit code: `1`.
- Checker decision: `KILL_OM_PHASE3_INTERVENTION_FAILS`.
- Artifact status: `artifact_complete=true`.
- Kill reason: primary median recovery `0.000000000000` is below the
  preregistered `0.20` kill floor.

Primary intervention result:

| Regime | Median recovery | Bootstrap 95% CI |
|---|---:|---:|
| static-1% baseline | reference | reference |
| migration-aware union | `0.000000000000` | `[0.000000000000, 0.711143244199]` |
| static-2% matched budget | `0.000000000000` | `[0.000000000000, 0.356017001423]` |
| magnitude average | `0.061276118929` | `[0.000000000000, 0.512245438462]` |

Mandatory control outcome:

- `union_outperforms_both_controls=false`.
- The best control was magnitude averaging, which beat union by
  `0.061276118929` median recovery.
- The sprint stop condition did not fire because the margin is below `0.10`.

Position-grid sensitivity:

| Grid | Positions | Median recovery | Bootstrap 95% CI |
|---|---|---:|---:|
| sparse | `{100, 5000, 10000}` | `0.000000000000` | `[0.000000000000, 0.752281234025]` |
| primary | `{100, 1000, 5000, 10000}` | `0.000000000000` | `[0.000000000000, 0.711143244199]` |
| dense | `{100, 500, 1000, 2000, 5000, 7500, 10000}` | `0.000000000000` | `[0.000000000000, 0.920007799726]` |

Consequences:

- The OutlierMigrate paper must not claim migration-aware static protection as
  a successful positive method.
- Conditional Phase 3 follow-ups are skipped: Nemotron-3 intervention,
  within-set refresh pilot, and decode-length scaling.
- The paper integration path is now a characterization plus negative
  intervention result: migration is robust, strict set-leaving matters, and
  simple static union protection is insufficient.
- Diagnostic note:
  `experimental/outlier_migrate/phase3/diagnostic.md`.

Completed Phase 3 paper integration:

- Paper updated:
  `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`.
- PDF rebuilt:
  `experimental/outlier_migrate/paper/outlier_migrate_colm2026.pdf`.
- Reviewer pack updated:
  `experimental/outlier_migrate/paper/reviewer_pack.md`.
- Committee review:
  `experimental/outlier_migrate/paper/committee_reviews/20260510_phase3_negative_intervention.md`.
- Final committee scores after three rounds:
  - COLM area chair: `8/10`.
  - MLSys reviewer: `7/10`.
  - adversarial reviewer: `8/10`.

Paper status after integration:

- OutlierMigrate is a camera-ready candidate only under the characterization
  plus negative-intervention framing.
- It is not camera-ready final.
- It is not a positive-method or systems-efficiency paper.
- The paper now explicitly reports both Phase 3 kill paths: median recovery
  `0.000000000000 < 0.20`, and `10/24` no-recoverable-static-gap traces
  exceeding the preregistered `25%` no-gap kill condition.
- The reviewer pack includes a primary-source citation spot-check for Kimi
  Linear, Qwen3.6, and Quamba-SE, but not a complete citation audit.

Remaining human-facing caveats:

1. Qwen3.6 and Kimi Linear validation remain deferred.
2. No independent seed/model repeat beyond recorded bootstrap exists.
3. No complete contamination or exploratory-history audit exists for the
   top-1% fraction, rank-delta threshold, decode positions, prompt slice, and
   model choices.
4. TeX builds successfully but still emits underfull layout warnings from long
   artifact paths and dense tables.

## Phase 4 Intervention Sprint Status (2026-05-12)

Human approved Phase 4 as a parallel depth-1 pivot from OutlierMigrate Phase 0
after Phase 3 killed on Granite-Tiny. Phase 4 moved the intervention test to
Granite-4.0-H-Small with a 512-token scoring window while preserving the Phase
3 decision thresholds and mandatory controls.

Completed Phase 4 decision packet:

- Final run:
  `experimental/outlier_migrate/phase4/results/om_phase4_20260511T054000Z`.
- Preregistration:
  `experimental/outlier_migrate/phase4/preregister_om_phase4_intervention.md`
  approved at commit `80bfbc26` before quantized Phase 4 scoring.
- Runner/checker:
  `experimental/outlier_migrate/phase4/run_om_phase4_intervention.py` and
  `experimental/outlier_migrate/phase4/check_om_phase4_intervention.py`.
- Started at `2026-05-11T05:36:03Z`; completed at
  `2026-05-12T05:20:33Z`.
- GPU-hour delta charged to Phase 4: `23.7417`.
- New cumulative GPU hours used in `swarm/state.json`: `52.2225`.
- Estimated spend at `$1.89/hr`: `$98.70`.
- Checker exit code: `1`.
- Checker decision: `KILL_OM_PHASE4_INTERVENTION_FAILS`.
- Artifact status: `artifact_complete=true`.
- Control stop: `false`.
- Diagnostic:
  `experimental/outlier_migrate/phase4/diagnostic.md`.

Phase 4 primary result:

| Metric | Value |
|---|---:|
| median recovery | `0.000000000000` |
| mean recovery | `-0.772191770347` |
| bootstrap CI95 | `[0.000000000000, 0.069540641955]` |
| no-recoverable-static-gap traces | `9/24` (`0.375000000000`) |
| traces with recovery > `0.50` | `4/24` |

Mandatory control outcome:

| Regime | Median recovery | CI95 high | >0.50 traces |
|---|---:|---:|---:|
| migration-aware union | `0.000000000000` | `0.069540641955` | `4/24` |
| static-2% matched budget | `0.000000000000` | `0.509447743503` | `8/24` |
| magnitude average | `0.000000000000` | `0.047903829981` | `6/24` |

- `union_outperforms_both_controls=false`.
- No control-stop fired because neither control beat union by more than `0.10`
  median recovery; all medians were zero.

Position-grid sensitivity:

| Grid | Median recovery | Bootstrap 95% CI |
|---|---:|---:|
| sparse | `0.000000000000` | `[-0.337218160457, 0.057480709850]` |
| primary | `0.000000000000` | `[0.000000000000, 0.069540641955]` |
| dense | `0.000000000000` | `[-0.474740597754, 0.055115818031]` |

Interpretation:

- Phase 4 does not support migration-aware static union protection as a
  positive method.
- The Phase 3 measurement-artifact hypothesis is weakened. Granite-Small with a
  512-token scoring window still had a no-gap fraction above the preregistered
  `25%` ceiling, and traces with measurable gaps did not produce stable union
  recovery.
- The paper must retain the characterization plus negative-intervention
  framing unless a new, separately preregistered adaptive method is authorized
  later.
- Conditional Phase 4 follow-ups are skipped because the primary Phase 4 gate
  did not pass.

## ThoughtFlow-FP8 Polish Status (2026-05-10)

Additional read-only committee polish completed for ThoughtFlow-FP8 under the
authorized scope. The paper body was not modified.

- Committee review:
  `experimental/thoughtflow_fp8/paper/committee_reviews/20260510_polish_round3.md`.
- Reviewer pack updated:
  `experimental/thoughtflow_fp8/paper/reviewer_pack.md`.
- Final committee scores under falsification-methodology / negative-results
  workshop framing:
  - COLM area chair: `8/10`.
  - MLSys reviewer: `8/10` as an artifact diagnostic, not as a systems paper.
  - adversarial reviewer: `8/10`.
- Checker verification:
  `PASS_THOUGHTFLOW_PAPER_BUILDABLE`.
- Owned tests:
  `70 passed, 1 warning in 11.77s` with `.venv_gpu`,
  `TRITON_CPU_BACKEND=1`, and `TRITON_INTERPRET=1`.

ThoughtFlow remains only a falsification-methodology workshop diagnostic
candidate. It is not camera-ready final, not a positive-method paper, and not
an MLSys systems paper.
