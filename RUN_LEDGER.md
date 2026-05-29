# Positive-Method Sprint Run Ledger

Last updated: 2026-05-29T04:05Z

## Active Objective

Finalize the mechanism/regime paper. Positive-method search is stopped for
this submission cycle. The remaining work is paper/audit/repro packaging and
systems-cost framing, not more GPU method exploration.

## Current GPU Lane

| Order | Gate | Status | Run / Artifact | Decision |
|---:|---|---|---|---|
| 0 | V1 ParoQuant-on-Nemotron | COMPLETE_HEADLINE_CHANGING | `experimental/outlier_migrate/phase9/results/om_v1_paroquant_nemotron_20260528T0318Z` | `PASS_V1_PAROQUANT_NEMOTRON_ROTATION_DOMINATES`; median 1.047, CI95 [1.007, 1.292], +0.232 over Nemotron M11b top-10 |
| 1 | ParoQuant Falcon smoke | COMPLETE_PASS | `experimental/outlier_migrate/phase9/results/om_v1_paroquant_falcon_20260528T154653Z` | `PASS_V1_PAROQUANT_FALCON_ROTATION_RESCUE`; median 0.381, CI95 [0.0645, 0.547], +0.337 over Falcon M11b top-10 |
| 2 | ParoQuant DeepSeek smoke | COMPLETE_PASS | `experimental/outlier_migrate/phase9/results/om_v1_paroquant_deepseek_20260528T162858Z` | `PASS_V1_PAROQUANT_DEEPSEEK_ROTATION_DOMINATES`; median 0.756, CI95 [-0.246, 0.855], +0.379 over DeepSeek static-top10 |
| 3 | DriftRot Scale/CVaR/Clip smoke | PASS_GRANITE_ONLY_AMBIG_DEEPSEEK_FALCON | `artifacts/scale_cvar_clip/`, `experimental/outlier_migrate/phase9/results/om_driftrot_falcon_clip_tight_20260528T2343Z` | Tight clip improves all recoverable Granite positive traces (median 0.754 -> 0.922, worst -26.37 -> 0.601), but DeepSeek median regresses 0.756 -> 0.518 and Falcon is flat-to-worse against Falcon ParoQuant; safe claim is Granite-specific tail control |
| 4 | DriftRot ResidualCorrection smoke | KILL_CURRENT_DESIGN | `artifacts/rot_resid_correction/` | Top-8x32 MoE residual correction worsened Granite tail trace; only reopen with a bounded or KLLOOK-gated design |
| 5 | Drift-aware Pairing smoke | CPU_GATED | `artifacts/drift_pairing/` | Run only if pairings materially differ from static high-low baseline |
| 6 | Falcon BranchRot diagnostic/protection | CONDITIONAL | `artifacts/falcon_branch_rotation/` | Promote if branch-local drift/covariance range is materially lower than post-mixer |
| 7 | M-SURFACE DriftRot diagnostic | KILL_OR_DEFER_SURFACE_NO_LOWER_DRIFT | `experimental/outlier_migrate/phase9/results/om_phase9_msurface_granite_sanity_20260529T0032Z` | Granite hook sanity found internal Mamba/attention projection-input surfaces drift more than same-run post-block; do not run Falcon M-SURFACE from this surface family |
| 8 | Falcon LAMBDA/HYST fallback | DEFERRED_AFTER_ROTATION | `artifacts/falcon_channel_fallbacks/` and prior HYST/LAMBDA artifacts | Run only if ParoQuant Falcon and BranchRot do not solve Falcon |
| 9 | Restricted KLLOOK | NOT_RUN_FINAL | `artifacts/kllook_residual_oracle/` | Rotated-residual oracle runner exceeded the one-hour construction gate; no KLLOOK claim is made. |
| 10 | Partial / full eval for finalists | STOPPED | `artifacts/final_path_decision.md` | No finalist beyond prior-work ParoQuant baseline; positive-method search stopped. |

## CPU Artifact Tasks

| Task | Output Directory | Status | Gate Use |
|---|---|---|---|
| WJAC prefilter | `artifacts/wjac_prefilter/` | COMPLETE | `KILL_WJAC_PREFILTER`; no WJAC GPU endpoint scoring |
| Funnel prefilters | `artifacts/funnel_prefilters/` | COMPLETE | Run LAMBDA/HYST smoke only on fixed DeepSeek/Falcon traces |
| RISKGUARD trigger calibration | `artifacts/riskguard/` | COMPLETE_DEFERRED | In-sample trigger crosses CI lower bound, but LOOCV robustness fails; no GPU confirm |
| Falcon LAMBDA budget prior | `artifacts/lambda_falcon/` | COMPLETE_DEFERRED | Reallocation exists, but `run_smoke=false`; no causal per-layer headroom |
| Falcon HYST thresholds | `artifacts/hyst_falcon/` | COMPLETE_READY | Run Falcon HYST smoke with margin `m=5`; runner supports `--methods hyst` |
| TRACE-ROUTER classifier | `artifacts/trace_router/` | COMPLETE_WEAK | Offline weak signal on Granite/DeepSeek only; needs preregistered larger slice before GPU evidence |
| M-SURFACE diagnostic refresh | `artifacts/msurface/` | COMPLETE_KILL_OR_DEFER | Granite `mamba_out_projection_input` and attention `o_proj` input were measured and both drifted more than same-run post-block |
| M-BRANCH diagnostic refresh | `artifacts/mbranch/` | COMPLETE_DEFERRED | No branch-local cache and no GPU diagnostic recommended |
| LayerKeep / no-gap detector | `artifacts/layerkeep_nogap/` | COMPLETE_CONDITIONAL | Falcon-only LayerKeep candidate `30-35`; no-gap filter not recommended |
| Kernel design | `artifacts/kernel_design/` | COMPLETE | Design only; no GPU kernel until finalist |
| C1 Covariance headroom | `artifacts/covariance_headroom/` | RUNNING | Gate ScaleRefresh/ClipRetune vs RotationRefresh/Pairing/KLT |
| C2 Scale CVaR clip | `artifacts/scale_cvar_clip/` | RUNNING | Candidate clip configs for Granite/Nemotron smoke |
| C3 Rotation config grid | `artifacts/rotation_config_grid/` | RUNNING | ParoQuant baseline plus group/rotation/clip grid |
| C4 Drift-aware pairing | `artifacts/drift_pairing/` | RUNNING | Promote only if pairings differ materially from ParoQuant baseline |
| C5 Rotated residual correction | `artifacts/rot_resid_correction/` | RUNNING | Candidate protected columns and PyTorch reference only |
| C6 Falcon branch rotation | `artifacts/falcon_branch_rotation/` | RUNNING | Branch-local rotation diagnostic/protection config |
| C7 M-SURFACE DriftRot | `artifacts/msurface_driftrot/` | COMPLETE_KILL_OR_DEFER | Granite two-trace diagnostic did not find a lower-drift internal surface |
| C8 Falcon channel fallbacks | `artifacts/falcon_channel_fallbacks/` | COMPLETE_DEFERRED | LAMBDA/HYST configs remain available, but Falcon ParoQuant rescue lowers channel-fallback priority |
| C9 Paper delta | `artifacts/paper_delta/` | RUNNING | Abstract/contribution options and figure plan |
| C10 Novelty audit | `artifacts/novelty_audit/` | RUNNING | Safe claims vs ParoQuant, QuaRot, SpinQuant, Quamba2, SmoothQuant/AWQ, MambaQuant, RRS |
| C11 Repro audit | `artifacts/repro_audit/` | RUNNING | V1/ParoQuant/M11b/cache provenance and final paper tables |
| C12 Kernel spec | `artifacts/kernel_spec/` | RUNNING | Protected-column correction spec only |

## Rotation-First CPU Gate Summary

| Gate | Decision | Consequence |
|---|---|---|
| C1 covariance headroom | `NEEDS_GPU_OR_CACHE_FOR_TRUE_COVARIANCE` | Existing compact caches support only block-output diagonal/magnitude proxy; do not claim off-diagonal rotation headroom yet |
| C2 Scale/CVaR/Clip | `PASS_GRANITE_ONLY_AMBIG_DEEPSEEK_FALCON` | Tight clip fixes Granite tails but is not universal; DeepSeek lower CI improves while median/worst regress, and Falcon median is flat while worst trace/lower CI regress |
| C3 rotation grid | `READY_FOR_GATED_GPU_SMOKE_TEMPLATE` | Do not run full 36-config grid before clip-only smoke shows held-out value |
| C4 drift-aware pairing | `NEEDS_ACTIVATION_CACHE` | Needs exact rotation-surface activation cache before GPU smoke |
| C5 residual correction | `KILL_TOP8X32_MOE_RESIDUAL_DIAGNOSTIC` | Delta columns were materialized, but valid Granite prompt-4 diagnostic scored -12.29 recovery versus tight ParoQuant 5.94; do not run 3-trace smoke for this candidate |
| C6 Falcon BranchRot | `NEEDS_GPU_DIAGNOSTIC` | Branch hooks feasible; only after ParoQuant Falcon is weak |
| C7 M-SURFACE DriftRot | `KILL_OR_DEFER_SURFACE_NO_LOWER_DRIFT` | Granite same-run diagnostic: post-block mean leaving 0.578, Mamba out-proj input 0.887, attention o-proj input 0.780; do not promote this surface family |
| C8 Falcon channel fallback | `DEFER_AFTER_FALCON_ROTATION_PASS` | Falcon ParoQuant rescue lowers priority for HYST/LAMBDA channel fallbacks |
| C9 paper delta | `ROTATION_FIRST_REGIME_AWARE` | Use rotation-first framing options; do not claim ParoQuant as ours |
| C10 novelty audit | `SAFE_CLAIMS_ONLY` | BranchRot is most defensible; scale/clip/residual correction require held-out drift-specific gain |
| C11 repro audit | `TABLES_READY` | Current method matrix, rotation summary, failed branch table prepared |
| C12 kernel spec | `SPEC_READY_KERNEL_BLOCKED_ON_METHOD` | No kernel implementation until residual correction passes held-out eval |

## Current Smoke Trace Gate

The next GPU smoke packet should be rotation-first. Falcon and DeepSeek smoke
must compare static ParoQuant-style rotation against the current best baselines
before running Falcon channel fallbacks.

| Model | Prompt indices | Gate rationale |
|---|---:|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 5, 11, 8 | positive-gap plus high-drift/representative traces |
| Falcon-H1-0.5B-Instruct | 7, 1, 11 | positive-gap plus high-drift/representative traces |

WJAC, M-PRED/AR/Kalman prediction, hard bins/switches, and naive
ParoQuant+M11b composition remain excluded because those branches were killed
or sub-additive under prior gates.

## DriftRot+ Residual-Correction CPU Gate

Timestamp: 2026-05-29T02:34Z. Commit context: `6688f4eb`.

The paper has been updated and pushed with the rotation-first / DriftRot
framework. The next CPU-gated sprint was executed into disjoint artifacts:

| Task | Artifact | Decision | Consequence |
|---|---|---|---|
| C1 K-RES residual headroom | `artifacts/k_res/` | `KILL_CURRENT_TOP8X32_PROXY_NO_GPU` | Do not run P1/P2 residual GPU for the current proxy. Residual energy is weak and the valid top-8x32 smoke worsened Granite tail recovery (-12.29) versus tight ParoQuant (5.94). |
| C2 K-CHURN rotated-set churn | `artifacts/k_churn/` | `INCOMPLETE_NO_POSITIONAL_ROTATED_SET_CACHE` | No dynamic residual P2/P3/P9 gate until a position-resolved residual-benefit top-k cache exists. |
| C3 K1 covariance drift | `artifacts/k1_cov/` | `P8_NOT_PROMOTED_NO_TRUE_COVARIANCE` | Clip/config tuning is closed as a headline; online scale-refresh needs true covariance evidence. |
| C4 M-SURFACE completion | `artifacts/msurface/` | `PARTIAL_COMPLETE_NO_PROMOTE_CHEAP_SURFACES_HIGHER_DRIFT` | Mamba out-proj and attention o-proj inputs drift more than post-block; SSM input and B/C remain unmeasured, not killed. |
| C5 K-BRANCH Falcon | `artifacts/k_branch/` | `DEFER_NO_BRANCH_LOCAL_CACHE` | BranchRot is not next on GPU without branch-local drift evidence. |
| C6 Residual systems spec | `artifacts/rot_resid/` | `SPEC_READY_KERNEL_BLOCKED_ON_METHOD_PASS` | No Triton/CUDA implementation until a residual policy passes quality. |
| C7 Novelty lock | `artifacts/novelty_lock/` | `SAFE_CLAIMS_LOCKED` | ParoQuant remains a baseline; safe DriftRot novelty requires held-out long-decode drift-aware selection/correction gain. |

Current next exact gate: either write a bounded/KLLOOK-gated residual selector
before any residual GPU work, or accept the mechanism/regime paper path unless
a different surface/branch gate produces lower drift.

## Final Bounded Positive-Method Gate

Timestamp: 2026-05-29T02:57Z. Commit context: `05d80f6c`.

The final bounded gate after the K-RES kill has been artifactized:

| Gate | Artifact | Decision | Consequence |
|---|---|---|---|
| K-RES failure audit | `artifacts/k_res_audit/` | `VALID_IMPLEMENTATION_KEEP_K_RES_PROXY_KILLED` | The top-8x32 residual-energy proxy was compared against the correct tight ParoQuant baseline on the same Granite tail trace/window; no implementation mismatch was found. Do not rerun this proxy. |
| Restricted residual KLLOOK oracle | `artifacts/kllook_residual_oracle/` | `NOT_EXECUTED_NO_ROTATED_RESIDUAL_KLLOOK_RUNNER` | The repo has an original-basis M-KLLOOK runner, not a post-ParoQuant rotated-residual oracle. No residual-oracle evidence exists, so residual correction is not promoted. |
| Complete M-SURFACE readout | `artifacts/msurface_complete/` | `KILL_MSURFACE_NO_LOWER_INTERNAL_SURFACE` | Complete Granite internals run measured SSM input/B/C plus projection-input surfaces. Best internal means were SSM input 0.484 and SSM C 0.486 versus post-block 0.546, not enough to promote. |
| Falcon BranchRot readout | `artifacts/falcon_branchrot_diagnostic/` | `DEFERRED_TO_FUTURE_WORK_NOT_RUN_THIS_PASS` | BranchRot was deferred unless M-SURFACE promoted. M-SURFACE did not promote, and branch-local Falcon activations are not cached. |
| Final path decision | `artifacts/final_path_decision.md` | `MECHANISM_REGIME_PAPER` | Stop the current DriftRot positive-method search. |

Current next exact gate: finalize the mechanism/regime paper and audit/repro
package. Do not claim ParoQuant as our method, and do not claim residual
KLLOOK evidence.

## Final Mechanism/Regime Paper Packaging

Timestamp: 2026-05-29T04:05Z. Commit context before packaging:
`7f20ce0b`.

| Item | Artifact | Status | Notes |
|---|---|---|---|
| Final PDF/source | `experimental/outlier_migrate/paper/outlier_migrate_colm2026.pdf` | BUILT | LaTeX build succeeds with underfull-box warnings only. |
| Claim audit | `CLAIM_AUDIT.md` | COMPLETE | 26 headline numerical claims mapped to source artifacts and check commands. |
| Systems envelope | `artifacts/systems_cost/` | COMPLETE | Analytical HBM/MAC/working-set/pJ table for hypothetical residual correction; no profiler claim. |
| Failure taxonomy | `artifacts/paper_delta/failure_taxonomy_table.tex` | COMPLETE | Converts killed branches into mechanism/design logic. |
| Regime map | `artifacts/paper_delta/regime_map.pdf` | COMPLETE | Model-by-remedy status map for the descriptive calibration protocol. |
| Novelty audit | `artifacts/novelty_audit/final_rotation_positioning.md` | COMPLETE | Safe distinction from ParoQuant, QuaRot, SpinQuant, MambaQuant, Quamba2, and DecDEC. |
| Final pack | `artifacts/final_mechanism_regime_pack/om_mechanism_regime_final_pack_20260529_0405.tar.gz` | COMPLETE | Includes PDF, TeX, ledgers, decisions, K-RES, M-SURFACE, V1 summaries, provenance, scripts, and claim map. |

Verification:

- `bash build.sh` in `experimental/outlier_migrate/paper` passed.
- `.venv_gpu/bin/python -m pytest release/tests -q` passed (`7 passed`).
- Python compile checks passed for final systems reference and key Phase 9 scripts.
- Pack validation reports zero missing required files and 96 decision JSON files included.

## Experiment Artifact Contract

Every GPU experiment packet must emit:

- `config.json`
- `command.sh`
- `metrics.json`
- `traces_used.json`
- git commit hash or diff reference
- pass/fail decision with reason

Existing Phase 9 packets may also emit richer legacy files; the required files
above are the minimum for the orchestrated sprint.
