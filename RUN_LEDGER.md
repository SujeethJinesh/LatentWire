# Positive-Method Sprint Run Ledger

Last updated: 2026-05-28T15:18Z

## Active Objective

Find one clean positive method or a stronger mechanism/protocol result without
wasting GPU. After V1, the primary question is whether drift-aware rotation
choices can beat or robustify static ParoQuant. GPU jobs require an explicit
orchestrator gate. CPU-only filter work writes to `artifacts/<task_name>/` and
does not edit this ledger.

## Current GPU Lane

| Order | Gate | Status | Run / Artifact | Decision |
|---:|---|---|---|---|
| 0 | V1 ParoQuant-on-Nemotron | COMPLETE_HEADLINE_CHANGING | `experimental/outlier_migrate/phase9/results/om_v1_paroquant_nemotron_20260528T0318Z` | `PASS_V1_PAROQUANT_NEMOTRON_ROTATION_DOMINATES`; median 1.047, CI95 [1.007, 1.292], +0.232 over Nemotron M11b top-10 |
| 1 | ParoQuant Falcon smoke | NEXT_GATED | `artifacts/rotation_config_grid/` / TBD smoke packet | Run first to test whether rotation resolves Falcon before channel rescue |
| 2 | ParoQuant DeepSeek smoke | NEXT_GATED | `artifacts/rotation_config_grid/` / TBD smoke packet | Run after or alongside Falcon only if the runner/checker is ready; tests four-model rotation-dominance |
| 3 | DriftRot Scale/CVaR/Clip smoke | CPU_GATED | `artifacts/scale_cvar_clip/`, `artifacts/rotation_config_grid/` | Granite + Nemotron; must include exact ParoQuant baseline config |
| 4 | DriftRot ResidualCorrection smoke | CPU_GATED | `artifacts/rot_resid_correction/` | Granite tail/weak trace plus two representative traces; no kernel until pass |
| 5 | Drift-aware Pairing smoke | CPU_GATED | `artifacts/drift_pairing/` | Run only if pairings materially differ from static high-low baseline |
| 6 | Falcon BranchRot diagnostic/protection | CONDITIONAL | `artifacts/falcon_branch_rotation/` | Promote if branch-local drift/covariance range is materially lower than post-mixer |
| 7 | M-SURFACE DriftRot diagnostic | CONDITIONAL | `artifacts/msurface_driftrot/` | Cheap if hooks ready; promote stable internal surfaces only |
| 8 | Falcon LAMBDA/HYST fallback | DEFERRED_AFTER_ROTATION | `artifacts/falcon_channel_fallbacks/` and prior HYST/LAMBDA artifacts | Run only if ParoQuant Falcon and BranchRot do not solve Falcon |
| 9 | Restricted KLLOOK | CONDITIONAL | TBD | Only if Falcon channel/rotation branches remain ambiguous |
| 10 | Partial / full eval for finalists | GATED | TBD | 6-trace partial; 12-trace + BCa for at most two finalists |

## CPU Artifact Tasks

| Task | Output Directory | Status | Gate Use |
|---|---|---|---|
| WJAC prefilter | `artifacts/wjac_prefilter/` | COMPLETE | `KILL_WJAC_PREFILTER`; no WJAC GPU endpoint scoring |
| Funnel prefilters | `artifacts/funnel_prefilters/` | COMPLETE | Run LAMBDA/HYST smoke only on fixed DeepSeek/Falcon traces |
| RISKGUARD trigger calibration | `artifacts/riskguard/` | COMPLETE_DEFERRED | In-sample trigger crosses CI lower bound, but LOOCV robustness fails; no GPU confirm |
| Falcon LAMBDA budget prior | `artifacts/lambda_falcon/` | COMPLETE_DEFERRED | Reallocation exists, but `run_smoke=false`; no causal per-layer headroom |
| Falcon HYST thresholds | `artifacts/hyst_falcon/` | COMPLETE_READY | Run Falcon HYST smoke with margin `m=5`; runner supports `--methods hyst` |
| TRACE-ROUTER classifier | `artifacts/trace_router/` | COMPLETE_WEAK | Offline weak signal on Granite/DeepSeek only; needs preregistered larger slice before GPU evidence |
| M-SURFACE diagnostic refresh | `artifacts/msurface/` | COMPLETE_CONDITIONAL | Granite `mamba_out_projection_input` 2-trace hook diagnostic is cheap and gateable |
| M-BRANCH diagnostic refresh | `artifacts/mbranch/` | COMPLETE_DEFERRED | No branch-local cache and no GPU diagnostic recommended |
| LayerKeep / no-gap detector | `artifacts/layerkeep_nogap/` | COMPLETE_CONDITIONAL | Falcon-only LayerKeep candidate `30-35`; no-gap filter not recommended |
| Kernel design | `artifacts/kernel_design/` | COMPLETE | Design only; no GPU kernel until finalist |
| C1 Covariance headroom | `artifacts/covariance_headroom/` | RUNNING | Gate ScaleRefresh/ClipRetune vs RotationRefresh/Pairing/KLT |
| C2 Scale CVaR clip | `artifacts/scale_cvar_clip/` | RUNNING | Candidate clip configs for Granite/Nemotron smoke |
| C3 Rotation config grid | `artifacts/rotation_config_grid/` | RUNNING | ParoQuant baseline plus group/rotation/clip grid |
| C4 Drift-aware pairing | `artifacts/drift_pairing/` | RUNNING | Promote only if pairings differ materially from ParoQuant baseline |
| C5 Rotated residual correction | `artifacts/rot_resid_correction/` | RUNNING | Candidate protected columns and PyTorch reference only |
| C6 Falcon branch rotation | `artifacts/falcon_branch_rotation/` | RUNNING | Branch-local rotation diagnostic/protection config |
| C7 M-SURFACE DriftRot | `artifacts/msurface_driftrot/` | RUNNING | Surface map and 2-trace diagnostic config |
| C8 Falcon channel fallbacks | `artifacts/falcon_channel_fallbacks/` | RUNNING | LAMBDA/HYST configs, lower priority until ParoQuant Falcon known |
| C9 Paper delta | `artifacts/paper_delta/` | RUNNING | Abstract/contribution options and figure plan |
| C10 Novelty audit | `artifacts/novelty_audit/` | RUNNING | Safe claims vs ParoQuant, QuaRot, SpinQuant, Quamba2, SmoothQuant/AWQ, MambaQuant, RRS |
| C11 Repro audit | `artifacts/repro_audit/` | RUNNING | V1/ParoQuant/M11b/cache provenance and final paper tables |
| C12 Kernel spec | `artifacts/kernel_spec/` | RUNNING | Protected-column correction spec only |

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
