# Positive-Method Sprint Run Ledger

Last updated: 2026-05-28T13:22Z

## Active Objective

Find one clean positive method or a stronger regime-aware diagnostic result
without wasting GPU. GPU jobs require an explicit orchestrator gate. CPU-only
filter work writes to `artifacts/<task_name>/` and does not edit this ledger.

## Current GPU Lane

| Order | Gate | Status | Run / Artifact | Decision |
|---:|---|---|---|---|
| 0 | V1 ParoQuant-on-Nemotron | COMPLETE_HEADLINE_CHANGING | `experimental/outlier_migrate/phase9/results/om_v1_paroquant_nemotron_20260528T0318Z` | `PASS_V1_PAROQUANT_NEMOTRON_ROTATION_DOMINATES`; median 1.047, CI95 [1.007, 1.292], +0.232 over Nemotron M11b top-10 |
| 1 | Falcon HYST smoke | PAUSED_BY_V1_GATE | `artifacts/hyst_falcon/` -> `--methods hyst --hyst-exit-margin-pct-points 5` | Do not launch until the V1 rotation-dominance implication is reviewed/reframed |
| 2 | Minimal M-SURFACE sanity on Granite | CONDITIONAL | `artifacts/msurface/diagnostic_config_2trace.json` | Cheap 2-trace diagnostic; promote only if internal surface leaving <0.30 or at least 0.15 below same-run post-block |
| 3 | LayerKeep-Falcon smoke | CONDITIONAL | `artifacts/layerkeep_nogap/` | Falcon-only fallback; candidate layers `30-35`, only after HYST or if HYST is blocked/ambiguous |
| 4 | Restricted Falcon KLLOOK | CONDITIONAL | TBD | Only if HYST/M-SURFACE/LayerKeep remain ambiguous |
| 5 | Partial eval for survivors | GATED | TBD | 6 traces, only smoke survivors |
| 6 | Full 12-trace + BCa | GATED | TBD | At most two finalists |

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

## Current Smoke Trace Gate

The next GPU smoke packet is allowed to include only LAMBDA and HYST on the
fixed stratified traces selected by `artifacts/funnel_prefilters/`:

| Model | Prompt indices | Gate rationale |
|---|---:|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 5, 11, 8 | positive-gap plus high-drift/representative traces |
| Falcon-H1-0.5B-Instruct | 7, 1, 11 | positive-gap plus high-drift/representative traces |

WJAC is excluded from the smoke runner because the artifactized prefilter
reproduced the corrected kill rule across covered model surfaces.

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
