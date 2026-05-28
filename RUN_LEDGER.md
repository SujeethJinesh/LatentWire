# Positive-Method Sprint Run Ledger

Last updated: 2026-05-28T04:18Z

## Active Objective

Find one clean positive method or a stronger regime-aware diagnostic result
without wasting GPU. GPU jobs require an explicit orchestrator gate. CPU-only
filter work writes to `artifacts/<task_name>/` and does not edit this ledger.

## Current GPU Lane

| Order | Gate | Status | Run / Artifact | Decision |
|---:|---|---|---|---|
| 1 | V1 ParoQuant-on-Nemotron | RUNNING | `experimental/outlier_migrate/phase9/results/om_v1_paroquant_nemotron_20260528T0318Z` | Pending score cache/checker |
| 2 | LAMBDA/HYST smoke on DeepSeek/Falcon | GATED | `experimental/outlier_migrate/phase9/preregister_om_phase9_funnel_smoke.md` | CPU gates landed; prepare runner, wait for V1 GPU completion |
| 3 | Minimal M-SURFACE sanity on Granite | DEFERRED | `artifacts/msurface/decision.json` | Hookable but inconclusive; only after smoke unless a finalist needs surface evidence |
| 4 | Minimal M-BRANCH sanity on Falcon | DEFERRED | `artifacts/mbranch/decision.json` | Falcon branch-local GPU run not recommended by CPU diagnostic |
| 5 | Partial eval for survivors | GATED | TBD | Only smoke survivors |
| 6 | Full 12-trace + BCa | GATED | TBD | At most two finalists |

## CPU Artifact Tasks

| Task | Output Directory | Status | Gate Use |
|---|---|---|---|
| WJAC prefilter | `artifacts/wjac_prefilter/` | COMPLETE | `KILL_WJAC_PREFILTER`; no WJAC GPU endpoint scoring |
| Funnel prefilters | `artifacts/funnel_prefilters/` | COMPLETE | Run LAMBDA/HYST smoke only on fixed DeepSeek/Falcon traces |
| M-SURFACE diagnostic | `artifacts/msurface/` | COMPLETE | Inconclusive; optional tiny Granite hook sanity only after higher-priority smoke |
| M-BRANCH diagnostic | `artifacts/mbranch/` | COMPLETE | Do not request Falcon M-BRANCH GPU scoring now |
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
