# Positive-Method Sprint Run Ledger

Last updated: 2026-05-28T06:23Z

## Active Objective

Find one clean positive method or a stronger regime-aware diagnostic result
without wasting GPU. GPU jobs require an explicit orchestrator gate. CPU-only
filter work writes to `artifacts/<task_name>/` and does not edit this ledger.

## Current GPU Lane

| Order | Gate | Status | Run / Artifact | Decision |
|---:|---|---|---|---|
| 0 | V1 ParoQuant-on-Nemotron | RUNNING | `experimental/outlier_migrate/phase9/results/om_v1_paroquant_nemotron_20260528T0318Z` | Pending score cache/checker; 3/12 prompts complete as of 2026-05-28T05:40Z |
| 1 | Falcon LAMBDA smoke | GATED | `artifacts/lambda_falcon/` -> smoke config | Wait for C2 config and V1 GPU completion |
| 2 | Falcon HYST smoke | GATED | `artifacts/hyst_falcon/` -> smoke config | Wait for C3 config and V1 GPU completion |
| 3 | Falcon LAMBDA+HYST smoke | CONDITIONAL | TBD | Only if both single-factor smokes pass and combo beats the better single |
| 4 | RISKGUARD confirm on Granite tail traces | CONDITIONAL | `artifacts/riskguard/` | Only if C1 predicts deployable post-guard CI lower bound > 0 |
| 5 | Minimal M-BRANCH diagnostic on Falcon | CONDITIONAL | `artifacts/mbranch/` | Only if refreshed C5 gate shows branch/surface drift at least 0.15 below post-block or a tiny hook run is clearly decisive |
| 6 | Minimal M-SURFACE sanity on Granite/Falcon | CONDITIONAL | `artifacts/msurface/` | Only if refreshed C6 gate says hooks are cheap and can test lower-drift internal surface |
| 7 | Restricted Falcon KLLOOK | CONDITIONAL | TBD | Only if steps 1-6 are ambiguous |
| 8 | LayerKeep-Falcon | CONDITIONAL | `artifacts/layerkeep_nogap/` | Coarse fallback if channel selection is unstable and C7 finds layer-level separation |
| 9 | Partial eval for survivors | GATED | TBD | 6 traces, only smoke survivors |
| 10 | Full 12-trace + BCa | GATED | TBD | At most two finalists |

## CPU Artifact Tasks

| Task | Output Directory | Status | Gate Use |
|---|---|---|---|
| WJAC prefilter | `artifacts/wjac_prefilter/` | COMPLETE | `KILL_WJAC_PREFILTER`; no WJAC GPU endpoint scoring |
| Funnel prefilters | `artifacts/funnel_prefilters/` | COMPLETE | Run LAMBDA/HYST smoke only on fixed DeepSeek/Falcon traces |
| RISKGUARD trigger calibration | `artifacts/riskguard/` | RUNNING | Promote only if predicted deployable post-guard CI lower bound crosses 0 |
| Falcon LAMBDA budget prior | `artifacts/lambda_falcon/` | RUNNING | Produce Falcon-only smoke config from layer stability prior |
| Falcon HYST thresholds | `artifacts/hyst_falcon/` | RUNNING | Produce Falcon-only smoke config from churn/local-pool stability |
| TRACE-ROUTER classifier | `artifacts/trace_router/` | RUNNING | Granite/DeepSeek only if cross-validated routing gain is positive; Falcon not routable |
| M-SURFACE diagnostic refresh | `artifacts/msurface/` | RUNNING | Refresh hook complexity and 2-trace diagnostic gate |
| M-BRANCH diagnostic refresh | `artifacts/mbranch/` | RUNNING | Refresh Falcon hook map and 2-trace diagnostic gate |
| LayerKeep / no-gap detector | `artifacts/layerkeep_nogap/` | QUEUED | Start when a subagent slot frees; coarse Falcon fallback and Granite no-gap interpretation |
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
