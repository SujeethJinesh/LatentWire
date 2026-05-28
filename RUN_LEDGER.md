# Positive-Method Sprint Run Ledger

Last updated: 2026-05-28T04:25Z

## Active Objective

Find one clean positive method or a stronger regime-aware diagnostic result
without wasting GPU. GPU jobs require an explicit orchestrator gate. CPU-only
filter work writes to `artifacts/<task_name>/` and does not edit this ledger.

## Current GPU Lane

| Order | Gate | Status | Run / Artifact | Decision |
|---:|---|---|---|---|
| 1 | V1 ParoQuant-on-Nemotron | RUNNING | `experimental/outlier_migrate/phase9/results/om_v1_paroquant_nemotron_20260528T0318Z` | Pending score cache/checker |
| 2 | WJAC/LAMBDA/HYST smoke on DeepSeek/Falcon | GATED | `experimental/outlier_migrate/phase9/preregister_om_phase9_funnel_smoke.md` | Wait for artifactized CPU gates |
| 3 | Minimal M-SURFACE sanity on Granite | GATED | `artifacts/msurface/decision.json` | Only if hooks ready and surface drift lower |
| 4 | Minimal M-BRANCH sanity on Falcon | GATED | `artifacts/mbranch/decision.json` | Only if hooks ready and branch drift lower |
| 5 | Partial eval for survivors | GATED | TBD | Only smoke survivors |
| 6 | Full 12-trace + BCa | GATED | TBD | At most two finalists |

## CPU Artifact Tasks

| Task | Output Directory | Status | Gate Use |
|---|---|---|---|
| WJAC prefilter | `artifacts/wjac_prefilter/` | QUEUED | Kill WJAC only with at least 2/4 diagnostics |
| Funnel prefilters | `artifacts/funnel_prefilters/` | QUEUED | Decide LAMBDA/HYST and fixed smoke traces |
| M-SURFACE diagnostic | `artifacts/msurface/` | QUEUED | Decide whether to request tiny Granite hook run |
| M-BRANCH diagnostic | `artifacts/mbranch/` | QUEUED | Decide whether to request tiny Falcon hook run |
| Kernel design | `artifacts/kernel_design/` | QUEUED | Design only; no GPU kernel until finalist |

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
