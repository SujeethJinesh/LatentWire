# Codex handoff bundle — how to use

Unzip at the **root of the target repo**. Layout:

```text
AGENTS.md                              # binding project instructions (read first; < 32 KiB)
README_HANDOFF.md                      # this file
scripts/check_handoff.py               # Step-0 integrity check (run before anything)
plans/CODEX_NEXT_72H.md                # the work order (the active execution plan)
plans/TEAM_OPERATING_SYSTEM.md         # Conductor, WIP, anomaly-mining, lineage, behavioral contract
plans/EXPERIMENT_SELECTION_SPEC.md     # EVI scheduling + anti-gaming + sequential futility + positive-method bias
plans/PLANTED_TESTS_SPEC.md            # positive controls (the pipeline must recover a planted effect)
plans/DATA_RETENTION_SPEC.md           # write-once evidence + provenance
plans/MAC_PRELAUNCH.md                 # if pre-launching on a Mac (no CUDA): Mac-vs-GPU split + cache-leakage protocol
plans/CAMPAIGN_BACKGROUND.md           # full rationale — read for *why*, NOT the work order
source_papers/latentwire_colm2026.pdf  # source material / sanity checks (context, not the work order)
source_papers/outlier_migrate_colm2026.pdf
# empty runtime dirs are also present: queues/ registry/ reviews/ ideas/ lessons/ litwatch/ paper/ dashboard/ results/ caches/ splits/
```

`AGENTS.md` is binding. `plans/CODEX_NEXT_72H.md` is the work order. `plans/CAMPAIGN_BACKGROUND.md` and `source_papers/` are rationale/source material, never the active work order.

## Orientation: this is a *positive-method* campaign
**Primary goal: one paper-strength *positive* method per paper** (Channel-Set: C-A1 / C-F / CE13; LatentWire: L-ScoreComp / L-B1 / L-A2). A bounded negative is the **fallback only** if the hard baselines kill the positive path. Every registry card carries `contribution_role` + `paper_claim_eligible`; only `positive_method`/`positive_enabler` cards are claim-eligible, and the GPU foreground is biased ≥75% to positive methods. Diagnostics / probes / parity / baselines / controls are support, never the contribution.

## Before starting Codex
1. Commit or snapshot the repo before unzipping.
2. Unzip this bundle at the repo root.
3. Run the integrity check (stalls if anything is misplaced or `AGENTS.md` exceeds the cap):
   ```bash
   python scripts/check_handoff.py
   ```
4. Confirm the **local GPU box** has the model weights, the existing cached activations/traces/scores from the two source papers, and the datasets — or the access paths you expect. If a cache/model/credential is missing, Codex must mark the job `PARKED`, not improvise.
5. Start Codex **from the repo root, on the GPU box** (see the local-GPU caveat below).

## First Codex prompt (this is `CODEX_NEXT_72H.md §11`)
```text
Read AGENTS.md fully. Then open plans/CODEX_NEXT_72H.md, plans/TEAM_OPERATING_SYSTEM.md,
plans/EXPERIMENT_SELECTION_SPEC.md, plans/PLANTED_TESTS_SPEC.md, and plans/DATA_RETENTION_SPEC.md.

You are the Conductor (run this as a /goal). Execute only Hours 0–6 first.

Before coding:
1. Run scripts/check_handoff.py — verify all referenced files exist at the exact paths and AGENTS.md < 32 KiB.
2. Confirm plans/CAMPAIGN_BACKGROUND.md is marked rationale-only (do not treat it as the work order).
3. Write dashboard/conductor_state.md with current objective, top-5 risks, and the next commands.

Then spawn the flat wave (subagents only spawn when asked; no nested spawning):
infra-schema-agent, gpu-daemon-agent, cpu-stage1-agent, latentwire-score-cache-agent,
channel-cache-agent, baseline-adversary-agent, dashboard-agent, review-panel.

Hard constraints:
- No full experiment, no confirmation-split access, no method GPU job.
- Only: tiny infra, schemas, write-once dirs, access manifest, queue validator, paper.init skeleton,
  baseline tiny-lock scaffolding, planted-test harness (synthetic), and GPU backfill-queue code.
- GPU runs backfill (repro+leakage-reviewed) only; report unexpected idle > 5 min.

Return at the Hour-24 gate with: dashboard/morning_brief.md, breakthrough_board.md, kill_board.md,
conductor_state.md, and the next six commands for the local GPU runner.
```

## Local GPU caveat (non-negotiable)
Codex Cloud sandboxes have **no GPU, no local weights/caches, and no network in the agent phase** — they may only write code, schemas, runners, dashboards, and queue files. **GPU experiments run only through the local runner on the GPU box.** Drive the GPU side from a **local Codex CLI session on the GPU box** (the Conductor `/goal`), not a cloud sandbox. The scoop/litwatch/novelty agents need a *separate* internet-enabled, allowlisted session, isolated from experiment data. No Codex task may fabricate or substitute a missing GPU result.

## Human oversight policy (you cannot fully step away)
Reasonable to let it churn overnight **after** the tiny infra pass is clean and the GPU daemon/backfill is running. Check at Hour 0–6, the Hour-24 gate, and then daily. **The first 24 h is agent-built infrastructure — review the schemas/validators/planted-tests yourself before authorizing any real run; that is the single biggest risk in the plan.** You must intervene for:
- any PARKED item, guardrail conflict, or sentinel/leakage failure;
- missing model weights / datasets / permissions;
- approval of internet-enabled litwatch/novelty sessions;
- every ≤4 Stage-2 promotions (a hard checkpoint);
- any final positive paper claim or decision to read confirmation splits or submit results.

Review the **decision artifacts**, not raw logs:
```text
dashboard/morning_brief.md   dashboard/conductor_state.md
dashboard/breakthrough_board.md   dashboard/kill_board.md   queues/parked.yaml
```
Per OpenAI's own Goals guidance, review diffs and re-run tests — do not trust the agent's summary.

## Note on provenance
This bundle supersedes any earlier zip. It carries the **positive-method reframe** (contribution roles + ≥75% positive-method GPU bias), the **deployable Wyner–Ziv** split for L-ScoreComp (source-minus-target residual is an oracle upper bound only), the stricter **TrustPacket leakage gate** (MI < 0.10 bits), the **EVI anti-gaming caps + conditional-power futility rule**, concrete **planted-test thresholds**, the **job-type-tiered review gate**, and **`scripts/check_handoff.py`**. If you have an older zip with files at the repo root or without `plans/DATA_RETENTION_SPEC.md` / `scripts/check_handoff.py`, discard it and use this one.
