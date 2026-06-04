# plans/MAC_PRELAUNCH.md — do the free work on the Mac before the GPU node

**Host = MacBook (Apple Silicon, MPS, 64 GB, NO CUDA).** Goal: do all CPU / cached / small-model work here for free, arrive at the GPU node with a **trust-verified pipeline + screened methods**, and spend the expensive card only on quantized-forward confirmation + the (user-operated) profiler trace.

This clone is **cache-rich (verified):** `experimental/` (~6.3 G — activation / KL / ParoQuant / drift packets for Channel-Set) and `results/` (~5.6 G — LatentWire source-private score caches + controls; ~677 result dirs). So **both papers' Stage-1 screening can run here with no model forwards.**

## ⚠ Cache-leakage protocol — do this FIRST (the one new requirement for a cache-rich clone)
You are screening **new** methods against **pre-existing** caches from the prior (near-negative) experiments. The trap: tuning on the same rows the prior paper reported on → overfitting/leakage that fakes a positive. As **Stage 0**, before any screen:
1. **Inventory** every cache under `experimental/` and `results/`: report each one's rows and whether it already has a clean dev/gate/confirm partition, or the prior run used the full test set.
2. **Freeze** a dev/gate/confirm partition over the cached rows; write `splits/*_hash`. **Quarantine the confirm slice** — the wrapped-`open()` access guard treats cached confirm-row files as off-limits during Stage 1/2 (the §14.5 leakage rule, now applied to cached files).
3. **Fit ALL selection on dev/gate only** — CVaR α, clip bounds, survival hazard threshold, WZ codebook, the combine-decode λ. Never on confirm rows.
4. **Run the planted controls on the cached data** (`planted_source_copy`, `planted_beyond_score`) to prove the pipeline catches overfitting/leakage *before* trusting any screen.
5. **A Mac "promote" is PROVISIONAL** — pending a held-out confirmation. If the prior caches used the full test set with no clean held-out, the confirmation needs *fresh* data → that's a GPU-node / fresh-generation step, not Mac-final.

## Do on the Mac (free) — BOTH papers
- **Infra + verified pipeline on synthetic.** Build the `pmc/` package, schemas, write-once dirs, access manifest, queue validator, aggregator, dashboards, the `gpu_daemon` *code*; pass the tiny/synthetic end-to-end, the **planted positive controls**, the **null-method sentinels**, and the **leakage guard**. This de-risks the #1 risk — agent-built infra — for free, before any GPU hour.
- **LatentWire — to a promote/kill decision.** L-ScoreComp (deployable WZ), L-B1 (damage-avoidance), L-A1/L-A2, the equal-byte baselines, destructive controls, and the combine-not-replace decode. Scoring uses small Qwen 0.5B/0.6B-class models on CPU/MPS; the receiver is a public decoder (no live big model). Reuse the existing C2C/KVComm MPS smoke as the high-byte anchors. Generate small-model MMLU-Pro / generated-solution score caches for a dev/gate slice.
- **Channel-Set — Stage-1 selection + offline recovery.** C-A1 (CVaR/EVT over the cached clip grid), C-F (survival selection scored vs cached static / EMA / ParoQuant recovery), CE13, CE21 no-gap audit, OSC-vs-drift. **Report per method: reached a real recovery number offline, vs needs a GPU quantized-forward to confirm.**
- Author all registry cards (`contribution_role` / `lineage`); build the paper skeletons; run litwatch/scoop (internet on the Mac, in a session isolated from experiment data).

## Park for the GPU node (the only expensive compute) — tag `parked: needs_gpu`
- Channel-Set **quantized-forward recovery confirmation** on real W4A16 models (especially the 30B MoE); cache **generation** for any config not already in the grids; **ParoQuant parity / DecDEC reconciliation** on real models; **ReasonPacket** training.
- **W4A16 / bitsandbytes anything** (no CUDA on Apple Silicon).
- The **user-operated native profiler trace** (systems card) — only you can run it.

## Mac entry prompt (give this to your Codex agent, from the repo root)
```text
Read AGENTS.md, then plans/MAC_PRELAUNCH.md, then plans/LOCAL_ASSETS.md, then the other plans/ files.
HOST = MacBook (Apple Silicon, MPS, 64 GB, NO CUDA). You are the Conductor; run as a /goal.

STAGE 0 FIRST (cache-leakage protocol): inventory every cache under experimental/ and results/; report
whether each has a clean dev/gate/confirm partition or used the full test set; freeze a dev/gate/confirm
partition over the cached rows (write splits/*_hash) and quarantine the confirm slice (the wrapped-open
guard must block cached confirm files in Stage 1/2).

THEN, locally and free: (1) build all pmc/ infra and pass the tiny/synthetic end-to-end, the planted
positive controls, sentinels, and leakage guard; (2) author all registry cards; (3) run the full
LatentWire first wave (L-ScoreComp deployable-WZ, L-B1, L-A1/A2, equal-byte baselines, destructive
controls, combine-not-replace decode) toward a promote/kill decision, scoring on small Qwen 0.5B/0.6B-
class models on CPU/MPS; (4) run Channel-Set Stage-1 selection + offline recovery (C-A1, C-F, CE13,
CE21, OSC) and mark each method offline-scorable vs needs-GPU; (5) build the paper skeletons.

Fit ALL selection on dev/gate only; treat every positive as PROVISIONAL pending held-out confirmation.
PARK with `parked: needs_gpu` everything requiring CUDA / W4A16 / 30B / long 20K-token decode /
ParoQuant-on-real-models / native profiler / ReasonPacket — do NOT attempt them here.
No confirmation-split access. Produce dashboard/conductor_state.md, dashboards, and a ranked
promote / kill / park list I can review.
```

## Handoff to the GPU node
`git commit` the **code + small artifacts + synthetic fixtures + registry + dashboards + the frozen `splits/*_hash`**. **Do NOT git-commit the multi-GB activation caches** (they're already on disk; use `git-lfs`/`rsync` or regenerate on the node). On the node: `git pull`, and Codex resumes only the `parked: needs_gpu` jobs + confirmations + the profiler trace.

## Row-mixed cache handling (refinement — the metadata_file_level freeze is too coarse for method runners)
The Stage-0 freeze quarantines whole FILES; method runners need ROW granularity within dev/gate. Policy:
- A) preferred: parse row-level entity IDs and filter rows by dev/gate/confirm.
- B) safe fallback: if a file is row-mixed and a parser is hard, quarantine the whole file.
- C) FORBIDDEN: read a whole row-mixed file and trust the file-level split.
Write `dashboard/cache_parse_coverage.md` (cache_family, files_seen, rows_seen, rows_dev/gate/confirm,
parse_status, usable_for_methods, blocked_reason). The confirm guard is enforced at BOTH file and row level;
add a test proving Stage 1/2 read zero confirm files AND zero confirm rows. Timebox parsing — quarantine hard
families and screen the parseable ones rather than blocking all screening.

## Mac result status taxonomy (no PASSED on the Mac)
Every Mac result is exactly one of: KILLED | AMBIGUOUS | CPU_SCREENED | PARKED_NEEDS_GPU |
PROVISIONAL_PROMOTE_TO_GPU. PASSED / paper-positive is impossible on the Mac; positives are provisional
pending held-out confirmation (which needs fresh data if the prior caches used the full test set).
