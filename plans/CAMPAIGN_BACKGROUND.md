> # ⚠ THIS FILE IS BACKGROUND / RATIONALE — repo role: `plans/CAMPAIGN_BACKGROUND.md`
> **Do NOT hand this file to Codex as `AGENTS.md`.** It is ~50 KB, far over Codex's **32 KiB `project_doc_max_bytes`** cap, so it would be **silently truncated** (the project-specific guidance at the end drops first). The operational handoff is three lean files: **`AGENTS.md`** (binding rules), **`plans/CODEX_NEXT_72H.md`** (the current ExecPlan Codex acts on), and **`plans/DATA_RETENTION_SPEC.md`** (write-once/provenance). Read *this* file for *why*; act on the ExecPlan for *what to do now*. **Where this file's first-night/queue text disagrees with `plans/CODEX_NEXT_72H.md`, the ExecPlan wins.**
>
> **Execution locality (binding):** Codex Cloud/CLI tasks may only do *code* (runners, tests, schemas, cards, aggregator, dashboards, queue files, paper tables) — cloud sandboxes have **no GPU, no local weights/caches, and no network in the agent phase**. **GPU experiments run only through the local runner on the GPU box.** No Codex task may fabricate or substitute a missing GPU result. **Subagent reality:** Codex spawns subagents *only when explicitly asked* (`max_threads=6`, `max_depth=1`), so there is **no nested spawning** — a worker cannot spawn its own fix subagent; on failure it writes to `queues/parked.yaml` and the top-level session re-dispatches. Rewrite every "spawn subagents" instruction below as explicit Codex task prompts.

> ## ⟳ REVISION 2026-06-02 (post-scoop) — READ BEFORE ACTING; THESE OVERRIDE STALE PRIORITIES BELOW
> A scoop check + methods validation (early June 2026) changed the priority order and killed/added work. Where this banner conflicts with older tier/first-wave text further down, **this banner wins** (the body has been reconciled, but read this first).
>
> **Citation fixes (applied):** relative representations is **2209.15430** (Moschella et al., ICLR 2023), NOT 2305.13093 (that ID is an unrelated image-restoration paper); ParoQuant code is the **z-lab** project (<https://z-lab.ai/projects/paroquant/>), NOT thu-nics (thu-nics owns C2C) — verify exact repo; TurboQuant is **2504.19874** (ICLR 2026), not a 26xx preprint. ParoQuant ICLR'26 status confirmed; Qwen3-4B AIME-24 numbers verified (AWQ 62.2 / ParoQuant 73.3 / FP16 75.6).
>
> **New constraining neighbors:**
> - **Latent Cache Flow (2605.22863)** — no-text model-to-model communication that builds an accuracy-vs-communication-budget frontier on MMLU-Pro/ARC-C and reports LCF-256 *exceeding an oracle routing frontier*. This is a **near-full scoop of the LatentWire "byte–capability frontier" spine** and overlaps L-A. LatentWire must re-anchor on **score-coding (L-ScoreComp) + trust/routing (L-B)**, cite LCF as a high-byte anchor, and (caveat) note LCF's budget axis is adapter size / latent width / TTFT on small Qwen pairs (0.5–0.6B), not literal bytes/query, and is self-described as preliminary.
> - **OSC (2604.12782)** claims activation outliers are **token-persistent** (fixed channels across tokens) and **DecDEC (2412.20185)** already exploits **dynamically shifting** salient channels at decode with <0.0003% added memory / ~1.7% slowdown. The drift thesis must be defended against OSC (C-D1 is now load-bearing, not inoculation), and the ImpactSidecar is dominated by DecDEC.
>
> **Priority changes (binding):**
> - **PROMOTE:** C-A1 (CVaR clip — reframed as a *per-model* tail-robustness study, not a universal objective; tail/CVaR objective is genuinely open) · L-ScoreComp (reframed as **the lead method** — a Wyner–Ziv/Slepian–Wolf treatment of inter-model score/logit communication) · L-B (trust/routing — safest positive; UCCI 2605.18796 occupies only the *scalar* confidence point, so a multi-byte risk packet is still open).
> - **DOWN-SCOPE:** C-A2 (horizon rotation → run a 2-bin prefill-vs-decode ablation FIRST; escalate to a smooth mixture only if it beats global rotation by >1pt on ≥3/4 models — the draft's own broadband drift entropy 0.85–0.89 + √t KL growth predict no sharp regime boundaries; TR-DQ 2503.06564 / Q-Drift 2603.18095 succeed in diffusion only because of engineered timestep schedules) · C-A3 (BranchRot → cheap ablation only; Fig 2's comparable cross-branch drift predicts "global suffices").
> - **DROP / PARK:** C-B ImpactSidecar (all sidecar variants) — dominated by DecDEC and by its own 54 MiB / 7.5 MiB-per-token envelope; revive ONLY if streamed cost falls ~100× (to tens of KiB/token) AND the selector beats DecDEC on a headroom gate.
>
> **Second independent review (2026-06-02, folded in):** confirmed every priority above and added — (i) a **binding C-A2 correctness gate** (interpolated rotations are not orthogonal; use geodesic/library, test full-precision equivalence); (ii) a **frozen tail-validation split + EVT/GPD estimator** for C-A1 (CVaR is fragile at small trace counts); (iii) the rule that **L-ScoreComp must be a Wyner–Ziv side-information code** (raw VQ/CountSketch is dominated, and now crowded by CU-HLM/TK-SLT/SMP); (iv) **L-A2 verifier-rerank > L-A1** (higher candidate entropy); (v) **L-B reframed to damage-avoidance** (conformal routing is crowded by UCCI/RouteNLP/CP-Router/C3PO); and new baselines ReSpinQuant/SingleQuant/SERQ/ResComp/EAQuant (Channel-Set) and CU-HLM/SMP/LMNet/LatentMAS (LatentWire).
>
> **Third independent review (2026-06-02, consolidated):** three reviewers now converge. Added — a **new non-rotation Channel-Set bet (C-F StableCore + survival-selected core)** that minimizes the paper's own error integral and diversifies the rotation-heavy pool; the **combine-not-replace decode rule** for all LatentWire packets (value-of-information); the **data-processing-inequality** formal kill for source-copy packets (§2b); a **mandatory no-gap filter** (CE21); **DarkForest 2605.25188** as a new L-B threat; and the **V = 2H+2N+2B+C−2S−2F−G** exploratory ranking rubric (§13.3, exploratory only).
>
> Full reasoning: companion scoop-check & methods-validation report.

# Positive-Method Campaign v2: Execution Plan

**Papers:** (1) *Channel-Set Drift in Long-Reasoning W4A16* and (2) *LatentWire* (pivoting to the byte–capability frontier of inter-model communication).
**Target:** Efficient Reasoning workshop @ COLM 2026 — <https://wdlctc.github.io/efficient-reasoning-2026/> (deadline **July 12, 2026 AoE**, non-archival, welcomes negative/ongoing work, **self-contained 4–10pp main text** → decisive gates and limitations must live in the body, not appendices).
**Scoop landscape as of:** early June 2026 (revised). **Re-run the scoop check the week before submission** — the most constraining neighbors are Apr–Jun 2026 preprints that postdate the drafts: **Latent Cache Flow (2605.22863)** for LatentWire (near-full scoop of the frontier spine), **OSC (2604.12782)** as a direct challenge to the drift thesis, plus KL-Lens (2604.13440), UCCI (2605.18796), ConQuR (2605.10793), LAQuant (2605.08755), InfoQuant/PSOT (2605.26175).

---

## ▶ OPERATING DIRECTIVE — AUTONOMOUS AGENT (READ FIRST, ACT ON THIS)

**If you are an autonomous coding agent (e.g., Codex), this file is your standing work order, not just documentation.** Read it end to end, run the bootstrap, then run the continuous loop below until the termination condition is met. **Deploy parallel subagents** (roles in Section 13.4). **Do not stop after one task. Do not wait for a human prompt between tasks.** Keep iterating: while any *audited, promotable* work exists, the queue must not be empty. **Idle GPU is a bug; idle agents do idle-work (Section 14.7), never unregistered jobs.**

**Build the engineering layer first (Section 14, IMPLEMENTATION_SPEC).** This strategy file defines *what to test and why*; Section 14 defines *exactly what to build, which commands must pass, valid file formats, blocking vs non-blocking jobs, failure budgets, and leakage guards*. Do not launch any full experiment before Section 14's tiny end-to-end pass succeeds.

> **First PR must only build Section 14 Milestone 1 on `--tiny`.** No full experiment until the tiny end-to-end pass, the numeric result schema (14.4), the leakage guard (14.5), the failure budget (14.6), the null-method sentinels (14.6), and the baseline locks (Phase A) all work.

### Bootstrap (cold drop-in, do this once, in order)
1. Read this file fully. Treat Sections 7 (gates), 14 (orchestration), and the guardrails below as binding.
2. Scaffold the repo: `registry/`, `splits/`, `caches/`, `results/`, `paper/`, `logs/`, `dashboard/`.
3. **Stage 0 freeze** (Section 1): lock splits, write `manifest.json`, `baselines.lock`, `splits/*_hash`, and a dated `scoop_check.md`. Nothing is confirmation-eligible before this exists.
4. Author one **registry card** (Section 6 schema) for every method in Sections 3, 4, 9, and 11.
5. Build infra (Section 13.3): job launcher, runner contract, audit gate, aggregator, dashboard — each with a smoke test that must pass before use.
6. Run **Phase A** (below) — Wave-0 shared capture (Section 13.1) **and** baseline reproduction + `baselines.lock`. Then enter the loop.

### Phase order — baselines first (Phase A is blocking)
Run in three priority tiers. **Phase A is blocking: no method queues for a paper until that paper's baselines are reproduced and locked.** This is deliberate — every gate in Section 7 is defined *relative to a baseline*, so without locked baselines Stage-1 has no yardstick to kill against and the audit gate's baseline-parity check cannot fire.

- **Phase A — baselines + capture + framing (highest, blocking).** Build the harness; reproduce each baseline to within tolerance on the frozen splits; write `baselines.lock`; run Wave-0 shared capture; **and** draft the baseline / related-work / positioning prose + the Section 5 apples-to-apples table. This is the **baseline-adversary agent's** founding job.
  - **Minimal critical set that must reproduce to unblock methods** — Channel-Set: BF16, static W4A16, static top-1/top-10, and local-ParoQuant-style (or official ParoQuant if immediately available). LatentWire: source-index + source-index+confidence + equal-byte score-sketch. **DecDEC is NOT blocking** (see Section 14.7) — it is required before final claims, not before screening, so a missing DecDEC dependency never stalls night-1 DriftRot screening.
  - **Heavy baselines** (C2C, KVComm, LAQuant, ConQuR, ResQ, KL-Lens) reproduce in parallel, **non-blocking**: methods may screen once the critical set is locked, but a method is **never confirmed (Stage 3) against a baseline that is not yet locked**.
  - **Tolerance (default, editable per baseline in `baselines.lock`):** within ±2% accuracy / ±0.05 recovery of the paper's reported number on our split. A baseline that will not reproduce within tolerance is flagged for human review, not silently used as a target.
- **Phase B — the methods funnel** (Sections 1–11, 14): screen → gate → confirm, each scored against the locked baselines.
- **Phase C — methods/results prose (lowest, idle-fill only).** Paper work is split by priority: **baseline / related-work / positioning rides with Phase A (high); methods / results / limitations prose is Phase C idle-fill.**

### The continuous loop (run until termination; never block on a single item)
```
loop:
  # 1. keep the pipeline full
  if unbuilt cards exist:        builder subagents implement + smoke-test them (parallel, one per family)
  if built cards exist:          run Stage-1 CPU/cached screen (parallel) → emit runner-contract JSON
  baseline-adversary agent:      score every screened method vs hard baselines + controls
  audit agent:                   gate Stage-1 passers → only AUDITED jobs may enter the GPU queue

  # 2. saturate the one GPU (foreground first; backfill keeps it hot; review-gate before launch)
  while gpu_foreground non-empty:
      run next AUDITED + REVIEW-PASSED job (serialized; see §1.5 pre-launch review gate)
  while gpu_foreground empty AND experiments unblocked:
      run next pre-approved BACKFILL job (reusable cache / baseline / parity / profiler; repro+leakage-reviewed)  # never make-work, never KILLED methods
      # watchdog (§2.5) launches backfill on idle; backfill is preempted only at checkpoints by foreground
      on FAILURE: append to queues/parked.yaml; the TOP-LEVEL session re-dispatches a fix prompt (no nested spawning at max_depth=1) AND immediately advance
      record bookkeeping (hashes, wall-time, utilization)

  # 3. decide, then REFILL (this is what makes it "always iterating")
  aggregator: apply gates (Section 7) + Holm (confirmatory) / BH (exploratory dashboard) → promoted.md / killed.md / ambiguous.md / leaderboard.csv / morning_brief.md
  promote survivors to the next stage (Stage-2 → Stage-3 → Stage-4 systems card)
  if GPU queue is draining:  generate the NEXT WAVE of work, in this priority order:
       (a) promote current survivors to their next stage
       (b) instantiate more variants from the extended pool (Sections 11) — new seeds, budgets, models, lateral families
       (c) if no methods remain to screen or promote: switch ALL agents to idle-work (below)

  # 4. idle-work — agent time is never wasted
  # 4. standing teams (cadence-gated; see plans/CODEX_NEXT_72H.md §8)
  on each result wave:  Ideation Lab proposes new lateral methods (math+adversary+scoop) → gated registry cards
  on each KILL/PARK:    Failure-Learning team writes a post-mortem → lessons/LESSONS_LEDGER.md (consumed by Ideation + Review)
  continuously (network-isolated): Literature & Best-Practices Watch refreshes scoop_check + best_practices/
  idle-work: Paper team updates paper/ (methods, results, limitations, ablation + kill tables), regenerates figures;
             on milestones the Paper Peer-Review Board (mock COLM) scores + iterates the paper to target; write morning_brief.md

  if termination_condition(): break
  else: continue loop
```

### Subagent deployment
**Codex spawns subagents only when explicitly asked** (`max_threads=6`, `max_depth=1`), so issue the roles as **explicit, flat task prompts** from the top-level session — not as auto-spawned background daemons. The concrete prompts live in `plans/CODEX_NEXT_72H.md §1`. Roles: **infra**, **builder (one per method family)**, **baseline-adversary**, **audit (GPU gate)**, **GPU queue manager (main)**, **stats**, **repro**. **There is no nested "fix subagent"** (depth is capped at 1): a failing worker appends to `queues/parked.yaml` and the **top-level session re-dispatches** a fix prompt, then advances. Cloud workers do code only; GPU work runs on the local box (see the locality rule at the top).

### Termination condition (stop the loop only when ALL hold; otherwise keep iterating)
1. every registered method is in a terminal state (`PASSED`, `KILLED`, or `PARKED`); **and**
2. every `PASSED` survivor has a Stage-3 held-out confirmation and a Stage-4 systems card; **and**
3. `paper/` reflects current results (methods, results, limitations, gates table) **and has cleared the Paper Peer-Review Board's mock-COLM target** (all mock reviewers ≥ weak-accept, AC accept, Honesty/Trust veto passed); **and**
4. `scoop_check.md` was refreshed within the last 72 h; **or**
5. the deadline guard fires (≤ 24 h to July 12 AoE).
After termination, enter **watch mode**: periodic scoop refresh + paper polish, surface a final report, and await human review.

### Non-negotiable guardrails (the queue is emptied by reaching legitimate terminal states — NEVER by skipping these)
- **Never bypass the audit gate.** Unaudited work does not touch the GPU.
- **Phase-A gate:** no method enters the GPU queue for a paper until that paper's `baselines.lock` exists and its minimal-critical-set baselines reproduce within tolerance; no method is confirmed (Stage 3) against an unreproduced baseline.
- **Never read a confirmation split during screening.** A method's registry card must predate any confirmation read.
- **Never relax a gate, lower a threshold, or fabricate a number to make something pass or to drain the queue.** A method that cannot clear Section 7 honestly is `KILLED` or `PARKED`, not promoted.
- **Honor the do-not-queue list** (Section 13.7) and the kill-only tiers.
- **Repeated failure → `PARKED` for human review**, never silently dropped.
- **Apply Holm** (confirmatory paper claims) over the confirmed family — BH only for exploratory dashboards, labeled exploratory — and disclose how many variants were screened.
- **No "beats ParoQuant" without the parity card.** Methods may screen against local-ParoQuant-style, but no paper text claims "beats ParoQuant" unless `paroquant_parity_card.md` exists recording official-code parity or exact divergence (C-E1). ParoQuant is the strongest baseline; this is mandatory, not optional.
- **Human checkpoints:** emit `morning_brief.md` each cycle; any `PARKED` item, guardrail conflict, or "final positive" claim requires human sign-off before it is treated as a paper result.
- **No make-work, but never idle either.** The GPU runs an audited foreground job **or** a pre-approved, checkpointable **backfill** job (reusable caches, baselines, ParoQuant parity, OSC/DecDEC stress, candidate pools, profiler rows) — see `plans/CODEX_NEXT_72H.md §2.5`. **Never launch an unregistered job, and never re-run a `KILLED` method, merely to keep the GPU busy**, but **unexpected idle GPU > 5 min is a scheduler bug** (report it). CPU agents do idle-work (Section 14.7) when their lane is free.
- **No experiment launches without a passing multi-subagent pre-launch review record** (`reviews/<method_id>@<code_sha>.json`, all of repro/stats/leakage/baseline-adversary PASS; backfill needs repro+leakage only). This is *in addition to* the post-run audit gate, which re-verifies the record exists, matches the code hash, and predates the run. The review enforces reproducibility (seeds/hashes/write-once/no-confirm-access) **and significance** (pre-registered gate, BCa+Holm, and a minimum-detectable-effect/power check that refuses underpowered confirmations). See `plans/CODEX_NEXT_72H.md §1.5`.
- **Failure budget is hard (Section 14.6).** Respect `max_retries_per_job`, per-stage GPU-hour caps, and `max_wall_clock_per_baseline_repro`; on exhaustion → `PARKED`, do not keep grinding (e.g., do not spend a night fixing C2C when LatentWire score packets are unscreened).
- **Leakage is mechanically enforced (Section 14.5), not just promised.** Confirmation-split paths are absent from the environment during Stage 1/2; runners fail if they open any `*_confirm*` file; the audit agent checks the wrapped-`open()` access manifest. Critical for score-codebook methods, which overfit held-out rows easily.

---

## 0. Two hero hypotheses (the front page)

The campaign exists to produce **one paper-strength positive method per paper**, not a table of lightly tested heuristics. Everything below serves one of two hypotheses:

**Channel-Set hero hypothesis.** *Fixed rotation is the strongest baseline, but long-reasoning drift creates horizon / branch / tail regimes that a static rotation policy does not handle. **Adaptive DriftRot** — a horizon-conditioned, branch-aware, tail-risk-selected rotation policy — improves held-out recovery or tail risk over a ParoQuant-style baseline.*

**LatentWire hero hypothesis.** *Low-entropy 4-way MCQ collapses to source-index transfer; byte packets become useful only on high-entropy ranking or trust/routing tasks where a selected label is insufficient. A byte packet beats source-index, source-index+confidence, and same-byte score sketches on such a task under destructive controls.*

> **Post-scoop caveat (2026-06-02).** The cleanest realization of this — an accuracy-vs-bytes frontier on MMLU-Pro — is now partly occupied by **Latent Cache Flow (2605.22863)**, which reports beating an oracle routing frontier. Re-anchor the hero claim on the *least-scooped* axes: **score-vector compression as the method (L-ScoreComp, Wyner–Ziv framing)** and **byte-budgeted trust/routing (L-B)**. A bare "packet beats source-index on MMLU-Pro rerank" is no longer sufficient novelty on its own.

By submission, the **primary target** for each paper is a positive method clearing its hero hypothesis on a held-out split with destructive controls and a systems card. A sharply bounded negative — showing why the adaptive method fails after controlled gates — is the **fallback only if the hard baselines kill the positive path**, not a co-equal goal. (Two positive contributions of a *different kind* also stand on their own: the decode-time drift measurement, and the byte-capability frontier under destructive controls.) Everything else — diagnostics, probes, parity, baselines, controls — is support that exists to make the positive claim credible.

---

## 1. Campaign structure: the funnel

Screen ~20 variants cheaply and in parallel; promote only survivors to expensive compute. Three rules keep this from becoming 20 unfinished experiments:

1. **Pre-registration.** Every variant gets a registry card (Section 6) *before* it touches the confirmation split.
2. **Cheap-first ordering with a hard asymmetry: headroom can KILL, it cannot PROMOTE.** Stage 1 proxies only eliminate impossible methods; promotion requires real quality evidence at Stage 2+.
3. **Honest correction.** At confirmation, apply Holm (confirmatory) over the whole family that reached Stage 3 (BH only for exploratory dashboard ranking), and disclose how many variants were screened.

| Stage | Purpose | Run count | Output | Hard rule |
|-------|---------|-----------|--------|-----------|
| **0. Freeze** | Lock splits, caches, registry, baselines, seeds. | once | `manifest.json`, split hashes, method cards, `baselines.lock`, `scoop_check.md` | No method is paper-eligible unless its card existed before the confirmation split was read. |
| **1. Headroom / ceiling** | Kill impossible methods cheaply (CPU, parallel, near-free). | ~20 variants | proxy JSONs | **Headroom can kill; it cannot promote.** |
| **2. Small live gate** | First real quality check (CPU / 1 GPU). | ~6–8 survivors | recovery / metric + controls | Only **3–4** survive. |
| **3. Confirmation** | Paper-quality evidence (multi-model / held-out; GPU serialized on one card). | ~2–3 promoted | held-out, multi-model, BCa + Holm, tail/CVaR, controls | This decides the paper. |
| **4. Systems card** | Workshop credibility (GPU). | survivors only | tokens/s, HBM, bytes, MACs, ≥1 profiler row | One real profiler row if at all possible. |

```
   STAGE 1 (CPU, parallel)                STAGE 2 (CPU/1-GPU)      STAGE 3 (multi-model/held-out)   STAGE 4 (GPU)
   priority 20 + cached pool               ~6-8 survivors           ~2-3 promoted                    survivors
   headroom / info-ceiling / controls  →   recovery R / metric  →   held-out + 2-4 models        →   tokens/s, HBM,
   KILLS ONLY, no promotion                + destructive controls   BCa + Holm + tail/CVaR (1 GPU)   bytes, MACs
```

**Parallelism model.** Stage 1 is embarrassingly parallel: one job per (variant × model × seed) reading *cached* activations/traces or cached source scores. Use a job array (Slurm `--array`, GNU `parallel`, or Ray), one JSON per job into a shared dir keyed by `method_id`. A nightly aggregator applies the gates (Section 7) and emits the promote/kill list. Stage 3/4 jobs are scheduled only for survivors.

**Execution ordering (dependency-based, not calendar):** Freeze → cheap screens (kill ≥ half) → small live gates (promote ≤2 per paper) → held-out confirmation → systems cards + figures → final scoop re-check before assembly.

---

## 2. The two near-free pre-filters (run first; they gate everything, kill-only)

### 2a. Channel-Set: oracle-headroom filter (no training, cached activations only)
For any candidate selector S, compute the best achievable recovery if S's selected set were corrected perfectly, from cached BF16 vs W4A16 activations. **Kill** S if `oracle_headroom(S) ≤ oracle_headroom(norm)` **and** `≤ oracle_headroom(KL_Lens_layer_selector)` **and** `≤ oracle_headroom(random)`. Promotion requires ≥25% more headroom than norm/PCA/random (Section 7), but headroom alone never promotes. *Note:* KL-Lens is a layer/component sensitivity method, so `KL_Lens_layer_selector` is an **adapter** — define an explicit layer-sensitivity → per-column/sidecar score mapping (Section 14.1); it is not a native per-column selector.

### 2b. LatentWire: information-ceiling filter (no model, cached source scores only)
`I_beyond = I(source_score_vector ; correct_answer | source_top1)`. If `I_beyond ≈ 0` on a task (e.g. 4-way MCQ), **no** packet can beat source-index there — kill the (task, method) pair. This is the go/no-go that tells you which tasks have room.

**Companion statistic at confirmation:** `Δ_beyond_label = Metric(packet) − max(source-index, source-index+confidence, same-byte-score-sketch)`. If not positive on a frozen held-out split, the method is a packetized label — kill it.

**Formal kill (data-processing inequality, 2026-06-02 consolidation):** if the packet `Z` is a deterministic function of the source's selected answer `A_S`, then `I(Y; Z | X, A_S, R) = 0` — no decoder already holding the public input `X`, receiver state `R`, and `A_S` can be beaten by `Z`. This is the exact reason a 4-way packet that follows source top-1 (the old draft's 0.995–0.999) cannot beat source-index, and it generalizes: a packet helps **only** by carrying information about `Y` that survives conditioning on `(X, A_S, R)`.

---

## 3. Paper 1 — Channel-Set Drift

**Existing asset (the anchor):** strict top-1% channel-set leaving of 53–67% over long decode across four reasoning models, replicated on AIME-2025 and MATH-500. The new work adds a positive method on top.

**Fact pattern that constrains the method.** ParoQuant-style rotation recovers 0.754 (Granite), 1.05 (Nemotron), 0.756 (DeepSeek), 0.381 (Falcon); budgeted EMA is model-dependent; hard switching, plain EMA, DecDEC-style reactive selection, residual-norm correction, internal-surface selection, and naive ParoQuant+EMA stacking are weak or killed. So the headroom is concentrated in **tails** (Granite worst-trace, DeepSeek negative lower CI) and on **Falcon** (only 0.381).

**Scoop-adjusted framing (read before picking variants).**
- *"KL beats norm/MSE for quant sensitivity" is published* (KL-Lens, Slim-LLM, LLM-MQ). KL-selection is a **cited baseline**, not a headline; the sidecar survives only as a *rotated-basis, per-column, drift-conditioned* component.
- *"Optimized rotation" is saturated* (ParoQuant, ConQuR, DartQuant, DFRot, SpinQuant, QuaRot, RRS). Never pitch "we learn a better rotation." Only open axis: **decode-horizon / within-trace drift + tail-risk.**
- *"Step-conditioned rotation" exists in diffusion* (TR-DQ, Q-Drift); porting to LLM long-reasoning decode position is open but must cite them.

### C-A: Adaptive DriftRot — PRIMARY BET (one method family, three variants)

The hero method. **Do not** present CVaR / horizon / branch as three separate headline methods — they are one thesis: *fixed rotation helps, but long decode needs regime-specific rotation selection.*

| Variant | What changes | Stage-1 CPU proxy | Stage-2 gate |
|---------|--------------|-------------------|--------------|
| **A1. CVaR clip/scale selection** | choose clip/scale by worst-trace / worst-20% risk, not median | precomputed per-trace ΔNLL grid | improves CVaR vs ParoQuant without median loss |
| **A2. Horizon mixture** | early/mid/late decode rotations, smoothly interpolated (spline/EMA on position), no hard switch | bucketed reconstruction/KL | beats fixed rotation **and** wrong-horizon control |
| **A3. BranchRot** | separate rotations/scales for attention / SSM-Mamba / MoE branches | branch-local Procrustes/KL on Granite/Falcon captures | branch-local > global on Granite/Falcon |

**Promote only if:** median recovery improves meaningfully over ParoQuant on the same traces, **and** CVaR/worst-trace improves, **and** wrong-horizon/random-clip controls fail, **and** ≥2 models show non-negative transfer.
**Kill if:** it only re-fixes Granite's known tight-clip effect, hurts DeepSeek/Falcon tails, or cannot beat fixed ParoQuant at the same budget.

**Post-scoop structural & priority notes (2026-06-02, binding):**
- **A1 (CVaR) — PROMOTE, but reframe.** No quantization paper was found using a tail/CVaR objective (DartQuant 2511.04063 tries variance/kurtosis/Whip and finds them ~flat), so this axis is genuinely open. *But* the draft already shows tight-clip helps Granite and regresses DeepSeek/Falcon — a single CVaR target will re-fit whichever model dominates calibration. Run CVaR **per-model**, report whether the selected α transfers; a non-transferring α is itself a publishable negative. This is the highest-ROI Channel-Set bet. **(R2) CVaR is statistically fragile at small trace counts, and choosing the tail policy and evaluating the tail win on the same traces overfits.** Use a frozen tail-validation split, report the no-gap fraction, and prefer a *parametric* tail estimator: fit a Generalized Pareto Distribution (peaks-over-threshold) to the per-trace ΔNLL tail and select the clip on the GPD tail-quantile, which extrapolates from few samples far better than an empirical worst-20%. EVT-for-clip in this decode-tail setting is open (existing clip work — PACT, percentile — is not tail-parametric); note DFRot 2412.00648 already does *distribution*-tail rotation, so frame the novelty strictly as the *decode*-tail objective.
- **A2 (horizon mixture) — DOWN-SCOPE.** The draft's own diagnostics argue against sharp position structure: drift entropy 0.85–0.89 (broadband) and KL growth ≈√t (sublinear, no regime boundaries). Position-dependent rotations need sharp regimes to pay for themselves. **Run a 2-bin prefill-vs-decode ablation first**; escalate to a smooth spline/EMA mixture only if 2-bin beats a single global rotation by >1pt on ≥3/4 models. The diffusion analogues (TR-DQ 2503.06564, Q-Drift 2603.18095) work because diffusion has engineered timestep schedules; AR decode has none. **(R2, BINDING CORRECTNESS GATE — build before any A2 run):** a linear interpolation of two rotation matrices is **not orthogonal**, so naive "smoothly interpolate rotations" silently breaks full-precision equivalence. Any horizon mixture must (a) interpolate *on the rotation manifold* — geodesic/SLERP via matrix `exp`/`log` on SO(n), a Cayley-transform path, or a smooth-gated *library* of exact orthogonal rotations — never a convex combination of matrices; and (b) compose the chosen rotation consistently around **every** affected linear map, residual addition, normalization placement, KV/cache consumer, and branch merge. Add `tests/test_rotation_orthogonality.py` (interpolants orthogonal to tolerance) and a full-precision-equivalence test at the unquantized limit; A2 does not enter the GPU queue until both pass. Note the rotation-calibration space is saturated (ConQuR/DartQuant/SingleQuant/DFRot all use *exact* Givens/Procrustes transforms), so OT/Bures or geodesic constructions are A2 *implementations*, not independent novelty.
- **A3 (BranchRot) — CHEAP ABLATION ONLY.** Figure 2 shows comparable strict set-leaving across attention/SSM/MoE branches, so branch-differentiated *rotation* (which targets drift) likely won't beat global rotation; KL-Lens 2604.13440 shows branches differ in *sensitivity*, not drift. Expected outcome "global suffices"; run only to close the reviewer question, not as a headline bet.

### C-B: ImpactSidecar — DROPPED / PARKED (post-scoop 2026-06-02), revive only on a ~100× bandwidth cut

> **Why dropped.** **DecDEC (2412.20185, OSDI'25)** already performs dynamic salient-channel residual correction with <0.0003% added memory and ~1.7% slowdown — it dominates this concept on exactly the gain-per-byte axis the sidecar is gated on. The draft's own envelope (54 MiB stored, ≈7.5 MiB/token streamed if not cache-resident) reintroduces the memory-bandwidth bottleneck in the decode-bound regime quantization exists to relieve. No selector (KL/Fisher/output-impact) changes that arithmetic at fixed MiB/token byte cost. **Do not build any sidecar kernel** unless streamed cost is first cut ~100× (to tens of KiB/token) AND the selector wins a headroom gate against DecDEC. CE18 gain-per-byte stays in the screen as a cheap diagnostic only.

The prior residual screen failed because residual *energy* was the wrong selector; the draft itself says a future residual method must show KL-lookahead headroom, pick columns materially different from the killed proxy, and confirm held-out. **Do not build a kernel until the selector wins.**

> Select tiny rotated-basis residual sidecar columns by output-impact: KL reduction, finite-difference loss impact, or Fisher/activation sensitivity.

Selectors compared: norm · PCA/ResQ-like · KL-lookahead · finite-difference output-impact · random matched-layer · ParoQuant alone.
**Promote if:** `KL_headroom(selector) > 1.25 × KL_headroom(norm)` **AND** `KL_headroom_per_byte(selector) > KL_headroom_per_byte(norm/PCA/random)`, selected-column overlap with norm < 70%, and small-live-gate improves ParoQuant on positive-gap traces without tail regression. **Only the gain-per-byte sidecar reaches Stage 2** — the draft's own envelope (a Granite top-8×32 sidecar stores ≈54 MiB and streams ≈7.5 MiB/token if not cache-resident) means recovery alone is not enough; a reviewer will reject accuracy bought with an unrealistic sidecar.
**Kill immediately if:** KL selector ≈ norm columns, low output-impact headroom, or it improves reconstruction but not next-token KL/recovery.

### C-C: BudgetRouter — FALLBACK (pragmatic, if adaptive rotation is noisy)

> Allocate a fixed protection/sidecar budget across layers and decode horizons using measured drift entropy, recoverable gap, and no-gap fraction.

Builds directly on "top-10 budgeted EMA succeeds on Nemotron but not uniformly on DeepSeek/Falcon." **Promote if** matched-average-budget routing beats uniform top-10 or static top-K on ≥2 models.

### C-F: StableCore + DynamicFrontier — NEW non-rotation bet (consolidated 2026-06-02), survival-selected core

> Statically protect a **stable core** of channels cheaply; spend the *dynamic* budget only on the **leaving frontier**. The core is chosen by **channel survival / residence time**, not instantaneous magnitude: protect the K channels with the longest expected residence in top-K (lowest hazard of leaving). (This consolidates the v2.2 stable-core/dynamic-frontier idea CE20 with the survival/hazard selector.)

**Why it is principled (it minimizes the paper's own error integral).** With protected set `S(t₀)`, current important set `S(t)`, and `L(t₀,t)=1−|S(t₀)∩S(t)|/K`, the paper's model gives `E[err(t)] ≈ K{ε_p(1−L)+ε_q L}`. Total excess error `∝ ∫ L(t₀,t) dt`, and minimizing it over the choice of `S(t₀)` is exactly maximizing `Σ_{c∈S(t₀)} (residence time of c)` — i.e. **protect the lowest-hazard channels**. Instantaneous top-1% ignores residence, which is why it sheds 53–67%.

**Why it matters strategically.** It is the **only Channel-Set bet that is neither rotation nor sidecar** — it directly answers the three-reviewer concern that the CS pool is over-concentrated on rotation. It is **CPU-screenable on already-cached traces** (estimate per-channel residence/hazard from the cached top-K-membership time series; no new GPU). And it operationalizes the paper's central finding (drift = turnover) into a method.
**Promote if:** the survival-selected static core (+ a small dynamic frontier) matches or beats budgeted-EMA and static top-K at matched budget on ≥2 models held out, and composes with ParoQuant where ParoQuant+EMA was sub-additive.
**Kill if:** hazards are near-uniform (then core ≈ random — testable; the paper's 33–47% residence implies they are not), or it fails to beat the online EMA tracker it generalizes. **Pairs with CE21 (no-gap filter) and the EVT estimator of C-A1.**

### C-D: OSC static-cluster stress test — MANDATORY DEFENSE (elevated post-scoop 2026-06-02)

Run regardless, and treat as load-bearing, not mere inoculation. **OSC (2604.12782)** explicitly claims activation outliers are *token-persistent* (a fixed structural channel cluster across tokens) — a direct challenge to the 53–67% strict top-1% set-leaving result. **DecDEC (2412.20185)** independently observes that the salient-channel set *changes dynamically* each decode step, which supports drift but also means the drift observation alone is not novel. The defense figure must (a) distinguish prompt/token-position outlier persistence (OSC) from long-decode *protected-set membership* drift at the measured block-output surface, and (b) position the contribution as the *long-reasoning-horizon, strict-set-membership* accounting that neither OSC nor DecDEC reports. Short, but it is in the body.

### Channel-Set v2 tiered table

| Tier | ID | Method | Why run it | Compute |
|------|----|--------|-----------|---------|
| **Primary (top bet)** | C-A1 | CVaR clip/scale selection (per-model) | highest-ROI; tail/CVaR objective is genuinely open; report α transfer | cheap grid + small eval |
| **Secondary** | C-A2 | horizon rotation (2-bin prefill/decode FIRST) | down-scoped: weak position signal (√t KL, broadband entropy); escalate only if 2-bin >1pt over global | cheap→medium |
| **Ablation** | C-A3 | BranchRot | down-scoped: Fig 2 shows comparable cross-branch drift; expect "global suffices" | cheap |
| **Parked** | C-B1 | KL-lookahead sidecar (gain-per-byte gated) | dominated by DecDEC (2412.20185) at fixed byte cost — revive only on ~100× bandwidth cut | none until envelope shrinks |
| **Secondary** | **CE13** | **Warmup-adapted per-prompt policy** | per-prompt early diagnosis (≠ hard switch); could turn the draft's descriptive protocol into a positive method | CPU/med |
| **Diagnostic** | **CE18** | Gain-per-byte ImpactSidecar | cheap screen only; sidecars parked (see C-B), keep as headroom diagnostic vs DecDEC | cheap |
| **Parked** | C-B2 | finite-difference/Fisher sidecar | parked with C-B (DecDEC-dominated) | none |
| **Parked** | C-B3 | drift-aware low-rank sidecar (vs ResQ PCA) | parked with C-B; ResQ 2412.14363 already owns low-rank residual | none |
| **Fallback** | C-C1 | drift-aware layer budget router | likely a scoped positive | cheap |
| **Exploratory** | C-A4 | FFT/spectral drift predictor | pure-CPU, no forward pass | cheap |
| **Adjacent control** | C-A5 | ConQuR-style decode-bucket calibration | baseline/scoop control | medium |
| **Mandatory defense** | C-D1 | OSC static-cluster stress + DecDEC reconciliation | OSC 2604.12782 claims token-persistence (contradicts drift); load-bearing, not inoculation | cheap |
| **Parity (gates the claim)** | C-E1 | **Official ParoQuant parity** → `paroquant_parity_card.md` (official-code parity **or** exact divergence from official ParoQuant) | a method may *screen* against local-ParoQuant-style, but may **not** claim "beats ParoQuant" in paper text unless this card exists | medium |
| **Smoke** | C-E2 | LAQuant / ParoQuant / DriftRot / LAQuant+DriftRot composability | pre-empts "are you only fixing what LAQuant fixes?" | medium |
| **Diagnostic** | C-E3 | InfoQuant-style activation-shaping axis (range/dispersion/occupancy/outlier-token/tail) | shows DriftRot solves a *different* failure than distribution shaping | CPU |
| **Capture** | C-E5 | Falcon branch internals capture (for C-A3) | C-A3 fails if the surfaces were never cached | GPU capture |
| **Kill-only** | C-X | hard switching, plain EMA, norm-only residual, surface fishing | saturated in draft | none |

---

## 4. Paper 2 — LatentWire (pivot to the frontier)

**New spine (revised post-scoop 2026-06-02):** *the byte–capability frontier of model-to-model communication under destructive controls* — accuracy vs communicated bytes from 1 byte to full KV, with text / KVComm / C2C **and now Latent Cache Flow (2605.22863)** as high-byte anchors and the packet at the low-byte end. **"Nobody has plotted this" is no longer true:** LCF builds essentially this frontier on MMLU-Pro/ARC-C (reporting LCF-256 beating an oracle routing frontier). The defensible residue is (a) **byte-exact, source-private, no-text** accounting under **destructive controls** (LCF reports neither the destructive-control ladder nor literal bytes/query — its axis is adapter size / latent width / TTFT on small Qwen pairs), and (b) the **score-coding (L-ScoreComp) and trust (L-B)** axes LCF does not address. Cite LCF prominently; do not reintroduce it as a strawman.

**Fact pattern.** On ARC-Challenge source-index 0.346 vs packet 0.344; on OpenBookQA both 0.378; packet follows the source choice at 0.995–0.999. The failure is task entropy, not method. MMLU-Pro (10-way) is the fastest higher-entropy escape.

**Scoop-adjusted framing.** The frontier itself is open (BaKlaVa / "Don't Waste Bits!" are KV *memory* budgeting, not communication bandwidth). "Calibrated uncertainty for routing" is crowded (UCCI, the routing survey, "LLMs Should Express Uncertainty") — TrustPacket survives only as *byte-budgeted, no-text, source-private* transmission of trust, not as a novel calibration idea. Cross-family connector with destructive controls is open as a rigor contribution.

### L-A: High-entropy reranking packet — PRIMARY BET (first, before ReasonPacket)

Tasks: MMLU-Pro 10-way; generated MATH/GSM8K solution reranking (16–64 candidates); optional RAG passage reranking (20–100).

| Variant | What the packet carries | Main metric |
|---------|------------------------|-------------|
| **A1. Top-k + margin packet** | sparse candidate IDs + quantized margins | accuracy / MRR |
| **A2. Pairwise tournament packet** | compressed pairwise preferences | MRR / nDCG |
| **A3. Syndrome correction packet** | correction to the receiver's ranking (fits SW/WZ framing) | regret / disagreement repair |
| **A4. Bradley–Terry sketch** | quantized utility shape | MRR / nDCG |

**Promote if** `Δ_beyond_label > 0` on a frozen held-out split with destructive controls collapsing. **But the decisive bar is stricter** — see L-ScoreComp and `Δ_beyond_score` below: a hard reviewer asks not "does it beat source-index?" but "does it beat the best *equal-byte compression of the source scores*?"

> **Post-scoop (2026-06-02): L-A is now OVERLAP-risk, not a clean primary.** C2C (2510.03215) and especially Latent Cache Flow (2605.22863) already operate on MMLU-Pro reranking/communication. Run L-A only if it is differentiated by the destructive-control ladder and byte-exact accounting, OR if LCF fails to replicate at realistic bytes/query. Otherwise prefer L-ScoreComp and L-B.

### Decode rule for ALL LatentWire packets — combine, don't replace (cue-combination / value-of-information)

The old draft's packet *replaces* the receiver's decision (follows source 0.995–0.999); the optimal use *adds* a source statistic. If `s_S ⊥ s_R | Y`, then `log P(Y|s_S,s_R) = log P(Y|s_S) + log P(Y|s_R) − log P(Y) + c`, so the receiver should **add** the source log-posterior-odds to its own. The packet should carry a quantized, *combinable* statistic (top-k source LLRs, or the WZ residual), and the receiver overrides **only when the source's LLR magnitude exceeds its own** — i.e. transmit/trust **only the bits that change the receiver's decision** (value-of-information). This provably reduces the target-correct damage term (0.185 on ARC) versus "replace," because when the source is uncertain (small LLR) the receiver keeps its correct answer. **Adopt as the default decode for every LW packet (WZ residual, trust, rerank); near-zero cost. The "replace/follow-source" decode is killed.** (Falsifier: if `s_S, s_R` are strongly dependent given `Y`, additivity over-counts — use a learned fusion weight.)

### L-ScoreComp: score-vector compression — PRIMARY METHOD (promoted post-scoop 2026-06-02), not just the adversary

First-class, and now the **lead LatentWire method**, not merely the bar the packet must clear. The sharpest strategic move from the scoop check: the strongest adversary baseline (equal-byte compression of source scores) is itself a score-compression method, and **distributed source coding (Wyner–Ziv / Slepian–Wolf) for inter-model score/logit communication is an unclaimed gap** — the nearest prior art (top-K sparse-logit transmission for distributed speculative decoding, e.g. 2509.04576) targets sampling equivalence, not MCQ/reranking accuracy, and Latent Cache Flow communicates KV/latent state, not coded scores. Frame L-ScoreComp as a DSC treatment of source→target score transmission; the **source-minus-target residual code (L-E2)** is the likely strongest CPU-screenable instance. The current draft's validation-only score-sketch does not beat source-label transfer — that is the starting point to beat, with the leakage guard (Section 14.5) mandatory because VQ/codebook variants overfit held-out rows trivially.

> **(R2, sharpened — this is the crux): the winning variant MUST exploit receiver side information; it must be a Wyner–Ziv / side-information code, not raw score compression.** Logic: if the packet is a deterministic function of the source score vector and the receiver has no special side information, an optimized equal-byte code of the source scores dominates by construction — so **raw VQ / CountSketch / JL of the source scores cannot be the method**, and that lane is now also crowded by CU-HLM (2505.11788, transmits compressed top-k probabilities when uncertain), TK-SLT (2509.04576, top-k sparse-logit transmission), and Structured Message Passing (2606.00405, aggregates output distributions on MMLU-Pro/GPQA). The defensible instance is **deployable Wyner–Ziv binning (L_SCORECOMP_wz_bins_deployable)**: the encoder sees **source scores only** and sends a bin/coset index of its score codeword; the decoder uses its **own target scores as side information** to pick the compatible codeword (rate ≈ H(source | receiver), zero rate loss vs. side-info-at-both-ends in the quadratic-Gaussian case). **An actual `s_source − s_target` residual is an oracle upper bound — NOT deployable — because the source cannot see the receiver's target scores at encode time in a one-way protocol; it is claimable only if the target estimate is from a *public/dev-only predictor* available to the source (`L_SCORECOMP_predicted_residual`).** Label the actual-residual row as `oracle side-information-at-encoder upper bound`. Frame the novelty as DSC-with-receiver-side-information; cite Slepian–Wolf (1973) / Wyner–Ziv (1976) + neural DSC (Whang 2106.02797, Özyılkan 2305.04380); treat CU-HLM/TK-SLT/SMP as the hard equal-byte baselines this must beat. **Brutal logic check:** if `I_beyond ≈ 0` on a task then H(source|receiver,label) ≈ 0 too, so even the optimal WZ code sends ≈ the label — WZ only wins where `I_beyond > 0`, i.e. exactly the high-entropy regime the info-ceiling filter (§2b) must clear first.

| Variant | Description | Role |
|---------|-------------|------|
| Score top-k quantization | candidate IDs + quantized margins | hard baseline |
| **Deployable WZ binning** (L-E2) | encoder sees source scores only; send a bin/coset index; decoder disambiguates with its own target scores (rate ≈ H(source\|receiver)). *Actual `s_source−s_target` = oracle upper bound only.* | **likely strongest CPU-screenable method** |
| CountSketch / JL projection | random projection of the source score vector | cheap equal-byte comparator |
| VQ score codebook | learned codebook over source-score shapes | strong CPU-screenable method |
| Pairwise margin code | quantized pairwise preferences | reranking |

**Stricter promotion bar (supersedes `Δ_beyond_label` for LatentWire ranking claims):**
`Δ_beyond_score = Metric(packet) − max(all equal-byte score-compression baselines) > 0` on a frozen held-out split, controls collapsing. (Leakage guard in Section 14.5 is mandatory here — VQ/codebook methods overfit held-out rows trivially.)

### L-B: TrustPacket / cascade packet — PRIMARY (safest beyond-label positive)

> A byte packet communicates trust/reliability rather than answer identity, improving selective prediction or routing at equal byte budget.

Metrics: AURC, risk-coverage, ECE, AUROC-of-correctness, accuracy@coverage, cost-adjusted accuracy.
**Promote if** it beats source-index+confidence or equal-byte score sketch on risk-coverage, reduces source-induced damage on source-wrong/target-correct rows, and gains concentrate on disagreement/high-uncertainty rows. Distinctive claim is the **channel** (no-text, source-private, byte-budgeted), not the calibration.

> **Post-scoop (2026-06-02): L-B is the safest positive.** UCCI (2605.18796) occupies only the *scalar calibrated-confidence* routing point (and reports strong cost cuts there), so a **multi-byte** trust/risk packet that beats source-index+scalar-confidence on AURC/risk-coverage is still open. Modest but defensible; lowest scoop and lowest structural risk of the LatentWire bets. **(R2, claim change): raw accuracy is the wrong target, and calibrated-confidence routing is now very crowded** (UCCI 2605.18796, plus RouteNLP 2604.23577 / CP-Router 2505.19970 / C3PO 2511.07396 all do conformal/calibrated cascade routing — so a "conformal-set packet" is thin novelty and belongs as a *baseline*, not the method). The claim must be **damage avoidance / selective prediction**: reduce source-induced damage on source-wrong/target-correct rows and improve risk–coverage / AURC / cost-adjusted accuracy at equal bytes, while **never transmitting the candidate**. If the packet merely mirrors a confidence scalar, UCCI-style routing dominates it — drop it. **(R3) Also now threatened by DarkForest (2605.25188), which uses calibrated structured belief states with controlled communication to raise multi-agent accuracy — so a "belief-state trust packet" must differentiate on the byte-private, no-text channel + damage-avoidance, not on calibrated belief sharing.**

### L-C: ReasonPacket + KV-budget C2C-lite — HIGH-UPSIDE, SECOND WAVE

Do not make ReasonPacket the first deliverable unless connector-training infra already works. MVP: one source/receiver pair, one generative reasoning task, fixed byte/token budget, frozen endpoints, vs answer-only label + short text handoff + source scores + C2C/KV baseline. **Promote if** it improves generative Pass@1 over answer-only and same-byte text, controls collapse, and it shows information beyond the final answer. This is the ICLR-strength path, not the safest COLM path.

### LatentWire v2 tiered table

| Tier | ID | Method | Why run it | Compute |
|------|----|--------|-----------|---------|
| **Ceiling probe** | L-A1 | MMLU-Pro 10-way rerank | cheap I_beyond probe; C2C/LCF (2605.22863) **and Structured Message Passing (2606.00405)** crowd MMLU-Pro distribution communication; 10-way label is only ~3.3 bits — not a deep escape; not a headline | CPU/cached |
| **Primary (highest-ROI beyond-score)** | L-A2 | generated-solution **verifier** rerank (16–64 candidates) | larger candidate entropy than 10-way MCQ; score vector has real shape; best chance of beyond-top-1 info without ReasonPacket training | medium |
| **Primary (safest)** | L-B1 | trust / **routing-only** packet (never carries a candidate) | safest positive; UCCI 2605.18796 owns only the scalar point, multi-byte is open | CPU/cached |
| **Primary (lead method)** | **L-ScoreComp** | source-minus-target residual (Wyner–Ziv) / VQ / CountSketch score compression | promoted to THE method; DSC for inter-model scores is the least-scooped axis | CPU |
| **Secondary** | L-A3 | syndrome correction | best match to Slepian–Wolf framing | CPU |
| **Secondary** | L-A4 | pairwise tournament / Bradley–Terry | good ranking story | CPU |
| **Optional** | L-D1 | multi-source micro-packets | disagreement geometry | CPU/cached |
| **Optional** | L-D2 | damage-avoidance selector | reduces source-induced damage | CPU/cached |
| **High-upside** | L-C1 | ReasonPacket query bottleneck | strongest if it works | GPU |
| **Anchor (not blocker)** | L-C2 | KV-budget C2C-lite byte curve; C2C/KVComm **and Latent Cache Flow (2605.22863)** as Stage-4 frontier anchors | high-byte endpoints of the frontier | GPU |
| **Kill-only** | L-X | more 4-way ARC/OBQA packets | source-index trap | none |

**Byte-frontier accounting rule:** the frontier plot reports **both** framed implementation bytes **and** minimum code bits for label/top-k/score baselines (4-way "1 byte" is information-theoretically loose); otherwise a reviewer says the frontier exaggerates packet efficiency.

---

## 5. Apples-to-apples baseline protocols (the part reviewers attack)

For each external method, fix the comparison axis and hold everything else constant. **These baselines are reproduced and locked in Phase A (see the Operating Directive) before any method queues:** the minimal critical set per paper must reproduce within tolerance to unblock methods; heavy baselines reproduce in parallel and must be locked before any method is confirmed against them; per-baseline tolerances live in `baselines.lock`.

### 5a. Channel-Set baselines

| Baseline | Hold constant | Our matched axis | Fairness rule |
|----------|---------------|------------------|---------------|
| **ParoQuant** ([2511.10645](https://arxiv.org/abs/2511.10645), ICLR 2026; project <https://z-lab.ai/projects/paroquant/>; repo z-lab not thu-nics — verify) | model, W4A16 bit budget, calibration set, eval (AIME-2025/MATH-500, decode length, temp, #samples) | recovery R **and** CVaR/worst-trace | use authors' scaled-pairwise rotation; report median **and** tail; never change bit budget between methods |
| **ConQuR** ([2605.10793](https://arxiv.org/abs/2605.10793)) | activation bit budget, calibration tokens | activation quant error + recovery | match Procrustes objective; flag as our reimpl |
| **LAQuant** ([2605.08755](https://arxiv.org/abs/2605.08755)) | bit budget, reasoning calibration set | recovery + decode speedup | it's calibration-time QAT, ours is decode-time; **also run LAQuant+ours** as composable upper bound |
| **ResQ** ([2412.14363](https://arxiv.org/abs/2412.14363), code <https://github.com/utkarsh-dmx/project-resq>) | average bits incl. high-precision fraction | recovery at matched high-precision rank | match high-precision rank/bytes exactly |
| **DecDEC** ([2412.20185](https://arxiv.org/abs/2412.20185), code <https://github.com/SNU-ARC/DecDEC>) | dynamic residual budget (channels/token) | recovery + added bytes/MACs | match fetched-channel count; ours differs only in selector |
| **KL-Lens** ([2604.13440](https://arxiv.org/abs/2604.13440), code <https://github.com/jasonkongie/kl-ssm-quant>) | budget, models | does our drift/rotated-column selector beat their layer selector? | the selector head-to-head |
| **Static top-K / EMA** | from draft | recovery | the floor |

Models: existing four + ≥1 of Qwen3-4B/8B, Qwen3-MoE/Qwen3-Next, gpt-oss-20B. Benchmarks: AIME-2024/2025, MATH-500, GPQA-Diamond, optionally LiveCodeBench + WikiText2/C4 PPL.

### 5b. LatentWire baselines

| Baseline | Hold constant | Our matched axis | Fairness rule |
|----------|---------------|------------------|---------------|
| **Source top-1 index (1 byte)** | task, source model, eval rows | metric at ≤ equal bytes | the trivial baseline that sank the old draft |
| **Source top-k + quantized confidence** | byte budget | accuracy/MRR/AURC | the hard baseline |
| **Quantized score-vector @ equal bytes** | **exactly equal bytes** | metric | hardest: source's full info; our win = more efficient encoding at same budget |
| **Same-byte visible text** | equal bytes | accuracy | the "why not text" baseline |
| **C2C** ([2510.03215](https://arxiv.org/abs/2510.03215), code <https://github.com/thu-nics/C2C>) | model pair, task; report communicated bytes + state exposed | accuracy-per-byte / per-exposure Pareto | C2C is high-byte (full KV); we anchor the low-byte end; note its same-context-length constraint |
| **Latent Cache Flow** ([2605.22863](https://arxiv.org/abs/2605.22863)) | model pair, task (MMLU-Pro/ARC-C) | accuracy vs communication budget | **new must-cite frontier neighbor**; LCF-256 reportedly beats an oracle routing frontier; differentiate via destructive controls + literal byte accounting (LCF uses adapter-size/latent-width/TTFT) |
| **DarkForest** ([2605.25188](https://arxiv.org/abs/2605.25188)) | model pair/agents, task | accuracy vs communication | calibrated structured belief-state comms ("less talk, higher accuracy") — hard baseline for L-B / belief-state packets; we differentiate on byte-private no-text channel + damage-avoidance |
| **KVComm (ICLR'26 selective)** ([2510.03346](https://arxiv.org/abs/2510.03346), code <https://github.com/Zephyroam/KVComm>) | pair, task | bytes vs metric | sends ~30% of layers' KV; high-byte anchor |
| **KVCOMM (NeurIPS'25 reuse, distinct)** ([2510.12872](https://arxiv.org/abs/2510.12872)) | — | — | **naming collision** — cite separately |
| **CIPHER** ([2310.06272](https://arxiv.org/abs/2310.06272)) / **AC** (last-hidden) | pair | bytes vs metric | same-family embedding/activation comms predecessors |
| **DroidSpeak** ([2411.02820](https://arxiv.org/abs/2411.02820)) | — | layer-selectivity finding | fourth must-cite cross-LLM KV system |

Tasks/metrics off MCQ: MMLU-Pro (10-way), GSM8K/MATH-500/AIME generative Pass@1, plus AURC / risk-coverage / ECE / AUROC-of-correctness / accuracy@coverage. Cross-family pairs: Qwen3-4B ↔ Llama-3.1-8B, gpt-oss-20B ↔ Qwen3-8B, 1.5B ↔ 14B small→large.

---

## 6. Shared registry (one card per variant)

`registry/<method_id>.yaml`; the aggregator reads all cards + result JSONs and applies the Section 7 gates. **Each card now also carries a `lineage` block** (`parent_methods`, `nearest_killed_methods`, `mechanism_family`, `differs_from_parent_by`) and a `lineage_audit` fails any card that recreates a logged killed mechanism without a new falsifier or that names no nearest baseline / killed relative. Operating discipline around the registry — Conductor decision rights (run as a Codex `/goal`), EVI-scheduled foreground GPU with sequential futility, planted-signal positive controls gating Stage-2, claim-to-figure, per-method breakthrough pre-mortems, and the failure-packet repair loop — lives in `plans/TEAM_OPERATING_SYSTEM.md`, `plans/EXPERIMENT_SELECTION_SPEC.md`, and `plans/PLANTED_TESTS_SPEC.md`.

```yaml
method_id: C_A1_driftrot_cvar
paper: channel_set_drift
tier: primary
contribution_role: positive_method     # positive_method | positive_enabler | diagnostic_defense | baseline_adversary | ceiling_probe | kill_only | parked
paper_claim_eligible: true             # only positive_method / positive_enabler are claim-eligible
hero_hypothesis: adaptive_driftrot
hypothesis: >
  Choosing clip/scale by held-out CVaR (worst-20%) beats ParoQuant's
  median-optimal setting on tail metrics without losing median.
scoop_check:
  differs_from: [ParoQuant, ConQuR, DartQuant, LAQuant, ResQ, KL-Lens, TR-DQ, RotateKV]
  novelty_axis: "tail-risk objective for rotation/clip under long-decode drift"
stage1_headroom:                 # KILL-ONLY, near-free, parallel
  proxy: cvar_over_precomputed_dNLL_grid
  models: [Granite-Small, DeepSeek-R1-Distill]
  kill_if: ["no_tail_gap_exists", "worse_than_random_clip"]
stage2_live_gate:
  promote_if: ["cvar_improves_vs_paroquant", "median_not_worse", "wrong_horizon_control_fails"]
stage3_confirm:
  models: [Granite-Small, DeepSeek-R1-Distill, Falcon-H1, Nemotron-3]
  split: held_out_traces
  stats: BCa_paired_bootstrap + Holm_over_family
  metrics: [recovery_R, CVaR, worst_trace]
  controls: [wrong_horizon, random_clip]
stage4_systems: { report: [tokens_per_s, HBM_bytes, sidecar_bytes, added_MACs] }
primary_metric: cvar_recovery
```

```yaml
method_id: L_A1_high_entropy_rerank
paper: latentwire
tier: primary
hero_hypothesis: beyond_label_packet
prefilter: info_ceiling             # kill if I_beyond ≈ 0 on the task
closest_baselines: [source_top1, source_topk_plus_conf, quantized_score_vector_equal_bytes, same_byte_text, C2C, KVComm]
stage1_ceiling: { task: mmlu_pro_10way, kill_if: ["I_beyond_approx_zero"] }
stage2_live_gate:
  rows: 200-500
  promote_if: ["delta_beyond_label > 0", "controls_collapse"]
  kill_if: ["packet_follows_top1 > 0.95 AND no ranking gain"]
controls: [wrong_row, candidate_derangement, coordinate_shuffle]
primary_metric: delta_beyond_label
```

---

## 7. Promotion / kill gates (explicit thresholds)

### Channel-Set

**Stage 2 (may promote) uses point estimates; Stage 3 (paper claim) requires the lower-bound version of the same gate.**

| Gate | Stage-2 promote (point est.) | Stage-3 paper claim (lower bound) | Kill |
|------|------------------------------|-----------------------------------|------|
| Headroom | selector ≥25% more KL/recovery headroom than norm/PCA/random | — (screening only) | selector ≤ norm **and** ≤ random |
| Median recovery | `delta_vs_paroquant > 0.05` | paired-bootstrap 95% LB for median recovery **> 0**, **or** clear CVaR/worst-trace improvement without median loss | below ParoQuant on median |
| Tail risk | CVaR or worst-trace improves | CVaR/worst-trace improvement holds under paired bootstrap | creates a new catastrophic tail |
| Transfer | non-negative on ≥2 models | non-negative on ≥2 models held out | one-model-only (unless explicitly named as scoped) |
| Controls | wrong-horizon / random-clip / random-sidecar fail | same, on held-out | controls match the method |
| Systems | overhead plausibly <10–15%, or clear quality/byte win | measured profiler row | sidecar traffic dominates the gain |

### LatentWire

**Stage 2 (may promote) uses point estimates; Stage 3 (paper claim) requires the paired-bootstrap lower bound > 0** — this is exactly the gate the old draft failed (packet 0.344 vs source-index 0.346 on ARC; 0.378 vs 0.378 on OpenBookQA; follows source 0.995–0.999).

| Gate | Stage-2 promote (point est.) | Stage-3 paper claim (lower bound) | Kill |
|------|------------------------------|-----------------------------------|------|
| Information ceiling | source scores carry info beyond top-1 | — (screening only) | source top-1 explains everything |
| Beyond-score (decisive) | `Δ_beyond_score > 0` held out | paired-bootstrap 95% LB for `Δ_beyond_score` **> 0**, else **exploratory only** | packet tied with best equal-byte score sketch |
| Text baseline | beats same-byte text, or clearly wins on non-text exposure | holds under paired bootstrap | same-byte text matches it |
| Controls | wrong-row / derangement / shuffle collapse | same, on held-out | controls still work |
| Disagreement audit | repairs source-correct/target-wrong without excessive target damage; uses a **combine** (not replace) decode | holds under paired bootstrap | simply copies the source, or uses a replace/follow-source decode |
| Byte frontier | occupies a useful Pareto point | Pareto point survives min-code-bit accounting | dominated by score sketch or C2C/text |

---

## 8. Checking protocol

**Channel-Set — report four metrics, never median alone:** recovery R vs ParoQuant; **tail/CVaR/worst-trace**; headroom attribution (selector beats norm/PCA/KL-Lens-layer/random); systems envelope. If a method works on only one model, name it for that regime ("Falcon rotation rescue"), not "universal DriftRot." **Mandatory no-gap filter (CE21):** report the no-gap fraction and compute recovery ratios only on positive-gap traces — the estimand changes otherwise, so a method may not claim a ratio win that is an artifact of no-gap traces.

**LatentWire — five controls per method + the beyond-label statistic:** source-index/label; source-index+confidence/score sketch; same-byte text; wrong-row/row shuffle; candidate derangement/coordinate shuffle. Plus `Δ_beyond_label > 0` held out, or it is a packetized label. Plus the **combine-not-replace decode** (packet adds to receiver evidence; never overrides on a small source LLR).

**Both:** frozen splits chosen before readout; BCa paired bootstrap; Holm over the confirmed family for paper claims (BH exploratory-only); disclose how many variants were screened.

---

## 9. First-wave bets (the priority order the scheduler must follow)

The screened pool is large; these **9** are the true first-wave bets. Everything else is secondary/fallback/defense and runs only as cached screening capacity allows.

**Channel-Set first-wave (revised 2026-06-02):** **C-A1 CVaR clip/scale (per-model — top bet)** · C-A2 horizon rotation **as a 2-bin prefill/decode ablation** (escalate only on >1pt over global) · **CE13 warmup-adapted per-prompt policy** · C-A3 BranchRot **as a cheap ablation only** · **C-F StableCore/survival** (CPU screen on cached traces — the non-rotation diversifier). *(CE18 gain-per-byte sidecar drops out of the headline first-wave — sidecars parked; keep CE18 only as a cheap headroom diagnostic vs DecDEC.)*
**LatentWire first-wave (revised 2026-06-02):** **L-ScoreComp source-minus-target / VQ score compression (lead method, Wyner–Ziv framing)** · **L-B1 trust/routing packet (safest positive)** · L-A2 generated-solution verifier rerank · L-A1 MMLU-Pro 10-way rerank **only if differentiated from C2C / Latent Cache Flow (2605.22863)**. **All LW packets use the combine-not-replace decode by default.**

**Mandatory regardless of bets:** C-E1 ParoQuant parity (no "beats ParoQuant" claim without it) · **C-D1 OSC static-cluster stress + DecDEC reconciliation (load-bearing defense of the drift thesis, not just inoculation)** · the score-compression baseline family (the `Δ_beyond_score` adversary) · **explicit differentiation of the LatentWire frontier spine from Latent Cache Flow (2605.22863)** in the related-work/positioning prose.

Secondary/fallback/defense (run as capacity allows): C-B1/B2/B3 sidecars, C-C1 budget router, C-A4 FFT predictor, C-A5 ConQuR-bucket control, C-E2 LAQuant composability smoke, C-E3 InfoQuant-shaping diagnostic, C-E5 Falcon-internals capture; L-A3 syndrome, L-A4 pairwise/BT, L-D1 multi-source, L-D2 damage-avoidance, L-C1 ReasonPacket (2nd wave), L-C2 + C2C/KVComm frontier anchors (Stage-4, non-blocking).

---

## 10. Reference index (papers + code)

**Channel-Set:** ParoQuant [2511.10645](https://arxiv.org/abs/2511.10645) (ICLR 2026, <https://openreview.net/forum?id=1USeVjsKau>, project <https://z-lab.ai/projects/paroquant/> — repo is z-lab, **not** thu-nics) · ConQuR [2605.10793](https://arxiv.org/abs/2605.10793) · **InfoQuant / PSOT [2605.26175](https://arxiv.org/abs/2605.26175) — scoop watch: train-free orthogonal activation shaping + adaptive clipping; weakens generic "rotation + adaptive clipping" novelty, so our axis must stay long-decode horizon/branch/tail** · LAQuant [2605.08755](https://arxiv.org/abs/2605.08755) *(verify)* · KL-Lens [2604.13440](https://arxiv.org/abs/2604.13440) (code <https://github.com/jasonkongie/kl-ssm-quant>) · DecDEC [2412.20185](https://arxiv.org/abs/2412.20185) (OSDI'25 <https://www.usenix.org/conference/osdi25/presentation/park-yeonhong>, code <https://github.com/SNU-ARC/DecDEC>) · ResQ [2412.14363](https://arxiv.org/abs/2412.14363) (code <https://github.com/utkarsh-dmx/project-resq>) · DartQuant [2511.04063](https://arxiv.org/abs/2511.04063) · DFRot [2412.00648](https://arxiv.org/abs/2412.00648) · QuaRot [2404.00456](https://arxiv.org/abs/2404.00456) · SpinQuant [2405.16406](https://arxiv.org/abs/2405.16406) · RotateKV [2501.16383](https://arxiv.org/abs/2501.16383) · TR-DQ [2503.06564](https://arxiv.org/abs/2503.06564) · Q-Drift [2603.18095](https://arxiv.org/abs/2603.18095) · OSC [2604.12782](https://arxiv.org/abs/2604.12782) · MambaQuant [2501.13484](https://arxiv.org/abs/2501.13484) · Quamba2 [2503.22879](https://arxiv.org/abs/2503.22879) · MoBiQuant [2602.20191](https://arxiv.org/abs/2602.20191) · CMPQ [2410.13056](https://arxiv.org/abs/2410.13056) · BaKlaVa [2502.13176](https://arxiv.org/abs/2502.13176) · MPQ survey (Slim-LLM/LLM-MQ) [2510.16805](https://arxiv.org/abs/2510.16805) · PM-KVQ [2505.18610](https://arxiv.org/abs/2505.18610) · **(R2 additions) ReSpinQuant [2604.11080](https://arxiv.org/abs/2604.11080) (subspace residual-rotation — more rotation saturation) · SingleQuant [2511.22316](https://arxiv.org/abs/2511.22316) (closed-form Givens, optimization-free) · SERQ [2603.08185](https://arxiv.org/abs/2603.08185) (saliency low-rank error reconstruction — sidecar baseline) · ResComp [2604.07955](https://arxiv.org/abs/2604.07955) (residual-error redefinition — sidecar baseline) · EAQuant [2506.13329](https://arxiv.org/abs/2506.13329) (expert-aware MoE PTQ — crowds BranchRot/CE14)**.

**LatentWire:** **Latent Cache Flow [2605.22863](https://arxiv.org/abs/2605.22863) — new must-cite, near-full scoop of the frontier spine** · C2C [2510.03215](https://arxiv.org/abs/2510.03215) (ICLR 2026, <https://openreview.net/forum?id=LeatkxrBCi>, code <https://github.com/thu-nics/C2C>) · KVComm selective [2510.03346](https://arxiv.org/abs/2510.03346) (code <https://github.com/Zephyroam/KVComm>) · KVCOMM reuse [2510.12872](https://arxiv.org/abs/2510.12872) · DroidSpeak [2411.02820](https://arxiv.org/abs/2411.02820) · CIPHER [2310.06272](https://arxiv.org/abs/2310.06272) (<https://openreview.net/forum?id=sehRvaIPQQ>) · UCCI [2605.18796](https://arxiv.org/abs/2605.18796) · routing survey [2603.04445](https://arxiv.org/abs/2603.04445) · LLMs-Express-Uncertainty [2604.05306](https://arxiv.org/abs/2604.05306) · SoftCoT [2502.12134](https://arxiv.org/abs/2502.12134) · relative representations [**2209.15430**](https://arxiv.org/abs/2209.15430) (Moschella et al., ICLR 2023; the previously-cited 2305.13093 was an unrelated image-restoration paper) · LLMLingua [2310.05736](https://arxiv.org/abs/2310.05736) · gist tokens [2304.08467](https://arxiv.org/abs/2304.08467) · **(R2 additions) CU-HLM [2505.11788](https://arxiv.org/abs/2505.11788) (transmits compressed top-k probs when uncertain — hard L-ScoreComp/TrustPacket baseline) · Structured Message Passing [2606.00405](https://arxiv.org/abs/2606.00405) (aggregates output distributions on MMLU-Pro/GPQA — hard L-A1/L-B/L-ScoreComp baseline) · TK-SLT [2509.04576](https://arxiv.org/abs/2509.04576) (top-k sparse-logit transmission) · LMNet / dense communication [2505.12741](https://arxiv.org/abs/2505.12741) · LatentMAS [2511.20639](https://arxiv.org/abs/2511.20639) (pure latent collaboration — crowds ReasonPacket) · RouteNLP [2604.23577](https://arxiv.org/abs/2604.23577) / CP-Router [2505.19970](https://arxiv.org/abs/2505.19970) / C3PO [2511.07396](https://arxiv.org/abs/2511.07396) (conformal cascade routing — L-B is crowded) · DSC foundations: Slepian–Wolf (1973), Wyner–Ziv (1976), neural DSC (Whang [2106.02797](https://arxiv.org/abs/2106.02797), Özyılkan [2305.04380](https://arxiv.org/abs/2305.04380))**. · **(R3 additions) DarkForest [2605.25188](https://arxiv.org/abs/2605.25188) (calibrated belief-state multi-agent comms) · FusionRoute [2601.05106](https://arxiv.org/abs/2601.05106) (token-level logit correction) · Graph-of-Agents [2604.17148](https://arxiv.org/abs/2604.17148)**.

**Benchmarks:** MMLU-Pro [2406.01574](https://arxiv.org/abs/2406.01574) · GPQA [2311.12022](https://arxiv.org/abs/2311.12022) · lm-eval-harness <https://github.com/EleutherAI/lm-evaluation-harness>.

---

## 11. Extended method pool (v2.1 — lateral techniques to screen)

These extend, not replace, the real bets. Most are **CPU-screenable on cached activations / scores**, so they cost almost nothing to add to an overnight screen. Tiers: `sec` = secondary, `exp` = exploratory, `sys` = systems-contribution, `abl` = ablation/control. New IDs avoid collision with Sections 3–4.

### Channel-Set extended

| ID | Method | Lateral inspiration | Stage-1 proxy | CPU/GPU | Tier |
|----|--------|--------------------|---------------|---------|------|
| **CE1** | **Co-drift channel pairing** — pair Givens-rotation channels that *drift together*, so the rotation stays valid longer under decode | ParoQuant pairing + your drift finding | drift-correlation matrix on cached acts; measure pairing stability | CPU | **sec (high interest — directly couples drift to ParoQuant's mechanism)** |
| **CE2** | **Horizon rotation library + selector** — a small library of cheap fixed (Hadamard/Givens) rotations, pick per decode bucket | ensembles + horizon conditioning | per-bucket reconstruction over the library | CPU/med | sec |
| **CE3** | **Attention-sink-aware rotation/protection** — special handling of massive-activation / sink channels under drift | RotateKV sink-aware, massive-activations lit | sink-channel set-leaving vs non-sink | CPU | sec |
| **CE4** | **Lazy rotation refresh** — event-triggered re-calibration only when a cheap drift detector (set-leaving / KL spike) crosses a threshold | error-feedback / event-driven control | detector firing rate vs recovery on cached grid | GPU (amortized cost) | sys |
| **CE5** | **Predictive prefetch + double-buffer** — systems realization of C-A4: prefetch next-horizon residuals to hide PCIe/HBM latency | speculative prefetch | prediction accuracy already from C-A4 | GPU systems | sys |
| **CE6** | **Per-token entropy-gated protection** — raise protection budget only on high-entropy decode steps where drift bites | token-adaptive precision | correlation(step entropy, set-leaving) on cached traces | CPU | exp |
| **CE7** | **Low-rank temporal drift basis** — SVD the per-channel magnitude *trajectory*, protect/correct the top temporal modes (≠ FFT) | dynamical low-rank | variance explained by top-k temporal modes | CPU | exp |
| **CE8** | **Outlier-migration-aware tracking** — track precision-dependent outlier migration during decode | MoBiQuant migration | migration-rate stats on cached acts | CPU | exp |
| **CE9** | **Cross-token quant-error feedback** — accumulate and correct quant error forward across tokens (error feedback) | QEP error propagation, optimizer error-feedback | simulated error-feedback ΔNLL on cached grid | med | exp |
| **CE10** | **Multi-objective rotation calibration** — MSE + KL + tail penalty in one objective | multi-objective calibration | grid over objective weights on cached grid | CPU | abl |

### LatentWire extended

| ID | Method | Lateral inspiration | Stage-1 proxy | CPU/GPU | Tier |
|----|--------|--------------------|---------------|---------|------|
| **LE1** | **ECC/parity packet over candidate IDs** — formalize syndrome (L-A3) with real coding theory; receiver corrects its own top-k | Hamming/LDPC, Slepian–Wolf | regret-repair on cached source/receiver rankings | CPU | **sec (strong fit to original SW/WZ framing)** |
| **LE2** | **Adversarial packet-dropout training** — mask source top-1 during training so the packet *must* carry margin/ranking | adversarial dropout | does masked-top1 packet still beat label? | GPU (small training) | **sec (attacks "collapses to label" by construction)** |
| **LE3** | **Disagreement-gated adaptive byte budget** — spend bytes only when source and a cheap receiver-side proxy disagree | adaptive computation | bytes-vs-gain on disagreement rows (cached) | CPU | sec |
| **LE4** | **Conformal prediction-set packet** — send a few bytes encoding a conformal set, not a point | conformal selective prediction | coverage/size vs bytes (cached scores) | CPU | sec (beyond-label by construction) |
| **LE5** | **Logit-lens uncertainty-trajectory sketch** — compress an early-layer logit-lens read of the source's uncertainty path | logit lens / early exit | trajectory info beyond top-1 (cached) | CPU | exp |
| **LE6** | **Multi-round budgeted micro-dialogue** — 2–3 tiny packets vs one big one | iterative comms | accuracy vs total bytes over rounds | med | exp |
| **LE7** | **VQ codebook packet** — vector-quantize source evidence against a public codebook both models read | VQ / shared dictionary | codebook reconstruction + beyond-label (cached) | CPU (codebook train cheap) | exp |
| **LE8** | **Speculative-verification acceptance sketch** — source proposes, receiver verifies; packet carries acceptance geometry | speculative decoding | acceptance-rate vs bytes | GPU-ish | exp |
| **LE9** | **Routing-only packet** — packet decides *which model answers*, never carries the label; scored on cost–accuracy | cascade routing | cost–accuracy Pareto (cached) | CPU | sec (clean beyond-label) |
| **LE10** | **Cross-family relative-rep / Procrustes feature packet** — send shared relative-representation feature IDs across families | relative representations | cross-family alignment quality + controls | CPU/med | exp |

**How this expands the overnight screen.** All the `CPU` rows above slot straight into the Stage-1 job array at near-zero marginal GPU cost — adding them roughly doubles the screened pool without touching the GPU budget. The registry (Section 6) makes this mechanical: a second review pass can author more cards and the aggregator picks them up automatically. Keep the **Section 9 first-wave bets** unchanged; the extended pool is exploration that can *only* promote by clearing the same Section 7 gates.

---

### Extended method pool (v2.2 — wider lateral search for the nightly factory)

With an RTX Pro 6000 + agent orchestration, broaden Stage-1 aggressively (cached screening is cheap). These are additional cards; most are CPU/cached. Tiers as before.

**Channel-Set v2.2**

| ID | Method | Lateral family / why | CPU/GPU | Tier |
|----|--------|---------------------|---------|------|
| **CE11** | CUSUM / change-point rotation selector | detect drift-regime changes online instead of fixed horizons | CPU | exp |
| **CE12** | Activation-tail-index (kurtosis) router | choose rotate/protect/sidecar by tail index | CPU | exp |
| **CE13** | **Warmup-adapted per-prompt policy** | use first 256–512 decode tokens to pick a policy from a small library (≠ hard position switch) | CPU/med | **sec — priority lateral family** |
| **CE14** | Expert-local MoE protection | allocate protection by active expert (Nemotron/Granite) | med | exp |
| **CE15** | Head/submodule-local protection | test whether drift localizes to heads/submodules | CPU | exp |
| **CE16** | KV→activation outlier predictor | use K/V norms + attention stats to predict later activation outliers | med | exp |
| **CE17** | Quant-error-whitening rotation | minimize quant-error *covariance*, not magnitude | CPU/med | exp |
| **CE18** | **Knapsack / gain-per-byte sidecar** | select residual columns by gain-per-byte, not raw gain | CPU | **sec — priority lateral family** |
| **CE19** | Cache-locality sidecar | prefer columns reused across nearby tokens to cut traffic | CPU | sys |
| **CE20** | Stable-core + dynamic-frontier split | static-protect the stable core; sidecar only the drifting frontier | CPU | sec |
| **CE21** | **No-gap-aware filter (MANDATORY)** | report no-gap fraction; compute recovery only on positive-gap traces — prevents misleading ratio wins | CPU | **filter (mandatory)** |
| **CE22** | Phase classifier (reasoning vs final-answer) | different budget/rotation near the answer phase | CPU/med | exp |
| **CE23** | Cross-layer propagation selector | pick columns whose correction most reduces *next-layer* error | med | exp |
| **CE24** | Tiny learned router over cached drift stats | shallow MLP → rotate/protect/sidecar | CPU | exp |
| **CE25** | Soft union-of-topK w/ decayed history | optimize smooth protection weights, not hard EMA | CPU | exp |

**LatentWire v2.2**

| ID | Method | Lateral family / why | CPU/GPU | Tier |
|----|--------|---------------------|---------|------|
| **LE11** | **Score-vector compression family** | CountSketch/JL projection + learned VQ codebook + residual-vs-receiver coding, all aimed at beating the equal-byte score sketch | CPU | **sec — priority lateral family** |
| **LE12** | Differential ranking code | encode source-minus-target ranking (explicit side info) | CPU | sec |
| **LE13** | Negative-evidence packet | send candidates to rule out, not the chosen one | CPU | exp |
| **LE14** | Verifier margin packet | source verifies generated chains, sends a margin sketch | med | exp |
| **LE15** | Critique-category tags | non-text categorical error tags for candidate solutions | med | exp |
| **LE16** | Answer-format uncertainty packet | math/code final-answer extraction uncertainty | med | exp |
| **LE17** | Cross-family calibration map | map source margins into receiver confidence space | med | sec |
| **LE18** | Seed-the-chain codebook | byte code indexes a learned soft-prompt/prefix hint | GPU | exp |
| **LE19** | **C2C weak-sharer damage repair** | trust packet gates harmful source fusion — attacks C2C's named failure | med | **sec — priority lateral family** |

**Four priority lateral families (run these first within the extended pool):** (1) gain-per-byte sidecars (CE18/CE19) — the residual cost envelope demands gain-per-byte discipline; (2) prompt-warmup adaptation (CE13) — per-prompt calibration from early observed drift, distinct from hard switching; (3) score-vector compression as the hard adversary (LE11) — the natural path to a positive LatentWire result that stays byte-budgeted but moves beyond label transport; (4) damage-avoidance packets (L-D2 / LE19) — can win on cost/risk even when raw accuracy ties source-index.

---

## 12. Compute budget and overnight batching (one RTX Pro 6000 + an agent)

**Hardware reality.** The RTX Pro 6000 (Blackwell, ~96 GB) is **one** GPU: GPU work is essentially *serialized*, and parallelism lives on the CPU side. 96 GB comfortably holds a 1.5–8B model in BF16 or a ~30B MoE in 4-bit, so model size is not the constraint — GPU wall-clock is.

**What the agent (Codex / GPT-5.5) actually buys you.** It compresses *engineering* time — writing/debugging the runner, the job array, result parsing, the aggregator, baseline harnesses — which is normally the real bottleneck. It does **not** speed up GPU wall-clock. So the agent is what lets you *launch* 40 experiments before bed; the card is what decides how many of them are GPU-bound vs free CPU screens. Design overnight batches accordingly: **CPU-heavy screening early, GPU-light confirmation later.**

**The leverage point.** Most Stage-1 screening runs on *cached* activations / source scores — pure CPU linear algebra, seconds–minutes per job, embarrassingly parallel across cores. So "dozens of results by morning" is realistic for screening; the GPU only carries cache generation, quantized-forward confirmation, and connector training.

**Rough throughput (estimates — calibrate with a night-0 micro-benchmark):**

| Workload | Per-unit cost (1 GPU) | Notes |
|----------|----------------------|-------|
| Stage-1 CPU screen (one variant×model×seed, cached) | seconds–few min | hundreds run overnight on CPU cores |
| Long-trace generation, 1.5–8B, ~20K tok | ~3–7 min/trace | only for **new** models/benchmarks; existing 4 models already cached |
| Quantized-forward confirmation, 1.5–8B, ~24 held-out traces | ~15–45 min/config | config = method or control, per model |
| Same, ~30B MoE in 4-bit | ~1–3 h/config | the expensive Channel-Set rows |
| Source-score caching (MMLU-Pro 10-way etc.) | ~1–3 GPU-h total | then all packet/baseline work is CPU |
| ReasonPacket connector training (small, frozen pair) | ~3–8 GPU-h/run | the heavy LatentWire item |

**Campaign GPU budget (order of magnitude):** Channel-Set confirmation ≈ **25–40 GPU-h** (3 promoted × ~4 models × method+controls) + new-model trace caching ≈ **10–30 GPU-h**. LatentWire CPU methods ≈ **2–5 GPU-h** (just score caching); ReasonPacket if pursued ≈ **15–30 GPU-h**. **Core total ≈ 60–120 GPU-h; up to ~150–200** if you push many new models + ReasonPacket.

**Overnight batching model (~10 effective GPU-h/night):**
- **Night 0:** micro-benchmark throughput; agent builds/validates the runner + aggregator on a tiny slice.
- **Night 1:** full Stage-1 CPU screen for *all* ~30+ variants (Sections 9 + 11) in parallel on CPU + cheapest GPU score-caching. **Kills ≥ half by morning.** GPU lightly loaded.
- **Nights 2–4:** Stage-2 small live gates for survivors on the two small models (1.5–8B) — several configs per night fit.
- **Nights 5–10:** Stage-3 confirmation, including the 30B-MoE configs (1–3 each per night) + ReasonPacket training if promoted.
- **Final 1–2 nights:** Stage-4 systems profiling rows + figure regeneration.

**Calendar.** The GPU-bound portion is roughly **~6–12 overnight runs** (≈2 weeks of nights), front-loaded so the cheap screen returns most decisions on night 1. With the agent orchestrating, the binding constraint becomes **your decision latency between batches**, not compute — which is exactly what the funnel is built to minimize. This fits comfortably inside the window to July 12 with room for iteration, writing, and the final scoop check.

**One honest constraint to design around:** on a single card you cannot truly parallelize GPU confirmation — batching small configs into one process (MPS, 96 GB shared) gives modest concurrency but the same SMs. Plan GPU work as serialized and keep the parallelism where it is free (the CPU screen). That is why the extended pool above is deliberately CPU-weighted.

---

## 13. Orchestration: the overnight experiment factory (Codex queue + subagents)

The hardware is one GPU; the agent's job is to keep it saturated with **audited, replay-able** work while doing everything else (building, screening, fixing, bookkeeping, writing) off the critical path.

### 13.1 Core principle — capture once, replay cheaply
Separate **shared expensive capture** from **cheap method scoring**. No method runs its own 20K-token decode. Capture once in Wave 0; dozens of methods replay against the cache.
- **Channel-Set Wave-0 caches:** BF16 activations at measured surfaces · W4A16 outputs/losses · ParoQuant-style outputs/losses · rotation-transformed activations · per-position KL traces · per-layer drift statistics · candidate residual columns + cheap output-impact estimates.
- **LatentWire Wave-0 caches:** source top-1/top-k/margins/entropy/score-vectors · target ranking/confidence · candidate pools (MMLU-Pro, GPQA, generated MATH/GSM8K) · same-byte text controls · baseline predictions (source-index, +confidence, equal-byte score sketch) · wrong-row / candidate-derangement / coordinate-shuffle controls.

### 13.2 Three GPU modes (layer the parallelism on one card)
| Mode | What runs | Capacity/night |
|------|-----------|----------------|
| **CPU / cached** | selectors, packet encoders, score sketches, bootstraps, controls | hundreds of variants |
| **GPU replay** | cached prompts, short forward passes, fixed candidate pools | 8–16 serious variants |
| **GPU capture** | long 20K decode, activation capture, live generation | shared precompute only — never per-method |

### 13.3 The runner contract (every experiment emits one JSON; the aggregator decides, not a human)
**The authoritative result schema is Section 14.4 (NUMERIC fields only).** This subsection describes the *lifecycle* only. **No textual control labels (e.g. `"wrong_row": "collapsed"`) are accepted by the aggregator** — controls must be numeric metrics compared against a numeric `control_threshold`, so a method cannot "pass" with a label its numbers don't support.

**Aggregator outputs:** `promoted.md` · `killed.md` · `ambiguous.md` · `leaderboard.csv` · `family_correction.json` (**Holm** for confirmatory paper claims; BH only for exploratory dashboard ranking, labeled exploratory) · `plots/` · `morning_brief.md`.

**Exploratory ranking heuristic (orders `leaderboard.csv` only — NOT a promotion gate):** rank exploratory variants by `V = 2H + 2N + 2B + C − 2S − 2F − G` where H = headroom/info signal, N = novelty-after-scoop, B = baseline distance, C = cheapness, S = scoop risk, F = structural-failure risk, G = GPU burden. This orders the queue; **promotion remains governed solely by the Section 7 gates + Holm**, never by `V`.

### 13.4 Agent roles and the queue lifecycle
**Roles (run as parallel Codex subagents merging through one test suite + result schema):**
- **Infra agent** — registry schema, job launcher, result aggregator, dashboard.
- **GPU-daemon agent (local box only)** — owns the serialized `gpu_foreground` queue + the `gpu_backfill`/`gpu_cohostable` queues + `.gpu.lock` + the `nvidia-smi` watchdog (§2.5 of the ExecPlan). Keeps the GPU on backfill whenever no audited foreground job exists; never assumes Codex Cloud has a GPU.
- **Review panel (repro / stats / leakage / baseline-adversary reviewers)** — the **pre-launch** gate: no runner launches until `reviews/<method_id>@<code_sha>.json` shows all required reviewers PASS (incl. a minimum-detectable-effect/power check). Per-code-hash, run in parallel, overlap GPU backfill. See ExecPlan §1.5.
- **Builder subagents (one per method family)** — Channel-Set rotation/CVaR/horizon · Channel-Set residual/sidecar/budget · LatentWire reranking · LatentWire trust/cascade. Each implements its registry cards as runners **with a smoke test**, runs CPU/cached Stage-1, emits JSON.
- **Baseline-adversary agent** — owns the hard baselines + controls (source-index, +confidence, equal-byte score sketch, ParoQuant, random/wrong-row/derangement). Every method is scored against it. **Prevents fake wins. Phase-A founding job:** reproduce each baseline to within tolerance, write `baselines.lock`, and draft the baseline / related-work / positioning prose + the Section 5 table — before any method for that paper queues.
- **Audit agent (the GPU gate)** — nothing enters the GPU queue until it passes: **a passing pre-launch review record (`reviews/<method_id>@<code_sha>.json`) that matches the current code hash and predates the run** · runner-contract compliance · `split_hash`/`cache_hash` present and matching the frozen manifest (no test leakage) · controls actually wired and collapsing on a sanity check · baseline parity (re-derives a known baseline number) · Stage-1 sanity (didn't "pass" via a bug) · registry card exists and predates the confirmation split.
- **GPU queue manager (main GPU agent)** — pulls **audited** jobs by priority, runs serialized GPU work, monitors utilization/wall-time. **On failure → spawns a fix subagent and immediately advances to the next queue item (non-blocking).**
- **Fix subagent (spawned on failure)** — reads logs, diagnoses, patches, re-runs the smoke test, resubmits to the audit gate. Max N retries, then `PARKED` for human review.
- **Stats agent** — BCa paired bootstrap, Holm over the confirmed family (confirmatory; BH exploratory-only), dashboard.
- **Repro agent** — hashes, manifests, smoke tests, audit trail.
- **Standing research teams (cadence-gated; full spec in `plans/CODEX_NEXT_72H.md §8`)** — beyond build/run/review, the org runs: an **Ideation Lab** (creative lateral method generation as results arrive; domain scouts + mechanism-mathematician + adversary + scoop-checker; only math∧adversary∧scoop survivors author gated registry cards, capped to Stage-1 capacity); a **Failure-Learning team** (a post-mortem + `LESSONS_LEDGER` entry on every KILL/PARK, consumed so dead ideas are never re-proposed); a **Literature & Best-Practices Watch** (internet-isolated; constant scoop refresh + "best way to make X work" for live methods); a **Paper team** (writes both papers to the COLM standard, claims calibrated to evidence); and a **Paper Peer-Review Board** (independent mock-COLM reviewers + area chair scoring against the real COLM rubric — Quality/Significance/Originality/Clarity/Honesty-Trust/prior-work/non-conventional — and iterating to target, with an ethics/honesty veto on overclaiming). Idle GPU time is never wasted; idle agent time goes to ideation, learning, and the paper. **Operating behaviors (binding; `plans/CODEX_NEXT_72H.md §9`):** forecast-before-run + chase surprise (`anomaly_board`); clean-room re-implement every positive before believing it; protect a ≥20% exploration budget with WIP limits + a rabbit-hole detector; maintain `KNOWN.md` + `decisions/`; weekly premortem/backcast/reframe; cross-pollinate the two papers; numerics-regression-suite + always-green `--tiny`; and a process retrospective so the org improves its own prompts.

**Job state machine (non-blocking — one failure never stalls the card):**
```
DRAFTED → BUILT(+smoke) → CPU_SCREENED → AUDITED → GPU_QUEUED → RUNNING → {PASSED | KILLED | FAILED}
                                   ↑                                              │
                                   └──────── re-audit ←── FIX_SPAWNED ←───────────┘  (FAILED, retries<N)
                                                          → PARKED (retries=N, human review)
```

### 13.5 Nightly cadence and targets
The RTX box + agent makes broad cached screening cheap, so go **wide at Stage 1, narrow at confirmation**. Targets for week 1: **100–150 Stage-1 screens → 12–20 Stage-2 gates → 4–6 confirmations → 1–2 paper-positive methods.**
- **Night 1 — make the factory real.** Smoke tests + first cheap cached screens, per `plans/CODEX_NEXT_72H.md §2` (Channel-Set: **C-A1 CVaR/EVT grid, C-F survival StableCore, CE13 warmup selector, C-D1 OSC stress, CE21 no-gap filter, CE1 co-drift**; LatentWire: **L-ScoreComp source-minus-target residual + equal-byte baseline family, L-B1 damage-avoidance TrustPacket, L-A1 MMLU-Pro ceiling probe, L-A3 syndrome**). *(Sidecars are parked — CE18 is a CPU diagnostic only, not a Night-1 screen; this corrects the earlier draft that listed C-B1/CE18 here.)* By morning: is there headroom?
- **Night 2 — broaden laterally.** 40–60 more cached variants (horizon/BranchRot/CUSUM/stable-core/expert-local; pairwise/negative-evidence/conformal-set/multi-source/differential). By morning: a ranked promote list.
- **Nights 3–5 — GPU small gates** for the top 6–10 (Channel-Set C-A1/A2/A3/B1/CE18; LatentWire L-A1/A2/B1/A3/LE11).
- **Week 2 — confirmation + systems cards** for survivors only.

### 13.6 Sprint options (pick by available calendar)
| Sprint | Scope |
|--------|-------|
| **7-day fast** | infra + ~120 cached screens + small gates + a held-out re-run → know which paper has the strongest positive method |
| **14-day strong** | + full confirmation on 3–5 survivors, systems profiling, figures, claim-boundary audit → materially improve one or both papers |
| **21-day ambitious** | + one new Channel-Set model, one extra LatentWire high-entropy task + one risk/cascade task, a ReasonPacket/query-bottleneck pilot, native profiling |

### 13.7 Hard "do not queue" list
More 4-way ARC/OBQA packets · hard position-switch Channel-Set methods · norm-only residuals · naive ParoQuant+EMA stacking · unregistered "try-until-it-works" sweeps · full 20K live decode per method (use shared capture instead) · **ImpactSidecar kernels (any C-B variant) until the streamed-byte envelope drops ~100× and the selector beats DecDEC on a headroom gate** · **any LatentWire MMLU-Pro-frontier framing that does not explicitly differentiate from Latent Cache Flow (2605.22863)** · **smooth horizon-rotation mixtures before the 2-bin prefill/decode ablation has cleared >1pt over global on ≥3/4 models** · **(R2) raw score-vector compression (VQ/CountSketch/JL of source scores) framed as the LatentWire method — it must be a receiver-side-information (Wyner–Ziv) code or it is dominated by an equal-byte source-score code and by CU-HLM/TK-SLT/SMP** · **(R2) any horizon-rotation mixture that has not passed the orthogonality + full-precision-equivalence correctness gate** · **(R2) any DriftRot tail/CVaR claim that chooses and evaluates the tail policy on the same traces (use a frozen tail-validation split)** · **(R3) belief-state / structured-aggregation trust packets that do not differentiate from DarkForest (2605.25188) / Structured Message Passing (2606.00405)** · **(R3) any LatentWire packet using a replace/follow-source decode instead of combine-not-replace**.

---

## 14. IMPLEMENTATION_SPEC (engineering layer — build this before any full experiment)

This section is the concrete spec Codex compiles against. It is binding. **Milestone 1 is not a result — it is that the commands in 14.2 pass on a tiny slice.**

### 14.1 Pre-run corrections the agent must apply to its own working copy
- Stage 3 is **multi-model / held-out confirmation**, not "multi-GPU" — GPU work is serialized on one card; only CPU/cache work parallelizes.
- The screened set is **priority-20 + extended cached pool**; obey the Section 9 priority order, do not treat all variants as equal.
- **DecDEC is non-blocking** unless the local proxy already runs; the blocking Channel-Set baselines are BF16/static-W4A16, static top-K/top-10, and a ParoQuant-style (or official) baseline.
- `KL_Lens_layer_selector` is an **adapter**: implement an explicit layer-sensitivity → per-column/sidecar score map; KL-Lens is not natively per-column.
- Create `models.lock.yaml`: for each model — local path, HF ID, snapshot hash, license status, max context, quantization path, and `activation_capture_ok: true/false`. Do not chase unavailable models or surfaces.

### 14.2 Required commands (Milestone 1 — must run on `--tiny` before anything scales)
```bash
python -m pmc.freeze    --config configs/campaign.yaml
python -m pmc.baselines --paper channel_set --critical --tiny
python -m pmc.baselines --paper latentwire  --critical --tiny
python -m pmc.run_stage1 --paper channel_set --method C_A1_driftrot_cvar --tiny
python -m pmc.run_stage1 --paper latentwire  --method L_A1_high_entropy_rerank --tiny
python -m pmc.audit     --stage stage1 --results results/stage1
python -m pmc.aggregate --stage stage1 --results results/stage1
python -m pmc.plots     --results results/stage1 --out dashboard/
```
No full experiment launches before this tiny end-to-end pass succeeds.

### 14.3 Repo skeleton
```text
configs/   campaign.yaml  models.lock.yaml  baselines.yaml  stage1.yaml  stage2.yaml  stage3.yaml
registry/  channel_set/*.yaml  latentwire/*.yaml
splits/    channel_{dev,gate,confirm}.jsonl  latent_{dev,gate,confirm}.jsonl  split_hashes.json
caches/    channel_set/  latentwire/
results/   stage1/  stage2/  stage3/  systems/
src/pmc/   __init__.py freeze.py registry.py baselines.py cache_schema.py
           run_stage1.py run_stage2.py run_stage3.py audit.py aggregate.py stats.py systems.py plots.py
tests/     test_registry_schema.py test_result_schema.py test_split_hashes.py
           test_no_confirm_read.py test_latentwire_controls.py test_channel_baseline_parity.py
dashboard/ paper/ logs/
```

### 14.4 Result JSON must be NUMERIC, not prose (the aggregator gates on numbers)
Textual `"controls": {"wrong_row": "collapsed"}` is insufficient — an agent can "pass" with a label that the numbers don't support. Require numeric fields and let `aggregate.py` apply the Section 7 gates. Common envelope: `method_id, paper, stage, status, split_name, split_hash, cache_hash, registry_hash, git_commit, created_utc, wall_time_sec` plus a **`provenance` block** (`run_id`, `parent_run_ids`, `data_schema_version`, `cache_schema_version`, `method_card_hash`, `runner_git_commit`, `models_lock_hash`, `baselines_lock_hash`), a Channel-Set **`denominators` block** (`n_traces_total`, `n_traces_positive_gap`, `n_traces_no_gap`, ... — so a ratio win cannot be a no-gap artifact, enforcing CE21), a LatentWire **`leakage_audit` block** (`packet_predicts_source_top1_acc`, `mutual_info_packet_source_top1_bits`, `candidate_id_decodable_from_packet_acc` — proving the packet is not the source-copy failure), and result dirs are **write-once** — see `plans/DATA_RETENTION_SPEC.md` for the full schema and the per-example/per-trace raw-file requirement (aggregates are disposable; raw files are source of truth). Plus:
```jsonc
// LatentWire
"metrics":   { "primary_metric":0.031, "delta_beyond_label":0.012, "delta_beyond_score":0.006, "accuracy":0.421, "mrr":0.584, "ndcg_at_5":0.701 },
"baselines": { "source_top1":0.407, "source_topk_conf":0.414, "score_sketch_equal_bytes":0.409, "same_byte_text":0.392, "target_only":0.366 },
"controls":  { "wrong_row_metric":0.368, "candidate_derangement_metric":0.361, "coordinate_shuffle_metric":0.364, "control_threshold":0.375 },
"bytes":     { "payload_bytes":8, "framed_bytes":11, "baseline_equal_bytes":11, "min_code_bits_label":2.0 },
"decision":  { "promote":true, "kill":false, "park":false, "reason":"delta_beyond_score_positive_and_controls_below_threshold" }
```
```jsonc
// Channel-Set
"metrics":  { "median_recovery_R":0.812, "delta_vs_paroquant":0.058, "cvar20_recovery":0.491, "worst_trace_recovery":0.114, "no_gap_fraction":0.083 },
"controls": { "random_clip_recovery":0.702, "wrong_horizon_recovery":0.691, "random_sidecar_recovery":0.660 },
"systems_estimate": { "extra_hbm_bytes_per_token":0, "extra_macs_per_token":0, "estimated_overhead_pct":3.2 }
```

### 14.5 Leakage enforcement (mechanical, not promised)
- Confirmation-split paths are **absent from environment variables** during Stage 1/2.
- Stage 1/2 runners **fail on import/open of any path matching `*_confirm*`**.
- The audit agent verifies a wrapped-`open()` access manifest per job; any confirm-path touch → `KILLED`/`PARKED`. Especially load-bearing for VQ/score-codebook methods (L-ScoreComp), which overfit held-out rows trivially.

### 14.6 Failure budget (hard caps — `configs/campaign.yaml`)
```yaml
max_retries_per_job: 2
max_gpu_hours_per_stage1_method: 0      # Stage 1 is CPU/cached
max_gpu_hours_per_stage2_method: 2
max_gpu_hours_per_stage3_method: 12
max_wall_clock_per_baseline_repro: 8h
park_if_dependency_missing: true
```
On exhaustion → `PARKED` (never grind). Do not spend a night fixing C2C while LatentWire score packets are unscreened.

**Null-method sentinels (audit agent runs these every cycle; if any "passes" a gate, controls are broken or there is leakage → halt promotion, alert human):**
```yaml
sentinel_jobs:
  channel_set:
    - random_rotation_same_norm
    - shuffled_trace_policy
    - random_sidecar_same_budget
  latentwire:
    - label_permutation
    - row_shuffle
    - candidate_derangement
    - equal_byte_random_score_code
kill_run_if_sentinel_passes: true   # a sentinel clearing any promote gate => broken controls/leakage
```

### 14.7 Blocking vs non-blocking baselines, and idle-work
**Blocking (must reproduce within tolerance before methods queue):** Channel-Set → BF16, static W4A16, static top-1/top-10, local-ParoQuant-style (or official if immediately available). LatentWire → target-only, source-index, source-index+confidence, equal-byte score sketch, same-byte text.
**Non-blocking but required before final claims:** Channel-Set → DecDEC, official-ParoQuant parity (C-E1), ConQuR, LAQuant, ResQ, KL-Lens. LatentWire → C2C, KVComm, KVCOMM, DroidSpeak-style byte/state accounting (Stage-4 frontier anchors).
**Idle-work (when no audited job exists — never make-work):** run tests, update dashboards/figures, write `paper/` prose (methods/results/limitations idle-fill; baseline/related-work rides with Phase A), refresh `scoop_check.md`, summarize `PARKED` items into `morning_brief.md`.

### 14.8 First-night ("start here") task list
```text
1. Repo skeleton + schemas (registry parser, NUMERIC result-JSON validator).
2. Freeze tiny dev/gate/confirm splits; write split_hashes.json + models.lock.yaml.
3. Critical baselines on tiny cache:
   Channel-Set: BF16 / static / local-ParoQuant-style.
   LatentWire: target-only / source-index / +confidence / equal-byte score sketch / same-byte text.
4. First Stage-1 Channel-Set screens (per `plans/CODEX_NEXT_72H.md §2`): C-A1 CVaR/EVT clip grid; C-F survival StableCore; CE13 warmup policy selector; C-D1 OSC stress; CE21 no-gap filter; **C-A2 orthogonality + full-precision-equivalence + KV-cache-basis tests ONLY (Milestone 1.1, blocking — no A2 runner until they pass)**. *(No sidecar screen here — sidecars are parked; CE18 is a CPU diagnostic only.)*
5. First Stage-1 LatentWire screens: L-ScoreComp source-minus-target residual + equal-byte score baseline family; L-B1 damage-avoidance TrustPacket (combine-not-replace decode, candidate-leakage audit); L-A1 MMLU-Pro information-ceiling probe only; L-A2 candidate-pool cache build (GPU capture).
6. **Codex packaging (do this first):** confirm the lean root `AGENTS.md`, `plans/CODEX_NEXT_72H.md`, and `plans/DATA_RETENTION_SPEC.md` exist and that Codex loaded the intended files; write-once result dirs + access manifest are in place.
6. Run audit + aggregator; enforce leakage guard + failure budget.
7. Emit morning_brief.md: what ran / failed / passed schema / baseline parity / next GPU queue.
```

## 15. Caveats
- LAQuant, KL-Lens, UCCI, ConQuR, TR-DQ, Q-Drift are 2025–2026 preprints; **read KL-Lens, UCCI, and especially Latent Cache Flow (2605.22863), DarkForest (2605.25188), and OSC (2604.12782) in full before finalizing positioning** — LCF most constrains the LatentWire frontier spine and L-A; DarkForest most constrains L-B / belief-state packets; OSC (token-persistence claim) most constrains the Channel-Set drift thesis; DecDEC (2412.20185) dominates the ImpactSidecar. ParoQuant ICLR'26 status confirmed (z-lab project page; Qwen3-4B AIME-24: AWQ 62.2 / ParoQuant 73.3 / FP16 75.6).
- **Citation fixes applied 2026-06-02:** relative representations = 2209.15430 (Moschella, ICLR 2023), NOT 2305.13093; ParoQuant repo = z-lab project (<https://z-lab.ai/projects/paroquant/>), NOT thu-nics (which owns C2C); TurboQuant = 2504.19874 (ICLR 2026). The claimed repos github.com/jasonkongie/kl-ssm-quant and github.com/Zephyroam/KVComm were not directly confirmed (the underlying papers are real) — verify before depending on them.
- **Latent Cache Flow (2605.22863) caveat:** its "beats oracle routing frontier" result is on small Qwen pairs (0.5–0.6B) with budget measured as adapter size / latent width / TTFT, not literal bytes/query, and is self-described as preliminary — verify against its tables before treating the frontier spine as fully scooped; if LCF fails to replicate at realistic bytes/query, L-A reopens.
- Some code links are inferred from author orgs (thu-nics, etc.); confirm the repo exists and matches the paper before depending on it for apples-to-apples runs.
- Re-run the scoop check the week before July 12.
- "OPEN" verdicts reflect a targeted sweep, not proof of absence.
- Workshop main text is 4–10pp and self-contained: the decisive gates, controls, and limitations must be in the body, not only appendices.
