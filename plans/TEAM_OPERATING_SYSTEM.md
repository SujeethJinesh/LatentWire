# plans/TEAM_OPERATING_SYSTEM.md — how the Codex research org behaves

Make the team behave like a ruthless, high-output research lab: fast evidence, hard baselines, aggressive falsification, clean memory, paper-ready claims. Binding. Companions: `EXPERIMENT_SELECTION_SPEC.md` (what to run next), `PLANTED_TESTS_SPEC.md` (positive controls). The science org/roles are in `CODEX_NEXT_72H.md §8`; the breakthrough behaviors in `§9`. This file is the *execution discipline*.

## Conductor — the only role with decision rights
The top-level session is the **Conductor**. It is the ONLY role that may: move a method between states (`DRAFTED/BUILT/CPU_SCREENED/AUDITED/GPU_QUEUED/PASSED/KILLED/PARKED`); change queue priority; approve a new registry card (after the Ideation Lab's math+adversary+scoop review); promote a result to paper-candidate; declare a human checkpoint. **All other agents produce evidence / patches / reviews / recommendations — they never change method state directly.**

Run the Conductor as a Codex **Goal** (`/goal …`; GA, CLI ≥ 0.128.0) — a persistent, verifiable completion contract, not a bigger prompt. Define the outcome + verification surface + constraints (preserve `AGENTS.md`, `CODEX_NEXT_72H.md`, `DATA_RETENTION_SPEC.md`, and the no-confirm-access rule). Manage with `/goal pause|resume|clear`. Goals **increase** the need for oversight: on each check-in, review the diff and run the tests — never trust the agent's summary.

Each cycle the Conductor writes `dashboard/conductor_state.md`: current objective · top-5 open risks · GPU queue head · CPU queue head · blockers · methods promoted/killed/parked since last cycle · next command for the local GPU runner.

## WIP limits (ruthless — activity ≠ progress)
- ≤ **2 method families per paper** in "serious push" at once; everything else is CPU-screen, diagnostic, or parked.
- ≤ **3 new registry cards graduated / 24 h**; ≤ **4 Stage-2 promotions total** before a human checkpoint.
- **No GPU job without a claim-to-figure mapping** (below).

## Cycle rhythm
1 ingest results → 2 audit guardrails → 3 update `leaderboard.csv` → 4 update breakthrough/kill boards → 5 mine anomalies → 6 write lessons for kills/parks → 7 create/reject ideas → 8 schedule next CPU/GPU by **EVI** (`EXPERIMENT_SELECTION_SPEC.md`) → 9 update paper claims + limitations → 10 write `morning_brief.md`.

## Claim-to-figure discipline (from day one)
Every Stage-2 promotion must include: `proposed_claim_sentence` · `nearest_disallowed_overclaim` · `figure_or_table_slot` · `required_baselines_in_same_figure` · `required_controls_in_same_figure` · `limitation_sentence`. A result that maps to no figure/table is probably not worth GPU. Every builder thinks like a reviewer: *"what figure would make this undeniable?"*

## Hardest-baseline champion (a standing duty of the baseline-adversary)
Make the strongest *fair* baseline as strong as possible **before** any positive claim. For each promoted method, implement the nearest stronger baseline variant, tuned **only** on the same dev/gate data as the method; maintain `baseline_upgrade_log.md`; rerun deltas after each upgrade. LatentWire focus: equal-byte score sketch, source-index+confidence, same-byte text, calibrated fusion. Channel-Set focus: local/official ParoQuant parity, static top-K/10, EMA, OSC/DecDEC reconciliation. **A method is demoted if a baseline upgrade erases its gate — and the team treats that as the paper getting more honest, not a loss.**

## Anomaly-mining agent (conditional wins, not just global wins)
Trigger: after every Stage-1/2 result wave. Reads `per_example.jsonl`/`per_trace.jsonl` + `result.json` + baselines + controls for the *same rows*. Output: `dashboard/anomaly_report_<cycle>.md` and `ideas/anomaly_seed_<id>.md` when warranted.
- **Channel-Set:** traces where ParoQuant fails but C-A1/C-F/CE13 helps; tail-helps-but-median-hurts; layer/model/horizon slices with unusual rescue; no-gap→positive-gap under a method; wrong-horizon controls that unexpectedly work.
- **LatentWire:** source-wrong/target-correct damage rows; source-correct/target-wrong repair rows; rows where residual-WZ beats all sketches; rows where the packet helps but source-index does not; rows where same-byte text unexpectedly wins.
Rules: never promote from anomaly mining alone; an anomaly becomes a registry card only after math + scoop + adversary review. *"Works on disagreement rows with target entropy > τ and source margin > m"* beats *"mean +0.3%."*

## Method lineage (machine-enforced dedup)
Every registry card also carries `contribution_role` + `paper_claim_eligible` (see `CODEX_NEXT_72H.md §4.5`): **only `positive_method`/`positive_enabler` cards may be promoted to a paper claim**; `diagnostic_defense`, `baseline_adversary`, `ceiling_probe`, `kill_only`, and `parked` cards are support and are never presented to the paper team as methods to claim. Plus the lineage block:
Every registry card carries:
```yaml
lineage:
  parent_methods: []
  nearest_killed_methods: []
  shared_failure_modes: []
  mechanism_family: [rotation_tail | survival_core | warmup_policy | wz_score_residual | trust_damage_avoidance | ...]
  differs_from_parent_by: "one sentence"
  would_be_killed_if_parent_failure_reappears: true
```
**`lineage_audit` fails the card if:** it recreates a logged killed mechanism without a *new* falsifier · it has no nearest baseline · it has no nearest killed relative. Every ideator must answer: *"what killed the nearest ancestor, and why does this avoid that mechanism?"*

## Breakthrough pre-mortem (every win is presumed fake)
Before any Stage-2 GPU job, write `reviews/premortem_<method_id>@<code_sha>.md`: (1) most likely trivial explanation if it wins; (2) hardest baseline that could erase it; (3) most likely leakage path that fakes it; (4) structural/theoretical reason it may fail; (5) fastest falsifier; (6) paper claim if it wins; (7) bounded-negative claim if it fails. **The method is believed only after the pre-mortem's falsifiers are run and fail to kill it.**

## Repair protocol (fix from a packet, not from scratch)
On failure the worker writes `queues/failures/<run_id>.md`: `run_id · method_id · code_sha · command · last_good_artifact · stderr_tail · suspected_failure_class · reproducible_on_tiny · proposed_minimal_fix · tests_to_run_after_fix · retry_count`. The top-level session re-dispatches a fix prompt that **consumes the packet and produces one minimal patch** (no scratch debugging). Honor `max_retries=2`, then PARK.

## Context hygiene (no reliance on chat memory)
Every subagent prompt states: exact file(s) it may edit · exact file(s) it must read · exact output artifact path · success criteria · forbidden actions · max wall time · whether it may touch experiment data. Every subagent response ends with: `files_changed · tests_run · artifacts_written · blockers · next_command`. **No agent relies on conversation memory for a claim-bearing decision**; the restart source of truth is `AGENTS.md + CODEX_NEXT_72H.md + the registry card + result files + reviews`. Prefer small, checkable diffs over big "I built the system" patches.

## Agent-improvement loop (every repeated mistake becomes a test)
Every 24 h the Conductor writes `dashboard/agent_retro.md`: top-5 agent failures · which guardrail caught each · which escaped to human review · repeated? · new eval/lint to prevent recurrence · proposed patch to AGENTS/ExecPlan/tests. **Add a new eval when:** a schema bug occurs twice · an agent proposes a killed method · an internet agent emits an uncited claim · a GPU job idles unexpectedly · a result lacks raw per-example/per-trace data.

## Morning brief = decision artifact (`dashboard/morning_brief.md`)
1 Executive decision (continue / pivot / halt / human-checkpoint). 2 GPU util (foreground hrs / backfill hrs / idle min / unexpected-idle incidents). 3 CPU util (Stage-1 done / queue depth / failed-parked). 4 Evidence (top promoted / killed / ambiguous + the exact next falsifier). 5 Baseline status (blocking locked? hard baselines upgraded?). 6 Guardrails (leakage / sentinel / schema / write-once-raw). 7 Paper movement (figures / claims / limitations). 8 Next 6 commands.

## Breakthrough standard (a candidate must clear ALL)
beat the hardest fair baseline · survive destructive controls · **pass the planted-signal suite** · have raw per-example/per-trace evidence · have a mechanism explanation · map to a paper figure · carry a limitation sentence · pass red-team review · be **clean-room replicated**.

## Research-agent behavioral contract (binding)
1 **Evidence first** — never narrate a win before showing the raw result path, baseline, control, split hash. 2 **Smallest falsifier first** — before improving a method, run the cheapest test that could kill it. 3 **Attack wins harder than losses** — a positive triggers red-team + leakage + baseline-upgrade review immediately. 4 **Prefer bounded negatives to ambiguous positives.** 5 **Never optimize against confirmation** — another tuning pass goes back to dev/gate. 6 **Make baselines stronger before claiming.** 7 **Separate mechanism from metric** — say what moved and what did not. 8 **Preserve optionality** — CPU wide, GPU narrow. 9 **No orphan artifacts** — every output reachable from a dashboard / registry card / run_id / paper table. 10 **Write the limitation while the result is fresh.** 11 **Disagree explicitly** — reviewers state the strongest reason to reject, not "PASS." 12 **Activity ≠ progress** — more variants are bad if they don't move the decision frontier.
