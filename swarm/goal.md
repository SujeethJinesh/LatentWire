# COLM 2026 Swarm Goal

Drive every alive project in this repo to camera-ready submission state for COLM 2026 (with MLSys 2026 / ICLR 2027 follow-up positioning). Read AGENTS.md and experimental/HANDOFF.md first; treat them as authoritative. Prioritize scientific integrity and submission quality over speed.

## Operating environment

- Full read/write authority over /workspace except where this document forbids modification.
- One RTX PRO 6000 Blackwell GPU with 96GB VRAM. Use it to its full capacity. Saturate compute when running profiler/inference.
- /workspace has 500GB total. /workspace/hf_cache holds model weights (~225GB used). If disk pressure occurs, you may delete cached models you do not currently need (Phase 2 weights for example may be re-downloaded later when needed). Do not delete preregistration files, paper drafts, or result packets.
- Authenticated git push (HTTPS via gh credential helper) and authenticated HuggingFace download. If either breaks, treat it as an infra failure and surface to human rather than papering over.

## Execution loop

For each entry in swarm/queue.yml in priority order:

1. Read its preregistration file. Do not modify it.
2. Run its runner script with full output captured to experimental/<project>/phase{0,1,2}/results/<run_id>/.
3. Run its checker. The checker's exit code and decision string are the ground truth — never override.
4. On PASS: append on_pass entries to queue.yml; spawn a subagent committee review on the result packet; update the project's paper draft with real numbers from the packet; rebuild the PDF; update reviewer_pack.md.
5. On KILL: write experimental/KILLED_<project>/README.md with the full kill manifest including artifact SHAs; update HANDOFF.md status table only (not preregs); commit.
6. Update swarm/state.json after every gate.
7. Commit after every gate result with message `AUTO: <project> <PASS|KILL> + paper update`. Push every commit (do not batch).
8. Continue to next queue entry.

## Self-unblocking

You are expected to unblock yourself when problems arise, but the unblocking must be principled and never hacky.

- When a tool, library, or model fails: read the actual error message, understand what it means, search the web for known issues with that exact error and version, and apply the documented fix. Do not silence errors with try/except blocks. Do not skip steps with TODO/FIXME placeholders. Do not lower standards (never reduce required row counts, relax statistical thresholds, or substitute simpler models without preregistration amendment) to make a check pass.
- When you encounter a known-issue scenario (vLLM version mismatch with a model architecture, missing CUDA library, conflicting torch versions): search the web for the upstream issue tracker, find the documented resolution, and apply it. If the documented fix requires upgrading a pinned dependency, evaluate the blast radius first: will it break torch/triton compatibility with the GPU? If yes, find a different path. If no, upgrade and proceed. Document the change in your commit message.
- When you need information you do not have: web-search before guessing. Cite sources in commit messages or comments where relevant.
- When a runner or checker script does not exist for an entry in queue.yml (this is true for most Phase 0/1/2 entries): you author it. The runner must implement what its preregistration specifies, no more and no less. The checker must apply the preregistered decision rule mechanically. Both must produce auditable artifacts. Spawn a subagent to draft, then spawn an independent subagent to code-review for adherence to the prereg before running.
- When a paper section needs writing: draft, build the PDF, run committee review, fix issues, rebuild, repeat up to 3 times per update.
- When you genuinely cannot proceed (true infra failure, ambiguous prereg requiring human judgment, conflicting evidence requiring human decision): write a clear handoff note to swarm/blocked_<entry_id>_<date>.md describing exactly what's blocked and what decision the human must make, update state.json with status=BLOCKED, commit, and pause. Do not guess.

## Reproducibility (non-negotiable)

- Every result packet must include: full environment snapshot (pip freeze, nvidia-smi output, CUDA version, driver version), exact model SHA from HuggingFace, exact prompt set with SHA-256, all random seeds used, exact command line invocation, full stdout/stderr logs.
- Every paper claim must cite an exact artifact path that resolves to a passing artifact_check.json.
- Every preregistered threshold appears verbatim in the paper alongside the measured value.
- No "approximately" numbers in tables — exact measured values with bootstrap CI.
- Code in the repo must be runnable end-to-end by a third party reading only the README and the paper.

## COLM 2026 submission conventions

- Use the official COLM 2026 LaTeX template (web-search for it; download if not in repo). Page limit per the call for papers (verify via web search).
- Anonymize for double-blind review. No author names, no acknowledgments, no GitHub URLs that reveal identity in the camera-ready candidate.
- Required sections: Abstract, Introduction, Related Work, Method, Experiments, Results, Discussion, Limitations, Reproducibility Statement, Ethics Statement (where applicable). Reproducibility Statement must explicitly point to the artifact paths.
- Tables: format with booktabs. No raster screenshots of numbers. All numbers come from artifact paths cited in caption.
- Figures: matplotlib + tikz/pgfplots preferred. PDF figures, not raster, where possible.
- Citations: BibTeX. Hand-verify every citation actually says what you claim it says — no hallucinated citations. If you cite a 2026 paper, web-search confirm it exists and matches your description.
- Limitations section is genuinely critical — reviewers read it carefully. List real limitations, not strawmen.
- Reproducibility Statement names the exact commit SHA, GPU model, and runtime hours required to reproduce.

## Subagent usage — use aggressively

- Spawn a fresh subagent for each gate's artifact-checker run, so the main agent's context stays clean.
- Spawn a fresh subagent for each paper rebuild and reviewer-pack update.
- Spawn a 3-member committee review subagent every time a paper is updated. The committee plays:
  (a) a COLM area chair scoring novelty/rigor/clarity 1-10 and writing the meta-review;
  (b) an MLSys reviewer focused on systems contribution, reproducibility, and engineering rigor;
  (c) an adversarial reviewer hunting for unjustified claims, missing controls, p-hacking, scope creep, hallucinated citations, and statistical sleight-of-hand.
  Each writes ≤300 words. Append committee output to experimental/<project>/paper/committee_reviews/<date>_<round>.md and commit. If any committee member scores below 6/10 or flags a fixable issue, spawn a fix subagent before continuing.
- Spawn a fresh subagent for every git commit and push, so the main loop never blocks on git.
- Spawn periodic "audit subagents" every 4 hours that re-verify: (i) no preregistration file has been modified since started_at_sha, (ii) every paper claim cites a passing artifact_check.json, (iii) no TODO/FIXME/XXX/hardcoded-stub strings in alive .tex files, (iv) every reviewer_pack.md entry resolves to an existing artifact path, (v) every BibTeX entry in alive papers has been web-search-confirmed (sample 3 random citations per audit).
- Spawn a "reproducibility audit" subagent before any paper is marked camera-ready candidate. It attempts to follow the paper's Reproducibility Statement from a fresh shell and reports whether it can reproduce the headline number. If it cannot, the paper is not camera-ready.

## Paper iteration loop

For each alive paper after every gate update:

1. Subagent rewrites the affected sections incorporating the new numbers. Cites exact artifact paths.
2. Builds the PDF (look for Makefile, build.sh, or `latexmk -pdf` in the paper dir; if none exists, author one and commit).
3. Committee review subagent (3 members above) reviews the new draft.
4. If any committee member identifies a fixable issue, spawn a fix subagent. Iterate up to 3 review→fix cycles per paper update.
5. Mark paper "camera-ready candidate" only when: (a) PDF builds clean, (b) all committee members score ≥7/10, (c) audit subagent passes, (d) reviewer_pack.md is current, (e) reproducibility audit passes.
6. Camera-ready candidates still require human review — do not mark final.

## Alive projects to drive to camera-ready

HybridKernel, ThoughtFlow-FP8, plus any Phase 0 branch (OutlierMigrate, Residual Migration, SSM-State Lifecycle, Cross-Layer Error Compounding) that passes its Phase 0 gate. Killed projects do not get papers — they get KILL manifests.

## Scope — may modify

- `experimental/<project>/phase{0,1,2}/results/`
- `experimental/<project>/paper/` (tex, md, figures, committee_reviews/, reviewer_pack.md, build scripts)
- `experimental/KILLED_<project>/README.md`
- `swarm/queue.yml` (only to append on_pass conditionals or mark entries done)
- `swarm/state.json`
- `swarm/final_report.md`
- `swarm/blocked_*.md` (handoff notes when truly stuck)
- `swarm/progress_*.md` (4-hour progress notes)
- `experimental/HANDOFF.md` (status-table updates only, never operating rules)
- Runner and checker scripts you author for not-yet-implemented queue entries
- Any non-protected files in /workspace as needed for environment maintenance

## Scope — must NEVER modify

- any `preregister*.md` file
- `AGENTS.md`
- any file under `experimental/KILLED_*/` except creating new KILL manifests
- any paper draft under `experimental/thoughtflow_fp8/` EXCEPT for adding new committee reviews and updating reviewer_pack.md (the paper itself is in copyedit-only mode per HANDOFF)

## Stop rules

Any of these pauses /goal and surfaces to human:

- Three consecutive FAIL_INFRA queue results.
- Any preregistration file modified during execution (audit subagent detects).
- `gpu_hours_used` in state.json exceeds 160 → graceful wrap-up: finalize current entry, write swarm/final_report.md, push.
- Any committee member detects p-hacking, post-hoc cherry-picking, or scope creep that requires preregistration amendment.
- A paper test that previously passed starts failing.
- A git push fails twice in a row.
- A reproducibility audit fails on a paper marked camera-ready candidate.
- A web search for a cited 2025/2026 paper returns no matching publication (likely hallucinated citation).

## Goal achieved when all of

- `swarm/audit_swarm_completion.py` exits 0 with `SWARM_COMPLETE`
- every alive paper builds clean
- every alive paper has committee reviews with all members ≥7/10
- every alive paper's reviewer_pack.md is current
- every alive paper has passed reproducibility audit
- `swarm/final_report.md` exists summarizing what survived, what was killed, what humans need to review next, and the exact commit SHAs for each camera-ready candidate

## Goal unmet (acceptable terminal state)

If budget runs out before the above. In that case, the wrap-up writes swarm/final_report.md with current state, explicit next-action list for the human, and partial camera-ready candidates clearly labeled.

## Reporting

- Write a "progress note" to `swarm/progress_<date_hour>.md` every 4 hours containing: current entry, what's done, what's running, expected next 4 hours, any concerns.
- The final `swarm/final_report.md` must include: paper-by-paper status, GPU hours actually used per entry, total wall-clock, total dollar spend estimate (~$1.89/hr), any preregistration that was killed and why, any committee feedback that's still unresolved.

## Persistence and pivot behavior — keep digging for positive methods

The objective is to ship at least one positive-method paper to COLM 2026. A positive-method paper proposes a new method, shows it works, and explains why. This is what publishes at top venues.

ThoughtFlow-FP8 is a separate case: it is a falsification paper that is already drafted and represents the safe submission. Everything else in the portfolio is being driven toward positive-method outcomes.

The default ranking of outcomes for this swarm, from best to worst:

1. Multiple positive-method papers from different branches.
2. One strong positive-method paper plus the ThoughtFlow falsification paper.
3. The ThoughtFlow falsification paper alone, with documentation of why other branches did not yield positive methods despite genuine effort.
4. No submission — last-resort outcome, requires explicit justification in swarm/final_report.md.

Outcomes 1-3 all count as goal-achieved or partial-success. Outcome 4 is acceptable only when evidence genuinely does not support any positive-method paper after the effort floor below has been exhausted.

### When a queue entry kills

A kill is data, not a conclusion. After writing the KILL manifest, before moving to the next queue entry, spawn a "diagnostic subagent" that produces diagnostic.md in the killed project's directory. The diagnostic must answer:

- What was the proximate failure? Distinguish between (a) the hypothesis was wrong, (b) the experimental setup was insufficient to detect the effect, (c) infrastructure issue, (d) prereg ambiguity.
- If (b): what setup change would give the original hypothesis a fair test? Smaller granularity, different decode positions, layer-stratified analysis, different model class. If a fair retest is feasible, propose it as a fresh preregistration with new thresholds. This is not a re-run of the killed experiment; it is a new experiment with a related but distinct hypothesis.
- If (a): the data probably contains hints about what *would* work. Examine the result packet for unexpected patterns. Examples: "outliers don't migrate but they cluster" suggests a clustering-based method. "States don't age uniformly but layer-13 states age strongly" suggests a layer-targeted method. List up to 3 alternative positive hypotheses informed by what the data actually showed. Each requires a fresh preregistration with new thresholds and a clear positive-method gate.
- For each alternative hypothesis, evaluate plausibility: how likely is this to yield a positive result, what would the resulting paper claim, and is it competitive at COLM 2026.

If the diagnostic identifies a plausible alternative positive hypothesis, author a fresh preregistration for it (new file, new branch directory, new gate criteria), draft runner/checker, and add to the queue with priority such that it runs before lower-priority original entries. Pivot depth limit: do not chain pivots beyond depth 2 from any original branch (a pivot of a pivot is fine; a pivot of a pivot of a pivot is not).

Do not spawn paper iteration on a kill itself. Papers are only drafted for entries that PASS their gates. The diagnostic.md is internal documentation, not a paper draft.

### When an entry hits a hard infra wall

Before declaring FAIL_INFRA, exhaust this checklist:

- Web-search the exact error message with the version of every relevant tool (vllm, torch, transformers, triton, cuda).
- Check the upstream issue tracker for that tool. If a known fix exists, apply it (per the self-unblocking section's blast-radius evaluation).
- Try alternative configurations: smaller batch size, different attention backend, eager mode instead of cudagraph, FP8 instead of BF16 if supported.
- Try a different model in the same architectural class if the original is buggy in vllm. Document the substitution and its scientific equivalence to the preregistration.
- Spawn a fresh subagent with no context to attempt the same task — sometimes context contamination causes loops.

Only after the checklist is exhausted does FAIL_INFRA count toward the three-strikes stop rule.

### Forbidden pivots (these are p-hacking, never do these)

- Re-running a killed experiment with different random seeds hoping for better statistics.
- Loosening any preregistered threshold to convert a kill into a pass.
- Substituting models or datasets after seeing kill results, without a fresh preregistration.
- Cherry-picking a subset of layers, positions, or prompts that happen to show a positive effect.
- Recharacterizing a killed hypothesis as a "different finding" without a fresh preregistration with new thresholds set before observing the new data.
- Pivoting to a method that has already been killed elsewhere in the portfolio.
- Writing a paper that omits the failure modes documented in diagnostic.md or committee_reviews.

If a subagent proposes any of the above, the supervising agent kills the proposal and logs the attempt to swarm/integrity_violations.md. Three integrity violations pause /goal and surface to human.

### Effort floor before declaring goal-unmet

Before declaring goal-unmet with zero positive-method camera-ready candidates, you must have:

- Run every original Phase 0 entry in the queue (not just stopped at the first kill).
- Authored a diagnostic.md for every killed entry.
- Pursued at least one pivot per killed branch where the diagnostic identified a plausible alternative positive hypothesis.
- Continued pivoting until either a positive method emerges OR the gpu_hours_used budget is genuinely close to exhaustion (within 10 hours of the 160-hour limit).
- Written swarm/final_report.md with explicit justification for why no positive-method submission is possible despite the pivots attempted.

ThoughtFlow's falsification paper does not count toward the positive-method requirement; it is a separate path. If ThoughtFlow is the only paper that ships, that is outcome 3 above (acceptable but not optimal).

Anything less than this effort floor means /goal continues working. Stopping early without exhausting the effort floor is a failure mode worse than a genuine negative outcome.

### Quality bar for positive-method papers

A positive-method paper must satisfy all of:

- The method is novel (web-search audit confirms no 2025/2026 paper has proposed it).
- The method works on the primary model with measurable improvement past a preregistered threshold.
- The improvement reproduces on at least one cross-family control.
- The paper has a clear story: what the method does, why it works, when it fails.
- Committee review scores ≥7/10 from all three reviewers.
- Reproducibility audit passes.

Do not ship a paper that fails any of these. A paper that ships and gets desk-rejected is worse for your reputation than no submission. Better to extend the swarm by another pivot cycle than to ship a weak paper.

## Preregistration drift vs authorized pivot creation — disambiguation

The audit subagent's "no preregistration file modified since started_at_sha" rule is a stop rule, but it must distinguish between two kinds of preregistration change:

1. MODIFICATION of an existing preregistration file (any preregister_*.md that existed at started_at_sha) is FORBIDDEN. This includes editing thresholds, changing model IDs in cross-family slots, adjusting position grids, or any text change. This rule remains absolute and unchanged.

2. CREATION of a fresh preregistration file in a new branch directory (new path, new SHA) IS PERMITTED if and only if all of:
   - It was authored as part of an authorized pivot per the "When a queue entry kills" section.
   - The killed branch's diagnostic.md exists and explicitly identifies the alternative positive hypothesis the new prereg targets.
   - The new prereg's thresholds were set without observing any data for the new hypothesis.
   - The new prereg has fresh gate criteria, not relaxed versions of the killed branch's criteria.
   - The new prereg lives in a new branch directory (not the killed branch's directory).
   - Pivot depth from the original branch is <=2.

Audit subagents must apply this disambiguation when checking for preregistration drift:
- A modification of an existing preregistration file ALWAYS triggers stop.
- A creation of a new preregistration file triggers stop ONLY if any of the six pivot conditions above is violated.
- When uncertain, the audit subagent writes its uncertainty to swarm/audit_uncertain_<date>.md and pauses, surfacing to human for review rather than silent decision.

This disambiguation resolves the May 8 2026 audit pause caused by the Decode Microkernel pivot preregistrations being flagged as drift. The Decode Microkernel pivot satisfies all six conditions and is therefore authorized; subsequent audits should not flag it.

## Authorized post-start preregistration manifest

The following preregistration files were created after started_at_sha and are explicitly authorized as fresh pivot preregistrations:

- experimental/decode_microkernel/phase0/preregister_dmc_phase0.md
- experimental/decode_microkernel/phase1/preregister_dmc_phase1.md
- experimental/decode_microkernel/phase2/preregister_dmc_phase2.md

These preregistrations were authored in response to the HybridKernel KILL_HYBRIDKERNEL_BELOW_SHELF decision. The HybridKernel diagnostic identified decode-path microkernel optimization as a plausible alternative positive hypothesis. Decode Microkernel preregistrations were authored before any decode-microkernel-specific data was observed. Pivot depth is 1 from HybridKernel (within the limit).

Audit subagents must consider the files in this manifest as authorized creations and not flag them as preregistration drift.
