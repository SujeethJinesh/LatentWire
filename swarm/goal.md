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

## 10-hour authorized work window — May 8 2026

Context: Swarm is BLOCKED on Phase 2 cross-model validation requiring Qwen3.6 (vLLM 0.10.2 incompatibility). Human is unreachable for approximately 10 hours. The Qwen3.6 block remains. Do NOT upgrade vLLM. Do NOT download Qwen3.6 or Kimi Linear weights. The following work is authorized and high-value for this window.

### Authorized work (priority order)

1. **OutlierMigrate related-work reframing**: Earlier portfolio audits missed prior dynamic-outlier-in-Mamba literature. Update experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex related work section to cite:
   - QMamba (Li et al., arXiv 2501.13624, January 2025): first PTQ for vision SSMs; identifies highly dynamic hidden-state behavior and proposes Temporal Group Quantization
   - OuroMamba (Ramachandran et al., arXiv 2503.10959, ICCV 2025): explicitly contrasts dynamic VMM outlier patterns vs static ViT patterns; proposes adaptive outlier selection in hybrid Transformer-Mamba vision models
   - Quamba (arXiv 2410.13229), Quamba-SE (arXiv 2601.09451), Mamba-PTQ (arXiv 2407.12397), MambaQuant (arXiv 2501.13484): language Mamba PTQ work that assumes static outlier behavior

   Reframe the OutlierMigrate contribution: the language Mamba PTQ literature treats outliers as statistically stationary; the vision Mamba PTQ literature found them dynamic. We test which regime applies to hybrid LLMs in long reasoning traces, find the dynamic regime at 84% migration, and position the implications for static-protection methods (BlockDialect, AWQ, SmoothQuant, QuaRot, KVQuant) used in language deployments.

   Forbidden claims: do NOT claim "first to find dynamic outliers in Mamba" — vision Mamba literature established this. The novel empirical contribution is the language reasoning quantification on hybrid LLMs.

   Run a fresh 3-member committee review after reframing. Iterate up to 3 review→fix cycles. Do not mark camera-ready-final.

2. **Continue queued Phase 0 entries in priority order** (all preregistered, all unaffected by the Qwen3.6 block):
   - residual_migration_phase0
   - ssm_lifecycle_phase0
   - cross_layer_error_theoretical

   For each: execute runner, run checker, apply PASS/KILL decision per the preregistration, update queue.yml on_pass conditionals, drive paper iteration on any pass.

3. **Add Kimi Linear to cross-validation queue.yml entries** (queue-only edit, not preregistration). The model `moonshotai/Kimi-Linear-48B-A3B-Instruct` should appear in cross_model_validation_outlier_migrate, cross_model_validation_residual_migration, and cross_model_validation_ssm_lifecycle entries' models blocks alongside the existing entries. This is operational layer scope and is permitted. Do not download the weights; this is a queue-config change only.

4. **Partial Phase 2 cross-validation on Nemotron-3 only**: When cross_model_validation_outlier_migrate becomes runnable (after RM/SSML/CLE Phase 0 complete), execute it with ONLY nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (vLLM 0.10.2 supports NemotronHForCausalLM). Document explicitly in the result packet and any paper updates that Qwen3.6 and Kimi Linear validation are deferred pending vLLM compatibility resolution. Do not claim full cross-validation; claim partial cross-validation on one cross-family hybrid model.

   Allowed: download Nemotron-3-Nano-30B-A3B-BF16 weights (~60GB), execute inference matching the Phase 1 protocol, produce migration metrics with full bootstrap CI, integrate results into OutlierMigrate paper.
   Forbidden: download Qwen3.6 or Kimi Linear weights, modify any preregistration to relax full-cross-validation requirements, claim the Nemotron-only result is full cross-validation.

5. **ThoughtFlow paper polish**: If committee scores are not yet >=7/10 from all three reviewers across recent rounds, run additional committee review rounds and polish accordingly. Do not mark camera-ready-final.

### Forbidden during this window (hard rules)

- Upgrading vLLM (would break validated torch/triton/CUDA stack)
- Downloading Qwen3.6 or Kimi Linear weights (cannot be loaded with current vLLM)
- Marking any paper "camera-ready final" (candidate is the ceiling without human review)
- Modifying any preregistration file (drift policy unchanged)
- Acting on Decode Microkernel beyond documenting current Phase 2 infra-block status (engineering resources required, not LLM resources)
- Making decisions about HybridFPGA collaboration or any Tambe lab outreach (human-only)
- Issuing any new pivot preregistration without explicit diagnostic.md justification

### Window-specific stop rules (additive to existing rules)

- Any model download fails twice for non-network reasons → blocked note, pause
- vLLM produces a load error that cannot be self-resolved by configuration changes within the no-upgrade constraint → blocked note, pause
- gpu_hours_used would exceed 100 cumulative → graceful wrap-up, write swarm/final_report.md, push, pause
- An audit subagent identifies that one of the previously committee-passed papers regressed → pause
- Three consecutive infra failures (existing rule, restated)

### Reporting during this window

Write swarm/progress_<date>_<hour>.md every 2 hours (denser than the usual 4-hour cadence) so the human has clear visibility on landing. Each progress note must include: what completed, what's currently running, any concerns, cumulative gpu_hours_used, and the camera-ready-candidate status of every alive paper.

## 15-20 hour authorized work window — May 8/9 2026

Context: Swarm is RUNNING. Cross_model_validation_outlier_migrate Nemotron-3 partial run is in flight. Human is unreachable for 15-20 hours. Portfolio is now a 2-paper bet: OutlierMigrate (positive-method, awaiting Phase 2) and ThoughtFlow-FP8 (falsification, paper-ready). Five branches killed cleanly via preregistration gates. One branch (Decode Microkernel) infra-blocked, deferred. This window supersedes the 10-hour authorized window from May 8.

### Authorized work — priority order

1. **Drive cross_model_validation_outlier_migrate Nemotron-3 partial run to completion**:
   - Let runner finish all decode positions
   - Run checker
   - Apply PASS/KILL/AMBIGUOUS decision per the existing preregistration; do not invent new decision rules
   - Commit and push the result packet, checker output, and decision string
   - Update swarm/state.json with completion

2. **Branch on the Phase 2 outcome** (handle all four outcomes without human input):

   **2a. If Phase 2 PASSES**:
   - Spawn paper iteration loop on OutlierMigrate
   - Update related work, results, discussion sections with Nemotron-3 numbers
   - Add the QMamba (arXiv 2501.13624), OuroMamba (arXiv 2503.10959), MambaQuant (arXiv 2501.13484) citations and reframing per the previous window's instructions if not already present
   - Write cross-validation section honestly: Nemotron-3 confirms; Qwen3.6 and Kimi Linear deferred pending vLLM upgrade
   - Run 3 committee review rounds with the standard 3-member committee (COLM area chair, MLSys reviewer, adversarial reviewer)
   - Iterate up to 3 review->fix cycles per round
   - Mark as camera-ready-candidate (NOT camera-ready-final) when all committee scores >= 7/10

   **2b. If Phase 2 KILLS** (Nemotron-3 shows <5% migration):
   - Write experimental/outlier_migrate/phase2/diagnostic.md analyzing why
   - Specifically address: (i) is this a Granite-family architectural property rather than a hybrid-reasoner property, (ii) is the Nemotron-3 architecture sufficiently different (Mamba-2 + 6 GQA + 128-expert MoE vs Granite's 9:1 Mamba-2:GQA) to explain the divergence, (iii) what extended within-Granite work could recover scientific value
   - Do NOT claim cross-architectural validation in any paper update
   - Demote OutlierMigrate paper framing from "hybrid reasoner property" to "Granite-family characterization with cross-architectural validation as future work"
   - Update limitations section to be explicit about scope
   - Run committee review on the demoted draft
   - Submit-track decision is "Granite-family characterization paper" — still publishable but more modest

   **2c. If Phase 2 is AMBIGUOUS** (1-5% migration on Nemotron-3):
   - This is preregistered as a kill of cross-validation. Apply the kill decision.
   - Same actions as 2b but with explicit acknowledgment that the result is ambiguous, not negative
   - Frame paper as "strong on Granite family, intermediate on Nemotron-3, deferred on Qwen3.6/Kimi"

   **2d. If Phase 2 hits infrastructure failure**:
   - Apply standard infra-fail handling (max 3 retries with config changes, no vLLM upgrade)
   - If unrecoverable, document in swarm/blocked_phase2_nemotron_infra.md
   - Continue to authorized work item 3 with OutlierMigrate as Granite-only

3. **Within-set vs set-membership migration disambiguation**:
   On the existing Phase 0 and Phase 1 result packets (data already collected; no new experiments), produce a post-hoc analysis distinguishing:
   - (a) fraction of top-1% channels at position 100 that have LEFT the top-1% set entirely by position 10000/20000 (set-membership migration)
   - (b) fraction of top-1% channels that remained IN the set but moved by more than 2 positions in rank (within-set rank shuffling)
   This is data-only post-processing, no new model runs. Output: experimental/outlier_migrate/phase{0,1}/results/<run_id>/migration_decomposition.md
   Update OutlierMigrate paper to report both numbers separately. Reviewer-anticipated objection neutralized.

4. **OutlierMigrate related-work modernization**:
   Ensure the OutlierMigrate paper draft cites the following with correct framing (do NOT claim "first to find dynamic outliers in Mamba" — vision Mamba literature established this):
   - LLM.int8 (arXiv 2208.07339) - origin of static channel outlier protection
   - SmoothQuant (arXiv 2211.10438) - canonical static-protection method
   - AWQ (arXiv 2306.00978) - 1% salient channel protection paradigm
   - Mamba-PTQ (arXiv 2407.12397) - first Mamba LLM PTQ characterization
   - Quamba (arXiv 2410.13229) - language Mamba PTQ baseline
   - MambaQuant (arXiv 2501.13484) - PScan amplification mechanism
   - QMamba (arXiv 2501.13624) - first vision Mamba dynamic outlier identification
   - OuroMamba (arXiv 2503.10959) - vision Mamba adaptive online outlier list
   - Kimi Linear (arXiv 2510.26692) - architectural mechanism for migration via channel-wise gating
   Frame contribution as: vision Mamba literature established dynamics, language Mamba literature assumed static, we test which regime applies to language hybrid reasoners and find dynamic at quantified rates.

5. **ThoughtFlow paper polish**:
   Run additional committee review rounds if scores not yet >=7/10 across all three reviewers. Iterate. Do NOT mark camera-ready-final (human must do final framing review on landing). The paper is currently at ~90-93% readiness; aim for 95%+ via committee polish.

6. **Kill manifest completion**:
   For each of the 5 killed branches (HybridKernel, Residual Migration, SSM-State Lifecycle, SSM Shape Codec, Cross-Layer Error), verify the KILLED_*/README.md exists and contains:
   - Decision string
   - Measured value vs preregistered threshold
   - Artifact SHAs from the result packet
   - Date of kill
   - One paragraph explaining why the result is not publishable as a negative result (this is for the human's reference, not for a paper)
   Any missing manifest fields, fill in from the result packets. Documentation only, zero scientific judgment required.

7. **HANDOFF.md status table refresh**:
   Update experimental/HANDOFF.md status table to reflect current portfolio reality (5 kills, 1 infra-block, 2 alive, 1 in-flight). Status updates only, no operating-rule changes.

8. **Draft swarm/final_report.md**:
   Pre-draft the final report so the human can read it as the executive summary on landing. Include:
   - Portfolio status: alive papers, killed branches with kill manifests linked, infra-blocked branches
   - Phase 2 outcome (whichever of 2a/2b/2c/2d landed) with measured numbers
   - Camera-ready-candidate status of each alive paper
   - Committee review scores
   - GPU hours used and unused
   - Workshop submission readiness assessment (1 paper / 2 papers / partial)
   - Explicit "human must decide on landing" list with prioritized actions
   - The exact commit SHAs for each camera-ready-candidate
   - Reproducibility statement scaffold for each alive paper

9. **Presubmission checklist drafting** (paper-iteration adjacent, no scientific decisions):
   For each alive paper, create paper/presubmission_checklist.md with:
   - Anonymization audit (author names, GitHub URLs, acknowledgments)
   - Page limit check (vs likely COLM workshop limit of 4-8 pages)
   - BibTeX entry verification (every cited paper exists, abstract roughly matches what we claim)
   - Reproducibility Statement exact wording
   - Limitations section completeness check
   - Ethics Statement applicability check
   - Tables: booktabs format, no raster screenshots
   - Figures: PDF format, not raster
   These are checklists, not changes. Human applies the checklist on landing.

### Forbidden during this window — hard rules

- Upgrading vLLM (would break validated torch/triton/CUDA stack)
- Downloading Qwen3.6 or Kimi Linear weights (cannot be loaded with current vLLM)
- Marking any paper "camera-ready final" (candidate is the ceiling without human review)
- Modifying any preregistration file (drift policy unchanged)
- Authoring new pivot preregistrations from any of the 5 kills (the portfolio has absorbed enough pivot complexity; further pivots in this window are forbidden)
- Acting on Decode Microkernel beyond the existing infra-block status
- Making decisions about HybridFPGA collaboration or any Tambe lab outreach
- Adjusting OutlierMigrate's preregistered thresholds, decision rules, position grids, or metrics in response to observed data
- Re-running any killed Phase 0/1 with different seeds or configurations
- Cherry-picking Phase 2 result subsets if Nemotron-3 produces partial pass-partial-fail across decode positions

### Window-specific stop rules — additive to existing

- gpu_hours_used would exceed 60 cumulative within this window (preserve reserve for after human returns and for cross-validation completion if/when Qwen3.6 becomes runnable)
- Any push fails twice in a row
- Disk free on /workspace drops below 50GB
- An audit subagent fires for any reason
- A committee review identifies p-hacking, scope creep, or methodology drift
- Any subagent proposes to amend OutlierMigrate preregistration

If any stop condition fires: write swarm/blocked_<reason>_<date>.md, transition state.json to BLOCKED, push, pause until human resumes.

### Reporting cadence

Write swarm/progress_<date>_<hour>.md every 2 hours. Each progress note must include:
- Current entry / what completed since last note
- Phase 2 status (most recent decode position reached, ETA to completion)
- Cumulative gpu_hours_used (delta since last note)
- Camera-ready-candidate status of each alive paper
- Any concerns or near-stop-condition events
- Next 2-hour expected work
- Disk free, GPU temperature, GPU utilization

Write swarm/final_report.md at the end of the window OR when goal.md achieved condition is met OR when any stop rule fires.

## Phase 4 window — always-available fallback work queue

If Phase 4 main completes before the 10-hour window and either (a) passed with conditional follow-ups running, or (b) killed/ambiguous with no follow-ups, Codex MUST continue working through the following fallback queue in priority order rather than idling. Each item is bounded, safe, and paper-strengthening. None require GPU; all are LLM-time.

PRIORITY ORDER

1. **Citation verification pass**: For every BibTeX entry cited in outlier_migrate_colm2026.tex and thoughtflow_fp8_colm2026.tex, web-search the title, verify the paper exists, the authors match, the year matches, and the abstract roughly matches what the paper claims about the work. Build experimental/<paper>/paper/citation_audit.md with a row per citation showing (cited claim, verified abstract excerpt, match Y/N). Any mismatch is a desk-reject risk; surface to human in final_report.md.

2. **Anonymization audit**: For each alive paper, scan all .tex files and figures for author names, GitHub URLs, acknowledgments referencing institutions, and any identifying information. Build experimental/<paper>/paper/anonymization_audit.md with findings. Replace any identifiable references with anonymous placeholders.

3. **Reproducibility statement audit**: For each alive paper, verify that every claim referencing a result packet path actually resolves to that path, that every cited artifact_check.json file exists, and that the commit SHAs named in the reproducibility section actually exist in git history. Build experimental/<paper>/paper/repro_audit.md.

4. **Limitations section completeness audit**: For each alive paper, list every reviewer-anticipated objection from the latest committee reviews. Check whether each is addressed in the limitations section. Where gaps exist, draft proposed limitations-section text additions (do NOT apply them; collect them in a draft document for human review on landing).

5. **Presubmission checklist authoring**: For each alive paper, create paper/presubmission_checklist.md covering: page limit (verify against COLM workshop limit when known), table format (booktabs check), figure format (PDF not raster check), BibTeX completeness, ethics statement applicability, broader impact statement requirement check (workshop-dependent), conflict-of-interest declarations.

6. **Ablation summary table consolidation**: For OutlierMigrate, consolidate all ablation results (Phase 3 grid sensitivity, Phase 3 controls, Phase 4 grid sensitivity if completed, Phase 4 controls if completed) into a single comparison table for the paper's main results section. This is post-hoc presentation, not new analysis.

7. **ThoughtFlow committee fifth-round polish**: Run a fifth committee round on ThoughtFlow targeting scores >=8/10 across all reviewers. Address wording-only feedback. Framing changes still forbidden.

8. **Cross-phase comparison narrative for OutlierMigrate paper**: If Phase 4 ran, draft the Phase 3 vs Phase 4 comparison narrative for the discussion section. Include: methodological lesson learned, what changed and why, what the comparison tells the field about quantization regime sensitivity.

9. **Future work section drafting**: For OutlierMigrate, draft the future-work section explicitly covering: Qwen3.6 and Kimi Linear cross-validation (pending vLLM compatibility resolution), within-set rank shuffling adaptive interventions, full AWQ-style activation-aware integration, hardware co-design implications for migration-aware static protection.

10. **Repository hygiene**: Run a final pass deleting any temporary files, intermediate cache files, or orphaned artifacts in experimental/outlier_migrate/phase4/results/ that are not referenced by metrics.json or other manifest files. Do NOT delete result packets or any artifact referenced by the paper.

FALLBACK QUEUE RULES

- Each item must complete before the next begins.
- Each item commits and pushes its outputs.
- Items must be done thoroughly, not rushed. A half-finished citation audit is worse than no citation audit.
- The fallback queue continues until the 10-hour window expires OR cumulative gpu_hours hits the 60-hour cap.
- If Codex completes all 10 items before the window expires, it begins a second round of citation verification (different sample of citations, deeper check on the abstract-matching part) and then re-runs the anonymization audit. Repetition is acceptable as long as each pass is genuinely thorough.
- Codex must NOT use the fallback queue to invent new experiments, propose new pivots, or modify any preregistration.
- Codex must NOT mark camera-ready-final during fallback work. Camera-ready-candidate marks are permitted only if committee scores meet the >=7/10 threshold per the goal.

When the window approaches its end (Codex must track wall-clock from the time of Phase 4 prompt acceptance), write a final summary to swarm/window_end_report.md capturing: which fallback items completed, what their outputs say, what humans must review on landing.

## Phase 5'/6 sprint authorization — chained after Phase 4

This section authorizes a three-experiment sprint that begins ONLY after Phase 4 has landed (PASS, KILL, AMBIGUOUS, or FAIL_INFRA). Do NOT begin any item in this section while Phase 4 is still executing. The goal is to produce a COLM 2026 workshop submission with FOUR contributions: decomposition, layer-uniformity, cross-architectural measurement, and Rank-Shuffling-Aware Periodic Refresh (RSPR) as a positive method.

DROP Phase 5 (calibration aging on Granite-Small) in its current form. It is substantially scooped by PM-KVQ (arXiv 2505.18610, May 2025) for RoPE-driven KV cache drift and MaCa (arXiv 2602.07465, February 2026) for sequence-length-driven weight Hessian sensitivity. If a Phase 5 calibration-aging preregistration draft exists, mark it SUPERSEDED in its file header and do NOT delete it (preserve scoop-audit evidence).

EXECUTION GATE

Before beginning any work in this section, Codex must verify:
1. swarm/state.json shows Phase 4 status is one of {PASS_OM_PHASE4_MIGRATION_AWARE_RECOVERS, KILL_OM_PHASE4_INTERVENTION_FAILS, KILL_OM_PHASE4_AMBIGUOUS, FAIL_INFRA_OM_PHASE4}
2. Phase 4 final commit is pushed to origin/main
3. swarm/final_report.md has been updated with the Phase 4 outcome

If any of the above is unmet, wait. Do not proceed.

EXPERIMENT 5' (FIVE-PRIME): Pure-Transformer reasoning control

Purpose: Either broaden the migration finding to all reasoning LLMs (if pure Transformers also migrate ~80%) OR sharpen the Mamba-2-hybrid-specificity claim (if pure Transformers migrate substantially less).

Author preregistration: experimental/outlier_migrate/phase5_prime/preregister_om_phase5_prime_transformer_control.md. Pivot depth: 1 from OutlierMigrate Phase 0 (parallel pivot, not child of Phase 3 or Phase 4).

Hypothesis: Same migration metric as Phase 0/1/2 measured on a pure-Transformer reasoning model with RoPE and no Mamba components.

Primary model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B at HuggingFace snapshot commit recorded in model_provenance.json before any inference. Fallback: Qwen/Qwen2.5-1.5B-Instruct if R1-Distill has vLLM 0.10.2 compatibility issues. Codex must test load both models before committing the prereg and record which is being used.

Trace set: 24 AIME-2025 traces, indices 0-23, prompt SHA matching Phase 1 (sha256:aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e).

Decode position grid: {100, 500, 1000, 5000, 10000, 20000}. Identical to Phase 1 for direct comparability.

Migration metric and bootstrap procedure: identical to Phase 0/1/2. Bootstrap seed: 20260511.

Decision rule (preregistered, non-directional):
- DYNAMIC_REGIME_TRANSFORMER: migration > 0.05 AND CI lower > 0.05. Implication: finding broadens to all reasoning LLMs.
- STATIC_REGIME_TRANSFORMER: migration < 0.05 AND CI upper < 0.05. Implication: finding is hybrid-specific.
- AMBIGUOUS_TRANSFORMER: in-between or wide CI. Implication: report honestly, no strong claim.
- FAIL_INFRA_TRANSFORMER: infrastructure issue.

Forbidden: choosing the decision rule directionality post-hoc.

Estimated 3-4 GPU hours.

EXPERIMENT D (DECOMPOSITION): No-GPU analytical cleanup

Purpose: Formalize the set-leaving vs rank-shuffling decomposition with publication-ready statistics.

Inputs: existing Phase 0/1/2 packets plus Phase 5' packet if landed.

Outputs (write to experimental/outlier_migrate/decomposition_analysis/):
1. kendall_tau_by_position.json: Kendall's tau rank correlation between position 100 and each scoring position, per layer, per trace, with bootstrap CIs
2. component_decomposition.md: formal decomposition table per phase per model with explicit definitions, CI for each component, layer-type breakdown
3. cross_tabulation.json: layer-type x decomposition-component table (does set-leaving concentrate in any layer type? does rank-shuffling concentrate?)
4. trace_difficulty_regression.json: regression of migration vs trace difficulty (AIME problem difficulty, trace length, BF16 perplexity at scoring position)

This is pure post-hoc analysis. Estimated 1-2 hours LLM time.

EXPERIMENT 6: Rank-Shuffling-Aware Periodic Refresh (RSPR) - POSITIVE METHOD

Purpose: Produce the COLM submission's positive method contribution. RSPR addresses both decomposition components: set-leaving via union-set protection AND rank-shuffling via periodic scale refresh.

Author preregistration: experimental/outlier_migrate/phase6/preregister_om_phase6_rspr.md. Pivot depth: 2 from OutlierMigrate Phase 0.

EXECUTION CONDITION: Run RSPR ONLY IF Phase 4 established a measurable static-protection gap (fewer than 25% of Phase 4 traces had no_recoverable_static_gap, and Phase 4 was not killed for measurement-design reasons). If Phase 4 also had the Granite-tiny-style gap-detectability issue, escalate to human and DO NOT run RSPR. Document the escalation in swarm/blocked_phase6_testbed_selection.md.

Method definition:
- Protected channel set: union of top-1% channels across decode positions {100, 1000, 5000, 10000}, identical to Phase 4's union set
- Refresh policy: every K decode positions, recompute quantization scales for the protected channel set using activation magnitudes from the most recent K tokens
- Quantization: W4A16 symmetric INT4 weights, FP16 activations, BF16 protected channels

Five regimes evaluated head-to-head:
1. BF16 baseline (oracle ceiling)
2. Static-1% at position 100 (standard baseline)
3. Static union {100, 1000, 5000, 10000} (Phase 4's method)
4. RSPR K=2000 (proposed method)
5. RSPR K=1000 (sensitivity ablation)

Mandatory controls:
6. Static-2% at position 100 matched-budget control
7. RSPR with random channel set as refresh target (negative control)

Decision rule:
- PASS_OM_PHASE6_RSPR_BEATS_BOTH_BASELINES: median RSPR (K=2000) recovery >= 0.50 with CI lower > 0.30 AND RSPR median recovery exceeds static-union median recovery by at least 0.15 AND RSPR median recovery exceeds static-1% baseline median recovery by at least 0.25.
- KILL_OM_PHASE6_RSPR_NO_IMPROVEMENT: RSPR (K=2000) median recovery within 0.05 of static-union median recovery.
- KILL_OM_PHASE6_RSPR_AMBIGUOUS: middle outcomes with overlapping CIs.
- FAIL_INFRA_OM_PHASE6: infrastructure issue.
- KILL_OM_PHASE6_RANDOM_CONTROL_BEATS: if random-channel refresh control (regime 7) outperforms RSPR (regime 4) by more than 0.10 median recovery, the refresh mechanism is artifactual; stop and surface to human.

Estimated 8-12 GPU hours.

Forbidden:
- Adjusting K post-hoc
- Selecting which scoring position to report post-hoc
- Skipping the random-channel negative control
- Running RSPR if Phase 4 had the gap-detectability issue
- Modifying any prior preregistration

PAPER INTEGRATION

After all three experiments land, integrate into experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex:
- Restructure as 4-contribution positive-method paper
- Section 5: decomposition contribution (Experiment D plus Phase 0/1/2 data)
- Section 6: layer-uniformity contribution
- Section 7: cross-architectural contribution (Phase 0/1/2 plus Phase 5' transformer result, and Phase 5'' Qwen3.6 result if landed)
- Section 8: RSPR positive method (Experiment 6)
- Section 9: reconciliation with DecDEC, PM-KVQ, MaCa, KL Lens, Quamba2, Nemotron-3, OuroMamba

Run 3 committee review rounds targeting >= 7/10 across all reviewers.

MANDATORY CITATIONS

- DecDEC (Park et al., OSDI 2025, arXiv 2412.20185): motivating prior art for migration measurement
- PM-KVQ (Liu et al., arXiv 2505.18610, May 2025): RoPE-driven KV cache calibration drift, differentiated by mechanism
- MaCa (Son et al., arXiv 2602.07465, February 2026): multi-scale calibration in pure Transformers
- KL Lens (Kong et al., arXiv 2604.13440, April 2026): hybrid layer-stratified sensitivity, differentiated from migration
- Quamba2 (Chiang et al., ICML 2025, arXiv 2503.22879): SSM channel-order preservation at calibration time vs decode-time rank migration
- Nemotron-3 white paper (NVIDIA, arXiv 2512.20848, December 2025): empirical sensitivity ordering
- OuroMamba (Ramachandran et al., ICCV 2025, arXiv 2503.10959): vision-Mamba adaptive online list

FORBIDDEN CLAIMS

- "First to find dynamic outliers in language LLMs" (DecDEC has priority)
- "First to observe calibration aging" (PM-KVQ/MaCa have priority)
- "First layer-stratified analysis on hybrid Mamba-Transformer" without specifying "migration rate"

ALLOWED CLAIMS

- "First decomposition of migration into set-leaving and rank-shuffling components"
- "First measurement of migration-rate uniformity across attention/SSM/MoE layer types in hybrid LLMs"
- "First positive method using decomposition-justified periodic refresh"

STOP CONDITIONS

- gpu_hours_used cumulative exceeds 80
- Push fails twice in a row
- Audit fires
- Three consecutive infra failures
- Random-channel negative control beats RSPR by > 0.10
- Any subagent proposes additional pivots beyond Phase 5'/6/5''

REPORTING

Continue 2-hour swarm/progress_*.md cadence. Update swarm/final_report.md after each experiment lands.

EXECUTION ORDER

1. Verify Phase 4 has landed per the EXECUTION GATE.
2. Author Phase 5' preregistration. Commit and push.
3. Run Phase 5' on pure-Transformer model. Apply checker. Commit and push.
4. Run Experiment D (decomposition analysis, no GPU). Commit and push.
5. Author Phase 6 preregistration. Commit and push.
6. Run Phase 6 RSPR only if Phase 4 had measurable gap. Apply checker. Commit and push.
7. Integrate all four contributions into paper.
8. Run 3 committee review rounds.
9. Update swarm/final_report.md.

Begin item 1 (verification) only after Phase 4 lands.

## Architectural scope clarification (May 2026 model landscape audit)

Three architectural lineages dominate the open-weight reasoning LLM space:

LINEAGE 1 - Pure-Transformer reasoning with novel attention compression
- DeepSeek-V4-Pro/Flash (April 23, 2026, 1.6T/284B, MIT, CSA+HCA hybrid attention)
- Kimi K2.6 (April 20, 2026, 1T/32B active, Modified MIT, MLA attention, same arch as K2.5)
- GLM-5/5.1 (April 7, 2026, MIT)
Status: OUT OF SCOPE for our paper. Wrong architectural lineage (no SSM, no Gated Linear Attention). Cite in related work as concurrent pure-Transformer reasoning releases. NEVER measure.

LINEAGE 2 - Mamba-2 hybrid + MoE (OUR PRIMARY MEASUREMENT SET)
- Granite-4.0-H-Tiny (Phase 0)
- Granite-4.0-H-Small (Phase 1, Phase 4)
- Nemotron-3-Nano-30B-A3B (Phase 2)
- Falcon-H1 family (deferred to ICLR archival)
Status: PRIMARY. All workshop submissions use these models.

LINEAGE 3 - Gated DeltaNet hybrid + MoE (Phase 5'' CROSS-LINEAGE TARGET)
- Qwen3.5-{9B, 35B-A3B, 122B-A10B} (released Feb-Mar 2026)
- Qwen3.6-27B dense (April 22, 2026)
- Qwen3.6-35B-A3B (April 16, 2026) - Phase 5'' primary target
- Kimi Linear (October 2025, channel-wise gated KDA, distinct from K2.5/K2.6)
Status: ICLR ARCHIVAL TARGETS. Qwen3.6-35B-A3B is the stretch experiment for COLM workshop (Phase 5'').

IMPORTANT: Kimi K2.6 (April 2026, 1T/32B MLA) is LINEAGE 1, not LINEAGE 3. The October 2025 Kimi Linear release is the LINEAGE 3 model, distinct from K2.5/K2.6. Always verify architecture from config.json or technical report, not the model name.

Paper framing: use "hybrid Mamba-2 reasoning LLMs (Granite-4 and Nemotron-3 families)" rather than "hybrid Mamba-Transformer reasoning LLMs". If Phase 5'' lands, broaden to "hybrid Mamba-2 and Gated DeltaNet reasoning LLMs."

## Phase 5'' (FIVE-DOUBLE-PRIME): Qwen3.6 cross-lineage validation

PURPOSE
Validate the Phase 0/1/2 migration measurement on a Gated DeltaNet hybrid model (Lineage 3), establishing cross-architectural generality of the migration finding beyond the Mamba-2 hybrid family (Lineage 2).

PRIORITY
OPTIONAL stretch experiment for the COLM workshop submission. Time-boxed. If Phase 5'' lands by June 6, 2026, integrate into workshop paper. If not, the workshop paper falls back to Mamba-2-specific framing. Do NOT delay Phase 4/5'/6 sprint for Phase 5''.

EXECUTION GATE
Before beginning Phase 5'', Codex must verify:
1. Phase 4, Phase 5', Decomposition Analysis, and Phase 6 (if applicable) have all landed
2. Cumulative GPU hours used is below 70 (cap is 80)
3. Wall-clock date is on or before June 1, 2026
4. The SGLang venv at /workspace/.sglang exists and passed the smoke test (see infrastructure below)

If any condition is unmet, skip Phase 5'' and proceed to paper integration with the Mamba-2-specific framing.

INFRASTRUCTURE - venv approach on existing pod

Phase 5'' uses a SEPARATE Python venv at /workspace/.sglang on the EXISTING pod. The validated vLLM environment (Phase 4/5'/6) stays untouched in its original location. PyTorch 2.9.1+cu128 in the new venv works with the host CUDA 12.8 driver via PyTorch's bundled CUDA runtime.

CRITICAL HYGIENE: NEVER activate both venvs in the same shell session. Phase 4/5'/6 work uses the original environment (vLLM 0.10.2, torch 2.8, triton 3.4). Phase 5'' work uses /workspace/.sglang (SGLang 0.5.9, torch 2.9.1, triton 3.5.1). Use separate tmux panes.

The /workspace/.sglang venv must be created by a human (not Codex) before Phase 5'' execution. Codex's role for Phase 5'' is limited to:
1. Verifying the venv exists and the smoke test passed
2. Authoring the preregistration
3. Running the migration measurement script when activated in the .sglang venv via a script in /workspace/LatentWire that explicitly sources the venv

PREREGISTRATION
Author at experimental/outlier_migrate/phase5_double_prime/preregister_om_phase5dp_qwen36.md before any inference runs. Pivot depth: 1 from OutlierMigrate Phase 0 (parallel pivot to Phase 5'). Decision rules identical to Phase 5' (non-directional: DYNAMIC, STATIC, AMBIGUOUS, FAIL_INFRA).

HOOK ADAPTATION FOR GATED DELTANET
Qwen3.6 layer pattern: 10 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Gated Attention -> MoE))
The migration measurement reads residual-stream output at each layer block, which is architecture-independent. Hook target is the OUTPUT of each block (after MoE), not internal projection layers.

For comparability with Phase 0/1/2:
- Phase 5'' hooks Gated DeltaNet output (analog of SSM output in Lineage 2)
- Phase 5'' hooks Gated Attention output (analog of Attention output)
- Phase 5'' hooks MoE output (same as Lineage 2)

Layer-stratified analysis groups: Gated DeltaNet layers, Gated Attention layers, MoE layers.

INTEGRATION RULES
If Phase 5'' produces measurable migration consistent with Phase 0/1/2 (~80%): workshop paper's layer-uniformity contribution generalizes from "uniform across attention/SSM/MoE in Mamba-2 hybrids" to "uniform across attention, efficient-sequence-mixer, and MoE in hybrid reasoning LLMs spanning Mamba-2 and Gated DeltaNet families."

If Phase 5'' produces measurably DIFFERENT migration on Qwen3.6: paper keeps the Mamba-2-specific claim and adds Qwen3.6 as a counter-example with mechanism speculation in discussion.

If Phase 5'' fails infrastructure: paper falls back to Mamba-2-specific framing without Qwen3.6 mention beyond related work.

COST BUDGET
- Incremental GPU: ~3-4 hours = ~$8-10
- Incremental engineering: ~2-3 hours of LLM work
- Total incremental cost: ~$15-25

FORBIDDEN
- Modifying the original pod's vLLM/torch/triton versions
- Codex creating or modifying the /workspace/.sglang venv (humans only)
- Including Kimi K2.6 in Phase 5'' (Lineage 1)
- Including DeepSeek-V4 in Phase 5'' (Lineage 1)
- Including Falcon-H1 in Phase 5'' (same lineage as existing measurements)

If multiple Lineage 3 models can be measured within budget, prefer Qwen3.6-35B-A3B (MoE) over Qwen3.6-27B-dense. Kimi Linear is acceptable as a second Lineage 3 data point only if Qwen3.6-35B-A3B has landed and time remains.

