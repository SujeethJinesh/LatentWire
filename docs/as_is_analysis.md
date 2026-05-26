# As-Is Analysis: Archived v2 OutlierMigrate Paper

Source analyzed: `paper_archive_20260526T022823Z/experimental_outlier_migrate_paper/outlier_migrate_colm2026.tex` and the archived `figures/` directory.

This is an as-is structural diagnosis only. It does not propose a rewrite.

## Current Readiness, Story, And Blocking Gap

- Current paper readiness status: workshop-polished measurement/mechanism draft, but not ICLR-ready as a robust positive-method paper.
- Current story of the paper: long-decode W4A16 reasoning makes static top-1% activation-channel protection brittle; channel membership drifts across four measured reasoning models; several channel-set interventions fail controls; budget-tuned EMA partially helps, especially on Nemotron top-10; ParoQuant shows the endpoint is not intrinsically impossible for W4A16 PTQ.
- Exact gap blocking ICLR submission: the paper does not yet establish a robust, model-invariant, deployable positive method. The best channel-set method is model-dependent, Granite has wide uncertainty, ParoQuant is stronger on Granite, and the release path is FAST_VERIFY rather than a full public GPU rerun.

## Current Title

Current title:

> Channel-Set Drift in Long-Reasoning W4A16 LLMs: Mechanisms and a Budget-Tuned Remedy

What it foregrounds:

- The primary phenomenon is "Channel-Set Drift," not latent transfer or cross-model communication.
- The domain is long-reasoning W4A16 LLM inference.
- The title promises both mechanism analysis and a "Budget-Tuned Remedy."
- The paper is framed as a quantization/protected-channel paper with a partial remedy, not as a broad systems implementation or communication benchmark paper.

As-is tension: "Budget-Tuned Remedy" is more positive than the evidence structure that follows, where the strongest channel-set result is model-dependent and a rotation baseline is stronger on Granite.

## Current Abstract

Current abstract:

> Static per-channel protection is attractive for W4A16 reasoning inference, but it assumes that high-magnitude activation channels remain useful protection targets throughout decode. We measure decode-time channel-set drift across four reasoning LLMs and find strict top-1\% set-leaving of 53--67\% by horizons up to 20K tokens. We observe the effect at the measured block-output surface in hybrid Mamba-2, parallel-hybrid, and pure-Transformer models, extending DecDEC's short-horizon Transformer observation and showing that Quamba2's channel-order and activation-persistence observations do not imply stable top-$K$ block-output protection at this measured surface. We then test eight channel-set protection families with matched-cost and random controls. Hard position or scale switches are actively harmful, top-1\% smoothing and cross-tensor coupling have signal but insufficient recovery, and reactive DecDEC-style selection does not recover the W4A16 gap. Budget-tuned EMA protection is the strongest channel-set remedy: on Nemotron-3, top-10 protection recovers 0.815 of the BF16-vs-static gap and beats static top-10 by 0.220; on Granite-Small, top-5 recovers 0.449 with a wide CI. A ParoQuant-style rotation baseline recovers 0.754 on Granite-Small, scoping our negative results to channel-set protection rather than W4A16 PTQ generally. KL trajectories grow sublinearly with AR decays 0.44--0.51, weakening compound-error explanations in the measured regime.

What the abstract leads with:

- It opens with the static per-channel protection assumption and immediately defines the failure mode.
- The first numerical claim is the four-model drift result: 53-67% strict top-1% set-leaving by up to 20K tokens.
- The next move is prior-art differentiation against DecDEC and Quamba2.
- The method story comes after measurement: eight channel-set families, controls, and several killed mechanisms.
- The positive result is framed as "strongest channel-set remedy," but is scoped by model dependence and by ParoQuant outperforming Granite M11b.
- The abstract ends with mechanism interpretation: KL trajectories weaken runaway compound-error explanations.

## Current Section Structure

1. Front matter: COLM article setup, anonymous author block, title, abstract.
2. Introduction:
   - Motivates long-reasoning decode shift.
   - Defines the W4A16 static protected-channel assumption.
   - Positions against Quamba2 and cost pressure.
   - Lists four contributions.
3. Related Work:
   - Mamba and state-space quantization.
   - LLM quantization and rotations.
   - Dynamic and long-context policies.
   - Error accumulation.
4. Method:
   - Defines strict top-1% set-leaving.
   - Defines recovery of the BF16-vs-static W4A16 gap.
   - Lists eight channel-set method classes.
   - Separates ParoQuant as a rotation baseline, not a channel-set method.
5. Experiments:
   - Models and traces.
   - Controls.
6. Results:
   - Channel sets drift at long reasoning horizons.
   - Layer type does not explain the drift away.
   - Top-1% channel-set interventions fail despite control signal.
   - Budget matters, but is not model-invariant.
   - KL does not support runaway compounding.
7. Discussion:
   - Organizes the results into three mechanisms: harmful discontinuities, insufficient top-1% smoothing, and binding budget.
   - Discusses trace heterogeneity and no-gap traces.
   - Names composition of rotations plus budget-aware protection as the natural next step in the current paper's own framing.
8. Limitations:
   - Scopes to W4A16 PTQ, small reasoning models, and block-output activation channels.
   - Notes Quamba2 comparison is not internal-state exhaustive.
   - Notes only Granite has dense FFT/KL diagnostics.
   - Notes M11b is cross-model but not model-invariant.
9. Conclusion:
   - Restates static channel protection failure under drift.
   - Restates Nemotron top-10 success, Granite uncertainty, and rotations as a stronger path.
10. Bibliography.
11. Ethics Statement.
12. Reproducibility Statement.
13. LLM Use Disclosure.
14. Appendix:
   - Detailed Experimental Provenance.
   - Per-Method Detailed Results.
   - Extended Layer-Stratified Results.
   - Hallucination and Source Audit.

## Current Figure List

The current archived TeX includes four external PDF figures. All four are one-page PDF assets in `paper_archive_20260526T022823Z/experimental_outlier_migrate_paper/figures/`.

1. `fig:set-leaving` - `figures/set_leaving_decode_positions.pdf`
   - Caption claim: strict top-1% channel-set leaving relative to decode position 100; static protection loses 53-67% of its original protected set by the final measured position across four reasoning models.
   - Visual role: headline measurement figure. It plots four model curves over log-scaled decode positions.
   - Source manifest: activation magnitude packets from Granite-Small, Nemotron-3, DeepSeek-R1-Distill-Qwen-1.5B, and Falcon-H1.

2. `fig:components` - `figures/per_component_drift.pdf`
   - Caption claim: attention, SSM/Mamba, and MoE-classified measured block outputs show comparable strict set-leaving.
   - Visual role: component/localization figure. It argues no single layer type explains the drift.
   - Source manifest: per-component dissection and layer-stratified migration artifacts.

3. `fig:method-recovery` - `figures/method_recovery_comparison.pdf`
   - Caption claim: median recovery with 95% bootstrap CIs; long negative CI whiskers are clipped in the plot; the dashed line marks the 0.30 positive-method threshold.
   - Visual role: central method outcome figure. It shows mostly failed top-1% channel-set methods, ParoQuant's stronger Granite result, and Nemotron budget rows including static top-10 and M11b top-10.
   - Source manifest: Phase 9 method checker packets and the ParoQuant baseline packet.

4. `fig:kl` - `figures/kl_accumulation_trajectories.pdf`
   - Caption claim: per-position KL trajectories on Granite-Small have lowest residuals under sublinear square-root fits and moderate AR decay, not superlinear growth.
   - Visual role: mechanism/falsification figure for compound-error explanations.
   - Source manifest: dense Granite-Small KL summary and growth-model fit artifacts.

Figure-adjacent tables:

- Main results table `tab:methods`: intervention outcomes, including method family, model, median recovery, 95% CI, and key comparison.
- Reproducibility model-checkpoint table.
- Appendix provenance table.
- Appendix per-method detailed results table.
- Appendix extended layer-stratified results table.

Asset note: the archived folder also contains `figure_audit_20260523.md` and `figure_inventory_20260523.json`, but those older files describe a pre-polish state with no external `includegraphics`. The current TeX and `figure_source_manifest_20260526.json` are the accurate as-is source of truth for this archived v2.

## Current Contribution List

The introduction presents four contributions:

1. Empirical drift:
   - Measures decode-time channel-set drift across four models and architectures up to 20K tokens.
   - Reports 53-67% strict set-leaving across hybrid Mamba-2, parallel-hybrid, and pure-Transformer models.

2. Prior-art differentiation:
   - Extends DecDEC's short-horizon Transformer evidence to long reasoning.
   - Argues Quamba2-style protected-set persistence does not hold at the measured block-output surface for the hybrid models tested.

3. Mechanism tests:
   - Evaluates eight channel-set protection families with random and matched-cost controls.
   - Identifies harmful boundary discontinuities, insufficient smooth top-1% tracking, budget dependence, and lack of support for runaway KL compounding in the Granite packet.

4. Partial remedy and scope:
   - Claims budget-tuned EMA succeeds on Nemotron top-10 and partially recovers Granite top-5 with wide uncertainty.
   - Uses ParoQuant recovery on Granite to scope the negative result to channel-set protection rather than all W4A16 PTQ.

## Current Framing Of The Headline Result

The headline is framed as a measurement/mechanism result with one partial positive method, not as a general deployable recipe.

Primary headline:

- Static top-1% channel protection loses 53-67% of its original protected set by the final measured position across four reasoning models.

Secondary headline:

- The drift reaches measured block-output surfaces across attention, SSM/Mamba, and MoE-classified layers.

Method headline:

- Most channel-set methods fail or remain insufficient under controls.
- M11b budget-tuned EMA is the strongest channel-set remedy:
  - Nemotron top-10 median recovery is 0.815 with CI [0.254, 0.923] and 0.220 margin over static top-10.
  - Granite top-5 median recovery is 0.449 with CI [-1.30, 1.00], so it is positive in median but uncertain.
- ParoQuant rotation recovers 0.754 on Granite and is explicitly used to limit the claim: W4A16 PTQ is not intrinsically broken; channel-set protection is fragile.

Mechanism headline:

- Hard position or scale boundaries are actively harmful.
- Smooth top-1% tracking and activation+key coupling contain signal but do not recover enough quality at top-1% budget.
- Budget is binding but not model-invariant.
- KL trajectories weaken runaway compound-error explanations.

Reviewer-feedback context read for this audit:

- The latest archived committee review round says the stable workshop claim is: channel-set protection is fragile under long-decode W4A16 reasoning, budget tuning partially helps, and rotation is a stronger baseline mechanism.
- That review has no load-bearing or substantive restructuring critiques for the workshop target, but still notes small sample sizes, ParoQuant beating M11b, and model-dependent M11b budget as the strongest attacks.

## Structural Strengths

- The paper has a clean measurement-first sequence: drift measurement, component analysis, intervention failures, partial budget remedy, KL mechanism check.
- The title, abstract, and first result figure all foreground the same phenomenon: decode-time top-1% channel-set drift.
- The strongest claim is scoped and repeatedly bounded: the target is channel-set protection under W4A16 long reasoning, not all PTQ and not all internal SSM state tensors.
- Controls are structurally central. The paper does not merely report failed methods; it compares them with random or matched-cost controls and uses those comparisons to infer mechanisms.
- The ParoQuant baseline is structurally useful because it prevents the paper from overclaiming that W4A16 is intrinsically unrecoverable.
- The section order mirrors the evidence ladder: measurement, localization, intervention, budget, then mechanism falsification.
- The figures are aligned with that evidence ladder. Each figure has a distinct job and the main results table gives exact numbers for the clipped method-recovery figure.
- The limitations section is honest about scope: model size, surface measured, dense diagnostics only on Granite, and lack of model-invariant M11b.
- The reproducibility and LLM-use disclosures are explicit about FAST_VERIFY, archived experimental runners, and LLM-assisted drafting/auditing.

## Structural Weaknesses

- The title's "Budget-Tuned Remedy" phrase is more affirmative than the body structure, which presents M11b as partial, model-dependent, and weaker than ParoQuant on Granite.
- The manuscript is not aligned with the LatentWire positive cross-model communication goal. It is a W4A16 quantization mechanism paper, not a latent-transfer method paper.
- The positive-method evidence is structurally secondary. The paper spends substantial space on measurement and killed method classes before the partial remedy appears.
- The method taxonomy is dense. Readers must track static unions, M2, M10, M11, M18, DecDEC proxy, M11b, M26, static top-10, and ParoQuant without a schematic.
- The central recovery metric has edge cases that complicate the story: negative recovery, recovery above one, no-gap traces, legacy static-union handling, and positive-gap filtering. The paper explains these, but the burden is high.
- The strongest channel-set result is Nemotron top-10, while Granite's positive row has wide uncertainty and ParoQuant is stronger on Granite. This creates a structural tension between "channel-set remedy" and "rotation is stronger."
- The method-recovery figure clips long negative CIs, so the table is required to understand the full negative evidence. This is acceptable but increases dependence on cross-reading figure and table.
- The figure-generation path is partly manifest-backed and partly value-mirrored in the plotting script, rather than fully raw-artifact-driven. The current numbers check out, but the structure can go stale.
- The archived figure folder contains older audit files that contradict the current TeX figure state. The current paper is clear, but the surrounding archive metadata is noisy.
- The reproducibility statement is candid that the release package is FAST_VERIFY rather than a full public GPU rerun. For ICLR-level review, that is a structural weakness even if it is honestly disclosed.

## Saturated, Alive, And Highest-Priority Gap

- Saturated in the current structure: four-model strict set-leaving measurement, component-level block-output drift, failure of hard switching/top-1% channel-set families under controls, Granite dense KL non-compounding evidence, and ParoQuant as a scope baseline.
- Still alive in the current structure: budget-tuned EMA as a partial, model-dependent channel-set remedy; rotation-based conditioning as the stronger adjacent mechanism.
- Highest-priority gap: a robust positive method that survives larger frozen slices, seed repeats, paired uncertainty, and strict cross-family controls. This file does not propose how to rewrite or pursue that next gate.
