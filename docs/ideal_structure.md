# Ideal COLM Workshop Paper Structure

## Scope And Readiness

Current paper readiness: COLM workshop-scoped, not ICLR-ready as a positive-method paper. The current story is that decode-position channel-set drift is reproducible across several long-reasoning model families, while natural protected-channel remedies either fail, remain partial, or are currently outperformed by a rotation baseline. The exact submission-blocking gap for an ICLR positive-method paper is still a deployable method that survives larger frozen slices, seed repeats with paired uncertainty, and strict same-family/cross-family separation.

Evidence used here is limited to `experimental/outlier_migrate/phase*/results`, `swarm/external_collaboration_state.md`, `swarm/positive_method_ideation_2026_05_24.md`, `swarm/positive_method_ideation_2026_05_26.md`, and the two Phase 9 `final_decision.md` files. No current paper TeX/PDF or `paper_archive` material was consulted. No `stage-roadmap` decision documents were present in the checked workspace paths.

## Proposed Title

**Decode-Position Channel-Set Drift in Long-Reasoning W4A16 LLMs: Measurements, Failed Remedies, and Budget-Dependent Signals**

## Headline Framing

This should be framed as a measurement-and-mechanism workshop paper, not as a solved positive-method paper: static W4A16 protected-channel sets become stale during long decode traces; several intuitive repairs do not survive intervention gates; budgeted dynamic protection shows real but model-dependent signal; ParoQuant-style rotation is currently the strongest Granite baseline.

## Proposed Abstract

W4A16 inference schemes often protect high-magnitude activation channels selected near the start of generation, implicitly assuming that the important channel set remains stable through long reasoning. We audit that assumption on archived long-decode traces and find substantial decode-position channel-set drift: strict top-1% set-leaving is `0.566234756098` on Granite-Small, `0.533713200380` on Nemotron-3-Nano, `0.670572916667` on DeepSeek-R1-Distill-Qwen-1.5B, and `0.673611111111` on Falcon-H1, with layer-stratified analyses showing the effect is not isolated to one block type. We then evaluate a sequence of preregistered or decision-gated remedies. Static unions and hard switching fail, top-1% EMA is insufficient, DecDEC-style reactive top-1% selection is only an algorithmic baseline with negative median recovery, and activation+K coupling is diagnostically less harmful than controls but still fails as a positive method. The strongest channel-set clue is budget-dependent M11b: Granite top-5 median recovery is `0.449284091125` with wide uncertainty, while Nemotron top-10 median recovery is `0.814739798903`; however, ParoQuant on Granite reaches `0.753776848891`, so the evidence supports a scoped workshop claim about mechanisms and next method gates, not a deployable cross-model positive method.

## Contribution List

- A reproducible long-reasoning measurement of decode-position top-channel set drift across Granite, Nemotron, DeepSeek, and Falcon artifacts.
- A decomposition that separates strict set-leaving from within-set rank shuffling, making the systems relevance of static protected-channel failure clearer.
- A negative-results ladder showing which natural remedies are saturated or killed: static unions, hard position/scale switching, top-1% EMA alone, DecDEC-style top-1% reactive protection, and top-1% activation+K coupling.
- A conservative positive-signal analysis: budget matters for dynamic protection, but current evidence is model- and budget-dependent and must be compared against rotation baselines before claiming a method.

## Proposed Section Structure

### 1. Introduction

Open with the practical assumption behind W4A16 protected channels: the channels selected early in generation are treated as if they remain important later. State the workshop-scale finding directly: this assumption breaks in long reasoning, and the paper maps which remedy hypotheses fail or remain alive.

### 2. Problem Setup And Metric

Define decode-position channel-set drift, strict top-1% set-leaving, within-set rank shuffling, and positive-static-gap recovery. Emphasize why strict set-leaving is the load-bearing metric for static protected-channel policies, using the migration decomposition files in Phases 0, 1, 2, 5', and 7.

### 3. Evidence Corpus And Reproducibility Protocol

Describe the frozen artifact style: archived activation magnitudes, BF16 traces where available, prompt manifests, random seeds, checker outputs, bootstrap CIs, and final decisions. Keep the methods description audit-oriented and scoped to the result packets rather than presenting a new experiment.

### 4. Measurement Result: Channel Sets Drift During Long Reasoning

Present the four-model strict set-leaving result: Granite-Small `0.566234756098`, Nemotron `0.533713200380`, DeepSeek `0.670572916667`, and Falcon-H1 `0.673611111111`, with Granite-Tiny as the earlier smoke gate. This is the strongest main-body evidence and should carry the first result figure.

### 5. Where The Drift Lives

Use the layer-stratified Phase 3 analysis to show that drift is not isolated to one component class: Granite attention and SSM/Mamba outputs are close, and Nemotron attention, MoE, and SSM/Mamba rows all show substantial strict set-leaving. This section should also clarify the limitation that these are block-output tensors, not internal SSM state tensors.

### 6. Remedy Gates That Failed

Walk through the negative ladder in chronological mechanism order: Phase 3/4 static unions, M2 hard position switching, M10 hard scale bins, M11 top-1% EMA, DecDEC proxy, M18 activation+K, and M26 stable core. The tone should be falsification-oriented: these results rule out easy stories while preserving narrower diagnostic signals.

### 7. Budget-Dependent Signal And Baseline Pressure

Present M11b as the partial channel-set remedy: Granite top-5 median recovery `0.449284091125` with CI crossing zero, Nemotron top-5 positive but below static top-10, and Nemotron top-10 median recovery `0.814739798903`. Immediately compare ParoQuant Granite median recovery `0.753776848891`, making clear that rotation is currently the strongest Granite baseline and that budgeted channel-set protection is not yet a portable method.

### 8. Mechanism Synthesis

Summarize the evidence-backed mechanism claims: hard discontinuities are harmful, smoothing alone is insufficient, budget is load-bearing, stale selection is more supported than compound-error growth, and cross-tensor coupling contains signal but not enough recovery at top-1%. Use this section to connect the result ladder to the live next-method branches without overclaiming.

### 9. Limitations And Next Gates

State the workshop limitations plainly: small frozen intervention slices, incomplete deployability, no clean cross-model positive method, ParoQuant not fully displaced, and block-output rather than internal-state scope. The next exact gate should be rotation+budget or sensitivity-weighted budget composition with matched ParoQuant-only, static, random, and same-family/cross-family controls.

### 10. Artifact And Decision Ledger

Close with a compact reproducibility table mapping claims to run directories, checkers, metrics, and final decisions. This section should make the paper reusable for future ICLR work by preserving which hypotheses are killed, weakened, promoted, or still alive.

## Proposed Figure List

1. **Problem schematic: early protected channels become stale.** Diagram the position-100 static top-1% protected set and the later decode-position set-leaving metric; no new data required.
2. **Four-model strict set-leaving.** Bar plot/table from Phase 1, Phase 2, Phase 5', and Phase 7 migration decompositions, with Granite-Tiny Phase 0 as the smoke result if space allows.
3. **Layer-type stratified drift.** Phase 3 layer-stratified table/plot showing attention, SSM/Mamba, and MoE/block-output drift rates where available.
4. **Remedy falsification ladder.** Compact decision plot for static unions, hard switching/binning, EMA, DecDEC proxy, M18, M26, M11b, and ParoQuant, annotated as killed, diagnostic, partial, or baseline.
5. **Budget-dependent recovery.** M11b top-1/top-5/top-10 on Granite and Nemotron, paired with static top-10 where reported, to show that budget changes the conclusion.
6. **Baseline pressure from rotation.** Granite comparison of M11b top-5 and ParoQuant recovery, with the explicit caveat that ParoQuant is an algorithmic reproduction baseline rather than the project-owned channel-set method.

## Saturated, Alive, And Highest-Priority Branches

Saturated or killed under the current evidence: static union protection, hard position-conditional switching, hard position-binned scales, top-1% EMA alone, DecDEC-style top-1% reactive selection as a positive method, stable-core alone, and M18 activation+K at the tested top-1% budget.

Still alive but not paper-headline-ready: budgeted M11b-style protection, rotation+budget co-design, sensitivity-weighted budget allocation, rotated-basis EMA tracking, and a larger-budget cross-tensor halo only if reopened under a fresh preregistration.

Highest-priority next branch for an ICLR follow-up: rotation+budget composition with ParoQuant-only, M11b-only, static, random, and matched-budget controls, followed by a strict cross-family falsification pair.

## Why This Structure Fits The Evidence

The strongest evidence is the measurement result, not the remedy. The external collaboration state identifies the paper as a COLM efficient-reasoning workshop target, and the result packets support a clear measurement claim across model families. Starting with drift measurement prevents the paper from depending on a positive-method claim that the current artifacts do not justify.

The negative-results ladder is unusually valuable and should be in the main structure, not hidden in an appendix. Phase 3/4, M2, M10, M11, DecDEC, M18, and M26 collectively narrow the mechanism space: hard discontinuities and naive reactive top-1% protection are not enough, while budget and rotation remain live. This is a coherent workshop contribution because it turns a broad space of plausible fixes into a smaller set of defensible next gates.

The partial remedy section must be placed after the failed remedies because M11b is interpretable only in contrast. On Granite, its top-5 signal is positive but uncertain; on Nemotron, top-10 is strong, but top-5 does not beat static top-10. ParoQuant then creates baseline pressure: a Granite rotation baseline is stronger than the current project-owned channel-set remedy, so the honest conclusion is "budget matters, but not yet a deployable method."

This structure also preserves future ICLR optionality. It records what has been ruled out, what remains alive, and what the next rigorous branch must test, while avoiding the unsupported claim that LatentWire already has a benchmark-backed positive method.
