# Structural Audit: Ideal vs. Archived v2 Paper

Date: 2026-05-26

Inputs:

- `docs/ideal_structure.md`
- `docs/as_is_analysis.md`
- archived v2 paper at `paper_archive_20260526T022823Z/`

Current paper readiness: workshop-polished measurement/mechanism draft. It is
not ICLR-ready as a positive-method paper because the live method evidence is
partial, model-dependent, and not yet a robust deployable method.

Current story: decode-position channel-set drift is reproducible across the
measured long-reasoning W4A16 model set; several intuitive channel-set remedies
fail or remain partial under controls; budget helps but is not model-invariant;
ParoQuant-style rotation is the strongest Granite baseline.

Exact gap blocking ICLR submission: a positive method that survives larger
frozen slices, seed repeats, paired uncertainty, and same-family/cross-family
generalization gates.

## Gate Summary

| Section / Element | Severity | Assessment |
|---|---|---|
| Title | TARGETED_CHANGE | Current title over-foregrounds "Budget-Tuned Remedy" relative to model-dependent evidence and ParoQuant pressure. |
| Abstract | TARGETED_CHANGE | Correct ordering, but "strongest channel-set remedy" still reads more positive than the evidence warrants. |
| Introduction/contributions | TARGETED_CHANGE | Strong measurement-first spine; contribution 4 should say "budget-dependent signal" rather than "remedy" if no Stage 1 result is added. |
| Related work | POLISH_ONLY | Good source-aware framing after hallucination audit. No structural issue. |
| Method + Experiments | TARGETED_CHANGE | Definitions are clear, but recovery edge cases and method taxonomy are dense. A compact taxonomy/decision framing would help. |
| Results: drift + components | POLISH_ONLY | Current order and figures match the ideal evidence spine. |
| Results: intervention ladder | TARGETED_CHANGE | Strong content, but the negative ladder, M11b budget signal, and ParoQuant baseline pressure could be separated more cleanly in headings/prose. |
| Discussion | TARGETED_CHANGE | Mechanism synthesis is correct but could more explicitly mark saturated vs. alive branches. |
| Limitations | POLISH_ONLY | Scope caveats are honest and structurally appropriate. |
| Reproducibility/appendices | POLISH_ONLY | Candid and audit-complete. FAST_VERIFY limitation is a strength for honesty, not a structural flaw for workshop submission. |

Counts: 6 `TARGETED_CHANGE`, 4 `POLISH_ONLY`, 0 `STRUCTURAL_GAP`.

Gate decision: **Phase 4B, targeted structural changes**. The current paper is
not fundamentally wrong. A full rewrite is not justified.

## Per-Section Gap Analysis

### Title

What works: it foregrounds channel-set drift, long-reasoning, W4A16, and a
method direction.

Gap: "Budget-Tuned Remedy" implies a stronger positive method than the body can
support. The evidence is better described as budget-dependent signal or a
partial remedy. ParoQuant also outperforms the Granite M11b row, so the title
should not imply the channel-set remedy is the main solution.

Recommendation: change to a more diagnostic title, for example:

`Channel-Set Drift in Long-Reasoning W4A16 LLMs: Mechanisms, Failed Remedies, and Budget-Dependent Signals`

Estimated work: 5 minutes.

### Abstract

What works: it leads with the static-protection assumption, four-model drift,
DecDEC/Quamba2 positioning, eight channel-set families, M11b, ParoQuant, and KL.

Gap: "Budget-tuned EMA protection is the strongest channel-set remedy" is true
within the channel-set rows, but the phrase makes the paper sound more
method-positive than the evidence. The ideal structure suggests calling it the
"strongest channel-set clue" or "budget-dependent signal."

Recommendation: reword the M11b sentence to emphasize model dependence and
baseline pressure without changing numbers.

Estimated work: 10 minutes.

### Introduction And Contributions

What works: the intro has the right measurement-first sequence and clearly
scopes the Quamba2 comparison to measured block-output surfaces.

Gap: contribution 4 still uses "Partial remedy and scope." This is acceptable
for a workshop paper, but the ideal framing prefers "budget-dependent signal"
to avoid overselling M11b.

Recommendation: rename the contribution to "Budget-dependent signal and scope"
and keep ParoQuant in the same bullet as baseline pressure.

Estimated work: 10 minutes.

### Related Work

What works: after source verification, the section correctly separates Mamba
quantization, rotations, dynamic/KV-cache policies, and error accumulation.

Gap: none structural. KV-cache methods share a paragraph with DecDEC, but this
is acceptable because the paragraph is explicitly about dynamic/long-context
policies and the tensor distinction is stated.

Recommendation: no structural change.

Estimated work: none.

### Method And Experiments

What works: strict set-leaving and recovery are defined before use; controls are
clearly described.

Gap: the method taxonomy is dense, and recovery edge cases are nontrivial
(negative values, values above one, no-gap traces, positive-gap filtering).
Readers may need a compact map from method family to mechanism hypothesis.

Recommendation: add one short sentence before the method list explaining that
the method classes are grouped by the mechanism each tests: static coverage,
hard switching, smoothing, cross-tensor signal, reactive selection,
larger-budget tracking, and stable cores. Keep the full list unchanged.

Estimated work: 15 minutes.

### Results: Drift And Components

What works: drift result comes first, then component localization. This exactly
matches the ideal structure. Figures serve the correct roles.

Gap: none structural.

Recommendation: no change.

Estimated work: none.

### Results: Intervention Ladder, Budget Signal, ParoQuant

What works: the intervention ladder precedes M11b, which makes the M11b result
interpretable. ParoQuant is already used to prevent overclaiming.

Gap: the heading "Budget matters, but is not model-invariant" is accurate, but
the subsection combines three ideas: M11b budget signal, model dependence, and
ParoQuant baseline pressure. This is manageable in a short paper, but the
ParoQuant comparison should be flagged as baseline pressure in the subsection
opening or heading.

Recommendation: change the heading to "Budget helps, but rotation remains the
stronger Granite baseline" or add an opening sentence that separates the two
claims.

Estimated work: 15 minutes.

### Discussion

What works: the discussion synthesizes the three mechanisms and identifies
composition as next step.

Gap: it does not explicitly mark which branches are saturated vs. alive. The
ideal structure treats that as the reusable contribution for future ICLR work.

Recommendation: add a compact sentence: "The saturated branches are hard
switching, top-1% smoothing alone, DecDEC-style top-1% reactive selection,
stable core alone, and top-1% activation+key coupling; the live branch is
rotation plus budget-aware protection."

Estimated work: 10 minutes.

### Limitations

What works: scope boundaries are explicit and honest.

Gap: no structural issue.

Recommendation: no change.

Estimated work: none.

### Reproducibility And Appendices

What works: archive, FAST_VERIFY limitation, provenance, LLM use, and source
audit are explicit. This is structurally appropriate for workshop submission.

Gap: none for this pass.

Recommendation: no change.

Estimated work: none.

## Overall Recommendation

Proceed with **targeted structural changes only**:

1. Make the title less method-positive.
2. Reword abstract/contribution phrasing from "remedy" toward
   "budget-dependent signal" while preserving the numeric results.
3. Add a one-sentence mechanism grouping before the method list.
4. Make the M11b/ParoQuant subsection heading or first sentence clarify
   baseline pressure.
5. Add one discussion sentence identifying saturated vs. live branches.

Do not run a full rewrite. The current order is already aligned with the
evidence: measurement, localization, intervention falsification, budget signal,
KL mechanism, limitations.
