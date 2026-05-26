# OutlierMigrate Phase 3 Negative-Intervention Committee Review

Date: 2026-05-10
Paper: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`
PDF: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.pdf`
Primary Phase 3 packet:
`experimental/outlier_migrate/phase3/results/om_phase3_20260509T212000Z`
Decision: `KILL_OM_PHASE3_INTERVENTION_FAILS`

## Round 1

### COLM Area Chair

Score: 6/10.

The paper is borderline acceptable as characterization plus negative
intervention. The arc is coherent: Phase 0/1 show Granite migration, Phase 2
partially supports Nemotron-3, and Phase 3 kills the simplest static
union-protection intervention. Main fixes: make 10/24 no-gap traces central,
explain recovery values below 0 and above 1, signal failed intervention in the
title/abstract, add per-trace diagnostics, keep Qwen3.6/Kimi as future work,
and verify 2025/2026 citations.

### MLSys Reviewer

Score: 5/10.

Reproducibility is strong: preregistrations, frozen prompts, model commits,
hashes, checker outputs, bootstrap CIs, controls, grid sensitivity, and command
metadata are visible. The systems contribution is weak: no working method, no
latency/memory/throughput improvement, no kernel/runtime design, and no
deployable quantization recipe. Fixes requested: explicitly state no systems
method remains, explain no-gap traces and recovery behavior, complete or defer
Qwen3.6/Kimi more sharply, and add contamination/audit notes.

### Adversarial Reviewer

Score: 6/10.

The paper is responsibly scoped but not submission-safe yet. Biggest risk is
the Phase 3 recovery metric: 10/24 no-gap traces, clipped zeros, values above
1 and below 0, and a wide CI. The dynamic threshold is weak relative to the
observed 0.82 migration and needs to be called a nonstationarity screen.
Layer-stratified analysis must stay descriptive. No hard stop condition fires.

## Round 2

### COLM Area Chair

Score: 7/10.

The revision materially improves the framing. Title, abstract, and
introduction now say characterization plus failed intervention, not a
positive-method final. Phase 3 no-gap kill, recovery caveats, and per-trace
counts are visible. Remaining fixes: explain that the checker reports the
first kill reason while the paper identifies the second kill condition from
packet fields, justify median recovery over mean, and verify citations.

### MLSys Reviewer

Score: 6/10.

The paper no longer implies a systems method. Phase 3 controls and caveats are
explicit, and artifact traceability remains strong. It is still weak as an
MLSys systems contribution because it lacks a working mechanism, deployed
system, performance evaluation, or completed cross-family validation. Fixes:
add a contribution list separating measurement, negative intervention, and
not-claimed items; justify deterministic traces plus bootstrap; complete
contamination audit; soften architecture-general language.

### Adversarial Reviewer

Score: 7/10.

Overclaiming is much reduced. The remaining p-hacking risk is provenance:
top-1%, rank-delta >2, decode positions, AIME slice, Granite choice, and the
0.05 pass threshold need audit context. Phase 2 remains partial. Layer
stratification is post-hoc and must remain descriptive. No hard stop condition
fires.

## Round 3

### COLM Area Chair

Score: 8/10.

The final fixes land well. The title, abstract, and contribution list now
clearly position the work as measurement plus failed static intervention.
The measured-hybrids wording narrows scope, and the no-systems-method claim is
explicit: no deployable quantization recipe, kernel/runtime system, latency,
throughput, memory, or quality gain. Remaining minor issues: abstract density,
URL/layout polish, and final citation verification.

### MLSys Reviewer

Score: 7/10.

The draft now clears only as empirical systems characterization plus a negative
intervention result. The contribution list is explicit and narrow:
preregistered migration measurement, strict set-leaving decomposition, partial
Nemotron-3 replication, and failed static-union intervention. Artifact
traceability is strong, and Phase 3 caveats are honest. Remaining weaknesses:
missing Qwen3.6/Kimi validation, no independent seed repeat, no complete
contamination or exploratory-history audit, and limited cross-family scope.

### Adversarial Reviewer

Score: 8/10.

No hard stop condition fires. Claims are narrow, Phase 3 is clearly negative,
Qwen3.6/Kimi are deferred, and unmeasured architectures are motivation only.
Statistical caveats around recovery ratio, zero/no-gap traces, values below 0
and above 1, median choice, wide CI, and second no-gap kill path are explicit.
Remaining concerns are preregistration provenance, weak threshold framing, and
final primary-source citation verification.

## Committee Outcome

Final scores: COLM `8/10`, MLSys `7/10`, adversarial `8/10`.

The paper clears the requested committee threshold only under the
characterization plus negative-intervention framing. It does not clear as a
positive-method or systems-efficiency paper.
