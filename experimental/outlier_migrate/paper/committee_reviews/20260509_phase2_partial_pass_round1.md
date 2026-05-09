# Committee Review: OutlierMigrate Partial Phase 2 PASS Round 1

Reviewed:

- `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`
- `experimental/outlier_migrate/paper/reviewer_pack.md`
- `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/checker_result.json`
- `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/artifact_check.json`
- `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/metrics.json`
- `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z/migration_decomposition.md`

## COLM Area Chair

Current paper readiness: not camera-ready; roughly 45-55% of an
ICLR/COLM-ready positive-method paper. Current story: preregistered evidence
that high-magnitude decode channels migrate over long traces, replicated in
Granite and partially cross-family on Nemotron-3. Blocking gap: no completed
cross-validation, no intervention, no application metric.

Novelty: 5/10. Rigor: 6/10. Clarity: 8/10. Overall: 6/10.

The updated draft is appropriately scoped and unusually honest. The Phase 2
Nemotron-3 PASS strengthens the observational claim: migration fraction
0.820809591642925, CI [0.7865325261158594, 0.8544931149097815], artifact
complete, 24 traces, 52 layers. The decomposition is useful: strict
set-leaving is 0.533713200379867, so the result is not only within-top-set
rank noise.

However, this remains a measurement paper, not yet a positive method. The
manuscript explicitly says no efficiency, quality, memory, latency, or
robustness gain is shown. A fixable issue blocks camera-ready-candidate status:
either complete the deferred Qwen3.6/Kimi validation or narrow the paper to a
preliminary observational submission, which conflicts with the stated project
goal. No current p-hacking/prereg stop condition is triggered if the text
remains conservative; treating partial Phase 2 as full cross-validation would
trigger one.

## MLSys Reviewer

Score: 6/10.

The engineering packet is solid for a research artifact: model snapshot
commit, prompt SHA, activation artifact SHA, command metadata, random seed,
checker output, metrics, bootstrap CI, and artifact completeness are recorded.
The Phase 2 packet explicitly reports `full_cross_validation_complete=false`,
deferred Qwen3.6/Kimi models, and an artifact-complete partial Nemotron-3
validation. That is reproducible enough for audit and rerun by a motivated
reviewer.

Systems weakness: the paper still does not demonstrate a systems method. It
motivates static protection failure modes, but no kernel, routing policy,
migration-aware quantizer, update schedule, memory/latency overhead,
throughput, or quality tradeoff is evaluated. Existing systems such as
SmoothQuant, AWQ, KVQuant, QuaRot, and BlockDialect are correctly framed as
motivation rather than defeated baselines, which avoids overclaiming but also
leaves the systems contribution thin.

Fixable blocker for camera-ready-candidate status: yes. The paper needs either
a predeclared migration-aware intervention with measured overhead and benefit,
or it should not be positioned as a systems/positive-method candidate.
Reproducibility is ahead of contribution strength. No p-hacking stop condition
is apparent from the packet because the thresholds and metric are preserved
and the partial scope is disclosed. Scope creep would trigger a stop if
deferred runtime failures are used to imply broad hybrid-model validation.

## Adversarial Reviewer

Score: 5/10.

The draft mostly avoids the obvious traps, but I would still attack several
points. First, the pass threshold is extremely weak: migration fraction >0.05
with CI lower >0.05, while observed values are around 0.82. That suggests the
gate is easy once any nonstationarity exists; the paper should justify why
this threshold was meaningful before seeing data. Second, the metric conflates
set leaving and rank shuffling, and the stricter decomposition is post-hoc.
The draft discloses this, but conclusions should lean on strict set-leaving
only when clearly labeled interpretive, not preregistered.

Third, Phase 2 is partial. Nemotron-3 helps, but Qwen3.6 and Kimi are exactly
the more relevant delta-rule/gated-linear-attention targets discussed in the
motivation. Deferring them while retaining broad "hybrid LLM" language is a
scope risk. Fourth, no independent seed repeat beyond bootstrap exists;
bootstrap over 24 deterministic traces is not a substitute for new
prompt/model/runtime repeats. Fifth, cited 2025/2026 model-card and arXiv
claims should be checked carefully before submission, especially Qwen3.6 and
Kimi, to avoid hallucinated or unstable citations.

Fixable issue blocks camera-ready-candidate status: yes, the absence of an
intervention and incomplete cross-validation. Current p-hacking/prereg
amendment stop condition: not triggered by the written draft, because it flags
partial validation and does not retune metrics. Stop condition triggers
immediately if the authors relabel this as full Phase 2, tune
positions/top-1%/rank delta after inspection, or claim a positive method
without a preregistered intervention.

## Supervising Decision

Camera-ready candidate status is blocked. The immediate fix is textual scope
hardening: preserve the partial PASS, but make the draft and reviewer pack
unmistakably observational and not a completed positive-method/system paper.
No preregistration amendment, p-hacking, or scope-creep stop condition has
fired as long as the paper remains conservative.
