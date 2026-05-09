# ThoughtFlow-FP8 Committee Review: Additional Polish Round

Date: 2026-05-09

Reviewed files:

- `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.tex`
- `experimental/thoughtflow_fp8/paper/reviewer_pack.md`
- `experimental/thoughtflow_fp8/paper/presubmission_checklist.md`
- `experimental/thoughtflow_fp8/paper/committee_reviews/20260508_paper_polish_gate.md`
- `experimental/thoughtflow_fp8/phase2/current_decision_manifest_20260506.md`
- `experimental/thoughtflow_fp8/phase2/results/thoughtflow_paper_polish_20260508T0050Z/checker_result.json`
- `experimental/thoughtflow_fp8/phase2/results/thoughtflow_paper_polish_20260508T0050Z/metrics.json`

Context: ThoughtFlow-FP8 is copyedit-only. This review treats the paper as a
falsification-methodology / negative-results workshop diagnostic, not a
positive-method or systems paper.

## COLM Area Chair

Score: 7/10 under falsification-methodology framing.

Novelty is moderate: the paper is not novel as KV compression, FP8, or systems
work, but the repo-local stop ladder is a defensible workshop contribution.
Rigor is the strongest axis: frozen surfaces, one-shot registered successor
rules, proxy/stopped-family separation, paired intervals, achieved keep rates,
and explicit demotion of `rdu_topk`, `psi_topk`, and `vwac_topk` make the
negative result unusually auditable. Clarity is now mostly adequate because the
abstract, claim-boundary table, final status matrix, and reviewer pack
repeatedly say there is no live method claim.

Remaining weakness is framing fragility. "ThoughtFlow-FP8" still invites
reviewers to expect FP8 evidence, and the related-work paragraph uses a
crowded omnibus citation that should be human-checked. I would not call this
camera-ready final, but it is a plausible camera-ready candidate without new
experiments if a human performs final venue framing, title/abstract judgment,
citation verification, and copyedit.

## MLSys Reviewer

Score: 7/10 only as an artifact-backed falsification diagnostic; 3/10 as an
MLSys systems paper.

There is no systems contribution in the normal MLSys sense: no native FP8
serving, CUDA kernel, latency, throughput, memory-bandwidth result, or faithful
baseline implementation. The paper is appropriately explicit about that.
Reproducibility and engineering rigor are stronger: the polish checker passes,
the PDF builds, owned tests pass, packet artifacts are hashed, regeneration
rejects dirty ThoughtFlow state, and the reviewer pack identifies the exact
artifacts and limits.

The main remaining blocker is artifact replay polish, not experiment evidence.
A final human pass should verify clean-checkout replay, environment/version
notes, expected runtime, and table-to-artifact traceability. Without paper-body
edits, it can be a camera-ready candidate only if the target venue welcomes
negative artifact methodology and accepts the current "not a systems result"
framing.

## Adversarial Reviewer

Score: 7/10 under the narrowed framing.

The claims are mostly safe: the draft repeatedly denies positive compression,
FP8, CUDA, latency, throughput, and reasoning-benchmark claims. The provenance
story is useful but not airtight: repo-local hashes are audit aids, not
external timestamps, and the paper admits this. P-hacking risk is contained by
the consumed-signal rule and one-shot fresh surfaces, but cannot be eliminated
because earlier informal exploration may have shaped the ladder. Citation risk
remains the highest residual issue: the omnibus "recent KV-cache methods"
entry and method-name-proxy language require human source verification before
submission.

Explicit answers:

- All three scores are at least 7/10 only under the falsification-methodology
  framing.
- The paper is not camera-ready final; human final review is required.
- It can be a camera-ready candidate without changing the paper body only if
  the human reviewer accepts the current copy and resolves title/framing and
  citation confidence externally; otherwise the blockers are body-copyedit
  blockers that only a human should resolve under the stated scope.
- No hard stop condition fires: no clear p-hacking, post-hoc cherry-picking,
  scope-creep preregistration violation, citation hallucination, or paper-test
  regression was found in this read.

## Local Verification

- `experimental/thoughtflow_fp8/phase2/check_paper_buildable.py` returned
  `PASS_THOUGHTFLOW_PAPER_BUILDABLE` for
  `experimental/thoughtflow_fp8/phase2/results/thoughtflow_paper_polish_20260508T0050Z`.
- Owned tests passed on the GPU node:
  `70 passed, 1 warning in 11.46s`.
