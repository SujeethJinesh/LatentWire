# ThoughtFlow-FP8 Committee Review: Additional Polish Round 3

Date: 2026-05-10

Reviewed files:

- `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.tex`
- `experimental/thoughtflow_fp8/paper/reviewer_pack.md`
- `experimental/thoughtflow_fp8/paper/presubmission_checklist.md`
- `experimental/thoughtflow_fp8/paper/committee_reviews/20260509_polish_round2.md`
- `experimental/thoughtflow_fp8/paper/committee_reviews/20260509_reproducibility_audit.md`

Scope: read-only committee polish. The paper is evaluated only as a
falsification-methodology / negative-results workshop diagnostic, not as a
positive method or systems paper. No paper-body edits were made.

## COLM Area Chair

Score: 8/10 under falsification-methodology / negative-results workshop
framing.

This is strong as a diagnostic note. The paper visibly states what it is not:
not FP8 serving, not CUDA, not latency/throughput, not a live compression
method, and not a reasoning-model benchmark. The evidence ladder is clear:
`rdu_topk` looked promising and then failed stricter reproduction; `psi_topk`
and `vwac_topk` failed fresh surfaces. Rigor is strongest: registered
successor rules, frozen surfaces, paired intervals, achieved keep rates,
oracle/headroom where available, artifact hashes, dirty-tree guard, and
build/test audit. Copyedit-only issues remain around abstract density,
first-read jargon, omnibus citation cleanup, and consistent "local proxy"
wording.

## MLSys Reviewer

Score: 8/10 under falsification-methodology / artifact diagnostic framing.

This does not score as an MLSys systems paper, but that is explicit enough not
to penalize under the requested framing. The draft separates claimed and
not-claimed contributions, marks `rdu_topk` as failed-to-reproduce, reports
paired intervals and achieved keep rates, preserves historical rows without
promoting them, and states that repo-local hashes are audit aids rather than
external timestamps. Artifact traceability is strong: reviewer pack,
presubmission checklist, current manifest, diagnostic packet, buildability
gate, test command, hash prefixes, and committee audits line up.
Copyedit-only issues: "FP8" can still misroute expectations, "benchmark"
should remain "saved-trace falsification fixture", and "baseline" should be
"local proxy" for ThinKV/R-KV/LongFlow-like rows.

## Adversarial Reviewer

Score: 8/10 under the falsification-methodology / negative-results workshop
framing.

The claim boundary is unusually explicit: no positive method, no real FP8, no
CUDA/latency/throughput, no faithful external baseline comparison, and no
reasoning-benchmark claim. The first-surface `rdu_topk` win is repeatedly
marked historical and failed-to-reproduce, while PSI/VWAC are clearly killed.
Copyedit-only issues: "FP8" still appears in project/package wording, the
omnibus bibliography item remains the main citation-copyedit risk, "proxy
controls" may be safer than "proxy baselines", and final layout should prevent
historical rows from reading like a leaderboard.

## Outcome

Final scores: COLM `8/10`, MLSys `8/10`, adversarial `8/10`.

No stop condition fired. ThoughtFlow-FP8 remains a camera-ready candidate only
as a falsification-methodology / negative-results workshop diagnostic. It is
not camera-ready final, not a positive-method paper, and not an MLSys systems
paper.
