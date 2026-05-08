# ThoughtFlow-FP8 Committee Review - Paper Polish Gate

Date: 2026-05-08
Packet: `experimental/thoughtflow_fp8/phase2/results/thoughtflow_paper_polish_20260508T0050Z`
Checker decision: `PASS_THOUGHTFLOW_PAPER_BUILDABLE`

## COLM Area Chair

Scores: novelty 6/10, rigor 7/10, clarity 7/10.

Meta-review: This is not a positive KV-compression paper, but it is potentially acceptable as a negative-results / falsification-methodology note. The strongest aspect is that the paper explicitly bounds the claim: Mac-local saved-trace fixtures, proxy baselines, paired intervals, failed reproduction, and no FP8/CUDA/latency claims. The evidence ladder is clear enough to see that `rdu_topk`, `psi_topk`, and `vwac_topk` are killed rather than promoted.

Fixable blockers: make the venue framing unambiguous in title/abstract; "FP8" still invites the wrong review. Compress the many historical tables so the failed-to-reproduce row cannot be mistaken for a leaderboard. State why this ladder is novel relative to ordinary preregistration/negative-result reporting. The related-work citations mostly resolve to real primary pages, including ThinKV and LongFlow, but the bibliography should avoid omnibus "recent methods" entries where possible.

## MLSys Reviewer

Scores: systems contribution 3/10, reproducibility 7/10, engineering rigor 6/10.

Review: As a systems paper this should be rejected: it has no real FP8, CUDA, throughput, latency, or serving result, and it says so explicitly in the claim boundary. As an artifact-backed diagnostic note, it is much stronger. The reproducibility story is credible: tracked scripts, JSON/Markdown artifacts, hash prefixes, dirty-tree refusal for packet regeneration, and stable local test commands are described in the reviewer pack and paper.

Fixable blockers: package the artifact as a clean-checkout replay, with explicit environment lockfile, model/source versions, hardware/OS notes, expected runtime, and expected hash outputs. Add a one-command "verify paper tables" path. Rename or subtitle away from FP8 unless the venue is explicitly a negative systems lessons track.

## Adversarial Reviewer

Scores: claim safety 7/10, p-hacking resistance 6/10, citation/provenance risk 5/10.

Review: The paper is unusually careful about not claiming a method, but the weak points are still visible. Repo-local registration is not external timestamping; the paper admits this, but adversarial reviewers will still ask whether rules were written after seeing informal results. The proxy baselines are named after real methods but are not faithful implementations, which risks accidental scope creep despite repeated caveats. Achieved keep rates are reported, but some fresh-surface comparisons have non-identical keep rates, so promotion/failure language should be phrased as diagnostic rather than definitive.

Fixable blockers: replace all "baseline" language with "local proxy" where exactness matters; add a short audit paragraph explaining what cannot be proven by hashes alone; make oracle/headroom absence for PSI/VWAC more visible. Spot checks of primary citation pages for ThinKV, LongFlow, R-KV, KVzip, Q-Filters, and KV-Direct-related entries resolved, but the omnibus "recent KV-cache methods" citation should be split into exact BibTeX entries.

## Supervising Decision

The gate passes buildability and reviewer-pack currency, but this is not a camera-ready positive-method paper and not an MLSys systems paper. Per `swarm/goal.md` and HANDOFF scope, the ThoughtFlow paper body remains untouched in this gate. The safe status is: buildable diagnostic/falsification submission candidate requiring human copyedit and venue-framing review.
