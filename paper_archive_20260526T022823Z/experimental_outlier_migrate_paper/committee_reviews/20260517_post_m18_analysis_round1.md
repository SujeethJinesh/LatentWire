# Committee Review: Post-M18 Analysis Draft

Draft reviewed:
`experimental/outlier_migrate/paper/outlier_migrate_colm2026.pdf`

Artifact basis:
- M18 packet:
  `experimental/outlier_migrate/phase9/results/om_phase9_m18_granite_small_vac12_20260516T193500Z`
- Post-M18 reports:
  `experimental/outlier_migrate/phase9/post_m18_analysis/`

## COLM Area Chair

Score: 7/10

The draft has a coherent measurement-and-mechanism story. The post-M18
analysis is a useful addition because it distinguishes what the current
artifact set can identify from what would require new telemetry. The strongest
contribution remains the replicated decode-position channel drift result
across Granite, Nemotron, DeepSeek, and Falcon-H1. The method section is now
more honest: M18 is not a win, but it is not merely random-control failure.
The paper is plausible as a COLM workshop submission if the remaining DecDEC
and M11b results are integrated cleanly.

Main requested fix: make the distinction between trace-level set-leaving and
prompt-averaged diagnostic set-leaving explicit wherever the post-M18 reports
are summarized. Reviewers may otherwise notice that some layer-level
post-M18 numbers are smaller than the main decomposition numbers.

## MLSys Reviewer

Score: 6/10

The systems contribution is still below archival MLSys bar because no method
yet improves quality, latency, memory, or bytes. The negative evidence is
valuable and increasingly systematic: boundary discontinuity, smoothing, and
cross-tensor coupling have all been tested under fixed gates. However, a
systems reviewer will ask for DecDEC comparison and a budget sweep before
accepting the negative conclusion. The post-M18 analysis correctly says K/V
migration and recovery curves are not identifiable; that honesty helps, but it
also highlights missing telemetry.

Main requested fix: DecDEC must be reported as an algorithmic baseline with
the omitted CUDA/CPU-staging pieces stated in the table caption. M11b should
be framed as the budget test, not another method search.

## ICLR Reviewer

Score: 5/10

The empirical phenomenon is interesting, but the paper is not yet an ICLR
positive-method paper. The current draft is best understood as a careful
negative-result mechanism paper. It needs either a positive intervention or a
stronger theoretical/mechanistic account of why the tested intervention class
fails. The post-M18 analyses help, especially the smooth-overlap and
always-protected-core readouts, but they are still diagnostic.

Main requested fix: after DecDEC/M11b, the paper should choose one claim
surface. Either it becomes a negative-result paper about the limits of
inference-time channel-set protection, or it reports a positive budget/reactive
method. Do not leave both framings in tension.

## Adversarial Reviewer

Score: 7/10

The draft is careful about not overclaiming M18, which is good. I see two
remaining risks. First, the phrase "small stable core" could be misread as
supporting a static method even though the static methods failed; keep saying
that core is insufficient. Second, the post-M18 reports include some
infeasibility findings. That is acceptable only if the paper does not imply
those questions have been answered. The current text mostly handles this, but
table captions and abstract wording should be checked after DecDEC/M11b.

Main requested fix: add one sentence in limitations that K/V migration and
per-position recovery curves are not yet measured, not merely negative.

## Action Items

1. Integrate DecDEC and M11b before raising the contribution claim.
2. Keep post-M18 diagnostic numbers labeled as prompt-averaged diagnostics when
   they are not the trace-bootstrap gate metrics.
3. Add a limitations sentence after DecDEC/M11b lands: K/V migration and
   recovery curves require new telemetry.
