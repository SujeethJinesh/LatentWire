# Paper Review Log

## 2026-05-15 Draft 0 After M2

Review file:
`experimental/outlier_migrate/paper/committee_reviews/20260515_draft0_after_m2.md`

Scores:

| Reviewer | Score | Main critique |
| --- | ---: | --- |
| COLM area chair | 7/10 | Coherent measurement/mechanism paper, but not positive-method yet. |
| MLSys reviewer | 6/10 | Systems contribution remains diagnostic without a working intervention. |
| Adversarial reviewer | 7/10 | Needs citation/contamination audit and careful reduced-slice caveats. |

Top fixes/actions:

1. Add stronger industrial motivation and cost context to the introduction.
2. Add terminology discipline for the overloaded phrase "outlier migration."
3. Continue M10/M11/M17 method queue to seek a positive method or a clean
   negative-result mechanism story.

Status: actions 1 and 2 landed in commit `4fbf9a36`; action 3 is in progress.

## 2026-05-17 Post-M18 Analysis Round 1

Review file:
`experimental/outlier_migrate/paper/committee_reviews/20260517_post_m18_analysis_round1.md`

Scores:

| Reviewer | Score | Main critique |
| --- | ---: | --- |
| COLM area chair | 7/10 | Coherent workshop measurement/mechanism paper; distinguish trace-level set-leaving from prompt-averaged diagnostics. |
| MLSys reviewer | 6/10 | Still no systems improvement; DecDEC baseline and M11b budget sweep are required before the negative conclusion is strong. |
| ICLR reviewer | 5/10 | Not yet a positive-method paper; must choose positive-method or negative-mechanism framing after DecDEC/M11b. |
| Adversarial reviewer | 7/10 | Avoid implying K/V migration or recovery curves were measured; keep stable-core language caveated. |

Top fixes/actions:

1. Integrate DecDEC and M11b before raising the contribution claim.
2. Label post-M18 diagnostic numbers as prompt-averaged diagnostics when they
   are not trace-bootstrap gate metrics.
3. Add a limitations sentence after DecDEC/M11b lands saying K/V migration and
   recovery curves require new telemetry.

Status: DecDEC Granite-Small is currently running; M11b is queued next.
