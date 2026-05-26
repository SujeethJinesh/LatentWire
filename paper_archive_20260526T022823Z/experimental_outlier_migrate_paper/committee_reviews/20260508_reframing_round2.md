# OutlierMigrate Committee Review: Related-Work Reframing Round 2

Reviewed after Round 1 fix response:

- `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`
- `experimental/outlier_migrate/paper/reviewer_pack.md`
- `experimental/outlier_migrate/paper/committee_reviews/20260508_reframing_round1.md`

## A. Area Chair

Novelty 6/10, rigor 6/10, clarity 8/10.

Round 1 wording fixes are resolved. The draft now explicitly says
OutlierMigrate is not first for dynamic Mamba outliers, credits QMamba and
OuroMamba, and scopes the contribution to hybrid LLM long-reasoning traces.
The "static-outlier hypothesis" phrasing is narrowed to "on this Granite
rank-migration surface," which avoids overclaiming. No new stop-rule issue
found.

## B. MLSys Reviewer

Rigor 6/10, clarity 8/10.

The runbook/project wording issue is fixed; "authorized work window" and vLLM
compatibility prose are absent from the limitation section. The paper still
clearly states no kernel, cache policy, throughput result, or intervention
claim. That is a research gap, not a wording blocker. No new stop-rule issue
found.

## C. Adversarial Reviewer

Novelty 6/10, rigor 6/10, clarity 8/10.

The adaptive-baseline fairness issue is fixed: KVQuant and BlockDialect are
now described as containing adaptive components, and the draft says the
evidence motivates stress-testing, not declaring them failed. The limitations
preserve preregistered thresholds and warn against retuning. No new stop-rule
issue found.

## Decision

Another fix round required: no.
