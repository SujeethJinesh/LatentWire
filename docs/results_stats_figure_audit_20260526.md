# Results / Statistics / Figure Audit

Date: 2026-05-26

Scope: `experimental/outlier_migrate/paper/outlier_migrate_colm2026.tex`,
`experimental/outlier_migrate/paper/figures/generate_polish_figures.py`,
generated figure assets, and raw OutlierMigrate result JSON/JSONL packets.
No paper, release, swarm, or experiment files were edited.

## Executive Status

Current paper readiness: workshop-polished but not ICLR-ready. The evidence
supports a scoped measurement/mechanism story with one partial budget-tuned
positive result, not a robust deployable positive method.

Current story: long-decode top-1% activation channel membership drifts across
Granite-Small, Nemotron-3, DeepSeek-R1-Distill, and Falcon-H1; simple
channel-set interventions mostly fail controls; M11b budget tuning is positive
on corrected Nemotron top-10 and partial/high-variance on Granite; ParoQuant
shows the endpoint is not intrinsically impossible for W4A16 PTQ.

Blocking gap: the numerical evidence is mostly present, but the polished paper
still has figure/table/statistical-presentation inconsistencies that would be
easy reviewer targets. These should be fixed before treating the paper as
submission-ready.

## Headline Spot Checks

Checked against packet keys rather than relying on paper text:

| Claim | Paper location | Packet source | Audit result |
|---|---|---|---|
| Granite set-leaving `0.566` | `outlier_migrate_colm2026.tex:74` | `phase1/results/om_phase1_20260508T014959Z/migration_decomposition.md` | Matches rounded `0.566234756098`. |
| Nemotron set-leaving `0.534` | `outlier_migrate_colm2026.tex:74` | `phase2/results/om_phase2_nemotron3_20260508T231723Z/migration_decomposition.md` | Matches rounded `0.533713200380`. |
| DeepSeek set-leaving `0.671` | `outlier_migrate_colm2026.tex:74` | `phase5_prime/results/om_phase5p_20260512T053800Z/migration_decomposition.md` | Matches rounded `0.670572916667`. |
| Falcon-H1 set-leaving `0.674` | `outlier_migrate_colm2026.tex:74` | `phase7/results/om_phase7_falcon_h1_20260512T223600Z/migration_decomposition.md` | Matches rounded `0.673611111111`. |
| M11b Granite top-5 `0.449 [-1.30, 1.00]` | `outlier_migrate_colm2026.tex:119,131` | `phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z/metrics.json:results_by_regime.m11b_top5` | Matches rounded `0.449284091125`, CI `[-1.300900018791, 1.000790626547]`. |
| M11b Nemotron top-5/top-10 | `outlier_migrate_colm2026.tex:120-121,131` | `phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z/metrics.json:results_by_regime` | Matches corrected salvage packet: top-5 `0.456736183270`, top-10 `0.814739798903`. |
| M11b Nemotron top-10 margin over static top-10 `0.220` | `outlier_migrate_colm2026.tex:121,131` | `.../control_metrics.json:m11b_top10_minus_static_top10_median` | Matches `0.220356055286`. |
| ParoQuant Granite `0.754 [0.477, 1.00]` | `outlier_migrate_colm2026.tex:122,133` | `phase9/results/om_paroquant_granite_small_20260520T1555Z/metrics.json:results_by_regime.paroquant_w4a16` | Matches rounded median; CI high is actually `1.003769554323`, displayed as `1.00`. |
| M26 stable core `0.178 [-0.116, 1.00]` | `outlier_migrate_colm2026.tex:118` | `phase9/results/om_phase9_m26_granite_small_vac12_20260518T203000Z/metrics.json:results_by_regime.m26_core` | Matches rounded `0.177615807648`, CI `[-0.116448558090, 0.999915931324]`. |
| DecDEC proxy `-0.070 [-8.50, 0.674]` and random margin `1.19` | `outlier_migrate_colm2026.tex:117` | `phase9/results/om_phase9_decdec_granite_small_vac12_20260517T141500Z/{metrics,control_metrics}.json` | Matches `-0.070034781258` and `1.192570144867`. |
| M18 activation+K `-0.344 [-14.1, 0.741]` and random margin `4.32` | `outlier_migrate_colm2026.tex:116` | `phase9/results/om_phase9_m18_granite_small_vac12_20260516T193500Z/{metrics,control_metrics}.json` | Matches `-0.343590844126` and `4.320788886925`. |
| KL means and AR decays | `outlier_migrate_colm2026.tex:137` | `phase9/results/om_phase9_kl_granite_small_dense_20260519T085400Z/{kl_summary,growth_model_fits}.json` | Matches rounded means `0.150/0.133/0.131` and AR decays `0.51/0.45/0.44`; sublinear sqrt has lowest RSS for all three. |
| Per-component drift | `outlier_migrate_colm2026.tex:85,493-497` | `phase3/results/layer_stratified_migration.json:type_summary` | Matches rounded Granite and Nemotron values. |

## Findings

| Severity | Finding | Evidence | Concrete fix |
|---|---|---|---|
| CRITICAL | None found in the audited numerical claims. | The headline method medians/CIs, four-model set-leaving values, KL means/AR decays, M18/DecDEC/M26 random-control margins, ParoQuant, and per-component drift values all trace to committed artifacts within rounding. | No critical numerical correction required. |
| SUBSTANTIAL | `fig:method-recovery` contains a Nemotron static top-10 row, but the main table says exact rounded figure values are in the table and omits that row. | Generator row at `generate_polish_figures.py:135` plots `("Static top-10", "Nemotron", 0.594, 0.317, 0.814)`. Figure caption at `outlier_migrate_colm2026.tex:101` says exact rounded values are in `Table~\\ref{tab:methods}`. The table at `outlier_migrate_colm2026.tex:113-122` has no static top-10 row, although Nemotron top-5/top-10 comparisons depend on it. | Add `Static top-10 & Nemotron & 0.594 & [0.317, 0.814] & baseline/control` to the main table, or remove the row from the figure and change the caption. Prefer adding the row because line `131` discusses static top-10 directly. |
| SUBSTANTIAL | The appendix "all method families" table omits a main-text method result: M11b Nemotron top-5. It also omits the static top-10 control needed to interpret the Nemotron top-5/top-10 rows. | Main table reports Nemotron top-5 at `outlier_migrate_colm2026.tex:120`; main text discusses it at line `131`. Appendix table `outlier_migrate_colm2026.tex:456-477` includes only "Budget EMA top-10, Nemotron" at line `472`, while caption line `477` says it gives detailed outcomes for all method families discussed. | Add appendix rows for `Budget EMA top-5, Nemotron & 10 & 0.167 & 0.457 & [0.346, 0.795]` and `Static top-10, Nemotron & 10 & 0.167 & 0.594 & [0.317, 0.814]`, or narrow the caption to say the table lists selected headline outcomes only. |
| SUBSTANTIAL | The paper repeatedly says "seven" channel-set method families/classes but enumerates eight if static migration-aware unions are included. | Abstract line `19` says seven. Contribution line `34` says seven. Method line `62` lists: static migration-aware unions, position-conditioned sets, position-binned scales, M11 EMA, M18 coupling, DecDEC proxy, M11b budget EMA, and M26 stable core. Appendix lines `464-474` also include static unions plus the later method rows. | Either change "seven" to "eight", or say "seven Phase-9 channel-set classes after two earlier static-union failures" and keep static unions out of the counted list. |
| SUBSTANTIAL | No-gap handling is not consistently described for legacy static-union rows versus Phase 9 method rows. | Method line `60` says no-gap traces are excluded from ratio computation for method CIs. Phase 9 packets do this: e.g. M11b Granite has 8 included / 4 no-gap in `...m11b_granite.../metrics.json`. But static union rows in the appendix use 24 traces while also reporting no-gap fractions: `outlier_migrate_colm2026.tex:464-465`; Phase 3/4 packets store per-trace zero recoveries and no excluded positive-gap denominator. | Clarify that Phase 3/4 legacy static-union gates kept no-gap traces as zero-recovery rows, while Phase 9 method packets use positive-gap included traces; or recompute Phase 3/4 static-union CIs under the same positive-gap filtering and update the appendix. |
| SUBSTANTIAL | Figure generation is not fully auditable from raw packets because method and component figures hard-code plotted values. | `generate_polish_figures.py:126-137` hard-codes all method medians/CIs, and lines `167-172` hard-code per-component values. The values match current raw packets, but the script can silently go stale. The base Python environment also lacks `matplotlib`, so the figures could not be regenerated in this audit environment without additional setup. | Make `generate_polish_figures.py` load values from the corresponding `metrics.json`, `control_metrics.json`, `kl_summary.json`, `growth_model_fits.json`, and `layer_stratified_migration.json` files, then emit a small source manifest alongside the PDFs. Add a repo-local venv or documented figure-generation dependency path. |
| MINOR | The set-leaving figure generator and the text/release source use slightly different strict set-leaving implementations, causing sub-0.001 display drift for some final values. | Text line `74` uses migration decomposition values, e.g. DeepSeek `0.670572916667` and Falcon `0.673611111111`. Recomputing with the figure script's `top_set`/set-difference logic from `generate_polish_figures.py:32-65` gives final means about DeepSeek `0.669828869048` and Falcon `0.673190235690` under the current NumPy path. Rounded paper values remain visually consistent, but exact figure/text provenance differs. | Drive `set_leaving_decode_positions.pdf` from the same migration-decomposition artifacts used by the paper/release exact values, or document the figure as a direct raw-set recomputation and keep exact textual values sourced to decomposition packets. |
| MINOR | "Unclipped" CI presentation is technically true in the metric packets but rounded table display hides values above one. | Table caption line `125` says recovery is unclipped and may exceed one. ParoQuant CI high is `1.003769554323`, displayed as `1.00` at line `122`; M11b Granite top-5 CI high is `1.000790626547`, displayed as `1.00` at line `119`. | Use three decimals for CI endpoints near one, or add "rounded; some upper endpoints slightly exceed 1.0" to the caption. |
| MINOR | KL "select" language is inferable but not explicitly encoded in the fit artifact. | Text line `137` says all regimes "select" sublinear square-root. In `growth_model_fits.json`, there is no `selected_model` key; sublinear sqrt is the lowest-RSS fit for static, DecDEC, and M11. | Add `selected_model: sublinear_sqrt` to the fit artifact on the next regeneration, or phrase the paper as "sublinear square-root has the lowest RSS among tested fits." |

## Control-Comparison Notes

The following control comparisons checked out and should remain useful reviewer
defenses once the presentation issues above are fixed:

- M2 random-bin control beats intended position-conditioned assignment by
  `0.667548290168`; paper rounds to `0.668`.
- M10 random scale-bin assignment beats hard position-binned scales by
  `0.761487459046`; paper rounds to `0.761`.
- M11 beats random-walk protection by `1.254404832487`; paper rounds to `1.25`.
- M18 activation+K beats random coupled activation+K by `4.320788886925`; paper
  rounds to `4.32`.
- DecDEC proxy beats random reactive by `1.192570144867`; paper rounds to
  `1.19`.
- M26 stable core beats random matched core by `1.727084589454`; paper rounds
  to `1.73`.

## Saturated / Alive / Next Gate

Saturated: four-model set-leaving headline, KL means/AR decays on the Granite
dense packet, M18/DecDEC/M26 negative-with-signal controls, ParoQuant scope
baseline, and corrected Nemotron M11b top-10 packet.

Still alive: M11b as a partial budget-sensitive channel-set remedy, but only
with model-dependent framing and explicit static top-10 control context.

Highest-priority next gate: fix the paper/figure/table consistency issues
above, then rerun a clean figure-generation audit from raw metrics with a
source manifest. Do not widen the scientific claim until the presentation
layer is mechanically reproducible.
