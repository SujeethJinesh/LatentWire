# Figure Audit

Date: 2026-05-29

Scope: workshop-hardening audit of main-text figures and tables. No data points were changed.

## Inventory

| Item | File / location | Role | Status |
|---|---|---|---|
| Figure 1 | `experimental/outlier_migrate/paper/figures/set_leaving_decode_positions.pdf` | Main drift curve visual centerpiece | Regenerated with shared style and annotated 53--67% final-position band. |
| Figure 2 | `experimental/outlier_migrate/paper/figures/per_component_drift.pdf` | Layer/component decomposition | Regenerated with shared style. |
| Figure 3 | `experimental/outlier_migrate/paper/figures/method_recovery_comparison.pdf` | Recovery forest plot | Regenerated with shared style; caption already notes clipped negative whiskers and Table 1 exact values. |
| Figure 4 | `experimental/outlier_migrate/paper/figures/kl_accumulation_trajectories.pdf` | KL-growth diagnostic | Regenerated with shared style. |
| Table 1 | Main text | Method/failure taxonomy and baselines | Decimal/CIs preserved; caption states ParoQuant is a baseline and margins are descriptive unless separately tested. |
| Appendix tables | Appendix | Provenance, detailed results, novelty, surfaces, systems cost | Kept outside main-text page budget; captions are self-contained. |

## Styling Actions

- Updated `outlier_migrate_matplotlib.mplstyle` to use an Okabe-Ito colorblind-safe palette.
- Regenerated all four existing PDF figures from `generate_polish_figures.py`.
- Added a subtle grayscale-compatible band and label to Figure 1 for the final-position 53--67% drift range.
- Kept the number of main-text figures at four to preserve the 10-page main-text limit.

## Readability Checks

- All main-text figures are referenced in the prose.
- Axis labels include the relevant units or quantities.
- Figure 3 caption explicitly explains clipped negative whiskers and points readers to Table 1 for exact rounded values.
- Figures are vector PDFs with small file sizes and embedded text-compatible font settings.

## Deferred

No new mechanism schematic or decision-tree figure was added. Round 2 convergence indicated that remaining concerns are limitations rather than missing visual explanation, and a new figure would risk crowding the 10-page main text.

